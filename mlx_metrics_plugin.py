#!/usr/bin/env python3
"""
mlx_metrics_plugin.py - endpoint GET /v1/metrics per mlx_lm.server, senza
modificare il sorgente di mlx-lm (funziona con mlx-lm ufficiale da PyPI).

Si avvolgono (functools.wraps) tre punti di mlx_lm.server:

  ResponseGenerator.generate(request, args, progress_callback)
      passaggio fra il server HTTP e il thread di generazione. Restituisce
      (ctx, iteratore di Response) quando il thread di generazione prende la
      richiesta. Da qui: richiesta in coda, inizio, avanzamento del prefill
      (progress_callback), un token per Response, finish_reason, errori e
      richieste abbandonate dal client.
  APIHandler.do_GET
      risponde a /v1/metrics, il resto va all'originale.
  BatchGenerator.__init__ / close (facoltativo)
      contatori del batch (attributo privato _counters). Se mancano, nel JSON
      la sezione "batch" resta a zero; il resto funziona.

Se mlx-lm ha gia' un /v1/metrics, o se uno dei punti manca, il plugin non
viene installato e il server parte comunque.

Il server HTTP gira solo sul rank 0; il plugin non cambia le collettive.

Uso: lo carica mlx_leak_shim.py (install()); il self-test e' nelle fasi
metrics-* di `mlx_leak_shim.py --self-test`.
Variabili d'ambiente:
  MLX_METRICS_PLUGIN=0   non installa il plugin
"""

import functools
import importlib
import inspect
import json
import os
import sys
import threading
import time

TAG = "[metrics-plugin]"


def _log(msg):
    print(f"{TAG} rank={os.environ.get('MLX_RANK', '?')} {msg}",
          file=sys.stderr, flush=True)


def _tps(tokens, seconds):
    return round(tokens / seconds, 3) if seconds > 0 else 0.0


class RequestMetrics:
    """
    I token e i tempi di una richiesta. Tutti i tempi sono in secondi.

    Lo scrive solo il thread HTTP della richiesta.
    """

    def __init__(self, model, prompt_tokens, cached_tokens, queued_at=None):
        self.model = model
        self.created = int(time.time())
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.prefill_tokens = 0
        self.generation_tokens = 0
        self.finish_reason = None

        self._start = time.perf_counter()
        self._queued_at = self._start if queued_at is None else queued_at
        self._prefill_end = None
        self._first_token = None
        self._last_token = None
        self._end = None

    def prefill(self, processed, done=False):
        """Imposta i token di prompt elaborati finora."""
        self.prefill_tokens = processed
        if done and self._prefill_end is None:
            self._prefill_end = time.perf_counter()

    def token(self):
        """Conta un token generato."""
        now = time.perf_counter()
        if self._prefill_end is None:
            # Con un modello draft non arriva nessun report di avanzamento.
            self.prefill_tokens = self.prompt_tokens - self.cached_tokens
            self._prefill_end = now
        if self._first_token is None:
            self._first_token = now
        self._last_token = now
        self.generation_tokens += 1

    def finish(self, finish_reason):
        self.finish_reason = finish_reason
        self._end = time.perf_counter()

    @property
    def prefill_time(self):
        end = self._prefill_end or self._end or time.perf_counter()
        return end - self._start

    @property
    def generation_time(self):
        if self._last_token is None:
            return 0.0
        return self._last_token - self._prefill_end

    def to_dict(self):
        end = self._end or time.perf_counter()
        prefill_time = self.prefill_time
        generation_time = self.generation_time
        first_token = self._first_token
        return {
            "model": self.model,
            "created": self.created,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "prefill_tokens": self.prefill_tokens,
            "prefill_time": round(prefill_time, 3),
            "prefill_tps": _tps(self.prefill_tokens, prefill_time),
            "generation_tokens": self.generation_tokens,
            "generation_time": round(generation_time, 3),
            "generation_tps": _tps(self.generation_tokens, generation_time),
            "queue_time": round(self._start - self._queued_at, 3),
            "time_to_first_token": (
                None if first_token is None else round(first_token - self._queued_at, 3)
            ),
            "total_time": round(end - self._queued_at, 3),
        }


class ServerMetrics:
    """
    I contatori del server dall'avvio.

    Lo usano i thread HTTP e il thread di generazione (contatori del batch).
    """

    _TOTALS = (
        "prompt_tokens",
        "cached_tokens",
        "prefill_tokens",
        "prefill_time",
        "generation_tokens",
        "generation_time",
    )
    _BATCH = ("prompt_tokens", "prompt_time", "generation_tokens",
              "generation_steps", "decode_time")

    def __init__(self):
        self._lock = threading.Lock()
        self._start = time.perf_counter()
        self._queued = {}
        self._active = []
        self._last = None
        self._requests = {"received": 0, "completed": 0, "cancelled": 0, "failed": 0}
        self._totals = dict.fromkeys(self._TOTALS, 0)
        self._batch = dict.fromkeys(self._BATCH, 0)
        self._live_batch = None

    def enqueue(self, key):
        """Conta una richiesta che aspetta il thread di generazione."""
        with self._lock:
            self._requests["received"] += 1
            self._queued[key] = time.perf_counter()

    def start(self, key, model, prompt_tokens, cached_tokens):
        """Crea le metriche di una richiesta quando la generazione la prende."""
        with self._lock:
            request = RequestMetrics(
                model, prompt_tokens, cached_tokens, self._queued.pop(key, None)
            )
            self._active.append(request)
        return request

    def finish(self, request, finish_reason):
        """Aggiunge una richiesta ai totali. Puo' essere incompleta."""
        request.finish(finish_reason)
        outcome = {"cancelled": "cancelled", "error": "failed"}
        with self._lock:
            self._active.remove(request)
            self._requests[outcome.get(finish_reason, "completed")] += 1
            for k in self._TOTALS:
                self._totals[k] += getattr(request, k)
            self._last = request

    def reject(self, key):
        """Conta una richiesta fallita prima della generazione."""
        with self._lock:
            self._queued.pop(key, None)
            self._requests["failed"] += 1

    def follow_batch(self, counters):
        """Legge i contatori di un nuovo batch generator."""
        with self._lock:
            self._batch = self._batch_counters()
            self._live_batch = counters

    def end_batch(self, counters):
        """Somma ai totali i contatori di un batch generator chiuso."""
        with self._lock:
            if counters is not None and counters is self._live_batch:
                self._batch = self._batch_counters()
                self._live_batch = None

    def _batch_counters(self):
        total = dict(self._batch)
        if self._live_batch is not None:
            for k in self._BATCH:
                total[k] += getattr(self._live_batch, k, 0)
        return total

    def to_dict(self):
        with self._lock:
            requests = dict(self._requests)
            requests["queued"] = len(self._queued)
            requests["active"] = len(self._active)
            totals = dict(self._totals)
            active = [r.to_dict() for r in self._active]
            last = self._last.to_dict() if self._last is not None else None
            batch = self._batch_counters()

        prefill_time = totals["prefill_time"]
        generation_time = totals["generation_time"]
        return {
            "uptime": round(time.perf_counter() - self._start, 3),
            "requests": requests,
            "totals": {
                "prompt_tokens": totals["prompt_tokens"],
                "cached_tokens": totals["cached_tokens"],
                "prefill_tokens": totals["prefill_tokens"],
                "prefill_time": round(prefill_time, 3),
                "prefill_tps": _tps(totals["prefill_tokens"], prefill_time),
                "generation_tokens": totals["generation_tokens"],
                "generation_time": round(generation_time, 3),
                "generation_tps": _tps(totals["generation_tokens"], generation_time),
            },
            "last_request": last,
            "active_requests": active,
            "batch": {
                "prefill_tokens": batch["prompt_tokens"],
                "prefill_time": round(batch["prompt_time"], 3),
                "prefill_tps": _tps(batch["prompt_tokens"], batch["prompt_time"]),
                "generation_tokens": batch["generation_tokens"],
                "generation_steps": batch["generation_steps"],
                "decode_time": round(batch["decode_time"], 3),
                "decode_tps": _tps(batch["generation_tokens"], batch["decode_time"]),
            },
        }


METRICS = ServerMetrics()


# ---------------- aggancio a mlx_lm.server ----------------

def _sync_cached(request, ctx):
    # Nel percorso non batch prompt_cache_count arriva dopo ctx (-1 fino ad allora).
    if request.cached_tokens < 0:
        request.cached_tokens = max(getattr(ctx, "prompt_cache_count", -1), -1)


def _track(request, ctx, responses):
    """Iteratore di Response che aggiorna le metriche di una richiesta."""
    reason = None
    outcome = "error"
    try:
        for response in responses:
            _sync_cached(request, ctx)
            request.token()
            reason = getattr(response, "finish_reason", None) or reason
            yield response
        # Fine senza finish_reason: la generazione si e' fermata (ctx.stop()).
        outcome = reason or "cancelled"
    except GeneratorExit:
        # Il client ha smesso di leggere (es. disconnessione).
        outcome = reason or "cancelled"
        raise
    finally:
        request.cached_tokens = max(request.cached_tokens, 0)
        METRICS.finish(request, outcome)


def _model_id(response_generator):
    key = getattr(getattr(response_generator, "model_provider", None), "model_key", None)
    return key[0] if key else None


def _wrap_generate(orig):
    @functools.wraps(orig)
    def generate(self, request, generation_args, progress_callback=None, **kwargs):
        key = object()
        holder = []

        def progress(processed, total):
            if holder:
                _sync_cached(holder[0], ctx)
                holder[0].prefill(processed, processed >= total)
            if progress_callback is not None:
                progress_callback(processed, total)

        METRICS.enqueue(key)
        try:
            ctx, responses = orig(self, request, generation_args,
                                  progress_callback=progress, **kwargs)
        except BaseException:
            METRICS.reject(key)
            raise
        cached = getattr(ctx, "prompt_cache_count", -1)
        holder.append(METRICS.start(key, _model_id(self), len(ctx.prompt),
                                    cached if cached >= 0 else -1))
        return ctx, _track(holder[0], ctx, responses)

    generate._metrics_plugin = True
    return generate


def get_metrics(response_generator):
    """Le metriche del server. Qualunque thread puo' chiamarla."""
    import mlx.core as mx

    rg = response_generator
    provider = getattr(rg, "model_provider", None)
    cli_args = getattr(provider, "cli_args", None)
    model, adapter, draft = (tuple(getattr(provider, "model_key", None) or ())
                             + (None,) * 3)[:3]
    group = mx.distributed.init()
    mode = None
    if group.size() > 1:
        mode = "pipeline" if getattr(cli_args, "pipeline", False) else "tensor"
    device = mx.device_info()

    metrics = {"status": "ok" if getattr(rg, "is_healthy", True) else "unavailable"}
    metrics.update(METRICS.to_dict())
    metrics["model"] = {
        "id": model,
        "adapter": adapter,
        "draft": draft,
        "batchable": getattr(provider, "is_batchable", None),
    }
    metrics["distributed"] = {"size": group.size(), "rank": group.rank(), "mode": mode}
    # La memoria e la prompt cache sono solo quelle di questo nodo.
    metrics["memory"] = {
        "active": mx.get_active_memory(),
        "cache": mx.get_cache_memory(),
        "peak": mx.get_peak_memory(),
        "total": device.get("memory_size"),
        "max_recommended": device.get("max_recommended_working_set_size"),
    }
    # Il thread di generazione cancella la prompt cache quando si ferma.
    prompt_cache = getattr(rg, "prompt_cache", None)
    if prompt_cache is not None:
        stats = getattr(prompt_cache, "stats_by_type", None)
        metrics["prompt_cache"] = {
            "n_sequences": len(prompt_cache),
            "n_bytes": getattr(prompt_cache, "nbytes", None),
            "max_sequences": getattr(prompt_cache, "max_size", None),
            "max_bytes": getattr(cli_args, "prompt_cache_bytes", None),
            "by_type": stats() if callable(stats) else None,
        }
    return metrics


def handle_metrics_request(handler):
    """Risponde a GET /v1/metrics."""
    response = {
        "object": "metrics",
        "created": getattr(handler, "created", None),
        "system_fingerprint": getattr(handler, "system_fingerprint", None),
    }
    response.update(get_metrics(handler.response_generator))
    body = json.dumps(response).encode()

    handler._set_completion_headers(200)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


def _wrap_do_get(orig):
    @functools.wraps(orig)
    def do_GET(self):
        if self.path.split("?")[0] == "/v1/metrics":
            return handle_metrics_request(self)
        return orig(self)

    do_GET._metrics_plugin = True
    return do_GET


def _hook_batch_generator():
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate e' anche una funzione
    cls = getattr(generate, "BatchGenerator", None)
    if cls is None or not callable(getattr(cls, "close", None)):
        return False
    if getattr(cls.__init__, "_metrics_plugin", False):
        return True

    orig_init, orig_close = cls.__init__, cls.close

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        METRICS.follow_batch(getattr(self, "_counters", None))

    @functools.wraps(orig_close)
    def close(self, *args, **kwargs):
        METRICS.end_batch(getattr(self, "_counters", None))
        return orig_close(self, *args, **kwargs)

    __init__._metrics_plugin = True
    cls.__init__, cls.close = __init__, close
    return True


def install():
    """Installa /v1/metrics. Restituisce False (e il server parte lo stesso)
    se mlx-lm non ha i punti attesi."""
    if os.environ.get("MLX_METRICS_PLUGIN", "1") == "0":
        _log("MLX_METRICS_PLUGIN=0: /v1/metrics NON attivo")
        return False

    import mlx_lm
    from mlx_lm import server

    version = getattr(mlx_lm, "__version__", "?")
    rg, handler = server.ResponseGenerator, server.APIHandler
    if getattr(rg.generate, "_metrics_plugin", False):
        return True
    if callable(getattr(handler, "handle_metrics_request", None)):
        _log(f"mlx-lm {version} ha gia' /v1/metrics: plugin NON installato")
        return False
    try:
        params = inspect.signature(rg.generate).parameters
    except (TypeError, ValueError):
        params = {}
    if "progress_callback" not in params or not callable(getattr(handler, "do_GET", None)):
        _log(f"mlx-lm {version}: ResponseGenerator.generate/APIHandler.do_GET diversi "
             "da quelli attesi, /v1/metrics NON attivo")
        return False

    rg.generate = _wrap_generate(rg.generate)
    handler.do_GET = _wrap_do_get(handler.do_GET)
    if not _hook_batch_generator():
        _log(f"mlx-lm {version}: BatchGenerator diverso da quello atteso, "
             "sezione batch a zero")
    _log(f"/v1/metrics attivo (mlx-lm {version})")
    return True


# ---------------- self-test (nessun modello) ----------------

def _check_counters():
    """I conti di ServerMetrics: ciclo di vita, esiti, contatori del batch."""
    from types import SimpleNamespace as NS

    m = ServerMetrics()
    m.enqueue("q")
    r = m.start("q", "model", prompt_tokens=10, cached_tokens=4)
    r.prefill(6, done=True)
    for _ in range(5):
        time.sleep(0.001)
        r.token()
    m.finish(r, "stop")
    r = m.start("x", "model", prompt_tokens=8, cached_tokens=0)
    r.token()
    m.finish(r, "cancelled")
    m.enqueue("y")
    m.reject("y")
    d = m.to_dict()
    assert d["requests"] == {"received": 2, "completed": 1, "cancelled": 1, "failed": 1,
                             "queued": 0, "active": 0}, d["requests"]
    assert d["totals"]["prefill_tokens"] == 6 + 8, d["totals"]
    assert d["totals"]["generation_tokens"] == 6, d["totals"]

    first = NS(prompt_tokens=100, prompt_time=0.5, generation_tokens=20,
               generation_steps=20, decode_time=2.0)
    m.follow_batch(first)
    second = NS(prompt_tokens=50, prompt_time=0.5, generation_tokens=0,
                generation_steps=0, decode_time=0.0)
    m.follow_batch(second)
    m.end_batch(second)
    m.end_batch(first)  # gia' sommato: non conta due volte
    b = m.to_dict()["batch"]
    assert (b["prefill_tokens"], b["prefill_tps"], b["decode_tps"]) == (150, 150.0, 10.0), b


def self_test_phase(patched):
    """Vero ResponseGenerator.generate e vero APIHandler.do_GET su oggetti
    finti: tre richieste (completata, interrotta dal client, fallita), poi
    GET /v1/metrics. Senza plugin la risposta e' 404."""
    import queue
    from types import SimpleNamespace as NS
    from mlx_lm import server
    from mlx_lm.models.cache import LRUPromptCache

    if callable(getattr(server.APIHandler, "handle_metrics_request", None)):
        print("mlx-lm ha gia' /v1/metrics: il plugin non serve")
        return 0
    if patched:
        _check_counters()
        if not install():
            print("plugin non installabile")
            return 1

    PROMPT, CACHED, TOKENS = 12, 4, 6

    def generation_thread(requests):
        # Come il thread di generazione: ctx, avanzamento, Response, None.
        while True:
            item = requests.get()
            if item is None:
                return
            rqueue, request, _ = item
            if request == "fail":
                rqueue.put(RuntimeError("tokenizzazione fallita"))
                continue
            ctx = server.GenerationContext(
                has_tool_calling=False, has_thinking=False, tool_parser=None,
                text_sm=None, initial_state="normal", prompt=list(range(PROMPT)))
            rqueue.put(ctx)
            ctx.prompt_cache_count = CACHED
            rest = PROMPT - CACHED
            for done in (0, rest // 2, rest):
                rqueue.put((done, rest))
            for i in range(TOKENS):
                time.sleep(0.002)
                rqueue.put(server.Response("x", 1, 0.0,
                                           "length" if i == TOKENS - 1 else None, ()))
            rqueue.put(None)

    requests = queue.Queue()
    thread = threading.Thread(target=generation_thread, args=(requests,), daemon=True)
    thread.start()
    cache = LRUPromptCache()
    rg = NS(requests=requests, generation_available=lambda: True,
            _await_response=lambda q: q.get(timeout=5), is_healthy=True, prompt_cache=cache,
            model_provider=NS(model_key=("fake-model", None, None), is_batchable=True,
                              cli_args=NS(pipeline=False, prompt_cache_bytes=None)))
    generate = server.ResponseGenerator.generate

    # Completata: il client legge tutto.
    ctx, responses = generate(rg, "ok", None, progress_callback=lambda *a: None)
    n = sum(1 for _ in responses)
    # Interrotta: il client legge 2 token e se ne va.
    ctx, responses = generate(rg, "ok", None, progress_callback=lambda *a: None)
    next(responses), next(responses)
    responses.close()
    # Fallita prima della generazione.
    try:
        generate(rg, "fail", None, progress_callback=lambda *a: None)
    except RuntimeError:
        pass
    requests.put(None)

    class Handler:
        path, created, system_fingerprint, response_generator = "/v1/metrics", 0, "fp", rg
        status, body = None, b""

        def _set_completion_headers(self, code):
            self.status = code

        def send_header(self, *args):
            pass

        def end_headers(self):
            pass

        @property
        def wfile(self):
            return self

        def write(self, data):
            self.body += data

        def flush(self):
            pass

    h = Handler()
    server.APIHandler.do_GET(h)
    if h.status != 200:
        print(f"GET /v1/metrics: HTTP {h.status}")
        print("NESSUNA METRICA: mlx-lm non ha /v1/metrics")
        return 3

    d = json.loads(h.body)
    last = d["last_request"]
    print(f"GET /v1/metrics: requests={d['requests']}")
    print(f"ultima richiesta: prompt={last['prompt_tokens']} cached={last['cached_tokens']} "
          f"prefill={last['prefill_tokens']} token={last['generation_tokens']} "
          f"fine={last['finish_reason']}")
    expected = {"received": 3, "completed": 1, "cancelled": 1, "failed": 1,
                "queued": 0, "active": 0}
    first_ok = n == TOKENS and d["totals"]["generation_tokens"] == TOKENS + 2
    last_ok = (last["prompt_tokens"], last["cached_tokens"], last["prefill_tokens"],
               last["generation_tokens"], last["finish_reason"]) == \
        (PROMPT, CACHED, PROMPT - CACHED, 2, "cancelled")
    if d["requests"] != expected or not first_ok or not last_ok or d["object"] != "metrics":
        print("METRICHE ERRATE")
        return 5
    print("OK: metriche corrette")
    return 0
