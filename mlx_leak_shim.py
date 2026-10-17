#!/usr/bin/env python3
"""

execute on both mac nodes: /Volumes/Home/Alvise/miniforge3/envs/mlx/bin/python mlx_leak_shim.py --self-test

expected output:

--- fase unpatched ---
    CRASH allo step 10395: [metal::malloc] Resource limit (499000) exceeded.
--- fase patched ---
    OK: 12000 step x 48 layer, valori corretti
    [leak-shim] rank=? patch applicata (mlx-lm 0.32.0+metrics)
--- fase stop-unpatched ---
    token generati rank0=5 rank1=200, collettive stop rank0=0 rank1=0, errori=['NotImplementedError', None]
    DESYNC: i rank escono da punti diversi
--- fase stop-patched ---
    token generati rank0=9 rank1=9, collettive stop rank0=2 rank1=2, errori=[None, None]
    OK: entrambi i rank fermi al token 9
    [leak-shim] rank=? patch stop applicata (mlx-lm 0.32.0+metrics, sync ogni 8 token)
PASS: leak riprodotto senza patch, assente con la patch
PASS: desync dei rank allo stop riprodotto senza patch, assente con la patch

----------------

mlx_leak_shim.py - avvia mlx_lm.server con una patch al leak di buffer Metal
dei modelli ibridi (ArraysCache: qwen3_5 / Qwen3.6 / Qwen3.8, qwen3_next, ...).

Causa (mlx-lm 0.31.3): ArraysCache.advance() esegue `-= N` lazy su
left_padding/lengths a ogni step; per i layer il cui metadata non viene mai
letto la catena non viene mai valutata e ogni nodo trattiene un buffer Metal
-> "[metal::malloc] Resource limit (499000) exceeded" dopo ~10k token.

Fix: dopo ogni BatchGenerator.next() si valutano (async) solo left_padding e
lengths delle cache avanzate nello step. La catena collassa; i valori non cambiano.

----------------

Seconda patch (stop distribuito, mlx-lm 0.32.0): con --kv-bits le richieste
non sono batchable e passano da ResponseGenerator._serve_single. Se il client
si disconnette a meta' stream (BrokenPipeError), l'handler HTTP chiama
ctx.stop(), ma il flag esiste solo sul rank 0: il rank 0 esce dal ciclo
(raise NotImplementedError) e torna ad aspettare richieste con all_sum(0),
mentre il rank 1 continua a generare con le all_sum del modello. Le collettive
jaccl si desincronizzano e muoiono tutti i rank (exit 255).

Fix: lo stop viene condiviso dal rank 0 a tutti i rank con _share_object ogni
MLX_STOP_SHIM_EVERY token; tutti i rank escono dallo stesso token, salvano la
cache e tornano allineati in attesa della richiesta successiva.

Uso:
  python mlx_leak_shim.py [argomenti di mlx_lm.server]
  python mlx_leak_shim.py --self-test
Variabili d'ambiente:
  MLX_LEAK_SHIM=0         non applica la patch al leak (per confronti A/B)
  MLX_STOP_SHIM=0         non applica la patch allo stop distribuito
  MLX_STOP_SHIM_EVERY=n   ogni quanti token si condivide lo stop (default 8)
  MLX_LEAK_SHIM_STRICT=1  esce con errore se il codice installato non e' quello atteso
"""

import functools
import inspect
import os
import subprocess
import sys
import textwrap
import threading

TAG = "[leak-shim]"
EXPECTED = ("self.lengths -= N", "self.left_padding -= N")

# Blocco di ResponseGenerator._serve_single sostituito dalla patch stop
STOP_OLD = """\
                if ctx._should_stop:
                    if self._is_distributed:
                        raise NotImplementedError()
                    break

                if finish_reason is not None:
                    break
"""
STOP_NEW = """\
                if finish_reason is not None:
                    break

                # [stop-shim] ctx.stop() lo vede solo il rank 0: lo si
                # condivide ogni {every} token cosi' tutti i rank escono
                # dallo stesso token
                if self._is_distributed:
                    if (len(cache_key) - len(prompt) - 1) % {every} == 0:
                        if self._share_object(True if ctx._should_stop else None):
                            break
                elif ctx._should_stop:
                    break
"""

_pending = {}
_lock = threading.Lock()


def _log(msg):
    print(f"{TAG} rank={os.environ.get('MLX_RANK', '?')} {msg}",
          file=sys.stderr, flush=True)


def _flush_pending():
    import mlx.core as mx
    with _lock:
        caches = list(_pending.values())
        _pending.clear()
    arrays = [a for c in caches
              for a in (getattr(c, "left_padding", None), getattr(c, "lengths", None))
              if isinstance(a, mx.array)]
    if arrays:
        mx.async_eval(arrays)


def _mismatch(msg):
    if os.environ.get("MLX_LEAK_SHIM_STRICT") == "1":
        _log(msg + " (STRICT: esco)")
        sys.exit(2)
    _log(msg)
    return False


def apply_patch():
    if os.environ.get("MLX_LEAK_SHIM", "1") == "0":
        _log("MLX_LEAK_SHIM=0: patch NON applicata")
        return False

    import mlx_lm
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import ArraysCache

    if getattr(ArraysCache.advance, "_leak_shim", False):
        return True

    version = getattr(mlx_lm, "__version__", "?")
    try:
        src = inspect.getsource(ArraysCache.advance)
    except (OSError, TypeError):
        src = ""
    if not all(p in src for p in EXPECTED) or not callable(getattr(BatchGenerator, "next", None)):
        return _mismatch(f"mlx-lm {version}: codice diverso da quello atteso, patch NON applicata")

    orig_advance = ArraysCache.advance

    @functools.wraps(orig_advance)
    def advance(self, N):
        orig_advance(self, N)
        # Senza left_padding/lengths (cache non batch, es. con --kv-bits)
        # advance() non crea catene e BatchGenerator.next() non viene mai
        # chiamato: registrarle tratterrebbe le cache per sempre.
        if self.left_padding is None and self.lengths is None:
            return
        with _lock:
            _pending[id(self)] = self

    orig_next = BatchGenerator.next

    @functools.wraps(orig_next)
    def next_(self, *args, **kwargs):
        out = orig_next(self, *args, **kwargs)
        _flush_pending()
        return out

    advance._leak_shim = True
    next_._leak_shim = True
    ArraysCache.advance = advance
    BatchGenerator.next = next_
    _log(f"patch applicata (mlx-lm {version})")
    return True


def apply_stop_patch():
    if os.environ.get("MLX_STOP_SHIM", "1") == "0":
        _log("MLX_STOP_SHIM=0: patch stop NON applicata")
        return False

    import mlx_lm
    from mlx_lm import server

    rg = server.ResponseGenerator
    if getattr(rg._serve_single, "_stop_shim", False):
        return True

    version = getattr(mlx_lm, "__version__", "?")
    every = os.environ.get("MLX_STOP_SHIM_EVERY", "8")
    if not every.isdigit() or int(every) < 1:
        _log(f"MLX_STOP_SHIM_EVERY={every!r} non valido, uso 8")
        every = "8"
    try:
        src = inspect.getsource(rg._serve_single)
    except (OSError, TypeError):
        src = ""
    if src.count(STOP_OLD) != 1 or not callable(getattr(rg, "_share_object", None)):
        return _mismatch(f"mlx-lm {version}: _serve_single diverso da quello atteso, "
                         "patch stop NON applicata")

    # Ricompila _serve_single nel namespace di mlx_lm.server con il blocco
    # sostituito, cosi' risolve gli stessi nomi (stream_generate, Response, ...)
    src = textwrap.dedent(src.replace(STOP_OLD, STOP_NEW.format(every=int(every))))
    ns = {}
    exec(compile(src, f"<stop-shim {server.__file__}>", "exec"), vars(server), ns)
    serve_single = ns["_serve_single"]
    serve_single.__qualname__ = rg._serve_single.__qualname__
    serve_single._stop_shim = True
    rg._serve_single = serve_single
    _log(f"patch stop applicata (mlx-lm {version}, sync ogni {every} token)")
    return True


# ---------------- self-test (nessun modello, nessun jaccl) ----------------
N_LAYERS = 48      # layer lineari di Qwen3.8-27B (64 x 3/4)
N_STEPS = 12_000   # oltre 499000/48 ~ 10.400


def _phase(patched):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    if patched and not apply_patch():
        print("patch non applicabile")
        return 1

    caches = []
    for _ in range(N_LAYERS):
        c = ArraysCache(2)
        c.left_padding = mx.array([0], dtype=mx.int32)
        caches.append(c)

    step = 0
    try:
        for step in range(1, N_STEPS + 1):
            for c in caches:
                c.advance(1)
            if patched:
                _flush_pending()
    except RuntimeError as e:
        print(f"CRASH allo step {step}: {e}")
        return 3

    mx.eval([c.left_padding for c in caches])
    vals = {c.left_padding[0].item() for c in caches}
    if vals != {-N_STEPS}:
        print(f"VALORI ERRATI: {vals} (atteso {-N_STEPS})")
        return 5
    print(f"OK: {N_STEPS} step x {N_LAYERS} layer, valori corretti")
    return 0


STOP_TOKENS = 200  # token generati dal finto modello
STOP_AFTER = 5     # il client del rank 0 si disconnette dopo 5 token


def _stop_phase(patched):
    """Due rank simulati con due thread che eseguono il vero _serve_single su
    un finto modello; le collettive sono un rendezvous fra i thread."""
    import queue
    from types import SimpleNamespace as NS
    from mlx_lm import server

    if patched and not apply_stop_patch():
        print("patch non applicabile")
        return 1

    def fake_stream_generate(**kw):
        n = kw["max_tokens"]
        for i in range(n):
            yield NS(text="x", token=1, logprobs={1: NS(item=lambda: 0.0)},
                     finish_reason="length" if i == n - 1 else None)

    server.stream_generate = fake_stream_generate
    server.make_prompt_cache = lambda model: []
    server._make_sampler = lambda args, tokenizer: None
    server._make_logits_processors = lambda args: None

    barrier = threading.Barrier(2, timeout=5)
    slot = [None]
    shares, tokens, errors = [0, 0], [0, 0], [None, None]

    def share(rank, obj):  # come _share_object: vale l'oggetto del rank 0
        shares[rank] += 1
        if rank == 0:
            slot[0] = obj
        barrier.wait()
        val = slot[0]
        barrier.wait()
        return val

    class Client(queue.Queue):  # handler HTTP: il rank 0 perde il client
        def __init__(self, rank):
            super().__init__()
            self.rank, self.ctx = rank, None

        def put(self, item, *args, **kwargs):
            if isinstance(item, server.GenerationContext):
                self.ctx = item
            elif isinstance(item, server.Response):
                tokens[self.rank] += 1
                if self.rank == 0 and tokens[0] == STOP_AFTER:
                    self.ctx.stop()
            elif isinstance(item, Exception):
                errors[self.rank] = item
            super().put(item, *args, **kwargs)

    def rank_self(rank):
        tok = NS(has_thinking=False, has_tool_calling=False, tool_parser=None)
        stop_sequences = NS(matcher=lambda: NS(advance=lambda t: False))
        metric = NS(prefill=lambda *a: None, token=lambda: None, finish_reason=None)
        return NS(
            _is_distributed=True,
            _share_object=lambda obj: share(rank, obj),
            model_provider=NS(model=None, tokenizer=tok, draft_model=None,
                              model_key=("fake",)),
            _tokenize=lambda t, r, a: (list(range(10)), None, None, "normal"),
            _make_state_machine=lambda *a: (stop_sequences, None),
            _log_cache_stats=lambda: None,
            prompt_cache=NS(fetch_nearest_cache=lambda k, p: (None, p),
                            insert_cache=lambda *a, **k: None),
            metrics=NS(start=lambda *a: metric, finish=lambda *a: None,
                       reject=lambda *a: None),
            cli_args=NS(prefill_step_size=256, kv_bits=8, kv_group_size=64,
                        quantized_kv_start=0),
        )

    args = NS(seed=None, max_tokens=STOP_TOKENS, num_draft_tokens=None,
              top_logprobs=0, stop_words=[])
    threads = [threading.Thread(target=server.ResponseGenerator._serve_single,
                                args=(rank_self(r), (Client(r), None, args), None))
               for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    errs = [type(e).__name__ if e else None for e in errors]
    print(f"token generati rank0={tokens[0]} rank1={tokens[1]}, "
          f"collettive stop rank0={shares[0]} rank1={shares[1]}, errori={errs}")
    if any(t.is_alive() for t in threads):
        print("DEADLOCK")
        return 3
    if tokens[0] != tokens[1] or shares[0] != shares[1] or any(errors):
        print("DESYNC: i rank escono da punti diversi")
        return 3
    if tokens[0] >= STOP_TOKENS:
        print("lo stop non e' stato rispettato")
        return 5
    print(f"OK: entrambi i rank fermi al token {tokens[0]}")
    return 0


def _self_test():
    results = {}
    for phase in ("unpatched", "patched", "stop-unpatched", "stop-patched"):
        print(f"--- fase {phase} ---", flush=True)
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--self-test-phase", phase],
                capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print("    TIMEOUT")
            results[phase] = None
            continue
        for line in (p.stdout + p.stderr).strip().splitlines()[-5:]:
            print("   ", line)
        results[phase] = p.returncode

    rc = 0
    for name, what in (("", "leak"), ("stop-", "desync dei rank allo stop")):
        unpatched, patched = results.get(name + "unpatched"), results.get(name + "patched")
        if unpatched == 3 and patched == 0:
            print(f"PASS: {what} riprodotto senza patch, assente con la patch")
        elif unpatched == 0 and patched == 0:
            print(f"INCONCLUSIVO: la simulazione non riproduce {what} su questa versione")
            rc = rc or 4
        else:
            print(f"FAIL ({what}): codici di uscita {unpatched}/{patched}")
            rc = 1
    return rc


PHASES = ("patched", "unpatched", "stop-patched", "stop-unpatched")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--self-test"]:
        return _self_test()
    if args[:1] == ["--self-test-phase"]:
        if len(args) < 2 or args[1] not in PHASES:
            print(f"uso: --self-test-phase {'|'.join(PHASES)}", file=sys.stderr)
            return 2
        if args[1].startswith("stop-"):
            return _stop_phase(args[1] == "stop-patched")
        return _phase(args[1] == "patched")

    apply_patch()
    apply_stop_patch()
    sys.argv = ["mlx_lm.server"] + args
    try:
        from mlx_lm.server import main as server_main
    except ImportError:
        import runpy
        runpy.run_module("mlx_lm.server", run_name="__main__", alter_sys=True)
        return 0
    server_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
