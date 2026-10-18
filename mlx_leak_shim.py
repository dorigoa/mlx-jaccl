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
--- fase keepalive-unpatched ---
    token letti 106/106, keepalive prima della disconnessione 0, BrokenPipe=True, stop=True
    SILENZIO: disconnessione rilevata solo a generazione finita
--- fase keepalive-patched ---
    token letti 15/106, keepalive prima della disconnessione 1, BrokenPipe=True, stop=True
    OK: disconnessione rilevata 0 token dopo
    [leak-shim] rank=? patch keepalive applicata (mlx-lm 0.32.0+metrics, dopo 0.05s di silenzio)
--- fase cache-unpatched ---
    prompt cache: 10 sequenze, 10.73 GB (limite 4.83 GB)
    SFORA: il limite in byte non viene applicato
--- fase cache-patched ---
    prompt cache: 4 sequenze, 4.29 GB (limite 4.83 GB)
    OK: limite rispettato
    [leak-shim] rank=? patch limite cache applicata (mlx-lm 0.32.0+metrics)
    [leak-shim] rank=? prompt cache limitata a 4.83 GB
--- fase pin-unpatched ---
    /v1/models: ['mlx-community/GLM-4.7-Flash-8bit', 'mlx-community/Qwen3.8-27B-4bit', 'mlx-community/Qwen3.8-27B-8bit']; richiesta per mlx-community/GLM-4.7-Flash-8bit servita con mlx-community/GLM-4.7-Flash-8bit
    il client vede e puo' caricare modelli diversi da quello di avvio
--- fase pin-patched ---
    /v1/models: ['mlx-community/Qwen3.8-27B-4bit']; richiesta per mlx-community/GLM-4.7-Flash-8bit servita con mlx-community/Qwen3.8-27B-4bit
    OK: il client vede e usa solo il modello di avvio
    [leak-shim] rank=? patch modello bloccato applicata (mlx-lm 0.32.0+metrics)
    [leak-shim] rank=? richiesto il modello 'mlx-community/GLM-4.7-Flash-8bit': servito con 'mlx-community/Qwen3.8-27B-4bit'
--- fase kvbits-unpatched ---
    CRASH di glm4_moe_lite con --kv-bits: 'list' object has no attribute 'swapaxes'
--- fase kvbits-patched ---
    cache dopo la generazione: {'glm4_moe_lite': ['KVCache'], 'llama': ['QuantizedKVCache']}
    OK: GLM genera a precisione piena, Llama resta quantizzato
    [leak-shim] rank=? patch kv-bits applicata (mlx-lm 0.32.0+metrics)
    [leak-shim] rank=? glm4_moe_lite: attenzione MLA, la cache KV resta a precisione piena (--kv-bits ignorato)
PASS: leak riprodotto senza patch, assente con la patch
PASS: desync dei rank allo stop riprodotto senza patch, assente con la patch
PASS: silenzio durante le tool call riprodotto senza patch, assente con la patch
PASS: sforamento del limite della prompt cache riprodotto senza patch, assente con la patch
PASS: cambio di modello su richiesta del client riprodotto senza patch, assente con la patch
PASS: crash dei modelli MLA con --kv-bits riprodotto senza patch, assente con la patch

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

----------------

Terza patch (keepalive SSE, mlx-lm 0.32.0): mentre il modello scrive una tool
call APIHandler.handle_completion non manda nulla al client (il testo della
tool call parte tutto nel chunk finale). Con tool call lunghe la connessione
resta muta per minuti, proxy/socat/client la chiudono per inattivita' e il
server se ne accorge solo alla fine (BrokenPipeError a server.py:1811), dopo
aver generato per nessuno; intanto il retry del client resta in coda.

Fix: se in streaming non si e' scritto nulla da MLX_KEEPALIVE_SHIM_EVERY
secondi si manda un commento SSE ": keepalive" (lo stesso formato che il
server usa gia' durante il prefill). La connessione non e' mai inattiva e una
disconnessione emerge subito, cosi' la patch stop ferma la generazione.

----------------

Quarta patch (limite in byte della prompt cache, mlx-lm 0.32.0): il server crea
LRUPromptCache passando solo --prompt-cache-size; --prompt-cache-bytes viene
applicato solo nel percorso batch (trim_to dopo ogni insert). Con --kv-bits le
richieste passano da _serve_single e la cache cresce fino a 10 sequenze di
qualsiasi dimensione: con contesti da 60k token ~10.7 GB per rank, e la
richiesta successiva va in OOM (Metal) durante il prefill.

Fix: si imposta max_bytes della LRUPromptCache a --prompt-cache-bytes, cosi'
insert_cache scarta le voci piu' vecchie in entrambi i percorsi. La dimensione
e' quella delle cache locali, uguale sui due rank: le scelte di eviction (e
quindi le cache trovate) restano identiche.

----------------

Quinta patch (modello bloccato, mlx-lm 0.32.0): /v1/models elenca tutti i
modelli MLX nella cache di Hugging Face, non quello in esecuzione, e il server
carica (o scarica) qualunque modello il client chieda nel campo "model". Un
client che prende il primo della lista (es. Pi con GLM-4.7-Flash-8bit) fa
scaricare il modello avviato e caricarne un altro, che puo' non stare in memoria.

Fix: /v1/models elenca solo il modello passato con --model, e ogni richiesta
viene servita con quel modello qualunque nome indichi (con un avviso nel log
la prima volta che arriva un nome diverso).

----------------

Sesta patch (--kv-bits sui modelli MLA, mlx-lm 0.32.0): quando la cache supera
--quantized-kv-start (default 5000 token) generate_step la converte in
QuantizedKVCache, il cui update_and_fetch restituisce tuple (pesi, scale, bias).
L'attenzione di Qwen/Llama passa da scaled_dot_product_attention, che le
gestisce; i modelli che mettono in cache il latente MLA (glm4_moe_lite =
GLM-4.7-Flash, deepseek_v3, kimi, ...) lo usano invece come array
(k_pe.swapaxes) e ogni richiesta oltre i 5000 token muore con
"AttributeError: 'list' object has no attribute 'swapaxes'".

Fix: per questi modelli generate_step riceve kv_bits=None e la cache resta a
precisione piena (il latente MLA e' gia' piccolo). --kv-bits resta sulla riga
di comando, quindi le richieste continuano a passare da _serve_single, dove
agiscono le patch stop e limite cache. Gli altri modelli restano quantizzati.

Uso:
  python mlx_leak_shim.py [argomenti di mlx_lm.server]
  python mlx_leak_shim.py --self-test
Variabili d'ambiente:
  MLX_LEAK_SHIM=0              non applica la patch al leak (per confronti A/B)
  MLX_STOP_SHIM=0              non applica la patch allo stop distribuito
  MLX_STOP_SHIM_EVERY=n        ogni quanti token si condivide lo stop (default 8)
  MLX_KEEPALIVE_SHIM=0         non applica la patch keepalive
  MLX_KEEPALIVE_SHIM_EVERY=s   secondi di silenzio prima di un keepalive (default 5)
  MLX_CACHE_SHIM=0             non applica il limite in byte alla prompt cache
  MLX_PIN_SHIM=0               non blocca il modello (torna il comportamento originale)
  MLX_KVBITS_SHIM=0            quantizza la cache KV anche dei modelli MLA (crash oltre 5000 token)
  MLX_LEAK_SHIM_STRICT=1       esce con errore se il codice installato non e' quello atteso
"""

import functools
import hashlib
import importlib
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

# Blocchi di APIHandler.handle_completion modificati dalla patch keepalive
KEEPALIVE_BLOCKS = (
    ("""\
        try:
            for gen in response:
""", """\
        _ka_last = time.monotonic()  # [keepalive-shim]
        try:
            for gen in response:
"""),
    ("""\
                    self.wfile.flush()
                    reasoning_text = ""
""", """\
                    self.wfile.flush()
                    _ka_last = time.monotonic()  # [keepalive-shim]
                    reasoning_text = ""
"""),
    ("""\
                if gen.finish_reason is not None:
                    finish_reason = gen.finish_reason
""", """\
                # [keepalive-shim] durante le tool call non si scrive nulla: un
                # commento SSE dopo {every}s di silenzio tiene viva la
                # connessione e fa emergere subito la disconnessione del client
                if self.stream and time.monotonic() - _ka_last >= {every}:
                    self.wfile.write(b": keepalive\\n\\n")
                    self.wfile.flush()
                    _ka_last = time.monotonic()

                if gen.finish_reason is not None:
                    finish_reason = gen.finish_reason
"""),
)

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


def _patch_method(cls, name, blocks, what):
    """Sostituisce nel sorgente di cls.<name> ogni blocco (old, new), dove ogni
    old deve comparire una sola volta, e ricompila il metodo nel namespace del
    suo modulo, cosi' risolve gli stessi nomi (stream_generate, Response, ...)."""
    import mlx_lm

    version = getattr(mlx_lm, "__version__", "?")
    method = getattr(cls, name)
    try:
        src = inspect.getsource(method)
    except (OSError, TypeError):
        src = ""
    if not src or any(src.count(old) != 1 for old, _ in blocks):
        return _mismatch(f"mlx-lm {version}: {name} diverso da quello atteso, "
                         f"patch {what} NON applicata")

    for old, new in blocks:
        src = src.replace(old, new)
    module = sys.modules[method.__module__]
    ns = {}
    exec(compile(textwrap.dedent(src), f"<{what}-shim {module.__file__}>", "exec"),
         vars(module), ns)
    patched = ns[name]
    patched.__qualname__ = method.__qualname__
    patched._shim = what
    setattr(cls, name, patched)
    return True


def _env_number(var, default, cast):
    value = os.environ.get(var, str(default))
    try:
        if cast(value) > 0:
            return cast(value)
    except ValueError:
        pass
    _log(f"{var}={value!r} non valido, uso {default}")
    return default


def apply_stop_patch():
    if os.environ.get("MLX_STOP_SHIM", "1") == "0":
        _log("MLX_STOP_SHIM=0: patch stop NON applicata")
        return False

    import mlx_lm
    from mlx_lm import server

    rg = server.ResponseGenerator
    if getattr(rg._serve_single, "_shim", None) == "stop":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    every = _env_number("MLX_STOP_SHIM_EVERY", 8, int)
    if not callable(getattr(rg, "_share_object", None)):
        return _mismatch(f"mlx-lm {version}: manca _share_object, patch stop NON applicata")
    if not _patch_method(rg, "_serve_single", [(STOP_OLD, STOP_NEW.format(every=every))], "stop"):
        return False
    _log(f"patch stop applicata (mlx-lm {version}, sync ogni {every} token)")
    return True


def apply_keepalive_patch():
    if os.environ.get("MLX_KEEPALIVE_SHIM", "1") == "0":
        _log("MLX_KEEPALIVE_SHIM=0: patch keepalive NON applicata")
        return False

    import mlx_lm
    from mlx_lm import server

    handler = server.APIHandler
    if getattr(handler.handle_completion, "_shim", None) == "keepalive":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    every = _env_number("MLX_KEEPALIVE_SHIM_EVERY", 5, float)
    blocks = [(old, new.format(every=every)) for old, new in KEEPALIVE_BLOCKS]
    if not _patch_method(handler, "handle_completion", blocks, "keepalive"):
        return False
    _log(f"patch keepalive applicata (mlx-lm {version}, dopo {every:g}s di silenzio)")
    return True


def apply_cache_bytes_patch():
    if os.environ.get("MLX_CACHE_SHIM", "1") == "0":
        _log("MLX_CACHE_SHIM=0: patch limite cache NON applicata")
        return False

    import mlx_lm
    from mlx_lm import server
    from mlx_lm.models.cache import LRUPromptCache

    rg = server.ResponseGenerator
    if getattr(rg.__init__, "_shim", None) == "cache-bytes":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    try:
        src = inspect.getsource(LRUPromptCache.insert_cache)
        params = list(inspect.signature(rg.__init__).parameters)
    except (OSError, TypeError, ValueError):
        src, params = "", []
    if "self._n_bytes > self.max_bytes" not in src or \
            params[:3] != ["self", "model_provider", "prompt_cache"]:
        return _mismatch(f"mlx-lm {version}: LRUPromptCache/ResponseGenerator diversi da "
                         "quelli attesi, patch limite cache NON applicata")

    orig_init = rg.__init__

    # Il limite va impostato prima di orig_init, che avvia il thread di generazione
    @functools.wraps(orig_init)
    def __init__(self, model_provider, prompt_cache, *args, **kwargs):
        limit = getattr(model_provider.cli_args, "prompt_cache_bytes", None)
        if limit is not None:
            prompt_cache.max_bytes = limit
            _log(f"prompt cache limitata a {limit / 1e9:.2f} GB")
        orig_init(self, model_provider, prompt_cache, *args, **kwargs)

    __init__._shim = "cache-bytes"
    rg.__init__ = __init__
    _log(f"patch limite cache applicata (mlx-lm {version})")
    return True


def apply_pin_patch():
    """Va applicata dopo apply_keepalive_patch, perche' avvolge handle_completion."""
    if os.environ.get("MLX_PIN_SHIM", "1") == "0":
        _log("MLX_PIN_SHIM=0: modello NON bloccato")
        return False

    import json
    import mlx_lm
    from mlx_lm import server

    handler = server.APIHandler
    if getattr(handler.handle_models_request, "_shim", None) == "pin":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    try:
        src = inspect.getsource(handler.do_POST)
    except (OSError, TypeError):
        src = ""
    if 'self.requested_model = self.body.get("model"' not in src or \
            not callable(getattr(handler, "handle_models_request", None)):
        return _mismatch(f"mlx-lm {version}: APIHandler diverso da quello atteso, "
                         "patch modello bloccato NON applicata")

    orig_completion = handler.handle_completion
    orig_models = handler.handle_models_request
    warned = set()

    @functools.wraps(orig_completion)
    def handle_completion(self, *args, **kwargs):
        pinned = self.response_generator.cli_args.model
        if pinned and self.requested_model != pinned:
            if self.requested_model not in warned and self.requested_model != "default_model":
                warned.add(self.requested_model)
                _log(f"richiesto il modello {self.requested_model!r}: servito con {pinned!r}")
            self.requested_model = pinned
        return orig_completion(self, *args, **kwargs)

    @functools.wraps(orig_models)
    def handle_models_request(self):
        pinned = self.response_generator.cli_args.model
        if not pinned:
            return orig_models(self)
        self._set_completion_headers(200)
        self.end_headers()
        parts = self.path.split("/")
        wanted = "/".join(parts[3:]) if len(parts) > 3 else None
        models = []
        if wanted in (None, pinned):
            models.append({"id": pinned, "object": "model", "created": self.created})
        self.wfile.write(json.dumps({"object": "list", "data": models}).encode())
        self.wfile.flush()

    handle_models_request._shim = "pin"
    handler.handle_completion = handle_completion
    handler.handle_models_request = handle_models_request
    _log(f"patch modello bloccato applicata (mlx-lm {version})")
    return True


# Attenzione che mette in cache il latente MLA e ne usa il risultato come array
LATENT_CACHE = "update_and_fetch(kv_latent"
_latent_models = {}


def _caches_latent(model):
    """True se un modulo del modello mette in cache il latente MLA, che con una
    QuantizedKVCache tornerebbe come tupla. Calcolato una volta per classe."""
    cls = type(model)
    if cls not in _latent_models:
        found = False
        for mod_cls in {type(m) for m in model.modules()}:
            try:
                found = LATENT_CACHE in inspect.getsource(mod_cls)
            except (OSError, TypeError):
                continue
            if found:
                break
        _latent_models[cls] = found
        if found:
            name = getattr(model, "model_type", cls.__module__)
            _log(f"{name}: attenzione MLA, la cache KV resta a precisione piena "
                 "(--kv-bits ignorato)")
    return _latent_models[cls]


def apply_kvbits_patch():
    if os.environ.get("MLX_KVBITS_SHIM", "1") == "0":
        _log("MLX_KVBITS_SHIM=0: patch kv-bits NON applicata")
        return False

    import mlx_lm
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate e' anche una funzione

    if getattr(generate.generate_step, "_shim", None) == "kvbits":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    try:
        params = inspect.signature(generate.generate_step).parameters
        src = inspect.getsource(generate.stream_generate)
    except (OSError, TypeError, ValueError):
        params, src = {}, ""
    if "kv_bits" not in params or "generate_step(prompt, model" not in src:
        return _mismatch(f"mlx-lm {version}: generate_step/stream_generate diversi da "
                         "quelli attesi, patch kv-bits NON applicata")

    orig_step = generate.generate_step

    # stream_generate risolve generate_step nel namespace del modulo a ogni
    # chiamata: basta sostituirlo li'. kv_bits e' keyword-only.
    @functools.wraps(orig_step)
    def generate_step(prompt, model, *args, **kwargs):
        if kwargs.get("kv_bits") is not None and _caches_latent(model):
            kwargs["kv_bits"] = None
        return orig_step(prompt, model, *args, **kwargs)

    generate_step._shim = "kvbits"
    generate.generate_step = generate_step
    _log(f"patch kv-bits applicata (mlx-lm {version})")
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


KA_REASONING = 5      # token di ragionamento (scritti subito al client)
KA_TOOL_TOKENS = 100  # token della tool call (10 ms l'uno, nessuna scrittura)
KA_DISCONNECT = 10    # il client se ne va al 10o token della tool call


def _keepalive_phase(patched):
    """Vero APIHandler.handle_completion su un finto handler: il modello scrive
    una lunga tool call e a meta' il client se ne va. Con la patch la
    disconnessione deve emergere subito, non a generazione finita."""
    import time
    from types import SimpleNamespace as NS
    from mlx_lm import server

    os.environ["MLX_KEEPALIVE_SHIM_EVERY"] = "0.05"
    if patched and not apply_keepalive_patch():
        print("patch non applicabile")
        return 1

    class FakeTextStateMachine:  # lo stato e' la prima lettera del testo
        STATES = {"R": "reasoning", "T": "tool", "N": "normal"}

        @staticmethod
        def step(sm_state, text):
            return sm_state, text, FakeTextStateMachine.STATES[text[0]]

        @staticmethod
        def flush(sm_state):
            return sm_state, "", "normal"

        @staticmethod
        def discard(sm_state):
            return sm_state, "normal"

    server.TextStateMachine = FakeTextStateMachine
    server.ToolCallFormatter = lambda parser, tools, stream: (lambda calls: calls)

    total = KA_REASONING + KA_TOOL_TOKENS + 1
    pulled = [0]
    disconnected = [False]

    def generation():
        seq = ["R"] * KA_REASONING + ["T"] * KA_TOOL_TOKENS + ["N"]
        for i, text in enumerate(seq):
            if text == "T":
                time.sleep(0.01)
            pulled[0] += 1
            if pulled[0] == KA_REASONING + KA_DISCONNECT:
                disconnected[0] = True
            yield NS(text=text, token=1, logprob=0.0, top_tokens=(),
                     finish_reason="stop" if i == total - 1 else None)

    class WFile:  # socket verso il client: dopo la disconnessione ogni write fallisce
        keepalives = 0

        def write(self, data):
            if disconnected[0]:
                raise BrokenPipeError(32, "Broken pipe")
            if data.startswith(b": keepalive"):
                self.keepalives += 1
            return len(data)

        def flush(self):
            pass

    ctx = server.GenerationContext(
        has_tool_calling=True, has_thinking=True, tool_parser=None,
        text_sm=NS(make_state=lambda s: s), initial_state="reasoning", prompt=[0])

    class Handler:
        stream = True
        stream_options = None
        logprobs = False
        top_logprobs = 0
        wfile = WFile()
        response_generator = NS(generate=lambda request, args, progress_callback: (ctx, generation()))

        def __getattr__(self, name):  # gli altri parametri della richiesta
            return None

        def _set_stream_headers(self, code):
            pass

        def end_headers(self):
            pass

        def generate_response(self, *args, **kwargs):
            return {}

    handler = Handler()
    try:
        server.APIHandler.handle_completion(handler, NS(tools=None), [])
        broken = False
    except BrokenPipeError:
        broken = True

    print(f"token letti {pulled[0]}/{total}, keepalive prima della disconnessione "
          f"{handler.wfile.keepalives}, BrokenPipe={broken}, stop={ctx._should_stop}")
    if not broken or not ctx._should_stop:
        print("la disconnessione non e' stata rilevata")
        return 5
    if pulled[0] == total:
        print("SILENZIO: disconnessione rilevata solo a generazione finita")
        return 3
    print(f"OK: disconnessione rilevata {pulled[0] - KA_REASONING - KA_DISCONNECT} token dopo")
    return 0


CACHE_LIMIT = 4_831_838_208  # --prompt-cache-bytes di mlx_jaccl-start.sh
CACHE_ENTRY = 1_072_542_515  # ~1 GB: una conversazione da 40-60k token, per rank


def _cache_phase(patched):
    """Vero ResponseGenerator.__init__ (senza thread di generazione) e vera
    LRUPromptCache, riempita come nel log di produzione con 10 cache da ~1 GB."""
    from types import SimpleNamespace as NS
    from mlx_lm import server

    if patched and not apply_cache_bytes_patch():
        print("patch non applicabile")
        return 1

    server.ResponseGenerator._run_generate = lambda self: None
    cache = server.LRUPromptCache(10)
    provider = NS(cli_args=NS(prompt_cache_bytes=CACHE_LIMIT))
    server.ResponseGenerator(provider, cache).join()
    for i in range(10):
        entry = [NS(nbytes=CACHE_ENTRY, is_trimmable=lambda: False)]
        cache.insert_cache(("fake",), [i, 1, 2, 3], entry)

    print(f"prompt cache: {len(cache)} sequenze, {cache.nbytes / 1e9:.2f} GB "
          f"(limite {CACHE_LIMIT / 1e9:.2f} GB)")
    if cache.nbytes > CACHE_LIMIT:
        print("SFORA: il limite in byte non viene applicato")
        return 3
    print("OK: limite rispettato")
    return 0


PIN_STARTED = "mlx-community/Qwen3.8-27B-4bit"
PIN_CACHED = ("mlx-community/GLM-4.7-Flash-8bit", PIN_STARTED, "mlx-community/Qwen3.8-27B-8bit")


def _pin_phase(patched):
    """Vero handle_models_request su una finta cache HF con tre modelli, e una
    richiesta per GLM passata dal vero wrapper: il client deve vedere e usare
    solo il modello di avvio."""
    import json
    from pathlib import Path
    from types import SimpleNamespace as NS
    from mlx_lm import server

    files = [NS(file_path=Path(name)) for name in
             ("config.json", "model.safetensors.index.json", "tokenizer_config.json")]
    server.scan_cache_dir = lambda: NS(repos=[
        NS(repo_type="model", repo_id=repo, refs={"main": NS(files=files)})
        for repo in PIN_CACHED])
    served = []
    server.APIHandler.handle_completion = \
        lambda self, request, stop_words: served.append(self.requested_model)

    if patched and not apply_pin_patch():
        print("patch non applicabile")
        return 1

    class WFile:
        data = b""

        def write(self, data):
            self.data += data

        def flush(self):
            pass

    def handler(requested=None):
        return NS(path="/v1/models", created=0, requested_model=requested, wfile=WFile(),
                  response_generator=NS(cli_args=NS(model=PIN_STARTED)),
                  _set_completion_headers=lambda code: None, end_headers=lambda: None)

    h = handler()
    server.APIHandler.handle_models_request(h)
    listed = [m["id"] for m in json.loads(h.wfile.data)["data"]]
    server.APIHandler.handle_completion(handler(PIN_CACHED[0]), None, [])

    print(f"/v1/models: {listed}; richiesta per {PIN_CACHED[0]} servita con {served[-1]}")
    if listed != [PIN_STARTED] or served[-1] != PIN_STARTED:
        print("il client vede e puo' caricare modelli diversi da quello di avvio")
        return 3
    print("OK: il client vede e usa solo il modello di avvio")
    return 0


KV_PROMPT = 12  # token di prompt del test
KV_START = 8    # --quantized-kv-start del test (in produzione 5000)


def _kvbits_phase(patched):
    """Veri glm4_moe_lite e llama minuscoli (pesi casuali) con kv_bits=8 e la
    cache che supera quantized_kv_start: GLM deve generare a precisione piena,
    Llama deve continuare a quantizzare."""
    import mlx.core as mx
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate e' anche una funzione
    from mlx_lm.models import glm4_moe_lite, llama
    from mlx_lm.models.cache import make_prompt_cache

    if patched and not apply_kvbits_patch():
        print("patch non applicabile")
        return 1

    models = {
        "glm4_moe_lite": glm4_moe_lite.Model(glm4_moe_lite.ModelArgs(
            vocab_size=64, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            n_routed_experts=4, num_experts_per_tok=2, kv_lora_rank=64, q_lora_rank=32,
            qk_rope_head_dim=64, qk_nope_head_dim=32, v_head_dim=32)),
        "llama": llama.Model(llama.ModelArgs(
            model_type="llama", vocab_size=64, hidden_size=256, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, rms_norm_eps=1e-5)),
    }
    kinds = {}
    for name, model in models.items():
        cache = make_prompt_cache(model)
        try:
            for _ in generate.generate_step(
                    mx.array(list(range(KV_PROMPT))), model, prompt_cache=cache,
                    max_tokens=8, prefill_step_size=4, kv_bits=8, quantized_kv_start=KV_START):
                pass
        except AttributeError as e:
            print(f"CRASH di {name} con --kv-bits: {e}")
            return 3
        kinds[name] = sorted({type(c).__name__ for c in cache})

    print(f"cache dopo la generazione: {kinds}")
    if kinds != {"glm4_moe_lite": ["KVCache"], "llama": ["QuantizedKVCache"]}:
        print("cache diverse da quelle attese")
        return 5
    print("OK: GLM genera a precisione piena, Llama resta quantizzato")
    return 0


def _self_test():
    results = {}
    for phase in ("unpatched", "patched", "stop-unpatched", "stop-patched",
                  "keepalive-unpatched", "keepalive-patched",
                  "cache-unpatched", "cache-patched", "pin-unpatched", "pin-patched",
                  "kvbits-unpatched", "kvbits-patched"):
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
    for name, what in (("", "leak"), ("stop-", "desync dei rank allo stop"),
                       ("keepalive-", "silenzio durante le tool call"),
                       ("cache-", "sforamento del limite della prompt cache"),
                       ("pin-", "cambio di modello su richiesta del client"),
                       ("kvbits-", "crash dei modelli MLA con --kv-bits")):
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


PHASES = ("patched", "unpatched", "stop-patched", "stop-unpatched",
          "keepalive-patched", "keepalive-unpatched",
          "cache-patched", "cache-unpatched", "pin-patched", "pin-unpatched",
          "kvbits-patched", "kvbits-unpatched")


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
        if args[1].startswith("keepalive-"):
            return _keepalive_phase(args[1] == "keepalive-patched")
        if args[1].startswith("cache-"):
            return _cache_phase(args[1] == "cache-patched")
        if args[1].startswith("pin-"):
            return _pin_phase(args[1] == "pin-patched")
        if args[1].startswith("kvbits-"):
            return _kvbits_phase(args[1] == "kvbits-patched")
        return _phase(args[1] == "patched")

    # Le patch cambiano le collettive: i rank devono avere lo stesso shim
    with open(os.path.abspath(__file__), "rb") as f:
        _log(f"shim sha256 {hashlib.sha256(f.read()).hexdigest()[:12]}")
    apply_patch()
    apply_stop_patch()
    apply_keepalive_patch()
    apply_cache_bytes_patch()
    apply_pin_patch()
    apply_kvbits_patch()
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
