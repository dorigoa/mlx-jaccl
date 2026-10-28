#!/usr/bin/env python3
"""

execute on both mac nodes: /Volumes/Home/Alvise/miniforge3/envs/mlx/bin/python mlx_leak_shim_v2.py --self-test

expected output:

--- fase unpatched ---
    CRASH allo step 10395: [metal::malloc] Resource limit (499000) exceeded.
--- fase patched ---
    OK: 12000 step x 48 layer, valori corretti
    [leak-shim] rank=? patch applied (mlx-lm 0.32.2)
--- fase stop-unpatched ---
    token generati rank0=5 rank1=200, collettive stop rank0=0 rank1=0, errori=['NotImplementedError', None]
    DESYNC: i rank escono da punti diversi
--- fase stop-patched ---
    token generati rank0=9 rank1=9, collettive stop rank0=2 rank1=2, errori=[None, None]
    OK: entrambi i rank fermi al token 9
    [leak-shim] rank=? patch stop applied (mlx-lm 0.32.2, sync ogni 8 token)
--- fase keepalive-unpatched ---
    token letti 106/106, keepalive prima della disconnessione 0, BrokenPipe=True, stop=True
    SILENZIO: disconnessione rilevata solo a generazione finita
--- fase keepalive-patched ---
    token letti 19/106, keepalive prima della disconnessione 2, BrokenPipe=True, stop=True
    OK: disconnessione rilevata 4 token dopo
    [leak-shim] rank=? patch keepalive applied (mlx-lm 0.32.2, dopo 0.05s di silenzio)
--- fase cache-unpatched ---
    prompt cache: 10 sequenze, 10.73 GB (limite 4.83 GB)
    SFORA: il limite in byte non viene applicato
--- fase cache-patched ---
    prompt cache: 4 sequenze, 4.29 GB (limite 4.83 GB)
    OK: limite rispettato
    [leak-shim] rank=? patch limite cache applied (mlx-lm 0.32.2)
    [leak-shim] rank=? prompt cache limitata a 4.83 GB
--- fase pin-unpatched ---
    /v1/models: ['mlx-community/GLM-4.7-Flash-8bit', 'mlx-community/Qwen3.8-27B-4bit', 'mlx-community/Qwen3.8-27B-8bit']; richiesta per mlx-community/GLM-4.7-Flash-8bit servita con mlx-community/GLM-4.7-Flash-8bit
    il client vede e puo' caricare modelli diversi da quello di avvio
--- fase pin-patched ---
    /v1/models: ['mlx-community/Qwen3.8-27B-4bit']; richiesta per mlx-community/GLM-4.7-Flash-8bit servita con mlx-community/Qwen3.8-27B-4bit
    OK: il client vede e usa solo il modello di avvio
    [leak-shim] rank=? patch modello bloccato applied (mlx-lm 0.32.2)
    [leak-shim] rank=? richiesto il modello 'mlx-community/GLM-4.7-Flash-8bit': servito con 'mlx-community/Qwen3.8-27B-4bit'
--- fase kvbits-unpatched ---
    CRASH di glm4_moe_lite con --kv-bits: 'list' object has no attribute 'swapaxes'
--- fase kvbits-patched ---
    cache dopo la generazione: {'glm4_moe_lite': ['KVCache'], 'llama': ['QuantizedKVCache']}
    OK: GLM genera a precisione piena, Llama resta quantizzato
    [leak-shim] rank=? patch kv-bits applied (mlx-lm 0.32.2)
    [leak-shim] rank=? glm4_moe_lite: attenzione MLA, la cache KV resta a precisione piena (--kv-bits ignorato)
--- fase metrics-unpatched ---
    GET /v1/metrics: HTTP 404
    NESSUNA METRICA: mlx-lm non ha /v1/metrics
--- fase metrics-patched ---
    GET /v1/metrics: requests={'received': 3, 'completed': 1, 'cancelled': 1, 'failed': 1, 'queued': 0, 'active': 0}
    ultima richiesta: prompt=12 cached=4 prefill=8 token=2 fine=cancelled
    OK: metriche corrette
    [metrics-plugin] rank=? /v1/metrics attivo (mlx-lm 0.32.2)
--- fase ranks-unpatched ---
    rank partiti: stessa configurazione [True, True], MLX_STOP_SHIM_EVERY 8/4 [True, True]
    PARTONO: le collettive si desincronizzano al primo stop
--- fase ranks-patched ---
    rank partiti: stessa configurazione [True, True], MLX_STOP_SHIM_EVERY 8/4 [False, False]
    OK: entrambi i rank si fermano all'avvio indicando la differenza
    [leak-shim] rank=? DIVERSO fra i rank: env MLX_STOP_SHIM_EVERY: rank0='8', rank1='4'
    [leak-shim] rank=? DIVERSO fra i rank: env MLX_STOP_SHIM_EVERY: rank0='8', rank1='4'
--- fase mtp-unpatched ---
    forward del modello per token generato: 1.00 (120 token), nessuna testa MTP
--- fase mtp-patched ---
    forward del modello per token: 1.00 senza MTP, 0.98 con la testa casuale, 0.42 con bozze giuste a 35/51 verifiche (accettate per verifica {0: 16, 1: 1, 2: 34})
    OK: token identici al generate_step originale (greedy, kv-bits, logits processors), cache coerente fermandosi a meta' verifica
    [leak-shim] rank=? patch MTP applied (mlx-lm 0.32.2, blocco della testa)
    [leak-shim] rank=? MTP attivo: /var/folders/.../tmpXXXX, 2 bozze per verifica
PASS: leak riprodotto senza patch, assente con la patch
PASS: desync dei rank allo stop riprodotto senza patch, assente con la patch
PASS: silenzio durante le tool call riprodotto senza patch, assente con la patch
PASS: sforamento del limite della prompt cache riprodotto senza patch, assente con la patch
PASS: cambio di modello su richiesta del client riprodotto senza patch, assente con la patch
PASS: crash dei modelli MLA con --kv-bits riprodotto senza patch, assente con la patch
PASS: assenza di /v1/metrics riprodotto senza patch, assente con la patch
PASS: avvio con configurazioni diverse fra i rank riprodotto senza patch, assente con la patch
PASS: un forward del modello per ogni token riprodotto senza patch, assente con la patch

----------------

mlx_leak_shim_v2.py - avvia mlx_lm.server con una patch al leak di buffer Metal
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
Non si modifica il sorgente: durante handle_completion il wfile dell'handler
ricorda l'ultima scrittura e l'iteratore di response_generator.generate()
manda il keepalive fra un token e l'altro.

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

----------------

Metriche: GET /v1/metrics viene da mlx_metrics_plugin.py (stessa cartella),
che avvolge ResponseGenerator.generate e APIHandler.do_GET senza modificare il
sorgente. Se il file manca o mlx-lm e' diverso, il server parte senza metriche.

----------------

Settima patch (MTP, mlx-lm 0.32.2): Qwen3.5/3.6/3.8 hanno una testa MTP
(multi-token prediction) che mlx-lm scarta (qwen3_5.sanitize toglie "mtp.*").
mlx-community la pubblica a parte (es. mlx-community/Qwen3.8-27B-MTP-8bit,
model_type qwen3_5_mtp, fatta per mlx-vlm): un layer di attenzione completa su
fc([embedding del token, hidden del modello]), con embedding e lm_head del
modello target.

Fix: quando ModelProvider carica un modello qwen3_5 denso si cerca in locale
(cache HF, nessun download) la testa con "-MTP" prima della quantizzazione
(mlx-community/Qwen3.8-27B-8bit -> mlx-community/Qwen3.8-27B-MTP-8bit) e la si
aggancia al modello; generate_step passa allora da _mtp_generate. A ogni passo
la testa propone block-1 bozze (greedy), il modello verifica in un solo forward
l'ultimo token e le bozze, e si emettono le bozze uguali ai token che il modello
campiona li', piu' il primo token suo: i token emessi sono quelli che il modello
avrebbe campionato da solo, con qualunque temperatura. Lo stato GatedDeltaNet
(ArraysCache) non si puo' accorciare: durante la verifica la ricorrenza va un
token alla volta e salva lo stato dopo ogni token, cosi' i token rifiutati si
scartano senza ricalcolo; le cache KV si accorciano con trim. Nel cluster bozze
e token campionati sono quelli del rank 0 (due all_sum piccole per verifica) e
la testa e' attiva solo se tutti i rank l'hanno caricata, uguale. Quando
l'ultimo token emesso non e' ancora nella cache, insert_cache lo esclude dalla
chiave, cosi' la prompt cache resta coerente. Richieste con immagini o cache
di altro tipo passano dal generate_step originale. L'MTP vale per le richieste
servite da _serve_single, cioe' con --kv-bits come in start.py: il percorso
batch (BatchGenerator) non usa generate_step.

----------------

Controllo dei rank all'avvio: prima di avviare il server i rank si scambiano
(all_gather) sha256 di shim e plugin, versioni di mlx/mlx-lm, argomenti,
variabili MLX_*_SHIM* e patch applicate. Se qualcosa differisce (es.
MLX_STOP_SHIM_EVERY impostata su un solo nodo, che desincronizzerebbe le
collettive al primo stop) tutti i rank escono subito con l'elenco delle
differenze, invece di morire con exit 255 a meta' di una richiesta.
Se mlx_lm.server.main manca il server non parte: rieseguire il modulo con
runpy creerebbe classi nuove, senza nessuna delle patch.

Uso:
  python mlx_leak_shim_v2.py [argomenti di mlx_lm.server]
  python mlx_leak_shim_v2.py --self-test
Variabili d'ambiente:
  MLX_LEAK_SHIM=0              non applica la patch al leak (per confronti A/B)
  MLX_STOP_SHIM=0              non applica la patch allo stop distribuito
  MLX_STOP_SHIM_EVERY=n        ogni quanti token si condivide lo stop (default 8)
  MLX_KEEPALIVE_SHIM=0         non applica la patch keepalive
  MLX_KEEPALIVE_SHIM_EVERY=s   secondi di silenzio prima di un keepalive (default 5)
  MLX_CACHE_SHIM=0             non applica il limite in byte alla prompt cache
  MLX_PIN_SHIM=0               non blocca il modello (torna il comportamento originale)
  MLX_KVBITS_SHIM=0            quantizza la cache KV anche dei modelli MLA (crash oltre 5000 token)
  MLX_METRICS_PLUGIN=0         non attiva GET /v1/metrics
  MLX_MTP_SHIM=0               non usa la testa MTP
  MLX_MTP_SHIM_MODEL=repo|dir  testa MTP da usare al posto di quella derivata da --model
  MLX_MTP_SHIM_BLOCK=n         token per verifica, ultimo + n-1 bozze (default: block_size della testa, 3)
  MLX_LEAK_SHIM_STRICT=1       esce con errore se il codice installato non e' quello atteso
"""

import functools
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

TAG = "[leak-shim]"
# Variabili dello shim: devono essere uguali su tutti i rank
SHIM_ENV = ("MLX_LEAK_SHIM", "MLX_LEAK_SHIM_STRICT", "MLX_STOP_SHIM", "MLX_STOP_SHIM_EVERY",
            "MLX_KEEPALIVE_SHIM", "MLX_KEEPALIVE_SHIM_EVERY", "MLX_CACHE_SHIM", "MLX_PIN_SHIM",
            "MLX_KVBITS_SHIM", "MLX_METRICS_PLUGIN", "MLX_MTP_SHIM", "MLX_MTP_SHIM_MODEL",
            "MLX_MTP_SHIM_BLOCK")
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

class _KeepaliveWFile:
    """wfile dell'handler che ricorda l'ultima scrittura."""

    def __init__(self, wfile):
        self._wfile = wfile
        self.last_write = time.monotonic()

    def write(self, data):
        n = self._wfile.write(data)
        self.last_write = time.monotonic()
        return n

    def __getattr__(self, name):
        return getattr(self._wfile, name)


class _KeepaliveGenerator:
    """response_generator dell'handler: generate() restituisce un iteratore
    che fra un token e l'altro manda ": keepalive" dopo `every` s di silenzio."""

    def __init__(self, response_generator, handler, wfile, every):
        self._rg, self._handler, self._wfile, self._every = \
            response_generator, handler, wfile, every

    def generate(self, *args, **kwargs):
        ctx, responses = self._rg.generate(*args, **kwargs)
        return ctx, self._keepalive(responses)

    def _keepalive(self, responses):
        for response in responses:
            # Durante le tool call non si scrive nulla: il commento SSE tiene
            # viva la connessione e fa emergere subito una disconnessione.
            if self._handler.stream and \
                    time.monotonic() - self._wfile.last_write >= self._every:
                self._wfile.write(b": keepalive\n\n")
                self._wfile.flush()
            yield response

    def __getattr__(self, name):
        return getattr(self._rg, name)


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
        _log("MLX_LEAK_SHIM=0: patch NON applied")
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
        return _mismatch(f"mlx-lm {version}: codice diverso da quello atteso, patch NON applied")

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
    _log(f"patch applied (mlx-lm {version})")
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
                         f"patch {what} NON applied")

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
        _log("MLX_STOP_SHIM=0: patch stop NON applied")
        return False

    import mlx_lm
    from mlx_lm import server

    rg = server.ResponseGenerator
    if getattr(rg._serve_single, "_shim", None) == "stop":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    every = _env_number("MLX_STOP_SHIM_EVERY", 8, int)
    if not callable(getattr(rg, "_share_object", None)):
        return _mismatch(f"mlx-lm {version}: manca _share_object, patch stop NON applied")
    if not _patch_method(rg, "_serve_single", [(STOP_OLD, STOP_NEW.format(every=every))], "stop"):
        return False
    _log(f"patch stop applied (mlx-lm {version}, sync ogni {every} token)")
    return True


def apply_keepalive_patch():
    if os.environ.get("MLX_KEEPALIVE_SHIM", "1") == "0":
        _log("MLX_KEEPALIVE_SHIM=0: patch keepalive NON applied")
        return False

    import mlx_lm
    from mlx_lm import server

    handler = server.APIHandler
    if getattr(handler.handle_completion, "_shim", None) == "keepalive":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    every = _env_number("MLX_KEEPALIVE_SHIM_EVERY", 5, float)
    try:
        src = inspect.getsource(handler.handle_completion)
    except (OSError, TypeError):
        src = ""
    if "self.response_generator.generate(" not in src or "self.wfile.write(" not in src:
        return _mismatch(f"mlx-lm {version}: handle_completion diverso da quello atteso, "
                         "patch keepalive NON applied")

    orig_completion = handler.handle_completion

    # wfile e response_generator sono dell'istanza (una per richiesta): li si
    # sostituisce solo durante la chiamata.
    @functools.wraps(orig_completion)
    def handle_completion(self, *args, **kwargs):
        wfile, rg = self.wfile, self.response_generator
        self.wfile = _KeepaliveWFile(wfile)
        self.response_generator = _KeepaliveGenerator(rg, self, self.wfile, every)
        try:
            return orig_completion(self, *args, **kwargs)
        finally:
            self.wfile, self.response_generator = wfile, rg

    handle_completion._shim = "keepalive"
    handler.handle_completion = handle_completion
    _log(f"patch keepalive applied (mlx-lm {version}, dopo {every:g}s di silenzio)")
    return True


def apply_cache_bytes_patch():
    if os.environ.get("MLX_CACHE_SHIM", "1") == "0":
        _log("MLX_CACHE_SHIM=0: patch limite cache NON applied")
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
                         "quelli attesi, patch limite cache NON applied")

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
    _log(f"patch limite cache applied (mlx-lm {version})")
    return True


def apply_metrics_plugin():
    try:
        import mlx_metrics_plugin
    except ImportError:
        _log("mlx_metrics_plugin.py assente: /v1/metrics NON attivo")
        return False
    return mlx_metrics_plugin.install()


def apply_pin_patch():
    """Va applied dopo apply_keepalive_patch, perche' avvolge handle_completion."""
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
                         "patch modello bloccato NON applied")

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
    _log(f"patch modello bloccato applied (mlx-lm {version})")
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
        _log("MLX_KVBITS_SHIM=0: patch kv-bits NON applied")
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
                         "quelli attesi, patch kv-bits NON applied")

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
    _log(f"patch kv-bits applied (mlx-lm {version})")
    return True


# ---------------- controllo dei rank all'avvio ----------------
CONFIG_BYTES = 8192  # spazio per la configurazione di ogni rank nell'all_gather


def _file_sha(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        return None


def _rank_config(applied, args, env=None):
    """Cio' che decide le collettive: shim, plugin, versioni, argomenti,
    variabili dello shim e patch effettivamente applicate."""
    import mlx.core as mx
    import mlx_lm

    env = os.environ if env is None else env
    here = os.path.dirname(os.path.abspath(__file__))
    config = {
        "sha256 shim": _file_sha(os.path.abspath(__file__)),
        "sha256 plugin": _file_sha(os.path.join(here, "mlx_metrics_plugin.py")),
        "mlx": mx.__version__,
        "mlx-lm": getattr(mlx_lm, "__version__", "?"),
        "argomenti": list(args),
    }
    config.update({f"env {name}": env.get(name) for name in SHIM_ENV})
    config.update({f"patch {name}": ok for name, ok in applied.items()})
    return config


def _gather_configs(blob):
    """all_gather delle configurazioni (blob JSON) di tutti i rank."""
    import mlx.core as mx

    group = mx.distributed.init()
    if group.size() == 1:
        return [blob]
    data = mx.array(list(blob.ljust(CONFIG_BYTES, b"\0")), dtype=mx.int32)
    raw = bytes(mx.distributed.all_gather(data, group=group).tolist())
    return [raw[i * CONFIG_BYTES:(i + 1) * CONFIG_BYTES].rstrip(b"\0")
            for i in range(group.size())]


def _check_ranks(config, gather=_gather_configs):
    """True se tutti i rank hanno la stessa configurazione; altrimenti scrive
    le differenze. Va chiamata da tutti i rank."""
    blob = json.dumps(config, sort_keys=True).encode()
    if len(blob) > CONFIG_BYTES:
        blob = json.dumps({"sha256 configurazione": hashlib.sha256(blob).hexdigest()}).encode()
    configs = [json.loads(b) for b in gather(blob)]
    diff = sorted(k for k in set().union(*configs)
                  if len({json.dumps(c.get(k)) for c in configs}) > 1)
    for key in diff:
        values = ", ".join(f"rank{i}={c.get(key)!r}" for i, c in enumerate(configs))
        _log(f"DIVERSO fra i rank: {key}: {values}")
    if not diff and len(configs) > 1:
        _log(f"configurazione identica su {len(configs)} rank")
    return not diff


# ---------------- MTP (multi-token prediction) ----------------
# Ricorrenza di GatedDeltaNet.__call__ (qwen3_5) sostituita dalla patch MTP
MTP_GDN_OLD = """\
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
"""
MTP_GDN_NEW = """\
        if getattr(cache, "_mtp_record", False):
            # [mtp-shim] verifica MTP: stato dopo ogni token del blocco, per
            # scartare i token rifiutati senza ricalcolare
            out, state = _mtp_shim_record(
                cache, conv_input, q, k, v, a, b, self.A_log, self.dt_bias,
                state, mask, not self.training,
            )
        else:
            out, state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )
"""
# Argomenti di generate_step gestiti da _mtp_generate
MTP_STEP_ARGS = ("max_tokens", "sampler", "logits_processors", "max_kv_size", "prompt_cache",
                 "prefill_step_size", "kv_bits", "kv_group_size", "quantized_kv_start",
                 "prompt_progress_callback", "input_embeddings")
MTP_CACHES = ("ArraysCache", "KVCache", "QuantizedKVCache")
# Campi di text_config che testa MTP e modello devono avere uguali
MTP_MATCH = ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
             "head_dim", "intermediate_size", "rms_norm_eps")
MTP_QUANT = re.compile(r"(.+)-(\d+bit|bf16|fp16|mxfp\d+|nvfp\d+)")


def _mtp_record(cache, conv_input, q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel):
    """Ricorrenza di GatedDeltaNet un token alla volta: conv e stato dopo ogni
    token restano in cache._mtp_steps finche' _mtp_commit non sceglie."""
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_update

    S = q.shape[1]
    n_keep = conv_input.shape[1] - S
    outs, steps = [], []
    for j in range(S):
        t = slice(j, j + 1)
        out, state = gated_delta_update(
            q[:, t], k[:, t], v[:, t], a[:, t], b[:, t], A_log, dt_bias, state,
            None if mask is None else mask[:, t], use_kernel=use_kernel)
        outs.append(out)
        steps.append((mx.contiguous(conv_input[:, j + 1:j + 1 + n_keep]), state))
    cache._mtp_steps = steps
    return mx.concatenate(outs, axis=1), state


def _mtp_commit(prompt_cache, keep, total):
    """Dei `total` token appena verificati la cache del modello tiene i primi `keep`."""
    for c in prompt_cache:
        steps = c.__dict__.pop("_mtp_steps", None)
        if steps is not None:
            c[0], c[1] = steps[keep - 1]
        elif keep < total:
            c.trim(total - keep)


def _lm_logits(lm, hidden):
    """Coda di qwen3_5.TextModel.__call__: lm_head (shardato per vocabolario)."""
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(hidden)
    from mlx_lm.models.base import gather_last_axis

    out = lm.lm_head(hidden)
    if lm.sharding_group is not None:
        out = gather_last_axis(out, lm.sharding_group)
    return out


@functools.lru_cache(maxsize=None)
def _mtp_module_class():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.qwen3_5 import DecoderLayer

    class MTPModule(nn.Module):
        """Testa qwen3_5_mtp: fc([embedding del token, hidden]) e layer di
        attenzione completa; nessun peso condiviso col modello."""

        def __init__(self, args, n_layers):
            super().__init__()
            H, eps = args.hidden_size, args.rms_norm_eps
            self.fc = nn.Linear(2 * H, H, bias=False)
            self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
            self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
            # con questo layer_idx DecoderLayer usa l'attenzione completa
            self.layers = [DecoderLayer(args, args.full_attention_interval - 1)
                           for _ in range(n_layers)]
            self.norm = nn.RMSNorm(H, eps=eps)

        def __call__(self, embeddings, hidden, cache):
            h = self.fc(mx.concatenate([self.pre_fc_norm_embedding(embeddings),
                                        self.pre_fc_norm_hidden(hidden)], axis=-1))
            mask = create_attention_mask(h, cache[0])
            for layer, c in zip(self.layers, cache):
                h = layer(h, mask=mask, cache=c)
            return self.norm(h)

    return MTPModule


class _MTPDrafter:
    """Testa MTP agganciata al modello come attributo semplice: non e' un suo
    sotto-modulo, quindi resta fuori da parameters() e dallo sharding (ogni
    rank ne ha una copia intera, senza collettive)."""

    def __init__(self, module, block, name):
        self.module, self.block, self.name = module, block, name

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in self.module.layers]

    def forward(self, lm, tokens, hidden, cache):
        """hidden[:, j] (del modello o della testa) col token che lo segue."""
        return self.module(lm.model.embed_tokens(tokens), hidden, cache)

    def draft(self, lm, hidden, index):
        """Bozza greedy del token `index` della risposta dall'ultimo hidden."""
        import mlx.core as mx
        return mx.argmax(_lm_logits(lm, hidden[:, -1, :]), axis=-1)


def _mtp_source(model_path):
    """(nome, cartella locale) della testa MTP del modello; nessun download."""
    name = os.environ.get("MLX_MTP_SHIM_MODEL")
    if not name:
        m = MTP_QUANT.fullmatch(str(model_path))
        if m is None:
            return None, None
        name = f"{m[1]}-MTP-{m[2]}"
    if Path(name).is_dir():
        return name, Path(name)
    try:
        from huggingface_hub import snapshot_download
        return name, Path(snapshot_download(name, local_files_only=True,
                                            allow_patterns=["*.json", "*.safetensors"]))
    except Exception:
        return name, None


def _load_mtp(lm, path, name, block=None):
    import mlx.core as mx
    import mlx.nn as nn

    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config") or {}
    if config.get("model_type") != "qwen3_5_mtp":
        raise ValueError(f"model_type {config.get('model_type')!r}, atteso 'qwen3_5_mtp'")
    for key in MTP_MATCH:
        if key in text and text[key] != getattr(lm.args, key):
            raise ValueError(f"{key}={text[key]} ma il modello ha {getattr(lm.args, key)}")
    n_layers = int(text.get("mtp_num_hidden_layers", 1))
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    module = _mtp_module_class()(lm.args, n_layers)
    quant = config.get("quantization")
    if quant:
        nn.quantize(module, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant.get("mode", "affine"),
                    class_predicate=lambda p, _: f"{p}.scales" in weights)
    module.load_weights(list(weights.items()), strict=True)
    module.eval()
    mx.eval(module.parameters())
    return _MTPDrafter(module, block or int(config.get("block_size", n_layers + 2)), name)


def _attach_mtp(model, model_path, block=None):
    """Aggancia a model._mtp_shim la testa MTP del modello appena caricato (o
    None). Nel cluster va chiamata da tutti i rank: la testa resta attiva solo
    se tutti l'hanno caricata con lo stesso blocco."""
    import mlx.core as mx

    drafter, why = None, None
    lm = getattr(model, "language_model", None)
    model_type = getattr(model, "model_type", type(model).__name__)
    if model_type != "qwen3_5" or getattr(getattr(lm, "args", None), "num_experts", 1) != 0:
        why = f"{model_type}: supportato solo qwen3_5 denso"
    else:
        name, path = _mtp_source(model_path)
        if name is None:
            why = f"nessuna testa MTP associata a {model_path!r}"
        elif path is None:
            why = f"testa MTP {name!r} non presente in locale"
        else:
            try:
                drafter = _load_mtp(lm, path, name, block)
            except Exception as e:
                why = f"testa MTP {name!r} non caricata: {e}"

    group = mx.distributed.init()
    if group.size() > 1:
        mine = mx.array([[1, drafter.block] if drafter else [0, 0]], dtype=mx.int32)
        rows = mx.distributed.all_gather(mine, group=group).tolist()
        if drafter is not None and any(row != rows[0] for row in rows):
            drafter, why = None, f"testa MTP diversa fra i rank ([caricata, blocco] = {rows})"

    model._mtp_shim = drafter
    if drafter is not None:
        _log(f"MTP attivo: {drafter.name}, {drafter.block - 1} bozze per verifica")
    else:
        _log(f"MTP non attivo: {why}")


def _mtp_generate(prompt, model, drafter, stream, *, max_tokens=256, sampler=None,
                  logits_processors=None, max_kv_size=None, prompt_cache=None,
                  prefill_step_size=2048, kv_bits=None, kv_group_size=64,
                  quantized_kv_start=5000, prompt_progress_callback=None,
                  input_embeddings=None):
    """generate_step con la testa MTP. A ogni verifica il modello elabora
    l'ultimo token emesso e block-1 bozze; si emettono le bozze uguali al token
    che il modello campiona nella loro posizione, piu' il primo token del
    modello che le smentisce (o che segue l'ultima)."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, make_prompt_cache
    generate = importlib.import_module("mlx_lm.generate")

    lm = model.language_model
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)
    mtp_cache = drafter.make_cache()
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    progress = prompt_progress_callback or (lambda *_: None)
    group = mx.distributed.init()
    k = drafter.block - 1
    history = None  # token gia' nella cache, per i logits processors

    def quantize():
        for c in (prompt_cache, mtp_cache):
            generate.maybe_quantize_kv_cache(c, quantized_kv_start, kv_group_size, kv_bits)

    def from_rank0(x):
        # tutti i rank usano bozze e token del rank 0: non divergono neanche se
        # i calcoli replicati differissero di un bit. x * 0 e non zeros_like(x):
        # anche gli altri rank devono calcolare x, e le sue collettive, prima
        # di questa all_sum, altrimenti l'ordine delle collettive cambia
        x = x.astype(mx.int32)
        if group.size() == 1:
            return x
        return mx.distributed.all_sum(x * (group.rank() == 0), group=group)

    def sample(hidden, fed):
        logits = _lm_logits(lm, hidden)
        if logits_processors:
            rows = []
            for i in range(logits.shape[0]):
                seen = fed[:i + 1] if history is None else mx.concatenate([history, fed[:i + 1]])
                row = logits[i:i + 1]
                for processor in logits_processors:
                    row = processor(seen, row)
                rows.append(row)
            logits = mx.concatenate(rows)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return from_rank0(sampler(logprobs)), logprobs

    def remember(tokens):
        nonlocal history
        if logits_processors:
            history = tokens if history is None else mx.concatenate([history, tokens])

    with mx.stream(stream):
        # Prefill: la testa accoppia l'hidden della posizione j col token j+1
        total = len(prompt)
        progress(0, total)
        while len(prompt) > 1:
            n = min(prefill_step_size, len(prompt) - 1)
            hidden = lm.model(prompt[:n][None], cache=prompt_cache)
            drafter.forward(lm, prompt[1:n + 1][None], hidden, mtp_cache)
            quantize()
            mx.eval([c.state for c in prompt_cache + mtp_cache])
            remember(prompt[:n])
            prompt = prompt[n:]
            progress(total - len(prompt), total)
            mx.clear_cache()

        hidden = lm.model(prompt[None], cache=prompt_cache)[:, -1:, :]
        y, logprobs = sample(hidden[0], prompt)
        remember(prompt)
        quantize()
        last = drafter.forward(lm, y[None], hidden, mtp_cache)
        seed = drafter.draft(lm, last, 1)
        mx.eval(y)
        mx.async_eval(seed, last)
        progress(total, total)

        b = y.item()
        emitted, rounds, accepted = 1, 0, 0
        pending = True  # b emesso ma non ancora nella cache del modello
        try:
            yield b, logprobs[0]
            while max_tokens < 0 or emitted < max_tokens:
                # Bozze: la prima e' gia' pronta, le altre dalla testa sul suo hidden
                drafts, h, appended = [seed], last, 0
                for i in range(1, k):
                    h = drafter.forward(lm, drafts[-1][None], h, mtp_cache)
                    drafts.append(drafter.draft(lm, h, emitted + i))
                    appended += 1
                drafts = from_rank0(mx.concatenate(drafts))

                # Verifica: un forward su [b, bozze]
                fed = mx.concatenate([mx.array([b], dtype=mx.int32), drafts])
                gdn = [c for c in prompt_cache if isinstance(c, ArraysCache)]
                for c in gdn:
                    c._mtp_record = True
                try:
                    verified = lm.model(fed[None], cache=prompt_cache)
                finally:
                    for c in gdn:
                        del c._mtp_record
                targets, logprobs = sample(verified[0], fed)
                mx.eval(targets, drafts)
                t, d = targets.tolist(), drafts.tolist()
                a = 0
                while a < k and t[a] == d[a]:
                    a += 1
                new = t[:a + 1] if max_tokens < 0 else t[:min(a + 1, max_tokens - emitted)]
                rounds += 1
                accepted += a

                # Testa: tiene le voci delle bozze confermate e aggiunge le
                # altre posizioni con l'hidden vero del modello
                keep = min(a, appended)
                for c in mtp_cache:
                    c.trim(appended - keep)
                last = drafter.forward(lm, mx.array(d[keep:a] + [t[a]], dtype=mx.int32)[None],
                                       verified[:, keep:a + 1], mtp_cache)
                seed = drafter.draft(lm, last, emitted + a + 1)
                last = last[:, -1:, :]
                mx.async_eval(seed, last)

                received = 0
                try:
                    for i, token in enumerate(new):
                        received = i + 1
                        yield token, logprobs[i]
                finally:
                    # anche se il chiamante smette a meta': la cache del modello
                    # copre b e i token ricevuti, tranne l'ultimo se non e' una bozza
                    _mtp_commit(prompt_cache, min(received, a) + 1, k + 1)
                    pending = received > a
                    emitted += received
                quantize()
                remember(fed[:min(received, a) + 1])
                if emitted // 256 > (emitted - received) // 256:
                    mx.clear_cache()
                b = new[-1]
        finally:
            if pending:
                # insert_cache esclude dalla chiave il token non ancora in cache
                prompt_cache[0]._mtp_behind = 1
            if rounds and group.rank() == 0:
                _log(f"MTP: {emitted} token, {(emitted - 1) / rounds:.2f} per verifica, "
                     f"bozze accettate {accepted}/{rounds * k}")


def apply_mtp_patch():
    """Va applied dopo apply_kvbits_patch, perche' avvolge generate_step."""
    if os.environ.get("MLX_MTP_SHIM", "1") == "0":
        _log("MLX_MTP_SHIM=0: MTP NON attivo")
        return False

    import mlx_lm
    from mlx_lm import server
    from mlx_lm.models import qwen3_5
    from mlx_lm.models.cache import LRUPromptCache
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate e' anche una funzione

    if getattr(generate.generate_step, "_mtp_shim", False):
        return True

    version = getattr(mlx_lm, "__version__", "?")
    try:
        step = set(inspect.signature(generate.generate_step).parameters)
        load = list(inspect.signature(server.ModelProvider._load).parameters)
        insert = list(inspect.signature(LRUPromptCache.insert_cache).parameters)
        body = inspect.getsource(qwen3_5.Qwen3_5TextModel.__call__)
        head = inspect.getsource(qwen3_5.TextModel.__call__)
    except (OSError, TypeError, ValueError):
        step, load, insert, body, head = set(), [], [], "", ""
    if not set(MTP_STEP_ARGS) <= step or \
            load[:4] != ["self", "model_path", "adapter_path", "draft_model_path"] or \
            insert[:4] != ["self", "model", "tokens", "prompt_cache"] or \
            "return self.norm(hidden_states)" not in body or \
            "gather_last_axis(out, self.sharding_group)" not in head or \
            not callable(getattr(generate, "maybe_quantize_kv_cache", None)):
        return _mismatch(f"mlx-lm {version}: generate_step/ModelProvider/qwen3_5 diversi "
                         "da quelli attesi, patch MTP NON applied")

    block = None
    if os.environ.get("MLX_MTP_SHIM_BLOCK"):
        block = max(2, _env_number("MLX_MTP_SHIM_BLOCK", 3, int))

    qwen3_5._mtp_shim_record = _mtp_record  # usata dal sorgente ricompilato
    if not _patch_method(qwen3_5.GatedDeltaNet, "__call__", [(MTP_GDN_OLD, MTP_GDN_NEW)], "mtp"):
        return False

    orig_step = generate.generate_step

    # stream_generate chiama generate_step(prompt, model, stream, **kwargs)
    @functools.wraps(orig_step)
    def generate_step(prompt, model, *args, **kwargs):
        drafter = getattr(model, "_mtp_shim", None)
        cache = kwargs.get("prompt_cache")
        if drafter is None or len(args) > 1 or len(prompt) == 0 or \
                kwargs.get("input_embeddings") is not None or \
                kwargs.get("max_kv_size") is not None or not set(kwargs) <= set(MTP_STEP_ARGS) or \
                (cache is not None and (len(cache) != len(model.layers) or
                                        any(type(c).__name__ not in MTP_CACHES for c in cache))):
            return orig_step(prompt, model, *args, **kwargs)
        stream = args[0] if args else generate.generation_stream
        return _mtp_generate(prompt, model, drafter, stream, **kwargs)

    orig_load = server.ModelProvider._load

    @functools.wraps(orig_load)
    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        orig_load(self, model_path, adapter_path, draft_model_path)
        _attach_mtp(self.model, model_path, block)

    orig_insert = LRUPromptCache.insert_cache

    @functools.wraps(orig_insert)
    def insert_cache(self, model, tokens, prompt_cache, *args, **kwargs):
        if prompt_cache and getattr(prompt_cache[0], "__dict__", {}).pop("_mtp_behind", 0):
            # l'ultimo token emesso dall'MTP non e' ancora nella cache
            tokens = tokens[:-1]
            if not tokens:
                return None
        return orig_insert(self, model, tokens, prompt_cache, *args, **kwargs)

    generate_step._mtp_shim = True
    generate.generate_step = generate_step
    server.ModelProvider._load = _load
    LRUPromptCache.insert_cache = insert_cache
    _log(f"patch MTP applied (mlx-lm {version}"
         + (f", blocco {block})" if block else ", blocco della testa)"))
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
        # tok serve a TextStateMachine.from_tokenizer (mlx-lm > 0.32.2),
        # _make_state_machine a mlx-lm <= 0.32.2
        tok = NS(has_thinking=False, has_tool_calling=False, tool_parser=None,
                 structural_markers=(), response_transitions={})
        stop_sequences = NS(matcher=lambda: NS(advance=lambda t: False))
        metric = NS(prefill=lambda *a: None, token=lambda: None, finish_reason=None)
        return NS(
            _is_distributed=True,
            _share_object=lambda obj: share(rank, obj),
            model_provider=NS(model=None, tokenizer=tok, draft_model=None,
                              model_key=("fake",)),
            _tokenize=lambda t, r, a: (list(range(10)), None, None, "normal", None),
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

    # Vera TextStateMachine: il testo della tool call esce solo alla chiusura
    text_sm = server.TextStateMachine({
        "reasoning": [("</think>", "normal"), ("<tool_call>", "tool")],
        "normal": [("<tool_call>", "tool")],
        "tool": [("</tool_call>", "normal")],
    })

    total = KA_REASONING + KA_TOOL_TOKENS + 1
    pulled = [0]
    disconnected = [False]

    def generation():
        seq = (["Hmm "] * KA_REASONING + ["<tool_call>"] + ["x"] * (KA_TOOL_TOKENS - 1)
               + ["</tool_call>"])
        for i, text in enumerate(seq):
            if KA_REASONING <= i < KA_REASONING + KA_TOOL_TOKENS:
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
        has_tool_calling=True, has_thinking=True,
        tool_parser=lambda text, tools: {"name": "f", "arguments": {}},
        text_sm=text_sm, initial_state="reasoning", prompt=[0])

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


def _metrics_phase(patched):
    try:
        import mlx_metrics_plugin
    except ImportError:
        print("mlx_metrics_plugin.py assente")
        return 1
    return mlx_metrics_plugin.self_test_phase(patched)


def _ranks_phase(patched):
    """Due rank simulati (thread, all_gather come rendezvous) partono con
    MLX_STOP_SHIM_EVERY diverso, come con la variabile impostata su un solo
    nodo. Lo shim precedente scrive solo lo sha nel log e i rank partono; il
    controllo deve fermarli entrambi, ma non rank con la stessa configurazione."""
    import contextlib
    import io

    barrier = threading.Barrier(2, timeout=5)
    slots = [None, None]

    def gather_for(rank):
        def gather(blob):
            slots[rank] = blob
            barrier.wait()
            blobs = list(slots)
            barrier.wait()
            return blobs
        return gather

    applied = {"stop": True, "mtp": True}
    args = ["--model", PIN_STARTED, "--kv-bits", "8"]

    def start(envs):
        started = [None, None]

        def rank(r):
            config = _rank_config(applied, args, env=envs[r])
            started[r] = _check_ranks(config, gather_for(r)) if patched else True

        threads = [threading.Thread(target=rank, args=(r,)) for r in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        return started

    with contextlib.redirect_stderr(io.StringIO()):
        same = start([{"MLX_STOP_SHIM_EVERY": "8"}] * 2)
    diff = start([{"MLX_STOP_SHIM_EVERY": "8"}, {"MLX_STOP_SHIM_EVERY": "4"}])
    print(f"rank partiti: stessa configurazione {same}, MLX_STOP_SHIM_EVERY 8/4 {diff}")
    if same != [True, True]:
        print("FALSO ALLARME: fermati rank con la stessa configurazione")
        return 5
    if diff == [True, True]:
        print("PARTONO: le collettive si desincronizzano al primo stop")
        return 3
    if diff != [False, False]:
        print("si ferma un solo rank")
        return 5
    print("OK: entrambi i rank si fermano all'avvio indicando la differenza")
    return 0


MTP_PROMPT = 40   # token del prompt: prefill a blocchi di 16 (16 + 16 + 7 + 1)
MTP_TOKENS = 120  # token generati
MTP_WRONG = (2, 5)  # l'oracolo sbaglia le bozze con indice % 7 in MTP_WRONG


def _tiny_qwen3_5():
    """qwen3_5 minuscolo a pesi casuali (float32): 6 layer GatedDeltaNet e 2 di
    attenzione completa, lo schema 3:1 di Qwen3.8."""
    import mlx.core as mx
    from mlx_lm.models import qwen3_5

    mx.random.seed(0)
    model = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=dict(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        vocab_size=96, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32, full_attention_interval=4,
        rms_norm_eps=1e-6, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                            "partial_rotary_factor": 0.25})))
    mx.eval(model.parameters())
    return model


def _mtp_expected_rounds(n_tokens, k):
    """Accettazioni per verifica e voci finali della cache della testa attese
    con l'oracolo di _mtp_phase (prefill: una voce per token del prompt; ogni
    verifica: una per bozza accettata e una per il token del modello)."""
    emitted, hist, entries = 1, [0] * (k + 1), MTP_PROMPT
    while emitted < n_tokens:
        a = 0
        while a < k and (emitted + a) % 7 not in MTP_WRONG:
            a += 1
        hist[a] += 1
        entries += a + 1
        emitted += min(a + 1, n_tokens - emitted)
    return hist, entries


def _mtp_phase(patched):
    """Vero qwen3_5 minuscolo e testa MTP a pesi casuali, salvata e ricaricata
    come quella di mlx-community (8 bit). In greedy l'MTP deve dare gli stessi
    token del generate_step originale (anche con kv-bits e logits processors),
    con meno forward del modello, e lasciare la cache come se i token ricevuti
    fossero stati elaborati uno a uno, anche fermandosi a meta' verifica."""
    import contextlib
    import io
    import tempfile
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate e' anche una funzione

    model = _tiny_qwen3_5()
    lm = model.language_model
    V = lm.args.vocab_size
    prompt = [(7 * i + 3) % V for i in range(MTP_PROMPT)]
    greedy = lambda x: mx.argmax(x, axis=-1)
    orig_step = generate.generate_step

    if patched and not apply_mtp_patch():
        print("patch non applicabile")
        return 1

    calls = [0]
    orig_call = qwen3_5.Qwen3_5TextModel.__call__

    def counted(self, *args, **kwargs):
        calls[0] += 1
        return orig_call(self, *args, **kwargs)

    qwen3_5.Qwen3_5TextModel.__call__ = counted

    def run(step, stop=MTP_TOKENS, **kwargs):
        """(token, cache, forward del modello per token oltre ai 4 del prefill)"""
        calls[0] = 0
        cache = make_prompt_cache(model)
        gen = step(mx.array(prompt), model, max_tokens=MTP_TOKENS, sampler=greedy,
                   prompt_cache=cache, prefill_step_size=16, **kwargs)
        out = []
        with contextlib.redirect_stderr(io.StringIO()):  # statistiche MTP
            for token, _ in gen:
                out.append(token)
                if len(out) == stop:
                    break
            gen.close()
        return out, cache, (calls[0] - 4) / len(out)

    ref, _, plain = run(orig_step)
    if not patched:
        print(f"forward del modello per token generato: {plain:.2f} ({len(ref)} token), "
              "nessuna testa MTP")
        return 3

    # Testa MTP casuale nel formato di mlx-community, ricaricata da _attach_mtp
    with tempfile.TemporaryDirectory() as tmp:
        mx.random.seed(1)
        module = _mtp_module_class()(lm.args, 1)
        nn.quantize(module, group_size=64, bits=8)
        mx.save_safetensors(f"{tmp}/model.safetensors", dict(tree_flatten(module.parameters())))
        text = {key: getattr(lm.args, key) for key in MTP_MATCH}
        Path(tmp, "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_mtp", "block_size": 3, "text_config": text,
            "quantization": {"group_size": 64, "bits": 8, "mode": "affine"}}))
        os.environ["MLX_MTP_SHIM_MODEL"] = tmp
        _attach_mtp(model, "tiny-8bit")
        del os.environ["MLX_MTP_SHIM_MODEL"]
    loaded = model._mtp_shim
    if loaded is None:
        print("testa MTP non caricata")
        return 5

    class Oracle(_MTPDrafter):
        """Bozze dalla risposta attesa, sbagliate se indice % 7 in MTP_WRONG."""
        expected = ref

        def make_cache(self):
            self.cache = super().make_cache()
            return self.cache

        def draft(self, lm, hidden, index):
            token = self.expected[index] if index < len(self.expected) else 0
            if index % 7 in MTP_WRONG:
                token = (token + 1) % V
            return mx.array([token], dtype=mx.int32)

    oracle = Oracle(loaded.module, loaded.block, "oracolo")
    k = oracle.block - 1
    failures = []

    model._mtp_shim = None  # senza testa: generate_step originale
    out, _, rate = run(generate.generate_step)
    if out != ref or rate != plain:
        failures.append("senza testa")

    model._mtp_shim = loaded
    out, _, random_rate = run(generate.generate_step)
    if out != ref:
        failures.append("testa casuale")

    model._mtp_shim = oracle
    out, cache, oracle_rate = run(generate.generate_step)
    hist, entries = _mtp_expected_rounds(MTP_TOKENS, k)
    if out != ref:
        failures.append("oracolo")
    if round(oracle_rate * len(out)) != sum(hist):
        failures.append(f"verifiche {round(oracle_rate * len(out))} invece di {sum(hist)}")
    if oracle.cache[0].offset != entries:
        failures.append(f"cache della testa a {oracle.cache[0].offset} invece di {entries}")

    for name, kwargs in (("kv-bits", dict(kv_bits=8, kv_group_size=32, quantized_kv_start=16)),
                         ("logits processors", dict(logits_processors=make_logits_processors(
                             repetition_penalty=1.5, repetition_context_size=16)))):
        expected, _, _ = run(orig_step, **kwargs)
        oracle.expected = expected
        out, cache, _ = run(generate.generate_step, **kwargs)
        if out != expected:
            failures.append(name)
        if "kv_bits" in kwargs and "QuantizedKVCache" not in {type(c).__name__ for c in cache}:
            failures.append("kv-bits: cache non quantizzata")
    oracle.expected = ref

    def same_cache(c1, c2):
        for x, y in zip(c1, c2):
            if type(x) is not type(y) or getattr(x, "offset", 0) != getattr(y, "offset", 0):
                return False
            pairs = zip(x.cache, y.cache) if type(x).__name__ == "ArraysCache" else \
                zip(x.keys_and_values(), y.keys_and_values())
            if not all(mx.allclose(u, v, atol=1e-3, rtol=1e-3).item() for u, v in pairs):
                return False
        return True

    # Il chiamante si ferma dopo `stop` token: cache e prompt cache coerenti
    behind_seen = set()
    for stop in range(25, 33):
        out, cache, _ = run(generate.generate_step, stop=stop)
        behind = cache[0].__dict__.get("_mtp_behind", 0)
        behind_seen.add(behind)
        reference = make_prompt_cache(model)
        model(mx.array(prompt + out[:len(out) - behind])[None], cache=reference)
        if out != ref[:stop] or not same_cache(cache, reference):
            failures.append(f"cache fermandosi a {stop} token")
        lru = LRUPromptCache(4)
        lru.insert_cache(("tiny",), prompt + out, cache)
        _, rest = lru.fetch_nearest_cache(("tiny",), prompt + out)
        if len(rest) != behind:
            failures.append(f"prompt cache fermandosi a {stop} token")
    if behind_seen != {0, 1}:
        failures.append(f"ultimo token fuori cache solo {behind_seen}")

    print(f"forward del modello per token: {plain:.2f} senza MTP, {random_rate:.2f} con la "
          f"testa casuale, {oracle_rate:.2f} con bozze giuste a {sum(hist[1:])}/{sum(hist)} "
          f"verifiche (accettate per verifica {dict(enumerate(hist))})")
    if failures:
        print(f"ERRORI: {failures}")
        return 5
    print("OK: token identici al generate_step originale (greedy, kv-bits, logits processors), "
          "cache coerente fermandosi a meta' verifica")
    return 0


def _self_test():
    results = {}
    for phase in ("unpatched", "patched", "stop-unpatched", "stop-patched",
                  "keepalive-unpatched", "keepalive-patched",
                  "cache-unpatched", "cache-patched", "pin-unpatched", "pin-patched",
                  "kvbits-unpatched", "kvbits-patched",
                  "metrics-unpatched", "metrics-patched",
                  "ranks-unpatched", "ranks-patched", "mtp-unpatched", "mtp-patched"):
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
                       ("kvbits-", "crash dei modelli MLA con --kv-bits"),
                       ("metrics-", "assenza di /v1/metrics"),
                       ("ranks-", "avvio con configurazioni diverse fra i rank"),
                       ("mtp-", "un forward del modello per ogni token")):
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
          "kvbits-patched", "kvbits-unpatched", "metrics-patched", "metrics-unpatched",
          "ranks-patched", "ranks-unpatched", "mtp-patched", "mtp-unpatched")


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
        if args[1].startswith("metrics-"):
            return _metrics_phase(args[1] == "metrics-patched")
        if args[1].startswith("ranks-"):
            return _ranks_phase(args[1] == "ranks-patched")
        if args[1].startswith("mtp-"):
            return _mtp_phase(args[1] == "mtp-patched")
        return _phase(args[1] == "patched")

    # Le patch cambiano le collettive: i rank devono avere lo stesso shim
    _log(f"shim sha256 {_file_sha(os.path.abspath(__file__))}")
    applied = {
        "leak": apply_patch(),
        "stop": apply_stop_patch(),
        "keepalive": apply_keepalive_patch(),
        "cache": apply_cache_bytes_patch(),
        "pin": apply_pin_patch(),
        "kvbits": apply_kvbits_patch(),
        "mtp": apply_mtp_patch(),
        "metrics": apply_metrics_plugin(),
    }
    if not _check_ranks(_rank_config(applied, args)):
        _log("shim o configurazione diversi fra i rank: il server NON parte")
        return 3
    sys.argv = ["mlx_lm.server"] + args
    try:
        from mlx_lm.server import main as server_main
    except ImportError as e:
        # Rieseguire mlx_lm.server con runpy creerebbe classi nuove, senza le patch
        _log(f"mlx_lm.server.main non disponibile ({e}): il server NON parte")
        return 2
    server_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
