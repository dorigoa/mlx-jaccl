#!/usr/bin/env python3
"""

execute on both mac nodes: /Volumes/Home/Alvise/miniforge3/envs/mlx/bin/python mlx_leak_shim.py --self-test

expected output:

--- phase unpatched ---
    CRASH at step 10395: [metal::malloc] Resource limit (499000) exceeded.
--- phase patched ---
    OK: 12000 steps x 48 layers, correct values
    [leak-shim] rank=? patch applied (mlx-lm 0.32.2)
--- phase stop-unpatched ---
    tokens generated rank0=5 rank1=200, stop collectives rank0=0 rank1=0, errors=['NotImplementedError', None]
    DESYNC: the ranks exit at different points
--- phase stop-patched ---
    tokens generated rank0=9 rank1=9, stop collectives rank0=2 rank1=2, errors=[None, None]
    OK: both ranks stopped at token 9
    [leak-shim] rank=? stop patch applied (mlx-lm 0.32.2, sync every 8 tokens)
--- phase keepalive-unpatched ---
    tokens read 106/106, keepalives before the disconnection 0, BrokenPipe=True, stop=True
    SILENCE: disconnection detected only after the generation finished
--- phase keepalive-patched ---
    tokens read 15/106, keepalives before the disconnection 1, BrokenPipe=True, stop=True
    OK: disconnection detected 0 tokens later
    [leak-shim] rank=? keepalive patch applied (mlx-lm 0.32.2, after 0.05s of silence)
--- phase cache-unpatched ---
    prompt cache: 10 sequences, 10.73 GB (limit 4.83 GB)
    EXCEEDED: the byte limit is not applied
--- phase cache-patched ---
    prompt cache: 4 sequences, 4.29 GB (limit 4.83 GB)
    OK: limit respected
    [leak-shim] rank=? cache limit patch applied (mlx-lm 0.32.2)
    [leak-shim] rank=? prompt cache limited to 4.83 GB
--- phase pin-unpatched ---
    /v1/models: ['mlx-community/GLM-4.7-Flash-8bit', 'mlx-community/Qwen3.8-27B-4bit', 'mlx-community/Qwen3.8-27B-8bit']; request for mlx-community/GLM-4.7-Flash-8bit served with mlx-community/GLM-4.7-Flash-8bit
    the client sees and can load models other than the startup one
--- phase pin-patched ---
    /v1/models: ['mlx-community/Qwen3.8-27B-4bit']; request for mlx-community/GLM-4.7-Flash-8bit served with mlx-community/Qwen3.8-27B-4bit
    OK: the client sees and uses only the startup model
    [leak-shim] rank=? pinned model patch applied (mlx-lm 0.32.2)
    [leak-shim] rank=? requested model 'mlx-community/GLM-4.7-Flash-8bit': served with 'mlx-community/Qwen3.8-27B-4bit'
--- phase kvbits-unpatched ---
    CRASH of glm4_moe_lite with --kv-bits: 'list' object has no attribute 'swapaxes'
--- phase kvbits-patched ---
    cache after the generation: {'glm4_moe_lite': ['KVCache'], 'llama': ['QuantizedKVCache']}
    OK: GLM generates at full precision, Llama stays quantized
    [leak-shim] rank=? kv-bits patch applied (mlx-lm 0.32.2)
    [leak-shim] rank=? glm4_moe_lite: MLA attention, the KV cache stays at full precision (--kv-bits ignored)
--- phase metrics-unpatched ---
    GET /v1/metrics: HTTP 404
    NO METRICS: mlx-lm has no /v1/metrics
--- phase metrics-patched ---
    GET /v1/metrics: requests={'received': 3, 'completed': 1, 'cancelled': 1, 'failed': 1, 'queued': 0, 'active': 0}
    last request: prompt=12 cached=4 prefill=8 tokens=2 finish=cancelled
    OK: correct metrics
    [metrics-plugin] rank=? /v1/metrics active (mlx-lm 0.32.2)
PASS: leak reproduced without the patch, absent with the patch
PASS: rank desync on stop reproduced without the patch, absent with the patch
PASS: silence during tool calls reproduced without the patch, absent with the patch
PASS: prompt cache limit overrun reproduced without the patch, absent with the patch
PASS: model switch on client request reproduced without the patch, absent with the patch
PASS: crash of MLA models with --kv-bits reproduced without the patch, absent with the patch
PASS: missing /v1/metrics reproduced without the patch, absent with the patch

----------------

mlx_leak_shim.py - starts mlx_lm.server with a patch for the Metal buffer
leak of the hybrid models (ArraysCache: qwen3_5 / Qwen3.6 / Qwen3.8, qwen3_next, ...).

Cause (mlx-lm 0.31.3): ArraysCache.advance() runs a lazy `-= N` on
left_padding/lengths at every step; for the layers whose metadata is never
read the chain is never evaluated and every node holds a Metal buffer
-> "[metal::malloc] Resource limit (499000) exceeded" after ~10k tokens.

Fix: after every BatchGenerator.next() only left_padding and lengths of the
caches advanced in the step are evaluated (async). The chain collapses; the
values do not change.

----------------

Second patch (distributed stop, mlx-lm 0.32.0): with --kv-bits the requests
are not batchable and go through ResponseGenerator._serve_single. If the client
disconnects mid-stream (BrokenPipeError), the HTTP handler calls
ctx.stop(), but the flag exists only on rank 0: rank 0 leaves the loop
(raise NotImplementedError) and goes back to waiting for requests with
all_sum(0), while rank 1 keeps generating with the all_sums of the model. The
jaccl collectives get out of sync and all the ranks die (exit 255).

Fix: rank 0 shares the stop with all the ranks with _share_object every
MLX_STOP_SHIM_EVERY tokens; all the ranks exit at the same token, save the
cache and go back, aligned, to waiting for the next request.

----------------

Third patch (SSE keepalive, mlx-lm 0.32.0): while the model writes a tool
call APIHandler.handle_completion sends nothing to the client (the text of the
tool call goes out entirely in the final chunk). With long tool calls the
connection stays silent for minutes, proxy/socat/client close it for
inactivity and the server notices only at the end (BrokenPipeError at
server.py:1811), after generating for no one; meanwhile the retry of the client
waits in the queue.

Fix: if while streaming nothing has been written for MLX_KEEPALIVE_SHIM_EVERY
seconds, an SSE comment ": keepalive" is sent (the same format that the
server already uses during the prefill). The connection is never idle and a
disconnection surfaces at once, so the stop patch stops the generation.
The source is not modified: during handle_completion the wfile of the handler
remembers the last write and the iterator of response_generator.generate()
sends the keepalive between one token and the next.

----------------

Fourth patch (byte limit of the prompt cache, mlx-lm 0.32.0): the server
creates LRUPromptCache passing only --prompt-cache-size; --prompt-cache-bytes
is applied only in the batch path (trim_to after every insert). With --kv-bits
the requests go through _serve_single and the cache grows up to 10 sequences
of any size: with 60k-token contexts ~10.7 GB per rank, and the next request
goes OOM (Metal) during the prefill.

Fix: max_bytes of the LRUPromptCache is set to --prompt-cache-bytes, so
insert_cache discards the oldest entries in both paths. The size is that of
the local caches, the same on the two ranks: the eviction choices (and hence
the caches found) stay identical.

----------------

Fifth patch (pinned model, mlx-lm 0.32.0): /v1/models lists all the MLX
models in the Hugging Face cache, not the one running, and the server loads
(or downloads) whatever model the client asks for in the "model" field. A
client that takes the first one in the list (e.g. Pi with GLM-4.7-Flash-8bit)
makes the server unload the started model and load another one, which may not
fit in memory.

Fix: /v1/models lists only the model passed with --model, and every request
is served with that model whatever name it gives (with a warning in the log
the first time a different name arrives).

----------------

Sixth patch (--kv-bits on MLA models, mlx-lm 0.32.0): when the cache exceeds
--quantized-kv-start (default 5000 tokens) generate_step converts it to
QuantizedKVCache, whose update_and_fetch returns tuples (weights, scales,
biases). The attention of Qwen/Llama goes through scaled_dot_product_attention,
which handles them; the models that cache the MLA latent (glm4_moe_lite =
GLM-4.7-Flash, deepseek_v3, kimi, ...) use it instead as an array
(k_pe.swapaxes) and every request beyond 5000 tokens dies with
"AttributeError: 'list' object has no attribute 'swapaxes'".

Fix: for these models generate_step receives kv_bits=None and the cache stays
at full precision (the MLA latent is already small). --kv-bits stays on the
command line, so the requests keep going through _serve_single, where the
stop and cache limit patches act. The other models stay quantized.

----------------

Metrics: GET /v1/metrics comes from mlx_metrics_plugin.py (same folder),
which wraps ResponseGenerator.generate and APIHandler.do_GET without modifying
the source. If the file is missing or mlx-lm is different, the server starts
without metrics.

Usage:
  python mlx_leak_shim.py [mlx_lm.server arguments]
  python mlx_leak_shim.py --self-test
Environment variables:
  MLX_LEAK_SHIM=0              does not apply the leak patch (for A/B comparisons)
  MLX_STOP_SHIM=0              does not apply the distributed stop patch
  MLX_STOP_SHIM_EVERY=n        every how many tokens the stop is shared (default 8)
  MLX_KEEPALIVE_SHIM=0         does not apply the keepalive patch
  MLX_KEEPALIVE_SHIM_EVERY=s   seconds of silence before a keepalive (default 5)
  MLX_CACHE_SHIM=0             does not apply the byte limit to the prompt cache
  MLX_PIN_SHIM=0               does not pin the model (back to the original behavior)
  MLX_KVBITS_SHIM=0            quantizes the KV cache of the MLA models too (crash beyond 5000 tokens)
  MLX_METRICS_PLUGIN=0         does not enable GET /v1/metrics
  MLX_LEAK_SHIM_STRICT=1       exits with an error if the installed code is not the expected one
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
import time

TAG = "[leak-shim]"
EXPECTED = ("self.lengths -= N", "self.left_padding -= N")

# Block of ResponseGenerator._serve_single replaced by the stop patch
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

                # [stop-shim] only rank 0 sees ctx.stop(): it is
                # shared every {every} tokens so that all the ranks exit
                # at the same token
                if self._is_distributed:
                    if (len(cache_key) - len(prompt) - 1) % {every} == 0:
                        if self._share_object(True if ctx._should_stop else None):
                            break
                elif ctx._should_stop:
                    break
"""

class _KeepaliveWFile:
    """wfile of the handler that remembers the last write."""

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
    """response_generator of the handler: generate() returns an iterator
    that between one token and the next sends ": keepalive" after `every` s of silence."""

    def __init__(self, response_generator, handler, wfile, every):
        self._rg, self._handler, self._wfile, self._every = \
            response_generator, handler, wfile, every

    def generate(self, *args, **kwargs):
        ctx, responses = self._rg.generate(*args, **kwargs)
        return ctx, self._keepalive(responses)

    def _keepalive(self, responses):
        for response in responses:
            # Nothing is written during the tool calls: the SSE comment keeps
            # the connection alive and makes a disconnection surface at once.
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
        _log(msg + " (STRICT: exiting)")
        sys.exit(2)
    _log(msg)
    return False


def apply_patch():
    if os.environ.get("MLX_LEAK_SHIM", "1") == "0":
        _log("MLX_LEAK_SHIM=0: patch NOT applied")
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
        return _mismatch(f"mlx-lm {version}: code differs from the expected one, patch NOT applied")

    orig_advance = ArraysCache.advance

    @functools.wraps(orig_advance)
    def advance(self, N):
        orig_advance(self, N)
        # Without left_padding/lengths (non-batch cache, e.g. with --kv-bits)
        # advance() creates no chains and BatchGenerator.next() is never
        # called: registering them would hold the caches forever.
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
    """Replaces in the source of cls.<name> every block (old, new), where every
    old must appear exactly once, and recompiles the method in the namespace of
    its module, so that it resolves the same names (stream_generate, Response, ...)."""
    import mlx_lm

    version = getattr(mlx_lm, "__version__", "?")
    method = getattr(cls, name)
    try:
        src = inspect.getsource(method)
    except (OSError, TypeError):
        src = ""
    if not src or any(src.count(old) != 1 for old, _ in blocks):
        return _mismatch(f"mlx-lm {version}: {name} differs from the expected one, "
                         f"{what} patch NOT applied")

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
    _log(f"{var}={value!r} not valid, using {default}")
    return default


def apply_stop_patch():
    if os.environ.get("MLX_STOP_SHIM", "1") == "0":
        _log("MLX_STOP_SHIM=0: stop patch NOT applied")
        return False

    import mlx_lm
    from mlx_lm import server

    rg = server.ResponseGenerator
    if getattr(rg._serve_single, "_shim", None) == "stop":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    every = _env_number("MLX_STOP_SHIM_EVERY", 8, int)
    if not callable(getattr(rg, "_share_object", None)):
        return _mismatch(f"mlx-lm {version}: _share_object missing, stop patch NOT applied")
    if not _patch_method(rg, "_serve_single", [(STOP_OLD, STOP_NEW.format(every=every))], "stop"):
        return False
    _log(f"stop patch applied (mlx-lm {version}, sync every {every} tokens)")
    return True


def apply_keepalive_patch():
    if os.environ.get("MLX_KEEPALIVE_SHIM", "1") == "0":
        _log("MLX_KEEPALIVE_SHIM=0: keepalive patch NOT applied")
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
        return _mismatch(f"mlx-lm {version}: handle_completion differs from the expected one, "
                         "keepalive patch NOT applied")

    orig_completion = handler.handle_completion

    # wfile and response_generator belong to the instance (one per request):
    # they are replaced only during the call.
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
    _log(f"keepalive patch applied (mlx-lm {version}, after {every:g}s of silence)")
    return True


def apply_cache_bytes_patch():
    if os.environ.get("MLX_CACHE_SHIM", "1") == "0":
        _log("MLX_CACHE_SHIM=0: cache limit patch NOT applied")
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
        return _mismatch(f"mlx-lm {version}: LRUPromptCache/ResponseGenerator differ from "
                         "the expected ones, cache limit patch NOT applied")

    orig_init = rg.__init__

    # The limit must be set before orig_init, which starts the generation thread
    @functools.wraps(orig_init)
    def __init__(self, model_provider, prompt_cache, *args, **kwargs):
        limit = getattr(model_provider.cli_args, "prompt_cache_bytes", None)
        if limit is not None:
            prompt_cache.max_bytes = limit
            _log(f"prompt cache limited to {limit / 1e9:.2f} GB")
        orig_init(self, model_provider, prompt_cache, *args, **kwargs)

    __init__._shim = "cache-bytes"
    rg.__init__ = __init__
    _log(f"cache limit patch applied (mlx-lm {version})")
    return True


def apply_metrics_plugin():
    try:
        import mlx_metrics_plugin
    except ImportError:
        _log("mlx_metrics_plugin.py missing: /v1/metrics NOT active")
        return False
    return mlx_metrics_plugin.install()


def apply_pin_patch():
    """Must be applied after apply_keepalive_patch, because it wraps handle_completion."""
    if os.environ.get("MLX_PIN_SHIM", "1") == "0":
        _log("MLX_PIN_SHIM=0: model NOT pinned")
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
        return _mismatch(f"mlx-lm {version}: APIHandler differs from the expected one, "
                         "pinned model patch NOT applied")

    orig_completion = handler.handle_completion
    orig_models = handler.handle_models_request
    warned = set()

    @functools.wraps(orig_completion)
    def handle_completion(self, *args, **kwargs):
        pinned = self.response_generator.cli_args.model
        if pinned and self.requested_model != pinned:
            if self.requested_model not in warned and self.requested_model != "default_model":
                warned.add(self.requested_model)
                _log(f"requested model {self.requested_model!r}: served with {pinned!r}")
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
    _log(f"pinned model patch applied (mlx-lm {version})")
    return True


# Attention that caches the MLA latent and uses the result as an array
LATENT_CACHE = "update_and_fetch(kv_latent"
_latent_models = {}


def _caches_latent(model):
    """True if a module of the model caches the MLA latent, which with a
    QuantizedKVCache would come back as a tuple. Computed once per class."""
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
            _log(f"{name}: MLA attention, the KV cache stays at full precision "
                 "(--kv-bits ignored)")
    return _latent_models[cls]


def apply_kvbits_patch():
    if os.environ.get("MLX_KVBITS_SHIM", "1") == "0":
        _log("MLX_KVBITS_SHIM=0: kv-bits patch NOT applied")
        return False

    import mlx_lm
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate is also a function

    if getattr(generate.generate_step, "_shim", None) == "kvbits":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    try:
        params = inspect.signature(generate.generate_step).parameters
        src = inspect.getsource(generate.stream_generate)
    except (OSError, TypeError, ValueError):
        params, src = {}, ""
    if "kv_bits" not in params or "generate_step(prompt, model" not in src:
        return _mismatch(f"mlx-lm {version}: generate_step/stream_generate differ from "
                         "the expected ones, kv-bits patch NOT applied")

    orig_step = generate.generate_step

    # stream_generate resolves generate_step in the module namespace at every
    # call: replacing it there is enough. kv_bits is keyword-only.
    @functools.wraps(orig_step)
    def generate_step(prompt, model, *args, **kwargs):
        if kwargs.get("kv_bits") is not None and _caches_latent(model):
            kwargs["kv_bits"] = None
        return orig_step(prompt, model, *args, **kwargs)

    generate_step._shim = "kvbits"
    generate.generate_step = generate_step
    _log(f"kv-bits patch applied (mlx-lm {version})")
    return True


# ---------------- self-test (no model, no jaccl) ----------------
N_LAYERS = 48      # linear layers of Qwen3.8-27B (64 x 3/4)
N_STEPS = 12_000   # beyond 499000/48 ~ 10,400


def _phase(patched):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    if patched and not apply_patch():
        print("patch not applicable")
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
        print(f"CRASH at step {step}: {e}")
        return 3

    mx.eval([c.left_padding for c in caches])
    vals = {c.left_padding[0].item() for c in caches}
    if vals != {-N_STEPS}:
        print(f"WRONG VALUES: {vals} (expected {-N_STEPS})")
        return 5
    print(f"OK: {N_STEPS} steps x {N_LAYERS} layers, correct values")
    return 0


STOP_TOKENS = 200  # tokens generated by the fake model
STOP_AFTER = 5     # the client of rank 0 disconnects after 5 tokens


def _stop_phase(patched):
    """Two simulated ranks with two threads that run the real _serve_single on
    a fake model; the collectives are a rendezvous between the threads."""
    import queue
    from types import SimpleNamespace as NS
    from mlx_lm import server

    if patched and not apply_stop_patch():
        print("patch not applicable")
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

    def share(rank, obj):  # like _share_object: the object of rank 0 counts
        shares[rank] += 1
        if rank == 0:
            slot[0] = obj
        barrier.wait()
        val = slot[0]
        barrier.wait()
        return val

    class Client(queue.Queue):  # HTTP handler: rank 0 loses the client
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
        # tok is needed by TextStateMachine.from_tokenizer (mlx-lm > 0.32.2),
        # _make_state_machine by mlx-lm <= 0.32.2
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
    print(f"tokens generated rank0={tokens[0]} rank1={tokens[1]}, "
          f"stop collectives rank0={shares[0]} rank1={shares[1]}, errors={errs}")
    if any(t.is_alive() for t in threads):
        print("DEADLOCK")
        return 3
    if tokens[0] != tokens[1] or shares[0] != shares[1] or any(errors):
        print("DESYNC: the ranks exit at different points")
        return 3
    if tokens[0] >= STOP_TOKENS:
        print("the stop was not honored")
        return 5
    print(f"OK: both ranks stopped at token {tokens[0]}")
    return 0


KA_REASONING = 5      # reasoning tokens (written to the client right away)
KA_TOOL_TOKENS = 100  # tokens of the tool call (10 ms each, no writes)
KA_DISCONNECT = 10    # the client leaves at the 10th token of the tool call


def _keepalive_phase(patched):
    """Real APIHandler.handle_completion on a fake handler: the model writes
    a long tool call and halfway through the client leaves. With the patch the
    disconnection must surface at once, not when the generation has finished."""
    import time
    from types import SimpleNamespace as NS
    from mlx_lm import server

    os.environ["MLX_KEEPALIVE_SHIM_EVERY"] = "0.05"
    if patched and not apply_keepalive_patch():
        print("patch not applicable")
        return 1

    # Real TextStateMachine: the text of the tool call comes out only at the closing tag
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

    class WFile:  # socket to the client: after the disconnection every write fails
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

        def __getattr__(self, name):  # the other parameters of the request
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

    print(f"tokens read {pulled[0]}/{total}, keepalives before the disconnection "
          f"{handler.wfile.keepalives}, BrokenPipe={broken}, stop={ctx._should_stop}")
    if not broken or not ctx._should_stop:
        print("the disconnection was not detected")
        return 5
    if pulled[0] == total:
        print("SILENCE: disconnection detected only after the generation finished")
        return 3
    print(f"OK: disconnection detected {pulled[0] - KA_REASONING - KA_DISCONNECT} tokens later")
    return 0


CACHE_LIMIT = 4_831_838_208  # --prompt-cache-bytes of mlx_jaccl-start.sh
CACHE_ENTRY = 1_072_542_515  # ~1 GB: a conversation of 40-60k tokens, per rank


def _cache_phase(patched):
    """Real ResponseGenerator.__init__ (without generation thread) and real
    LRUPromptCache, filled as in the production log with 10 caches of ~1 GB."""
    from types import SimpleNamespace as NS
    from mlx_lm import server

    if patched and not apply_cache_bytes_patch():
        print("patch not applicable")
        return 1

    server.ResponseGenerator._run_generate = lambda self: None
    cache = server.LRUPromptCache(10)
    provider = NS(cli_args=NS(prompt_cache_bytes=CACHE_LIMIT))
    server.ResponseGenerator(provider, cache).join()
    for i in range(10):
        entry = [NS(nbytes=CACHE_ENTRY, is_trimmable=lambda: False)]
        cache.insert_cache(("fake",), [i, 1, 2, 3], entry)

    print(f"prompt cache: {len(cache)} sequences, {cache.nbytes / 1e9:.2f} GB "
          f"(limit {CACHE_LIMIT / 1e9:.2f} GB)")
    if cache.nbytes > CACHE_LIMIT:
        print("EXCEEDED: the byte limit is not applied")
        return 3
    print("OK: limit respected")
    return 0


PIN_STARTED = "mlx-community/Qwen3.8-27B-4bit"
PIN_CACHED = ("mlx-community/GLM-4.7-Flash-8bit", PIN_STARTED, "mlx-community/Qwen3.8-27B-8bit")


def _pin_phase(patched):
    """Real handle_models_request on a fake HF cache with three models, and a
    request for GLM passed through the real wrapper: the client must see and use
    only the startup model."""
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
        print("patch not applicable")
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

    print(f"/v1/models: {listed}; request for {PIN_CACHED[0]} served with {served[-1]}")
    if listed != [PIN_STARTED] or served[-1] != PIN_STARTED:
        print("the client sees and can load models other than the startup one")
        return 3
    print("OK: the client sees and uses only the startup model")
    return 0


KV_PROMPT = 12  # prompt tokens of the test
KV_START = 8    # --quantized-kv-start of the test (5000 in production)


def _kvbits_phase(patched):
    """Real tiny glm4_moe_lite and llama (random weights) with kv_bits=8 and the
    cache exceeding quantized_kv_start: GLM must generate at full precision,
    Llama must keep quantizing."""
    import mlx.core as mx
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate is also a function
    from mlx_lm.models import glm4_moe_lite, llama
    from mlx_lm.models.cache import make_prompt_cache

    if patched and not apply_kvbits_patch():
        print("patch not applicable")
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
            print(f"CRASH of {name} with --kv-bits: {e}")
            return 3
        kinds[name] = sorted({type(c).__name__ for c in cache})

    print(f"cache after the generation: {kinds}")
    if kinds != {"glm4_moe_lite": ["KVCache"], "llama": ["QuantizedKVCache"]}:
        print("caches differ from the expected ones")
        return 5
    print("OK: GLM generates at full precision, Llama stays quantized")
    return 0


def _metrics_phase(patched):
    try:
        import mlx_metrics_plugin
    except ImportError:
        print("mlx_metrics_plugin.py missing")
        return 1
    return mlx_metrics_plugin.self_test_phase(patched)


def _self_test():
    results = {}
    for phase in ("unpatched", "patched", "stop-unpatched", "stop-patched",
                  "keepalive-unpatched", "keepalive-patched",
                  "cache-unpatched", "cache-patched", "pin-unpatched", "pin-patched",
                  "kvbits-unpatched", "kvbits-patched",
                  "metrics-unpatched", "metrics-patched"):
        print(f"--- phase {phase} ---", flush=True)
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
    for name, what in (("", "leak"), ("stop-", "rank desync on stop"),
                       ("keepalive-", "silence during tool calls"),
                       ("cache-", "prompt cache limit overrun"),
                       ("pin-", "model switch on client request"),
                       ("kvbits-", "crash of MLA models with --kv-bits"),
                       ("metrics-", "missing /v1/metrics")):
        unpatched, patched = results.get(name + "unpatched"), results.get(name + "patched")
        if unpatched == 3 and patched == 0:
            print(f"PASS: {what} reproduced without the patch, absent with the patch")
        elif unpatched == 0 and patched == 0:
            print(f"INCONCLUSIVE: the simulation does not reproduce {what} on this version")
            rc = rc or 4
        else:
            print(f"FAIL ({what}): exit codes {unpatched}/{patched}")
            rc = 1
    return rc


PHASES = ("patched", "unpatched", "stop-patched", "stop-unpatched",
          "keepalive-patched", "keepalive-unpatched",
          "cache-patched", "cache-unpatched", "pin-patched", "pin-unpatched",
          "kvbits-patched", "kvbits-unpatched", "metrics-patched", "metrics-unpatched")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--self-test"]:
        return _self_test()
    if args[:1] == ["--self-test-phase"]:
        if len(args) < 2 or args[1] not in PHASES:
            print(f"usage: --self-test-phase {'|'.join(PHASES)}", file=sys.stderr)
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
        return _phase(args[1] == "patched")

    # The patches change the collectives: the ranks must have the same shim
    with open(os.path.abspath(__file__), "rb") as f:
        _log(f"shim sha256 {hashlib.sha256(f.read()).hexdigest()[:12]}")
    apply_patch()
    apply_stop_patch()
    apply_keepalive_patch()
    apply_cache_bytes_patch()
    apply_pin_patch()
    apply_kvbits_patch()
    apply_metrics_plugin()
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
