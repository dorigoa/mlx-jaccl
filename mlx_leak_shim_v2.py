#!/usr/bin/env python3
"""

execute on both mac nodes: /Volumes/Home/Alvise/miniforge3/envs/mlx/bin/python mlx_leak_shim_v2.py --self-test

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
    tokens read 19/106, keepalives before the disconnection 2, BrokenPipe=True, stop=True
    OK: disconnection detected 4 tokens later
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
--- phase penalty-unpatched ---
    (repetition, presence, logits processors): request without penalties (0.0, 0.0, 0), with 1.1/0.0 (1.1, 0.0, 1)
    IGNORED: the request without penalties has them off, not those of the server
--- phase penalty-patched ---
    [leak-shim] rank=? penalty patch applied (mlx-lm 0.32.2, server default: repetition_penalty 1, presence_penalty 1.5)
    (repetition, presence, logits processors): request without penalties (0.0, 1.5, 1), with 1.1/0.0 (1.1, 0.0, 1)
    OK: server penalties for the request without them, the client's values win
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
--- phase ranks-unpatched ---
    ranks started: same configuration [True, True], MLX_STOP_SHIM_EVERY 8/4 [True, True]
    THEY START: the collectives get out of sync at the first stop
--- phase ranks-patched ---
    ranks started: same configuration [True, True], MLX_STOP_SHIM_EVERY 8/4 [False, False]
    OK: both ranks stop at startup reporting the difference
    [leak-shim] rank=? DIFFERENT between the ranks: env MLX_STOP_SHIM_EVERY: rank0='8', rank1='4'
    [leak-shim] rank=? DIFFERENT between the ranks: env MLX_STOP_SHIM_EVERY: rank0='8', rank1='4'
--- phase mtp-unpatched ---
    model forwards per generated token: 1.00 (120 tokens), no MTP head
--- phase mtp-patched ---
    model forwards per token: 1.00 without MTP, 0.98 with the random head, 0.42 with correct drafts in 35/51 verifications (accepted per verification {0: 16, 1: 1, 2: 34})
    OK: tokens identical to the original generate_step (greedy, kv-bits, logits processors), cache consistent when stopping mid-verification
    [leak-shim] rank=? MTP patch applied (mlx-lm 0.32.2, block size of the head)
    [leak-shim] rank=? MTP active: /var/folders/.../tmpXXXX, 2 drafts per verification
PASS: leak reproduced without the patch, absent with the patch
PASS: rank desync on stop reproduced without the patch, absent with the patch
PASS: silence during tool calls reproduced without the patch, absent with the patch
PASS: prompt cache limit overrun reproduced without the patch, absent with the patch
PASS: model switch on client request reproduced without the patch, absent with the patch
PASS: server penalties ignored reproduced without the patch, absent with the patch
PASS: crash of MLA models with --kv-bits reproduced without the patch, absent with the patch
PASS: missing /v1/metrics reproduced without the patch, absent with the patch
PASS: startup with different configurations between the ranks reproduced without the patch, absent with the patch
PASS: one model forward per token reproduced without the patch, absent with the patch

----------------

mlx_leak_shim_v2.py - starts mlx_lm.server with a patch for the Metal buffer
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

----------------

Seventh patch (MTP, mlx-lm 0.32.2): Qwen3.5/3.6/3.8 have an MTP head
(multi-token prediction) that mlx-lm discards (qwen3_5.sanitize removes "mtp.*").
mlx-community publishes it separately (e.g. mlx-community/Qwen3.8-27B-MTP-8bit,
model_type qwen3_5_mtp, made for mlx-vlm): a full attention layer on
fc([token embedding, model hidden]), with the embedding and lm_head of the
target model.

Fix: when ModelProvider loads a dense qwen3_5 model, the head with "-MTP"
before the quantization is looked up locally (HF cache, no download)
(mlx-community/Qwen3.8-27B-8bit -> mlx-community/Qwen3.8-27B-MTP-8bit) and
attached to the model; generate_step then goes through _mtp_generate. At every
step the head proposes block-1 drafts (greedy), the model verifies the last
token and the drafts in a single forward, and the drafts equal to the tokens
that the model samples there are emitted, plus its own first token: the emitted
tokens are those that the model would have sampled on its own, at any
temperature. The GatedDeltaNet state (ArraysCache) cannot be shortened: during
the verification the recurrence goes one token at a time and saves the state
after every token, so the rejected tokens are discarded without recomputation;
the KV caches are shortened with trim. In the cluster, drafts and sampled
tokens are those of rank 0 (two small all_sums per verification) and the head
is active only if all the ranks have loaded it, the same one. When the last
emitted token is not yet in the cache, insert_cache excludes it from the key,
so the prompt cache stays consistent. Requests with images or with caches of
another type go through the original generate_step. MTP applies to the
requests served by _serve_single, i.e. with --kv-bits as in start.py: the
batch path (BatchGenerator) does not use generate_step.

----------------

Eighth patch (server penalties, mlx-lm 0.32.2): mlx_lm.server has options for
temp, top-p, top-k and min-p, but repetition_penalty and presence_penalty
come only from the request (do_POST: self.body.get(..., 0.0), i.e. off), so
the server cannot apply the values recommended for a model (Qwen3.8 instruct
mode: presence_penalty 1.5) to clients that do not send them.

Fix: before do_POST validates the request, the requests that do not set the
field get MLX_PENALTY_SHIM_REPETITION and MLX_PENALTY_SHIM_PRESENCE (set by
start.py on all the ranks, from --reasoning); the values sent by the client
win. repetition_penalty 1.0 leaves the logits unchanged and becomes
0.0 (off), so that no logits processor runs for nothing (with MTP it is
applied to every row of each verification). The request arguments are built
on rank 0 and shared with the other ranks as before.

----------------

Rank check at startup: before starting the server the ranks exchange
(all_gather) sha256 of shim and plugin, versions of mlx/mlx-lm, arguments,
MLX_*_SHIM* variables and applied patches. If anything differs (e.g.
MLX_STOP_SHIM_EVERY set on one node only, which would desync the collectives
at the first stop) all the ranks exit at once with the list of the
differences, instead of dying with exit 255 in the middle of a request.
If mlx_lm.server.main is missing the server does not start: re-running the
module with runpy would create new classes, without any of the patches.

Usage:
  python mlx_leak_shim_v2.py [mlx_lm.server arguments]
  python mlx_leak_shim_v2.py --self-test
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
  MLX_MTP_SHIM=0               does not use the MTP head
  MLX_MTP_SHIM_MODEL=repo|dir  MTP head to use instead of the one derived from --model
  MLX_MTP_SHIM_BLOCK=n         tokens per verification, last + n-1 drafts (default: block_size of the head, 3)
  MLX_PENALTY_SHIM_REPETITION=x  repetition_penalty of the requests that do not set it (default: off)
  MLX_PENALTY_SHIM_PRESENCE=x  presence_penalty of the requests that do not set it (default: off)
  MLX_PENALTY_SHIM=0           ignores the two variables above (back to the original behavior)
  MLX_LEAK_SHIM_STRICT=1       exits with an error if the installed code is not the expected one
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
# Shim variables: they must be the same on all the ranks
SHIM_ENV = ("MLX_LEAK_SHIM", "MLX_LEAK_SHIM_STRICT", "MLX_STOP_SHIM", "MLX_STOP_SHIM_EVERY",
            "MLX_KEEPALIVE_SHIM", "MLX_KEEPALIVE_SHIM_EVERY", "MLX_CACHE_SHIM", "MLX_PIN_SHIM",
            "MLX_KVBITS_SHIM", "MLX_METRICS_PLUGIN", "MLX_MTP_SHIM", "MLX_MTP_SHIM_MODEL",
            "MLX_MTP_SHIM_BLOCK", "MLX_PENALTY_SHIM", "MLX_PENALTY_SHIM_REPETITION",
            "MLX_PENALTY_SHIM_PRESENCE")
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


# Variables with the server default of the penalties that mlx_lm.server
# reads only from the request, with this line of do_POST
PENALTY_ENV = {"MLX_PENALTY_SHIM_REPETITION": "repetition_penalty",
               "MLX_PENALTY_SHIM_PRESENCE": "presence_penalty"}
PENALTY_GET = 'self.{0} = self.body.get("{0}", 0.0)'


def apply_penalty_patch():
    """Server default of the penalties for the requests that do not set them."""
    if os.environ.get("MLX_PENALTY_SHIM", "1") == "0":
        _log("MLX_PENALTY_SHIM=0: penalty patch NOT applied")
        return False

    defaults = {}
    for var, field in PENALTY_ENV.items():
        value = os.environ.get(var)
        if value:
            try:
                defaults[field] = float(value)
            except ValueError:
                _log(f"{var}={value!r} not a number: {field} off")

    import mlx_lm
    from mlx_lm import server

    handler = server.APIHandler
    if getattr(handler.validate_model_parameters, "_shim", None) == "penalty":
        return True

    version = getattr(mlx_lm, "__version__", "?")
    try:
        src = inspect.getsource(handler.do_POST)
    except (OSError, TypeError):
        src = ""
    validate = src.find("self.validate_model_parameters()")
    if validate < 0 or not all(0 <= src.find(PENALTY_GET.format(field)) < validate
                               for field in PENALTY_ENV.values()):
        return _mismatch(f"mlx-lm {version}: APIHandler.do_POST differs from the expected one, "
                         "penalty patch NOT applied")

    orig_validate = handler.validate_model_parameters

    @functools.wraps(orig_validate)
    def validate_model_parameters(self):
        # do_POST has read the penalties (0.0 when missing) and validates them now
        for field, value in defaults.items():
            if field not in self.body:
                setattr(self, field, value)
        # 1.0 leaves the logits unchanged: off, as 0.0, so no logits processor
        if self.repetition_penalty == 1:
            self.repetition_penalty = 0.0
        return orig_validate(self)

    validate_model_parameters._shim = "penalty"
    handler.validate_model_parameters = validate_model_parameters
    _log(f"penalty patch applied (mlx-lm {version}, server default: "
         + (", ".join(f"{field} {value:g}" for field, value in defaults.items()) or "none")
         + ")")
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


# ---------------- rank check at startup ----------------
CONFIG_BYTES = 8192  # room for the configuration of each rank in the all_gather


def _file_sha(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        return None


def _rank_config(applied, args, env=None):
    """What decides the collectives: shim, plugin, versions, arguments,
    shim variables and patches actually applied."""
    import mlx.core as mx
    import mlx_lm

    env = os.environ if env is None else env
    here = os.path.dirname(os.path.abspath(__file__))
    config = {
        "sha256 shim": _file_sha(os.path.abspath(__file__)),
        "sha256 plugin": _file_sha(os.path.join(here, "mlx_metrics_plugin.py")),
        "mlx": mx.__version__,
        "mlx-lm": getattr(mlx_lm, "__version__", "?"),
        "arguments": list(args),
    }
    config.update({f"env {name}": env.get(name) for name in SHIM_ENV})
    config.update({f"patch {name}": ok for name, ok in applied.items()})
    return config


def _gather_configs(blob):
    """all_gather of the configurations (JSON blobs) of all the ranks."""
    import mlx.core as mx

    group = mx.distributed.init()
    if group.size() == 1:
        return [blob]
    data = mx.array(list(blob.ljust(CONFIG_BYTES, b"\0")), dtype=mx.int32)
    raw = bytes(mx.distributed.all_gather(data, group=group).tolist())
    return [raw[i * CONFIG_BYTES:(i + 1) * CONFIG_BYTES].rstrip(b"\0")
            for i in range(group.size())]


def _check_ranks(config, gather=_gather_configs):
    """True if all the ranks have the same configuration; otherwise logs
    the differences. Must be called by all the ranks."""
    blob = json.dumps(config, sort_keys=True).encode()
    if len(blob) > CONFIG_BYTES:
        blob = json.dumps({"sha256 configuration": hashlib.sha256(blob).hexdigest()}).encode()
    configs = [json.loads(b) for b in gather(blob)]
    diff = sorted(k for k in set().union(*configs)
                  if len({json.dumps(c.get(k)) for c in configs}) > 1)
    for key in diff:
        values = ", ".join(f"rank{i}={c.get(key)!r}" for i, c in enumerate(configs))
        _log(f"DIFFERENT between the ranks: {key}: {values}")
    if not diff and len(configs) > 1:
        _log(f"identical configuration on {len(configs)} ranks")
    return not diff


# ---------------- MTP (multi-token prediction) ----------------
# Recurrence of GatedDeltaNet.__call__ (qwen3_5) replaced by the MTP patch
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
            # [mtp-shim] MTP verification: state after every token of the block,
            # to discard the rejected tokens without recomputing
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
# Arguments of generate_step handled by _mtp_generate
MTP_STEP_ARGS = ("max_tokens", "sampler", "logits_processors", "max_kv_size", "prompt_cache",
                 "prefill_step_size", "kv_bits", "kv_group_size", "quantized_kv_start",
                 "prompt_progress_callback", "input_embeddings")
MTP_CACHES = ("ArraysCache", "KVCache", "QuantizedKVCache")
# Fields of text_config that the MTP head and the model must have equal
MTP_MATCH = ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
             "head_dim", "intermediate_size", "rms_norm_eps")
MTP_QUANT = re.compile(r"(.+)-(\d+bit|bf16|fp16|mxfp\d+|nvfp\d+)")


def _mtp_record(cache, conv_input, q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel):
    """GatedDeltaNet recurrence one token at a time: conv and state after every
    token stay in cache._mtp_steps until _mtp_commit chooses."""
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
    """Of the `total` tokens just verified the model cache keeps the first `keep`."""
    for c in prompt_cache:
        steps = c.__dict__.pop("_mtp_steps", None)
        if steps is not None:
            c[0], c[1] = steps[keep - 1]
        elif keep < total:
            c.trim(total - keep)


def _lm_logits(lm, hidden):
    """Tail of qwen3_5.TextModel.__call__: lm_head (sharded by vocabulary)."""
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
        """qwen3_5_mtp head: fc([token embedding, hidden]) and a full
        attention layer; no weights shared with the model."""

        def __init__(self, args, n_layers):
            super().__init__()
            H, eps = args.hidden_size, args.rms_norm_eps
            self.fc = nn.Linear(2 * H, H, bias=False)
            self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
            self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
            # with this layer_idx DecoderLayer uses the full attention
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
    """MTP head attached to the model as a plain attribute: it is not one of
    its sub-modules, so it stays out of parameters() and of the sharding (every
    rank has a whole copy of it, without collectives)."""

    def __init__(self, module, block, name):
        self.module, self.block, self.name = module, block, name

    def make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in self.module.layers]

    def forward(self, lm, tokens, hidden, cache):
        """hidden[:, j] (of the model or of the head) with the token that follows it."""
        return self.module(lm.model.embed_tokens(tokens), hidden, cache)

    def draft(self, lm, hidden, index):
        """Greedy draft of the token `index` of the answer from the last hidden."""
        import mlx.core as mx
        return mx.argmax(_lm_logits(lm, hidden[:, -1, :]), axis=-1)


def _mtp_source(model_path):
    """(name, local folder) of the MTP head of the model; no download."""
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
        raise ValueError(f"model_type {config.get('model_type')!r}, expected 'qwen3_5_mtp'")
    for key in MTP_MATCH:
        if key in text and text[key] != getattr(lm.args, key):
            raise ValueError(f"{key}={text[key]} but the model has {getattr(lm.args, key)}")
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
    """Attaches to model._mtp_shim the MTP head of the model just loaded (or
    None). In the cluster it must be called by all the ranks: the head stays
    active only if all of them have loaded it with the same block."""
    import mlx.core as mx

    drafter, why = None, None
    lm = getattr(model, "language_model", None)
    model_type = getattr(model, "model_type", type(model).__name__)
    if model_type != "qwen3_5" or getattr(getattr(lm, "args", None), "num_experts", 1) != 0:
        why = f"{model_type}: only dense qwen3_5 is supported"
    else:
        name, path = _mtp_source(model_path)
        if name is None:
            why = f"no MTP head associated with {model_path!r}"
        elif path is None:
            why = f"MTP head {name!r} not present locally"
        else:
            try:
                drafter = _load_mtp(lm, path, name, block)
            except Exception as e:
                why = f"MTP head {name!r} not loaded: {e}"

    group = mx.distributed.init()
    if group.size() > 1:
        mine = mx.array([[1, drafter.block] if drafter else [0, 0]], dtype=mx.int32)
        rows = mx.distributed.all_gather(mine, group=group).tolist()
        if drafter is not None and any(row != rows[0] for row in rows):
            drafter, why = None, f"MTP head differs between the ranks ([loaded, block] = {rows})"

    model._mtp_shim = drafter
    if drafter is not None:
        _log(f"MTP active: {drafter.name}, {drafter.block - 1} drafts per verification")
    else:
        _log(f"MTP not active: {why}")


def _mtp_generate(prompt, model, drafter, stream, *, max_tokens=256, sampler=None,
                  logits_processors=None, max_kv_size=None, prompt_cache=None,
                  prefill_step_size=2048, kv_bits=None, kv_group_size=64,
                  quantized_kv_start=5000, prompt_progress_callback=None,
                  input_embeddings=None):
    """generate_step with the MTP head. At every verification the model processes
    the last emitted token and block-1 drafts; the drafts equal to the token
    that the model samples in their position are emitted, plus the first token
    of the model that contradicts them (or that follows the last one)."""
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
    history = None  # tokens already in the cache, for the logits processors

    def quantize():
        for c in (prompt_cache, mtp_cache):
            generate.maybe_quantize_kv_cache(c, quantized_kv_start, kv_group_size, kv_bits)

    def from_rank0(x):
        # all the ranks use the drafts and tokens of rank 0: they do not diverge
        # even if the replicated computations differed by one bit. x * 0 and not
        # zeros_like(x): the other ranks too must compute x, and its collectives,
        # before this all_sum, otherwise the order of the collectives changes
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
        # Prefill: the head pairs the hidden of position j with the token j+1
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
        pending = True  # b emitted but not yet in the cache of the model
        try:
            yield b, logprobs[0]
            while max_tokens < 0 or emitted < max_tokens:
                # Drafts: the first one is already ready, the others from the head on its hidden
                drafts, h, appended = [seed], last, 0
                for i in range(1, k):
                    h = drafter.forward(lm, drafts[-1][None], h, mtp_cache)
                    drafts.append(drafter.draft(lm, h, emitted + i))
                    appended += 1
                drafts = from_rank0(mx.concatenate(drafts))

                # Verification: one forward on [b, drafts]
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

                # Head: keeps the entries of the confirmed drafts and adds the
                # other positions with the real hidden of the model
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
                    # even if the caller stops halfway: the cache of the model
                    # covers b and the received tokens, except the last one if it is not a draft
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
                # insert_cache excludes from the key the token not yet in the cache
                prompt_cache[0]._mtp_behind = 1
            if rounds and group.rank() == 0:
                _log(f"MTP: {emitted} tokens, {(emitted - 1) / rounds:.2f} per verification, "
                     f"drafts accepted {accepted}/{rounds * k}")


def apply_mtp_patch():
    """Must be applied after apply_kvbits_patch, because it wraps generate_step."""
    if os.environ.get("MLX_MTP_SHIM", "1") == "0":
        _log("MLX_MTP_SHIM=0: MTP NOT active")
        return False

    import mlx_lm
    from mlx_lm import server
    from mlx_lm.models import qwen3_5
    from mlx_lm.models.cache import LRUPromptCache
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate is also a function

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
        return _mismatch(f"mlx-lm {version}: generate_step/ModelProvider/qwen3_5 differ "
                         "from the expected ones, MTP patch NOT applied")

    block = None
    if os.environ.get("MLX_MTP_SHIM_BLOCK"):
        block = max(2, _env_number("MLX_MTP_SHIM_BLOCK", 3, int))

    qwen3_5._mtp_shim_record = _mtp_record  # used by the recompiled source
    if not _patch_method(qwen3_5.GatedDeltaNet, "__call__", [(MTP_GDN_OLD, MTP_GDN_NEW)], "mtp"):
        return False

    orig_step = generate.generate_step

    # stream_generate calls generate_step(prompt, model, stream, **kwargs)
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
            # the last token emitted by MTP is not yet in the cache
            tokens = tokens[:-1]
            if not tokens:
                return None
        return orig_insert(self, model, tokens, prompt_cache, *args, **kwargs)

    generate_step._mtp_shim = True
    generate.generate_step = generate_step
    server.ModelProvider._load = _load
    LRUPromptCache.insert_cache = insert_cache
    _log(f"MTP patch applied (mlx-lm {version}"
         + (f", block {block})" if block else ", block size of the head)"))
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


def _penalty_phase(patched):
    """Real do_POST on a handler without socket, server started as by
    start.py --reasoning off: a request without penalties must get those of
    the server, with repetition 1.0 off (no logits processor), and one with
    its own penalties must keep them."""
    import io
    from types import SimpleNamespace as NS
    from mlx_lm import server
    from mlx_lm.sample_utils import make_logits_processors

    seen = []
    server.APIHandler.handle_completion = lambda self, request, stop_words: seen.append(
        (self.repetition_penalty, self.presence_penalty, len(make_logits_processors(
            None, self.repetition_penalty, 20, self.presence_penalty, 20))))

    os.environ.update(MLX_PENALTY_SHIM_REPETITION="1.0", MLX_PENALTY_SHIM_PRESENCE="1.5")
    if patched and not apply_penalty_patch():
        print("patch not applicable")
        return 1

    def post(body):
        raw = json.dumps(dict(prompt="hello", **body)).encode()
        h = object.__new__(server.APIHandler)
        h.path, h.headers, h.rfile = "/v1/completions", {"Content-Length": str(len(raw))}, \
            io.BytesIO(raw)
        h.wfile = io.BytesIO()
        h.response_generator = NS(cli_args=NS(num_draft_tokens=3, max_tokens=16, temp=0.7,
                                              top_p=0.8, top_k=20, min_p=0.0))
        h._set_completion_headers = h.end_headers = lambda *a: None
        server.APIHandler.do_POST(h)
        return seen.pop() if seen else h.wfile.getvalue()

    plain = post({})
    own = post({"repetition_penalty": 1.1, "presence_penalty": 0.0})
    print(f"(repetition, presence, logits processors): request without penalties {plain}, "
          f"with 1.1/0.0 {own}")
    if plain == (0.0, 0.0, 0):
        print("IGNORED: the request without penalties has them off, not those of the server")
        return 3
    if plain != (0.0, 1.5, 1) or own != (1.1, 0.0, 1):
        print("WRONG: expected (0.0, 1.5, 1) and (1.1, 0.0, 1)")
        return 1
    print("OK: server penalties for the request without them, the client's values win")
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


def _ranks_phase(patched):
    """Two simulated ranks (threads, all_gather as a rendezvous) start with
    different MLX_STOP_SHIM_EVERY, as with the variable set on one node only.
    The previous shim only writes the sha in the log and the ranks start; the
    check must stop both of them, but not ranks with the same configuration."""
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
    print(f"ranks started: same configuration {same}, MLX_STOP_SHIM_EVERY 8/4 {diff}")
    if same != [True, True]:
        print("FALSE ALARM: ranks with the same configuration were stopped")
        return 5
    if diff == [True, True]:
        print("THEY START: the collectives get out of sync at the first stop")
        return 3
    if diff != [False, False]:
        print("only one rank stops")
        return 5
    print("OK: both ranks stop at startup reporting the difference")
    return 0


MTP_PROMPT = 40   # prompt tokens: prefill in blocks of 16 (16 + 16 + 7 + 1)
MTP_TOKENS = 120  # generated tokens
MTP_WRONG = (2, 5)  # the oracle gets the drafts with index % 7 in MTP_WRONG wrong


def _tiny_qwen3_5():
    """Tiny qwen3_5 with random weights (float32): 6 GatedDeltaNet layers and 2
    full attention ones, the 3:1 pattern of Qwen3.8."""
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
    """Acceptances per verification and final entries of the head cache expected
    with the oracle of _mtp_phase (prefill: one entry per prompt token; every
    verification: one per accepted draft and one for the token of the model)."""
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
    """Real tiny qwen3_5 and MTP head with random weights, saved and reloaded
    like the mlx-community one (8 bit). In greedy mode MTP must give the same
    tokens as the original generate_step (also with kv-bits and logits processors),
    with fewer model forwards, and leave the cache as if the received tokens
    had been processed one by one, even when stopping mid-verification."""
    import contextlib
    import io
    import tempfile
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors
    generate = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate is also a function

    model = _tiny_qwen3_5()
    lm = model.language_model
    V = lm.args.vocab_size
    prompt = [(7 * i + 3) % V for i in range(MTP_PROMPT)]
    greedy = lambda x: mx.argmax(x, axis=-1)
    orig_step = generate.generate_step

    if patched and not apply_mtp_patch():
        print("patch not applicable")
        return 1

    calls = [0]
    orig_call = qwen3_5.Qwen3_5TextModel.__call__

    def counted(self, *args, **kwargs):
        calls[0] += 1
        return orig_call(self, *args, **kwargs)

    qwen3_5.Qwen3_5TextModel.__call__ = counted

    def run(step, stop=MTP_TOKENS, **kwargs):
        """(tokens, cache, model forwards per token beyond the 4 of the prefill)"""
        calls[0] = 0
        cache = make_prompt_cache(model)
        gen = step(mx.array(prompt), model, max_tokens=MTP_TOKENS, sampler=greedy,
                   prompt_cache=cache, prefill_step_size=16, **kwargs)
        out = []
        with contextlib.redirect_stderr(io.StringIO()):  # MTP statistics
            for token, _ in gen:
                out.append(token)
                if len(out) == stop:
                    break
            gen.close()
        return out, cache, (calls[0] - 4) / len(out)

    ref, _, plain = run(orig_step)
    if not patched:
        print(f"model forwards per generated token: {plain:.2f} ({len(ref)} tokens), "
              "no MTP head")
        return 3

    # Random MTP head in the mlx-community format, reloaded by _attach_mtp
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
        print("MTP head not loaded")
        return 5

    class Oracle(_MTPDrafter):
        """Drafts from the expected answer, wrong if index % 7 in MTP_WRONG."""
        expected = ref

        def make_cache(self):
            self.cache = super().make_cache()
            return self.cache

        def draft(self, lm, hidden, index):
            token = self.expected[index] if index < len(self.expected) else 0
            if index % 7 in MTP_WRONG:
                token = (token + 1) % V
            return mx.array([token], dtype=mx.int32)

    oracle = Oracle(loaded.module, loaded.block, "oracle")
    k = oracle.block - 1
    failures = []

    model._mtp_shim = None  # without head: original generate_step
    out, _, rate = run(generate.generate_step)
    if out != ref or rate != plain:
        failures.append("without head")

    model._mtp_shim = loaded
    out, _, random_rate = run(generate.generate_step)
    if out != ref:
        failures.append("random head")

    model._mtp_shim = oracle
    out, cache, oracle_rate = run(generate.generate_step)
    hist, entries = _mtp_expected_rounds(MTP_TOKENS, k)
    if out != ref:
        failures.append("oracle")
    if round(oracle_rate * len(out)) != sum(hist):
        failures.append(f"verifications {round(oracle_rate * len(out))} instead of {sum(hist)}")
    if oracle.cache[0].offset != entries:
        failures.append(f"head cache at {oracle.cache[0].offset} instead of {entries}")

    for name, kwargs in (("kv-bits", dict(kv_bits=8, kv_group_size=32, quantized_kv_start=16)),
                         ("logits processors", dict(logits_processors=make_logits_processors(
                             repetition_penalty=1.5, repetition_context_size=16)))):
        expected, _, _ = run(orig_step, **kwargs)
        oracle.expected = expected
        out, cache, _ = run(generate.generate_step, **kwargs)
        if out != expected:
            failures.append(name)
        if "kv_bits" in kwargs and "QuantizedKVCache" not in {type(c).__name__ for c in cache}:
            failures.append("kv-bits: cache not quantized")
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

    # The caller stops after `stop` tokens: cache and prompt cache consistent
    behind_seen = set()
    for stop in range(25, 33):
        out, cache, _ = run(generate.generate_step, stop=stop)
        behind = cache[0].__dict__.get("_mtp_behind", 0)
        behind_seen.add(behind)
        reference = make_prompt_cache(model)
        model(mx.array(prompt + out[:len(out) - behind])[None], cache=reference)
        if out != ref[:stop] or not same_cache(cache, reference):
            failures.append(f"cache when stopping at {stop} tokens")
        lru = LRUPromptCache(4)
        lru.insert_cache(("tiny",), prompt + out, cache)
        _, rest = lru.fetch_nearest_cache(("tiny",), prompt + out)
        if len(rest) != behind:
            failures.append(f"prompt cache when stopping at {stop} tokens")
    if behind_seen != {0, 1}:
        failures.append(f"last token out of the cache only {behind_seen}")

    print(f"model forwards per token: {plain:.2f} without MTP, {random_rate:.2f} with the "
          f"random head, {oracle_rate:.2f} with correct drafts in {sum(hist[1:])}/{sum(hist)} "
          f"verifications (accepted per verification {dict(enumerate(hist))})")
    if failures:
        print(f"ERRORS: {failures}")
        return 5
    print("OK: tokens identical to the original generate_step (greedy, kv-bits, logits processors), "
          "cache consistent when stopping mid-verification")
    return 0


def _self_test():
    results = {}
    for phase in ("unpatched", "patched", "stop-unpatched", "stop-patched",
                  "keepalive-unpatched", "keepalive-patched",
                  "cache-unpatched", "cache-patched", "pin-unpatched", "pin-patched",
                  "penalty-unpatched", "penalty-patched", "kvbits-unpatched", "kvbits-patched",
                  "metrics-unpatched", "metrics-patched",
                  "ranks-unpatched", "ranks-patched", "mtp-unpatched", "mtp-patched"):
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
                       ("penalty-", "server penalties ignored"),
                       ("kvbits-", "crash of MLA models with --kv-bits"),
                       ("metrics-", "missing /v1/metrics"),
                       ("ranks-", "startup with different configurations between the ranks"),
                       ("mtp-", "one model forward per token")):
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
          "penalty-patched", "penalty-unpatched",
          "kvbits-patched", "kvbits-unpatched", "metrics-patched", "metrics-unpatched",
          "ranks-patched", "ranks-unpatched", "mtp-patched", "mtp-unpatched")


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
        if args[1].startswith("penalty-"):
            return _penalty_phase(args[1] == "penalty-patched")
        if args[1].startswith("kvbits-"):
            return _kvbits_phase(args[1] == "kvbits-patched")
        if args[1].startswith("metrics-"):
            return _metrics_phase(args[1] == "metrics-patched")
        if args[1].startswith("ranks-"):
            return _ranks_phase(args[1] == "ranks-patched")
        if args[1].startswith("mtp-"):
            return _mtp_phase(args[1] == "mtp-patched")
        return _phase(args[1] == "patched")

    # The patches change the collectives: the ranks must have the same shim
    _log(f"shim sha256 {_file_sha(os.path.abspath(__file__))}")
    applied = {
        "leak": apply_patch(),
        "stop": apply_stop_patch(),
        "keepalive": apply_keepalive_patch(),
        "cache": apply_cache_bytes_patch(),
        "pin": apply_pin_patch(),
        "penalty": apply_penalty_patch(),
        "kvbits": apply_kvbits_patch(),
        "mtp": apply_mtp_patch(),
        "metrics": apply_metrics_plugin(),
    }
    if not _check_ranks(_rank_config(applied, args)):
        _log("shim or configuration differs between the ranks: the server does NOT start")
        return 3
    sys.argv = ["mlx_lm.server"] + args
    try:
        from mlx_lm.server import main as server_main
    except ImportError as e:
        # Re-running mlx_lm.server with runpy would create new classes, without the patches
        _log(f"mlx_lm.server.main not available ({e}): the server does NOT start")
        return 2
    server_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
