# Distributed MLX over RDMA on Thunderbolt 5 — 2× Mac Studio
## + Patch for metrics

Oct 22, 2026 · @Alvise Dorigo

Reference doc: https://ml-explore.github.io/mlx/build/html/usage/distributed.html

Verified setup: 2× Mac Studio M4 Max 36 GB over Thunderbolt 5, mlx 0.32.4 + mlx-lm 0.32.2, JACCL (RDMA) backend, tensor parallelism across 2 nodes. Qwen3.6-27B-8bit generates 22.4 tok/s. Qwen3.8-27B-8bit generates 15–18 tok/s, about 30–35 tok/s with its MTP head (`mlx_leak_shim_v2.py`, section 6).

# Table of contents
1. [Prerequisites and enabling RDMA](#1-prerequisites-and-enabling-rdma)
2. [Thunderbolt network: IP, MTU 9000, bridge](#2-thunderbolt-network-ip-mtu-9000-bridge)
3. [Python: Miniforge and the mlx environment](#3-python-miniforge-and-the-mlx-environment)
4. [Hugging Face CLI (hf): installation, token, download](#4-hugging-face-cli-hf-installation-token-download)
5. [Passwordless SSH](#5-passwordless-ssh)
6. [Helper scripts and launch](#6-helper-scripts-and-launch)
7. [Compatible models and memory budget](#7-compatible-models-and-memory-budget)
8. [Tuning and troubleshooting](#8-tuning-and-troubleshooting)
9. [Glossary and references](#9-glossary-and-references)

## 1. Prerequisites and enabling RDMA

Both Macs need macOS 26.2 or later, and a direct certified Thunderbolt 5 cable (80 Gbps) between the two.

Enabling RDMA, on both Mac:

1. Boot into recoveryOS: with the Mac off, hold the power button down until "Options" appears.
2. Menu Utility → Terminal.
3. Run `rdma_ctl enable`.
4. Reboot normally.

Verification, on both Macs:

```bash
sw_vers -productVersion
rdma_ctl status
ibv_devinfo | grep -E 'hca_id|state'
```

Expected: version ≥ 26.2, `enabled`, device `rdma_en3` (Mac1) / `rdma_en4` (Mac2) in state `PORT_ACTIVE` (`rdma_en3` and `rdma_en4` are for my specific setup).

## 2. Thunderbolt network: IP, MTU 9000, bridge

Mac1 is rank 0: all jobs are launched from it.

| Node | Interface | IP | RDMA device |
| --- | --- | --- | --- |
| Mac1 (rank 0) | en3 | 192.168.20.1/24 | rdma\_en3 |
| Mac2 (rank 1) | en4 | 192.168.20.2/24 | rdma\_en4 |

Disable the Thunderbolt Bridge (recommended by the MLX documentation even though RDMA doesn't use TCP/IP), on both Macs:

```bash
sudo networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off
```

IP and MTU (not persistent; to make them permanent use System Settings → Network on the en3/en4 service, hardware mapping via `networksetup -listallhardwareports`):

```bash
# Mac1
sudo ifconfig en3 inet 192.168.20.1 netmask 255.255.255.0 mtu 9000
# Mac2
sudo ifconfig en4 inet 192.168.20.2 netmask 255.255.255.0 mtu 9000
```

Verify jumbo frames from Mac1 (8972 = 9000 − 28 bytes of IP+ICMP headers, `-D` forbids fragmentation):

```bash
ping -c3 -D -s 7168 192.168.20.2
```

## 3. Python: Miniforge and the mlx environment

The environment must be identical and at the same path on both Macs: `mlx.launch` starts the same Python executable over SSH on both.

Installing Miniforge (on both Macs, if not present):

```bash
curl -fsSLO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init "$(basename "$SHELL")"
exec "$SHELL" -l
```

Environment and packages (on both Macs, versions pinned for reproducibility):

```bash
conda create -y -n mlx python=3.12
conda activate mlx
pip install mlx==0.32.4 mlx-lm==0.32.2
python -c "import mlx.core as mx, mlx_lm; print(mx.__version__, mlx_lm.__version__)"
which python
```

Expected: `0.32.4 0.32.2` (or similar numbers) and the path `$HOME/miniforge3/envs/mlx/bin/python` identical on both Macs (here `/Volumes/Home/Alvise/...`).

### 3.1 Updating mlx and mlx-lm

The patches in `mlx_leak_shim_v2.py` and the `/v1/metrics` plugin (`mlx_metrics_plugin.py`) don't modify any file of mlx or mlx-lm: they are applied in memory at every server start. They do hook into internals of mlx-lm 0.32.2 (private methods, signatures, exact code blocks), which a new release can change. Before applying a patch, the shim compares the installed code with the expected one:

- if it differs, the log says `… differs from the expected one, … patch NOT applied`; with `MLX_LEAK_SHIM_STRICT=1` (always set by `start.py`) the line ends with `(STRICT: exiting)` and the server does not start (exit 2);
- if the metrics plugin can't hook in, the server starts without `/v1/metrics` and `start.py` prints no `[tps]` lines;
- if the two Macs have different versions, the shim stops at startup with `DIFFERENT between the ranks: mlx-lm: rank0=…, rank1=…`;
- if `mlx.launch` (mlx 0.32.4) has changed, `start.py` prints `[start] mlx.launch not patched, …` and starts anyway, with Mac1 at 1-2 busy CPU cores (`mlx_launch_shim.py`, section 6).

Most fragile are the distributed stop and the MTP patches, which replace exact code blocks; the metrics plugin and the `[tps]` lines of `start.py` depend on the least. A release can also change behaviour without touching the checked code: the self-test is there for that.

Steps, on both Macs (`mlx` env active):

1. Note the current versions, to go back if needed:

   ```bash
   python -c "import mlx.core as mx, mlx_lm; print(mx.__version__, mlx_lm.__version__)"
   ```

2. Update both Macs to the same versions (`mlx-metal` follows `mlx`):

   ```bash
   pip install -U mlx mlx-lm
   ```

3. Run the self-test, from the repository folder:

   ```bash
   python mlx_leak_shim_v2.py --self-test
   ```

   For each patch it reproduces the problem without the patch and checks that the patch removes it:
   - `PASS`: the patch applies and is still needed;
   - `INCONCLUSIVE`: the patch applies but the new mlx-lm no longer shows the problem, it was probably fixed upstream; the patch can be switched off;
   - `FAIL`: the patch no longer applies (or no longer works) and the shim must be adapted; meanwhile switch that patch off or go back to the previous versions.

   Exit code 0 means all `PASS`, 1 at least one `FAIL`, 4 at least one `INCONCLUSIVE` and no `FAIL`.

4. Start the server (section 6) and check the log: one `… patch applied` line per patch on each rank, `identical configuration on 2 ranks`, `MTP active: …` and `/v1/metrics active`, and no `mlx.launch not patched` line.

5. To go back to the verified versions:

   ```bash
   pip install mlx==0.32.4 mlx-metal==0.32.4 mlx-lm==0.32.2
   ```

To switch off a single patch, the variable (listed at the top of `mlx_leak_shim_v2.py`, e.g. `MLX_STOP_SHIM=0`) must reach both ranks with the same value: add `"--env", "MLX_STOP_SHIM=0",` to the `mlx.launch` command in `build_command()` of `start.py`. A variable exported only in the local shell reaches rank 0 alone, and the shim stops at startup because the configuration differs between the ranks. Only `MLX_LEAK_SHIM` is already forwarded by `start.py` from the shell.

## 4. Hugging Face CLI (hf): installation, token, download

The `hf` command is part of the `huggingface_hub` package, already installed as a dependency of mlx-lm.

Update and verify (on both Macs, `mlx` env active):

```bash
pip install -U huggingface_hub
which hf
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

Token: huggingface.co → Settings → Access Tokens → New token, type **Read**. Export it as `HF_TOKEN` in the shell.

Login (on both Macs). The token is saved to `~/.cache/huggingface/token` and stays valid in non-interactive SSH sessions too, where `HF_TOKEN` may not be defined:

```bash
hf auth login --token "$HF_TOKEN"
hf auth whoami
```

For gated (access-restricted) models, accept the license on the model page first.

## 5. Passwordless SSH

Assumed to be already set up.

## 6. Helper scripts and launch
```bash
git clone git@github.com:dorigoa/mlx-jaccl.git # on both Macs, at the same path
cd mlx-jaccl
```

| File | Purpose |
| --- | --- |
| `start.py` | Builds and runs the `mlx.launch` command (JACCL backend, hostfile, shim, model options), prints the speed of each finished request, and when it is stopped stops the ranks on all the hosts |
| `mlx_leak_shim_v2.py` | Default shim of `start.py`: starts `mlx_lm.server` with in-memory patches (Metal buffer leak, distributed stop, SSE keepalive during tool calls, byte limit of the prompt cache, model pinned to `--model`, server default of repetition and presence penalty, `--kv-bits` on MLA models, rank configuration check at startup, MTP) |
| `mlx_leak_shim.py` | Previous shim, without rank check, MTP and server penalties: it doesn't accept `--presence-penalty`/`--repetition-penalty`, so to go back use the `start.py` of commit 8f31674 with `-leak-patch ~/mlx-jaccl/mlx_leak_shim.py` |
| `mlx_launch_shim.py` | Runs `mlx.launch` for `start.py` without its busy loop (in-memory patch: ~0% CPU on Mac1 instead of ~130%) |
| `mlx_metrics_plugin.py` | `GET /v1/metrics`, loaded by both shims |
| `hosts.json` | `mlx.launch` hostfile |
| `active-request.sh` | Live view of the running request |
| `rdma-monitor.sh`, `rdma_monitor.d` | RDMA traffic and latency of the rank on this Mac (DTrace, needs `sudo`) |

Self-test, especially after an update of mlx and mlx-lm (section 3.1):

```bash
conda activate mlx # if not already done
python mlx_leak_shim_v2.py --self-test
```

Launch from Mac1 (rank 0). `start.py` passes absolute paths (by default `hosts.json` and `mlx_leak_shim_v2.py` next to it), so the repository must be at the same path on both Macs, and the two ranks check at startup that they run the same shim, plugin, versions and variables:

```
./start.py -8bit -host 0.0.0.0
Server start: model=mlx-community/Qwen3.8-27B-8bit, port=8080, reasoning=on (temp 1.0, top-p 0.95, top-k 20, min-p 0.0, presence-penalty 0.0, repetition-penalty 1.0)
[INFO] Running /Volumes/Home/Alvise/miniforge3/envs/mlx/bin/python -X mlx_jaccl_tag=5675b25b35b6 /Volumes/Home/Alvise/mlx-jaccl/mlx_leak_shim_v2.py --model mlx-community/Qwen3.8-27B-8bit --host 0.0.0.0 --port 8080 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repetition-penalty 1.0 --max-tokens 262144 --log-level INFO --prompt-cache-bytes 0G --prefill-step-size 256 --chat-template-args '{"enable_thinking": true, "reasoning_effort": "medium"}' --kv-bits 8
[leak-shim] rank=0 penalty patch applied (mlx-lm 0.32.2, server default: repetition_penalty 1.0, presence_penalty 0.0)
[leak-shim] rank=0 MTP patch applied (mlx-lm 0.32.2, block size of the head)
[leak-shim] rank=0 identical configuration on 2 ranks
[leak-shim] rank=0 MTP active: mlx-community/Qwen3.8-27B-MTP-8bit, 2 drafts per verification
...
[leak-shim] rank=0 MTP: 1109 tokens, 2.61 per verification, drafts accepted 684/850
[tps] 17:07:23 last request: 34.6 tok/s (1109 tokens in 32.0s, stop), prefill 285.4 tok/s (153 tokens), average since start 34.6 tok/s
```

`./start.py -h` lists all the options. `--reasoning on` (the default) or `--reasoning off` switches thinking on or off and sets the sampling that [Unsloth recommends for Qwen3.8-27B](https://unsloth.ai/docs/models/qwen3.8) in that mode:

| `--reasoning` | temp | top-p | top-k | min-p | presence penalty | repetition penalty |
| --- | --- | --- | --- | --- | --- | --- |
| `on` (thinking mode) | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| `off` (instruct mode) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

These are the server defaults: a client that sends its own `temperature`, `top_p`, `presence_penalty`, … in the request overrides them for that request. `mlx_lm.server` has options for temp, top-p, top-k and min-p but not for the two penalties, which it takes only from the request (missing: off); `start.py` passes them as `--presence-penalty` and `--repetition-penalty` next to the other options, and the shim takes them out of the arguments of `mlx_lm.server` and gives them to the requests that don't set them (`MLX_PENALTY_SHIM=0` switches this off). A repetition penalty of 1.0 changes nothing, so the shim leaves it off (no work per token). In mlx-lm both penalties look only at the last 20 tokens (prompt included), unless the request sets `presence_context_size` / `repetition_context_size`. With `--reasoning on`, `-xhigh` raises the reasoning effort from `medium` to `xhigh`.

The `[tps]` line comes from `start.py` itself, which reads `/v1/metrics` on this machine every 3 seconds (`-tps-every SECONDS`, 0 switches it off) and prints the last finished request; requests finished within the same interval show only the last one (`N more not shown`), the average includes all of them.

Stopping: `start.py` stays alive as long as the server (its pid is the one to save, e.g. `nohup ./start.py … & echo $! > mlx-jaccl.pid`) and stops the whole cluster when it gets Ctrl-C, `kill`, or the closing of its terminal (unless started with `nohup`):

```
[start] SIGTERM received: stopping the ranks
[start] no rank left on 192.168.20.1, 192.168.20.2
```

It closes `mlx.launch` with its ssh connections, so every rank gets a hangup and exits; then it checks every host of the hostfile and kills (`kill -9`) the ranks of this launch still alive after 5 seconds. Each launch marks its ranks with `-X mlx_jaccl_tag=…`, so the stop touches no other process. After a `kill -9` of `start.py`, a companion process (the same one that prints `[tps]`) notices within a second and stops the cluster (`start.py ended without stopping the cluster: stopping it`). At startup `start.py` checks the hosts and refuses to start while a rank is already there (a server already running, or one left by an earlier launch), printing the `kill` to run.

CPU of `mlx.launch`: in mlx 0.32.4, `mlx.launch` forwards its stdin to the ranks in a loop that never waits, which keeps Mac1 at ~130% CPU for as long as the server runs. `start.py` never writes to that stdin, so it runs `mlx.launch` through `mlx_launch_shim.py`, which removes the loop in memory (~0% CPU, no installed file modified); in `ps` the process is `python -P …/mlx_launch_shim.py …/mlx.launch …`. If a new mlx has changed that code, the log says `[start] mlx.launch not patched, it can keep 1-2 CPU cores busy: …` and the unchanged `mlx.launch` runs: the server works as before.

MTP (multi-token prediction): the shim looks for the MTP head of the model only in the local Hugging Face cache, it never downloads it. Download it once on both Macs:

```bash
hf download mlx-community/Qwen3.8-27B-MTP-8bit
```

Without it the log says `MTP not active: MTP head … not present locally` and generation works as before. MTP needs `--kv-bits` (always set by `start.py`) and keeps the output identical in distribution to normal generation; in the `MTP:` line, `per verification` is the number of tokens produced per model forward (at most 3 with 2 drafts).

Querying metrics (requests to `/v1/metrics` are not written to the server log, so polling doesn't fill the console):

```bash
curl -s http://192.168.1.101:8080/v1/metrics | jq '.last_request'
```

returns the last finished request. For the request in progress (what `active-request.sh` shows):

```bash
curl -s http://192.168.1.101:8080/v1/metrics | jq '.active_requests[0]'
```

The `.batch` section stays at zero with `--kv-bits`, because requests are then served one at a time instead of in batch.

Monitoring memory usage:
```
watch -n 2 "curl -s http://localhost:8080/v1/metrics | jq '{cache_gb: (.prompt_cache.n_bytes/1e9), limite_gb: (.prompt_cache.max_bytes/1e9), ok: (.prompt_cache.n_bytes <= .prompt_cache.max_bytes), active_gb: (.memory.active/1e9)}'"
```

Monitoring RDMA traffic and latency: macOS has no byte counters for the `rdma_enX` devices (`netstat` on `en3` sees only IP traffic), so `rdma-monitor.sh` traces with DTrace the collectives (`all_sum`, `all_gather`, …) of the rank running on this Mac, while the server works. Run it on either Mac while a request is in progress; it asks for the `sudo` password, prints one line per operation every second and the totals at the end (also after Ctrl-C):

```bash
./rdma-monitor.sh        # 60 s; ./rdma-monitor.sh 0 runs until Ctrl-C, -p PID picks the process
```

```
   t(s)  op            calls/s      MB/s   avg us   max us
   58.0  all_gather         33      13.7       85      238
   58.0  all_sum          1432      43.2       21      227
...
Totals over 60.0 s
op               calls         MB     MB/s   avg us   min us   max us
all_gather        1983      820.7     13.7       91       37     1459
all_sum          86218     2601.5     43.4       20        4     1005
all                         3422.2     57.0
```

(Qwen3.8-27B-8bit with MTP generating at about 28 tok/s on Mac1, `rdma_en3`.) MB are the data passed to each operation, about what this rank sends to the other; latency is the time of the call, so it includes waiting for the other Mac. During generation the collectives take about 3% of the time: there the limit is the GPU, not the link. During prefill the operations are much larger (MB instead of KB). The tracing adds about 1 µs per call and changes nothing in the server's behaviour; when it ends the process is as before.
## 7. Compatible models and memory budget

A model runs in TP only if its architecture (`model_type`) implements `shard()` in mlx-lm. The architectures supported by the installed version can be listed as follows:

```bash
conda activate mlx

grep -l "def shard" "$(python -c 'import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))')"/models/*.py \
  | xargs -n1 basename | sed 's/\.py$//' | sort

deepseek_v2
deepseek_v3
deepseek_v32
deepseek_v41
exaone_moe
glm4_moe
glm4_moe_lite
gpt_oss
iquestloopcoder
kimi_k25
kimi_k3
llama
longcat_flash
longcat_flash_ngram
minimax
minimax_m3_vl
ministral3
mistral4
muse_spark
qwen2
qwen3
qwen3_5
qwen3_moe
step3p5
```

Then compare against the model's `model_type` .

Then to check if a model on HF, e.g. "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit" can be ran in distributed way by `mlx-lm` with `jaccl` backend do the following:
```
conda activate mlx
repo=mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/$repo/resolve/main/config.json" | grep -m1 '"model_type"'
    "model_type": "qwen2",
```

and match the answer (`qwen2` in this case) against the list above. The list above could change with future releases.

Budget: roughly 27 GB of GPU memory per node with the default limit (about 75% of 36 GB, estimated), i.e. about 50 GB for weights and KV cache across the two nodes; about 55–60 GB after raising `iogpu.wired_limit_mb` (section 9).

Estimated size of a quantized model: parameters × (bit + 0.5) / 8 bytes, where 0.5 bits accounts for quantization scales and biases (groups of 64).

KV cache per node in TP (KV heads are split across nodes; parameters in `config.json`):

```latex
M_{KV} = \frac{2 \cdot L_{att} \cdot H_{KV} \cdot d_{head} \cdot b \cdot T}{N_{nodes}}
```

Where 2 = keys and values, L\_att = full-attention layers, H\_KV = KV heads, d\_head = head dimension, b = bytes per value (2 at 16 bit), T = context tokens. The result is in bytes.

Candidates to check with the two commands (names and architectures from memory, except Qwen3-32B-8bit which was checked with `curl`):

| Model (mlx-community/…) | model\_type | Estimated size (GB) | Notes |
| --- | --- | --- | --- |
| Llama-3.3-70B-Instruct-4bit | llama | 40 | Dense; architecture of the official MLX TP example |
| Qwen3-Next-80B-A3B-Instruct-4bit | qwen3\_next | 45 | MoE, memory at the limit |
| Qwen3-32B-8bit | qwen3 | 35 | Dense |
| Qwen3-Coder-30B-A3B-Instruct-8bit | qwen3\_moe | 32 | MoE for code |
| Qwen3.6-27B-8bit | qwen3\_5 | 29 | Verified in TP, 22.4 tok/s |

Dense models benefit more from TP than MoE models with few active parameters.

Gemma: no Gemma architecture implements `shard()` in mlx-lm 0.31.3. Alternatives: update mlx-lm and re-run the `grep`; use a single Mac (Gemma 3 27B at 4 bit is about 15 GB, Gemma 4 31B at 4 bit is about 17 GB); or a `shard()` patch on the Llama model. Some Gemma 4 conversions (`gemma4_unified`) don't load even on a single node with mlx-lm 0.31.3.

## 8. Tuning and troubleshooting

More GPU memory, on both Macs (does not persist across reboots; 30000 MB leaves about 6 GB for macOS, a conservative estimate):

```bash
sudo sysctl iogpu.wired_limit_mb=30000
```

Processes left running after a crash, to clean up before relaunching (`start.py` refuses to start while they exist, and prints the exact `kill` to run):

```bash
pkill -f mlx_leak_shim; ssh 192.168.20.2 'pkill -f mlx_leak_shim'
```

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Permission denied` / `cannot execute` | Script launched as a command after `--` | Absolute path to Python after `--`, or `tp.sh` |
| `cat None … non-zero exit status 1` | Side effect of the shutdown after another error | Ignore, fix the first error |
| `No module named 'mlx_lm.examples'` | Examples not included in the pip package | Use `tp_generate.py` |
| `exec: mlx.launch: not found` | Corrupted command line from copy-paste | Use `tp.sh` |
| `errno 22` (RTR) or `errno 60` at startup | Wrong RDMA GID selection over Thunderbolt (known bug, PR mlx #3468) | Update mlx on both nodes |
| Hang with long prompts, no errors | Multi-block prefill in TP (reported on large models) | `--prefill-step-size` ≥ prompt length |
| Hang with `--temp > 0` | Ranks sampling different tokens | `mlx_lm.server` already syncs the random seed across ranks at startup; with MTP the shim also takes drafts and sampled tokens from rank 0 |
| `[jaccl] Changing queue pair to RTR failed with errno 14` on one rank and `Recv failed with errno=2` on the other, at startup | A rank of an earlier launch still running on a host keeps the RDMA devices busy (left by `mlx.launch` killed without stopping its ranks, which the `start.py` before the supervisor allowed) | Stop it with the command above, then relaunch |
| `Error: mlx_lm.server is already running, or left over …` from `start.py` | The check at startup found a rank on a host | Run the `kill` printed by `start.py`, or stop the other `start.py` |
| Server stops at startup with `DIFFERENT between the ranks: …` | Different shim, plugin, mlx/mlx-lm versions, arguments or `MLX_*_SHIM*` variables between the two Macs | Same files and versions on both Macs, variables through `--env` in `start.py` (section 3.1) |
| Server stops at startup with `… NOT applied (STRICT: exiting)` | The installed mlx-lm differs from the one the patches expect | Section 3.1 |
| Error `-12` in Send/Recv, `ibv_devinfo` hanging | Inconsistent RDMA state (reported on a 4-node cluster) | Reboot both Macs |
| HF: "Invalid username or password" | Repo non-existent or gated | `curl` with token (section 4), check the name |
| Out of memory | Weights + KV cache beyond the limit | `iogpu.wired_limit_mb`, `--kv-bits 8`, `--max-kv-size` |

## 9. Glossary and references

| Abbrev. | Meaning |
| --- | --- |
| RDMA | Remote Direct Memory Access: direct transfer between the memories of the two Macs without going through the TCP/IP stack |
| JACCL | Apple's collective communication library used by MLX for RDMA over Thunderbolt |
| TP | Tensor Parallelism: each node holds all layers, but with the weights split |
| Rank | Index of the process in the distributed group (0 = coordinator) |
| KV cache | Keys/values attention memory: retains the already-processed context |
| MoE | Mixture of Experts: for each token only a subset of the weights is active |
| MTU | Maximum size of a network packet (9000 = jumbo frame) |
| GID | Global identifier of an RDMA port |
| HF | Hugging Face |

References (read via search, not verified in detail):

- [MLX — Distributed Communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX — Launching Distributed Programs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html)
- [WWDC26 — Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/)
- [mlx PR #3468 — GID over Thunderbolt RDMA](https://github.com/ml-explore/mlx/pull/3468)
- [mlx discussion #3939 — prefill hang in TP](https://github.com/ml-explore/mlx/discussions/3939)
- [mlx discussion #4247 — error -12 in Send/Recv](https://github.com/ml-explore/mlx/discussions/4247)
