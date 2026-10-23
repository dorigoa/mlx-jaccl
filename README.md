# Distributed MLX over RDMA on Thunderbolt 5 — 2× Mac Studio
## + Patch for metrics

Oct 22, 2026 · @Alvise Dorigo

Reference doc: https://ml-explore.github.io/mlx/build/html/usage/distributed.html

Verified setup: 2× Mac Studio M4 Max 36 GB over Thunderbolt 5, mlx 0.32.4 + mlx-lm 0.32.2, JACCL (RDMA) backend, tensor parallelism across 2 nodes. Qwen3.6-27B-8bit generates 22.4 tok/s.

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
```
git clone git@github.com:dorigoa/mlx-jaccl.git # on both Macs
cd mlx-jaccl

# Execute a self test, especially after an update of mlx and mlx-lm packages (step 3)
conda activate mlx # if not already done
python mlx_leak_shim.py --self-test

# In the following command use the full paths ~/mlx-jaccl/hosts.json and ~/mlx-jaccl/mlx_leak_shim.py
# instead of ./hosts.json and ./mlx_leak_shim.py, because the two scripts must also be located on the remote Mac

./mlx_jaccl-start.sh -hostfile ~/mlx-jaccl/hosts.json -leak-patch ~/mlx-jaccl/mlx_leak_shim.py Qwen3.8-27B -max-tokens 131072
Server start: model=mlx-community/Qwen3.8-27B-8bit, port=8080, thinking=true
[INFO] Running /Volumes/Home/Alvise/miniforge3/envs/mlx/bin/python /Volumes/Home/Alvise/mlx-apple-doc/mlx_leak_shim.py --model mlx-community/Qwen3.8-27B-8bit --host 0.0.0.0 --port 8080 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --max-tokens 131072 --log-level INFO --prompt-cache-bytes 9663676416 --prefill-step-size 256 --chat-template-args '{"enable_thinking": true, "reasoning_effort": "medium"}'
[leak-shim] rank=0 patch applied (mlx-lm 0.32.0+metrics)
[leak-shim] rank=1 patch applied (mlx-lm 0.32.0+metrics)
...
...
```
Querying metrics:
```
curl -s http://192.168.1.101:8080/v1/metrics | jq '.last_request'
```
works only once a request has completed. Otherwise you must poll continuously:
```
curl -s http://192.168.1.101:8080/v1/metrics | jq '.batch'
```
Monitoring memory usage:
```
watch -n 2 "curl -s http://localhost:8080/v1/metrics | jq '{cache_gb: (.prompt_cache.n_bytes/1e9), limite_gb: (.prompt_cache.max_bytes/1e9), ok: (.prompt_cache.n_bytes <= .prompt_cache.max_bytes), active_gb: (.memory.active/1e9)}'"
```
## 7. Compatible models and memory budget

A model runs in TP only if its architecture (`model_type`) implements `shard()` in mlx-lm. The architectures supported by the installed version can be listed as follows:

```bash
grep -l "def shard" "$(python -c 'import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))')"/models/*.py \
  | xargs -n1 basename | sed 's/\.py$//' | sort
```

Then compare against the model's `model_type` .

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

Processes left running after a crash, to clean up before relaunching:

```bash
pkill -f tp_generate.py; ssh 192.168.20.2 'pkill -f tp_generate.py'
```

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Permission denied` / `cannot execute` | Script launched as a command after `--` | Absolute path to Python after `--`, or `tp.sh` |
| `cat None … non-zero exit status 1` | Side effect of the shutdown after another error | Ignore, fix the first error |
| `No module named 'mlx_lm.examples'` | Examples not included in the pip package | Use `tp_generate.py` |
| `exec: mlx.launch: not found` | Corrupted command line from copy-paste | Use `tp.sh` |
| `errno 22` (RTR) or `errno 60` at startup | Wrong RDMA GID selection over Thunderbolt (known bug, PR mlx #3468) | Update mlx on both nodes |
| Hang with long prompts, no errors | Multi-block prefill in TP (reported on large models) | `--prefill-step-size` ≥ prompt length |
| Hang with `--temp > 0` | Ranks sampling different tokens | Same `--seed` (already set by the script) |
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
