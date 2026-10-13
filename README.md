# MLX distribuito con RDMA su Thunderbolt 5 — 2× Mac Studio
## + Patch per metriche

Ott 13, 2026 · @Alvise Dorigo

Doc di riferimento: https://ml-explore.github.io/mlx/build/html/usage/distributed.html

Configurazione verificata: 2× Mac Studio M4 Max 36 GB in Thunderbolt 5, mlx 0.32.2 + mlx-lm 0.31.3, backend JACCL (RDMA), tensor parallel su 2 nodi. Qwen3.6-27B-8bit genera 22,4 tok/s.

## 1. Prerequisiti e abilitazione RDMA

Servono macOS 26.2 o successivo su entrambi i Mac e un cavo Thunderbolt 5 diretto tra i due.

Abilitazione RDMA, una volta per Mac:

1. Avvio in recoveryOS: Mac spento, tieni premuto il tasto di accensione fino a "Opzioni".
2. Menu Utility → Terminale.
3. Esegui `rdma_ctl enable`.
4. Riavvia normalmente.

Verifica (su entrambi i Mac):

```bash
sw_vers -productVersion
rdma_ctl status
ibv_devinfo | grep -E 'hca_id|state'
```

Atteso: versione ≥ 26.2, `enabled`, device `rdma_en3` (Mac1) / `rdma_en4` (Mac2) in stato `PORT_ACTIVE`.

## 2. Rete Thunderbolt: IP, MTU 9000, bridge

Mac1 è il rank 0: da qui si lanciano tutti i job.

| Nodo | Interfaccia | IP | Device RDMA |
| --- | --- | --- | --- |
| Mac1 (rank 0) | en3 | 192.168.20.1/24 | rdma\_en3 |
| Mac2 (rank 1) | en4 | 192.168.20.2/24 | rdma\_en4 |

Disattiva il Bridge Thunderbolt (consigliato dalla documentazione MLX anche se RDMA non usa TCP/IP), su entrambi i Mac:

```bash
sudo networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off
```

IP e MTU (non persistenti; per renderli permanenti usa Impostazioni di Sistema → Rete sul servizio di en3/en4, mappatura con `networksetup -listallhardwareports`):

```bash
# Mac1
sudo ifconfig en3 inet 192.168.20.1 netmask 255.255.255.0 mtu 9000
# Mac2
sudo ifconfig en4 inet 192.168.20.2 netmask 255.255.255.0 mtu 9000
```

Verifica jumbo frame da Mac1 (8972 = 9000 − 28 byte di header IP+ICMP, `-D` vieta la frammentazione):

```bash
ping -c3 -D -s 7168 192.168.20.2
```

Alternativa automatica: `mlx.distributed_config --backend jaccl --hosts <h1>,<h2> --over thunderbolt --auto-setup --output hosts.json` configura rete e hostfile, ma richiede sudo senza password e usa l'IP di en0 per il rank 0.

## 3. Python: Miniforge e ambiente mlx

L'ambiente deve essere identico e nello stesso percorso sui due Mac: `mlx.launch` avvia via SSH lo stesso eseguibile Python su entrambi.

Installazione Miniforge (su entrambi i Mac, se assente):

```bash
curl -fsSLO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init "$(basename "$SHELL")"
exec "$SHELL" -l
```

Ambiente e pacchetti (su entrambi i Mac, versioni fissate per riproducibilità):

```bash
conda create -y -n mlx python=3.12
conda activate mlx
pip install mlx==0.32.3 mlx-lm==0.32
python -c "import mlx.core as mx, mlx_lm; print(mx.__version__, mlx_lm.__version__)"
which python
```

Atteso: `0.32.3 0.32` (o numeri simili) e percorso `$HOME/miniforge3/envs/mlx/bin/python` uguale sui due Mac (qui `/Volumes/Home/Alvise/...`).

## 4. Hugging Face CLI (hf): installazione, token, download

Il comando `hf` fa parte del pacchetto `huggingface_hub`, già installato come dipendenza di mlx-lm.

Aggiornamento e verifica (su entrambi i Mac, env `mlx` attivo):

```bash
pip install -U huggingface_hub
which hf
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

Token: huggingface.co → Settings → Access Tokens → New token, tipo **Read**. Esportalo come `HF_TOKEN` nella shell.

Login (su entrambi i Mac). Il token viene salvato in `~/.cache/huggingface/token` e resta valido anche nelle sessioni SSH non interattive, dove `HF_TOKEN` potrebbe non essere definito:

```bash
hf auth login --token "$HF_TOKEN"
hf auth whoami
```

Per i modelli ad accesso limitato (gated) accetta prima la licenza sulla pagina del modello.

Architettura di un modello senza scaricarlo:

```bash
repo=mlx-community/Qwen3-32B-8bit
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/$repo/resolve/main/config.json" | grep -m1 '"model_type"'
```

Senza token HF risponde "Invalid username or password" anche per repo inesistenti; con `-f` curl mostra il codice reale (401/404).

La cache dei modelli è in `~/.cache/huggingface/hub`; si sposta con la variabile `HF_HOME`.

## 5. SSH senza password

Si assume già configurato.

## 6. Applicazione patch per metriche (su entrambi i Mac)
```
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MLX_LM=0.32.0+metrics ~/miniforge3/envs/mlx/bin/python -m pip install --no-deps --force-reinstall "git+https://github.com/dorigoa/mlx-lm.git@metrics"
```

## 7. Scripts di supporto e avvio
```
git clone git@github.com:dorigoa/mlx-jaccl.git # su entrambi i Mac
cd mlx-jaccl
# Nel comando seguente usare il path completo ~/mlx-jaccl/hosts.json e ~/mlx-jaccl/mlx_leak_shim.py
# invece di ./hosts.json e ./mlx_leak_shim.py, perche' i due script devono essere localizzati anche sul Mac remoto
./mlx_jaccl-start.sh -hostfile ~/mlx-jaccl/hosts.json -leak-patch ~/mlx-jaccl/mlx_leak_shim.py Qwen3.8-27B -max-tokens 131072
Server start: model=mlx-community/Qwen3.8-27B-8bit, port=8080, thinking=true
[INFO] Running /Volumes/Home/Alvise/miniforge3/envs/mlx/bin/python /Volumes/Home/Alvise/mlx-apple-doc/mlx_leak_shim.py --model mlx-community/Qwen3.8-27B-8bit --host 0.0.0.0 --port 8080 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --max-tokens 131072 --log-level INFO --prompt-cache-bytes 9663676416 --prefill-step-size 256 --chat-template-args '{"enable_thinking": true, "reasoning_effort": "medium"}'
[leak-shim] rank=0 patch applicata (mlx-lm 0.32.0+metrics)
[leak-shim] rank=1 patch applicata (mlx-lm 0.32.0+metrics)
...
...
```
Chiedere metriche:
```
curl -s http://192.168.1.101:8080/v1/metrics | jq '.last_request'
```
funziona solo quando la richiesta e' stata completata. Altrimenti bisogna interrogare continuamente:
```
curl -s http://192.168.1.101:8080/v1/metrics | jq '.batch'
```
## 7. Modelli compatibili e budget di memoria

Un modello gira in TP solo se la sua architettura (`model_type`) implementa `shard()` in mlx-lm. Le architetture supportate dalla versione installata si ricavano così:

```bash
grep -l "def shard" "$(python -c 'import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))')"/models/*.py \
  | xargs -n1 basename | sed 's/\.py$//' | sort
```

Poi confronta con il `model_type` del modello (comando `curl` della sezione 4).

Budget: circa 27 GB di memoria GPU per nodo con il limite predefinito (circa 75% di 36 GB, stima), quindi circa 50 GB tra pesi e KV cache sui due nodi; circa 55–60 GB alzando `iogpu.wired_limit_mb` (sezione 11).

Peso stimato di un modello quantizzato: parametri × (bit + 0,5) / 8 byte, dove 0,5 bit tiene conto di scale e bias di quantizzazione (gruppi da 64).

KV cache per nodo in TP (le teste KV sono divise tra i nodi; parametri in `config.json`):

```latex
M_{KV} = \frac{2 \cdot L_{att} \cdot H_{KV} \cdot d_{head} \cdot b \cdot T}{N_{nodi}}
```

Dove 2 = chiavi e valori, L\_att = strati ad attenzione completa, H\_KV = teste KV, d\_head = dimensione di una testa, b = byte per valore (2 a 16 bit), T = token di contesto. Il risultato è in byte.

Candidati da verificare con i due comandi (nomi e architetture a memoria, tranne Qwen3-32B-8bit controllato con `curl`):

| Modello (mlx-community/…) | model\_type | Peso stimato (GB) | Note |
| --- | --- | --- | --- |
| Llama-3.3-70B-Instruct-4bit | llama | 40 | Denso; architettura dell'esempio TP ufficiale MLX |
| Qwen3-Next-80B-A3B-Instruct-4bit | qwen3\_next | 45 | MoE, memoria al limite |
| Qwen3-32B-8bit | qwen3 | 35 | Denso |
| Qwen3-Coder-30B-A3B-Instruct-8bit | qwen3\_moe | 32 | MoE per codice |
| Qwen3.6-27B-8bit | qwen3\_5 | 29 | Verificato in TP, 22,4 tok/s |

I modelli densi traggono più vantaggio dal TP dei MoE con pochi parametri attivi.

Gemma: nessuna architettura Gemma implementa `shard()` in mlx-lm 0.31.3. Alternative: aggiornare mlx-lm e ripetere il `grep`; usare un solo Mac (Gemma 3 27B a 4 bit circa 15 GB, Gemma 4 31B a 4 bit circa 17 GB); oppure una patch `shard()` sul modello di Llama. Alcune conversioni Gemma 4 (`gemma4_unified`) non si caricano nemmeno su un nodo singolo con mlx-lm 0.31.3.

## 8. Ottimizzazione e risoluzione problemi

Più memoria GPU, su entrambi i Mac (non persiste al riavvio; 30000 MB lascia circa 6 GB a macOS, valore prudente stimato):

```bash
sudo sysctl iogpu.wired_limit_mb=30000
```

Processi rimasti attivi dopo un crash, da pulire prima di rilanciare:

```bash
pkill -f tp_generate.py; ssh 192.168.20.2 'pkill -f tp_generate.py'
```

| Sintomo | Causa | Soluzione |
| --- | --- | --- |
| `Permission denied` / `cannot execute` | Script lanciato come comando dopo `--` | Percorso assoluto di Python dopo `--`, o `tp.sh` |
| `cat None … non-zero exit status 1` | Effetto collaterale della chiusura dopo un altro errore | Ignorare, risolvere il primo errore |
| `No module named 'mlx_lm.examples'` | Esempi non inclusi nel pacchetto pip | Usare `tp_generate.py` |
| `exec: mlx.launch: not found` | Riga di comando corrotta nel copia-incolla | Usare `tp.sh` |
| `errno 22` (RTR) o `errno 60` all'avvio | Selezione errata del GID RDMA su Thunderbolt (bug noto, PR mlx #3468) | Aggiornare mlx su entrambi i nodi |
| Blocco con prompt lunghi, senza errori | Prefill a più blocchi in TP (segnalato su modelli grandi) | `--prefill-step-size` ≥ lunghezza del prompt |
| Blocco con `--temp > 0` | Rank che campionano token diversi | Stesso `--seed` (già impostato dallo script) |
| Errore `-12` in Send/Recv, `ibv_devinfo` bloccato | Stato RDMA incoerente (segnalato su cluster a 4 nodi) | Riavviare entrambi i Mac |
| HF: "Invalid username or password" | Repo inesistente o gated | `curl` con token (sezione 4), controllare il nome |
| Memoria esaurita | Pesi + KV cache oltre il limite | `iogpu.wired_limit_mb`, `--kv-bits 8`, `--max-kv-size` |

## Connettere con Claude Code
Installare un proxy:
```
git clone https://github.com/dorigoa/claude-code-mlx-proxy.git   && cd claude-code-mlx-proxy && ls -1 && (command -v uv && uv --version || echo "uv ASSENTE")
```
deve stampare la versione uv disponibile (altrimenti: `python -m pip install uv`)

Creare il file `.env`:
```
# .env - Example configuration for the proxy server
HOST=127.0.0.1
PORT=8888
# URL of the mlx-lm OpenAI-compatible server (must include /v1)
MLX_SERVER_URL=http://192.168.1.101:8080/v1
# API-exposed model name (what Claude Code sees)
API_MODEL_NAME=claude-4-sonnet-20250514
# Tool mode: full / slim / none
# TOOL_MODE=slim
# Enable verbose debug logging
VERBOSE=false
MODEL_NAME=default_model
```
Quindi scaricare i pacchetti:
```
uv sync
```
Runnare il proxy:
```
uv run main.py
```

## 9. Glossario e riferimenti

| Sigla | Significato |
| --- | --- |
| RDMA | Remote Direct Memory Access: trasferimento diretto tra memorie dei due Mac senza passare dallo stack TCP/IP |
| JACCL | Libreria di comunicazione collettiva di Apple usata da MLX per RDMA su Thunderbolt |
| TP | Tensor Parallelism: ogni nodo tiene tutti gli strati, ma con i pesi divisi |
| Rank | Indice del processo nel gruppo distribuito (0 = coordinatore) |
| KV cache | Memoria delle chiavi/valori di attenzione: conserva il contesto già elaborato |
| MoE | Mixture of Experts: per ogni token lavora solo una parte dei pesi |
| MTU | Dimensione massima di un pacchetto di rete (9000 = jumbo frame) |
| GID | Identificativo globale di una porta RDMA |
| HF | Hugging Face |

Riferimenti (letti tramite ricerca, non verificati in dettaglio):

- [MLX — Distributed Communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX — Launching Distributed Programs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html)
- [WWDC26 — Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/)
- [mlx PR #3468 — GID su Thunderbolt RDMA](https://github.com/ml-explore/mlx/pull/3468)
- [mlx discussion #3939 — prefill bloccato in TP](https://github.com/ml-explore/mlx/discussions/3939)
- [mlx discussion #4247 — errore -12 in Send/Recv](https://github.com/ml-explore/mlx/discussions/4247)
