#!/usr/bin/env bash

readonly MODELS=("Qwen3.8-27B" "GLM-4.7-Flash" "Qwen3.6-35B-A3B" "Llama-3.3-70B-Instruct")
readonly SCRIPT_DIR=${HOME}/mlx-jaccl
readonly PYTHON_BIN="${HOME}/miniforge3/envs/mlx/bin/python"

usage() {
    cat >&2 <<EOF
Uso: $(basename "$0") <model> [-8bit] [-nothink] [-port <1-65535>] [-max-token <integer>] [-hostfile <file>]
  <modello>          Mandatory: $(IFS='|'; echo "${MODELS[*]}")
  -8bit              Optional: use 8bit quantization (default: 4bit)
  -nothink           Optional: disable thinking (default: enable)
  -xhigh             Optional: enable high reasoning effort (default: medium)
  -port <n>          Optional: listening port for API server(default: 8080)
  -max-tokens <n>    Optional: max number of tokens to be generated (default: 8192)
  -hostfile <file>   Optional: path to hostfile containing the host defition for mlx cluster
  -leak-patch <file> Optional: change the patch path for memory leak with qwen models
  -temp              Optional: change temperature to be passed to the llm
  -top-p             Optional: change top-p parameter to be passed to the llm
  -cache-gbytes <n>  Optional: size in GB of prompt cache bytes
  -debug             Optional: show debug messages
  
  -h         Show this help
EOF
    exit "${1:-1}"
}

die() { echo "Error: $*" >&2; exit 1; }

is_model() {
    local m
    for m in "${MODELS[@]}"; do [[ "$1" == "$m" ]] && return 0; done
    return 1
}

MNAME=""
BIT_DEPTH="4bit"
THINK_MODE=1
PORT=8080
MAX_TOKENS=262144
LOG_LEVEL="INFO"
REAS_DEPTH="medium"
HOSTFILE="${SCRIPT_DIR}/hosts.json"
SHIM="${SCRIPT_DIR}/mlx_leak_shim.py"
TEMP="1.0"
TOPP="0.95"
PROMPT_CACHE_BYTES=4.5G

while [[ $# -gt 0 ]]; do
    case "$1" in
        -8bit)
            BIT_DEPTH="8bit"; shift ;;
        -nothink)
            #[[ "${2:-}" =~ ^[01]$ ]] || die "-think requires 0 or 1"
            THINK_MODE=0; shift ;;
        -xhigh)
            REAS_DEPTH="xhigh"; shift ;;
        -port)
            [[ "${2:-}" =~ ^[0-9]{1,5}$ ]] && (( 10#$2 >= 1 && 10#$2 <= 65535 )) \
                || die "-port requires an integer in the range 1024-65535"
            PORT="$((10#$2))"; shift 2 ;;
        -max-tokens)
            MAX_TOKENS="$((10#$2))"; shift 2 ;;
        -hostfile)
            HOSTFILE="$2"; shift 2 ;;
        -leak-patch)
            SHIM="$2"; shift 2 ;;
        -temp)
            TEMP="$2"; shift 2 ;;
        -top-p)
            TOPP="$2"; shift 2 ;;
        -debug)
            LOG_LEVEL="DEBUG"; shift ;;
        -cache-gbytes)
            [[ "${2:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "-cache-gbytes requires a number of GB (e.g. 2 or 4.5)"
            PROMPT_CACHE_BYTES="${2}G"; shift 2 ;;
            #PROMPT_CACHE_BYTES="${2}G"; shift 2 ;;
        -h|--help)
            usage 0 ;;
        -*)
            echo "Error: unknown option: $1" >&2; usage ;;
        *)
            is_model "$1" || { echo "Error: unknown model: $1" >&2; usage; }
            [[ -z "$MNAME" ]] || die "Model specified more than once"
            MNAME="$1"; shift ;;
    esac
done

[[ -n "$MNAME" ]]      || { echo "Error: missing model" >&2; usage; }
[[ -f "$HOSTFILE" ]]   || die "hostfile not found: $HOSTFILE"
[[ -x "$PYTHON_BIN" ]] || die "Python interpreter not found: $PYTHON_BIN"
[[ -f "$SHIM" ]] || { echo "ERROR: $SHIM not found" >&2; exit 1; }

MODEL="mlx-community/${MNAME}-${BIT_DEPTH}"
(( THINK_MODE )) && THINK_BOOL=true || THINK_BOOL=false

command -v conda >/dev/null 2>&1 || die "conda not found in the PATH"
eval "$(conda shell.bash hook)"  || die "conda init failed"
conda activate mlx               || die "could not activated conda env 'mlx'"
command -v mlx.launch >/dev/null 2>&1 || die "mlx.launch not found in 'mlx' env"

echo "Server start: model=$MODEL, port=$PORT, thinking=$THINK_BOOL" >&2

server_args=(
    "$SHIM"
    --model "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --temp "$TEMP" --top-p "$TOPP" --top-k 20 --min-p 0.0
    --max-tokens "$MAX_TOKENS"
    --log-level "$LOG_LEVEL"
    --prompt-cache-bytes "$PROMPT_CACHE_BYTES"
    --prefill-step-size 256
    --chat-template-args "{\"enable_thinking\": ${THINK_BOOL}, \"reasoning_effort\": \"$REAS_DEPTH\"}"
    --kv-bits 8
)

exec mlx.launch --verbose --backend jaccl --hostfile "$HOSTFILE" \
    --env MLX_METAL_FAST_SYNCH=0 --env HF_HUB_OFFLINE=0 \
    --env MLX_LEAK_SHIM="${MLX_LEAK_SHIM:-1}" --env MLX_LEAK_SHIM_STRICT=1 \
    -- "$PYTHON_BIN" "${server_args[@]}"
