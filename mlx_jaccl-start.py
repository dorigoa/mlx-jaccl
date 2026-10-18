#!/usr/bin/env python3

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

MODELS = ("Qwen3.8-27B", "GLM-4.7-Flash", "Qwen3.6-35B-A3B", "Llama-3.3-70B-Instruct")
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_BIN = Path.home() / "miniforge3" / "envs" / "mlx" / "bin"
PYTHON_BIN = ENV_BIN / "python"
MLX_LAUNCH = ENV_BIN / "mlx.launch"


def die(msg):
    sys.exit(f"Error: {msg}")


def port(value):
    if not re.fullmatch(r"\d{1,5}", value) or not 1 <= int(value) <= 65535:
        raise argparse.ArgumentTypeError("requires an integer in the range 1-65535")
    return int(value)


def number(value):
    """Controlla che sia un numero ma lo passa al server cosi' come e' scritto."""
    try:
        float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    return value


def gbytes(value):
    if not re.fullmatch(r"\d+(\.\d+)?", value):
        raise argparse.ArgumentTypeError("requires a number of GB (e.g. 2 or 4.5)")
    return value


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Start mlx_lm.server on the jaccl cluster (through mlx_leak_shim.py).",
        allow_abbrev=False,
    )
    parser.add_argument("model", choices=MODELS, help="model to serve")
    parser.add_argument("-8bit", dest="eight_bit", action="store_true",
                        help="use 8bit quantization (default: 4bit)")
    parser.add_argument("-nothink", action="store_true",
                        help="disable thinking (default: enabled)")
    parser.add_argument("-xhigh", action="store_true",
                        help="high reasoning effort (default: medium)")
    parser.add_argument("-port", type=port, default=8080,
                        help="listening port of the API server (default: %(default)s)")
    parser.add_argument("-host", type=str, default="127.0.0.1",
                        help="listening on specified host of the API server (default: 127.0.0.1)")
    parser.add_argument("-max-tokens", type=int, default=262144,
                        help="max number of tokens generated per request (default: %(default)s)")
    parser.add_argument("-hostfile", default=str(SCRIPT_DIR / "hosts.json"),
                        help="hostfile of the mlx cluster (default: %(default)s)")
    parser.add_argument("-leak-patch", default=str(SCRIPT_DIR / "mlx_leak_shim.py"),
                        help="shim with the mlx-lm patches (default: %(default)s)")
    parser.add_argument("-temp", type=number, default="1.0",
                        help="sampling temperature (default: %(default)s)")
    parser.add_argument("-top-p", type=number, default="0.95",
                        help="sampling top-p (default: %(default)s)")
    parser.add_argument("-cache-gbytes", type=gbytes,
                        help="prompt cache size in GB (default: 0 for Qwen, 4.5 otherwise)")
    parser.add_argument("-debug", action="store_true", help="show debug messages")
    parser.add_argument("-dry-run", action="store_true",
                        help="print the mlx.launch command instead of running it")
    return parser.parse_args(argv)


def model_id(args):
    return f"mlx-community/{args.model}-{'8bit' if args.eight_bit else '4bit'}"


def build_command(args):
    cache = args.cache_gbytes
    if cache is None:
        # Qwen3.x sono ibridi: con --kv-bits la prompt cache non fa mai hit
        cache = "0" if args.model.startswith("Qwen") else "4.5"
    chat_template_args = json.dumps({
        "enable_thinking": not args.nothink,
        "reasoning_effort": "xhigh" if args.xhigh else "medium",
    })
    server_args = [
        args.leak_patch,
        "--model", model_id(args),
        "--host", str(args.host), "--port", str(args.port),
        "--temp", args.temp, "--top-p", args.top_p, "--top-k", "20", "--min-p", "0.0",
        "--max-tokens", str(args.max_tokens),
        "--log-level", "DEBUG" if args.debug else "INFO",
        "--prompt-cache-bytes", f"{cache}G",
        "--prefill-step-size", "256",
        "--chat-template-args", chat_template_args,
        "--kv-bits", "8",
    ]
    return [
        str(MLX_LAUNCH), "--verbose", "--backend", "jaccl", "--hostfile", args.hostfile,
        "--env", "MLX_METAL_FAST_SYNCH=0", "--env", "HF_HUB_OFFLINE=0",
        "--env", f"MLX_LEAK_SHIM={os.environ.get('MLX_LEAK_SHIM') or '1'}",
        "--env", "MLX_LEAK_SHIM_STRICT=1",
        "--", str(PYTHON_BIN), *server_args,
    ]


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.hostfile).is_file():
        die(f"hostfile not found: {args.hostfile}")
    if not Path(args.leak_patch).is_file():
        die(f"shim not found: {args.leak_patch}")
    for exe in (PYTHON_BIN, MLX_LAUNCH):
        if not os.access(exe, os.X_OK):
            die(f"not found or not executable: {exe}")

    cmd = build_command(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    thinking = "false" if args.nothink else "true"
    print(f"Server start: model={model_id(args)}, port={args.port}, thinking={thinking}",
          file=sys.stderr)
    env = dict(os.environ, PATH=f"{ENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}")
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main())
