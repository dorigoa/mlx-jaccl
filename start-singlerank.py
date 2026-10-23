#!/usr/bin/env python3

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

#MODELS = ("Qwen3.8-27B", "GLM-4.7-Flash", "Qwen3.6-35B-A3B", "Llama-3.3-70B-Instruct")
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_BIN = Path.home() / "miniforge3" / "envs" / "mlxv" / "bin"
PYTHON_BIN = ENV_BIN / "python"
MLX_LAUNCH = ENV_BIN / "mlx_vlm.server"# "


def die(msg):
    sys.exit(f"Error: {msg}")


def port(value):
    if not re.fullmatch(r"\d{1,5}", value) or not 1 <= int(value) <= 65535:
        raise argparse.ArgumentTypeError("requires an integer in the range 1-65535")
    return int(value)


def number(value):
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
    parser.add_argument("-temp", type=number, default="1.0",
                        help="sampling temperature (default: %(default)s)")
    parser.add_argument("-top-p", type=number, default="0.95",
                        help="sampling top-p (default: %(default)s)")
    # parser.add_argument("-debug", action="store_true", help="show debug messages")
    parser.add_argument("-dry-run", action="store_true",
                        help="print the mlx-vlm command instead of running it")
    return parser.parse_args(argv)

def build_command(args):
    chat_template_args = json.dumps({
        "enable_thinking": not args.nothink,
        "reasoning_effort": "xhigh" if args.xhigh else "medium",
    })
    server_args = [
        "--host", str(args.host), "--port", str(args.port),
        "--temp", args.temp, "--top-p", args.top_p, "--top-k", "20", "--min-p", "0.0",
        "--max-tokens", str(args.max_tokens),
        "--log-level", "DEBUG" if args.debug else "INFO"
    ]
    return [
        str(MLX_LAUNCH), "--verbose", 
        "--model", "mlx-community/Qwen3.8-27B-4bit",
        "--draft-model", "mlx-community/Qwen3.8-27B-MTP-4bit",
        "--draft-kind mtp",
        "--env", "HF_HUB_OFFLINE=0",
        "--", str(PYTHON_BIN), *server_args,
    ]


def main(argv=None):
    args = parse_args(argv)
    # if not Path(args.hostfile).is_file():
    #     die(f"hostfile not found: {args.hostfile}")
    # if not Path(args.leak_patch).is_file():
    #     die(f"shim not found: {args.leak_patch}")
    for exe in (PYTHON_BIN, MLX_LAUNCH):
        if not os.access(exe, os.X_OK):
            die(f"not found or not executable: {exe}")

    cmd = build_command(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    thinking = "false" if args.nothink else "true"
    print(f"Server start: model=Qwen3.8-27B-4bit, port={args.port}, thinking={thinking}",
          file=sys.stderr)
    env = dict(os.environ, PATH=f"{ENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}")
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main())
