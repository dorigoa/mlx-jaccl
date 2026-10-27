#!/usr/bin/env python3

import argparse
import http.client
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


#MODELS = ("Qwen3.8-27B", "GLM-4.7-Flash", "Qwen3.6-35B-A3B", "Llama-3.3-70B-Instruct")
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_BIN = Path.home() / "miniforge3" / "envs" / "mlx" / "bin"
PYTHON_BIN = ENV_BIN / "python"
MLX_LAUNCH = ENV_BIN / "mlx.launch"
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5")
# -X option that marks the Python of every rank of one launch, so that the stop
# kills those ranks and nothing else (Python ignores unknown -X options)
TAG_OPTION = "mlx_jaccl_tag"
# Any rank started by start.py, also by an older launch; the bracket keeps
# pgrep from matching the shell that runs it
RANK_PATTERN = "[m]lx_leak_shim[^ /]*[.]py --model "


def die(msg):
    sys.exit(f"Error: {msg}")


def log(msg):
    print(f"[start] {msg}", file=sys.stderr, flush=True)


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


def seconds(value):
    try:
        return max(float(value), 0.0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number of seconds: {value!r}") from None


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Start mlx_lm.server on the jaccl cluster (through mlx_leak_shim_v2.py). "
                    "Stopping start.py (kill, kill -9, Ctrl-C) stops the ranks on all the hosts.",
        allow_abbrev=False,
    )
    #parser.add_argument("model", choices=MODELS, help="model to serve")
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
    parser.add_argument("-leak-patch", default=str(SCRIPT_DIR / "mlx_leak_shim_v2.py"),
                        help="shim with the mlx-lm patches (default: %(default)s)")
    parser.add_argument("-temp", type=number, default="1.0",
                        help="sampling temperature (default: %(default)s)")
    parser.add_argument("-top-p", type=number, default="0.95",
                        help="sampling top-p (default: %(default)s)")
    parser.add_argument("-cache-gbytes", type=gbytes,
                        help="prompt cache size in GB (default: 0 for Qwen, 4.5 otherwise)")
    parser.add_argument("-tps-every", type=seconds, default=3.0,
                        help="poll /v1/metrics on this machine every SECONDS and print the "
                             "generation speed of each finished request (0: off; "
                             "default: %(default)s)")
    parser.add_argument("-debug", action="store_true", help="show debug messages")
    parser.add_argument("-dry-run", action="store_true",
                        help="print the mlx.launch command instead of running it")
    return parser.parse_args(argv)


def model_id(args):
    #return f"mlx-community/{args.model}-{'8bit' if args.eight_bit else '4bit'}"
    return f"mlx-community/Qwen3.8-27B-{'8bit' if args.eight_bit else '4bit'}"


def build_command(args, tag):
    cache = args.cache_gbytes
    if cache is None:
        # Qwen3.x are hybrid: with --kv-bits the prompt cache never does hit
        cache = "0" #if args.model.startswith("Qwen") else "4.5"
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
        "--", str(PYTHON_BIN), "-X", f"{TAG_OPTION}={tag}", *server_args,
    ]


class TpsReport:
    """Read /v1/metrics (mlx_metrics_plugin.py) and, when last_request changes,
    print its generation_tps. Requests finished between two reads show only
    the last one."""

    def __init__(self, host, port, every):
        self.host, self.port, self.every = host, port, every
        self.last, self.finished = None, 0

    def read(self):
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
            conn.request("GET", "/v1/metrics")
            metrics = json.loads(conn.getresponse().read())
            conn.close()
        except (OSError, ValueError, http.client.HTTPException):
            return  # model still loading, server down, or no metrics plugin
        request = metrics.get("last_request")
        if not request or request == self.last:
            return
        counts = metrics.get("requests") or {}
        done = sum(counts.get(k, 0) for k in ("completed", "cancelled", "failed"))
        skipped = f", {done - self.finished - 1} more not shown" if done - self.finished > 1 else ""
        self.last, self.finished = request, done
        average = (metrics.get("totals") or {}).get("generation_tps", 0.0)
        print(f"[tps] {time.strftime('%H:%M:%S')} last request: "
              f"{request['generation_tps']:.1f} tok/s ({request['generation_tokens']} tokens "
              f"in {request['generation_time']:.1f}s, {request['finish_reason']}), "
              f"prefill {request['prefill_tps']:.1f} tok/s ({request['prefill_tokens']} tokens), "
              f"average since start {average:.1f} tok/s{skipped}", flush=True)


def ssh_hosts(hostfile):
    try:
        return [h["ssh"] for h in json.loads(Path(hostfile).read_text())]
    except (OSError, ValueError, TypeError, KeyError) as e:
        die(f"cannot read the hosts of {hostfile}: {e}")


def on_hosts(hosts, script, timeout):
    """Run the shell `script` with ssh on all `hosts` at once and return
    {host: (exit code, or None without an answer within `timeout` s, output)}.
    Each ssh has a session of its own, so Ctrl-C on the console doesn't cut it."""
    procs = {host: subprocess.Popen([*SSH, host, script], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, start_new_session=True)
             for host in hosts}
    deadline = time.monotonic() + timeout
    results = {}
    for host, proc in procs.items():
        try:
            out, _ = proc.communicate(timeout=max(deadline - time.monotonic(), 0))
            results[host] = (proc.returncode, out.strip())
        except subprocess.TimeoutExpired:
            proc.kill()
            results[host] = (None, proc.communicate()[0].strip())
    return results


def running_ranks(hosts):
    """{host: ["pid command", ...]} of the ranks already on the hosts: a server
    still running, or ranks left by a launch that wasn't stopped properly,
    which keep the RDMA devices busy (jaccl errors at the next start)."""
    found = {}
    for host, (code, out) in on_hosts(hosts, f"pgrep -lf {shlex.quote(RANK_PATTERN)}", 15).items():
        if code == 0:
            found[host] = out.splitlines()
        elif code != 1:  # 1: pgrep found nothing
            log(f"{host}: cannot check for running ranks: {out or 'no answer'}")
    return found


def signal_group(pgid, sig=signal.SIGTERM):
    """Send `sig` to the process group `pgid`; False if it no longer exists."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def stop_cluster(launch, hosts, tag):
    """Stop mlx.launch and its ssh connections (process group `launch`): each
    rank gets SIGHUP when its connection closes. Then wait up to 5 s for the
    ranks of this launch to exit on every host, and kill -9 those left."""
    signal_group(launch)
    pattern = shlex.quote(f"[{TAG_OPTION[0]}]{TAG_OPTION[1:]}={tag}")
    script = (f"for i in $(seq 20); do pgrep -f {pattern} >/dev/null || exit 0; sleep 0.25; done; "
              f"echo $(pgrep -f {pattern}); pkill -KILL -f {pattern}; exit 3")
    stopped = []
    for host, (code, out) in on_hosts(hosts, script, 20).items():
        if code == 0:
            stopped.append(host)
        elif code == 3:
            log(f"{host}: rank still running 5 s after the stop, killed with kill -9 (pid {out})")
        else:
            log(f"{host}: cannot check that the ranks have stopped: {out or 'no answer'}")
    if stopped:
        log(f"no rank left on {', '.join(stopped)}")


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def watch(pid, launch, hosts, tag, tps):
    """Companion of start.py (`pid`): print the [tps] lines when `tps` is set
    and, if start.py ends without stopping the cluster (kill -9), stop it."""
    step = min(tps.every, 1.0) if tps else 1.0
    next_read = time.monotonic() + (tps.every if tps else 0)
    alive = True
    while alive:
        time.sleep(step)
        alive = process_alive(pid)
        if tps and (not alive or time.monotonic() >= next_read):
            tps.read()  # also once more after start.py has gone
            next_read = time.monotonic() + tps.every
    if signal_group(launch, 0):
        log("start.py ended without stopping the cluster: stopping it")
        stop_cluster(launch, hosts, tag)


def start_watcher(pid, launch, hosts, tag, tps):
    """Run watch() in a detached grandchild that writes to this console."""
    sys.stdout.flush()
    sys.stderr.flush()
    child = os.fork()
    if child:
        os.waitpid(child, 0)
        return
    try:
        # New session: Ctrl-C or the closing of the console don't reach it
        os.setsid()
        if os.fork() == 0:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            watch(pid, launch, hosts, tag, tps)
    except Exception as e:
        print(f"[start] watcher stopped: {e}", file=sys.stderr, flush=True)
    finally:
        os._exit(0)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.hostfile).is_file():
        die(f"hostfile not found: {args.hostfile}")
    if not Path(args.leak_patch).is_file():
        die(f"shim not found: {args.leak_patch}")
    for exe in (PYTHON_BIN, MLX_LAUNCH):
        if not os.access(exe, os.X_OK):
            die(f"not found or not executable: {exe}")

    hosts = ssh_hosts(args.hostfile)
    tag = uuid.uuid4().hex[:12]
    cmd = build_command(args, tag)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    running = running_ranks(hosts)
    if running:
        for host, procs in running.items():
            for proc in procs:
                log(f"{host}: {proc[:160]}")
        stop = "; ".join(f"ssh {host} kill {' '.join(p.split()[0] for p in procs)}"
                         for host, procs in running.items())
        die("mlx_lm.server is already running, or left over by a launch that wasn't "
            f"stopped properly (it keeps the RDMA devices busy). Stop it with:\n  {stop}")

    thinking = "false" if args.nothink else "true"
    print(f"Server start: model={model_id(args)}, port={args.port}, thinking={thinking}",
          file=sys.stderr)
    env = dict(os.environ, PATH=f"{ENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}")
    received = []
    launch = None

    def on_signal(signum, frame):
        received.append(signum)
        if launch is not None:
            signal_group(launch.pid)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        # Left ignored when inherited so: nohup (SIGHUP), & in a script (SIGINT)
        if signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, on_signal)
    # Own session and process group: the stop reaches mlx.launch and its ssh
    # connections, nothing else. stdin is a pipe never written: mlx.launch
    # forwards stdin to the ranks and would spin on /dev/null (nohup, GUI)
    launch = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, start_new_session=True)
    if received:
        signal_group(launch.pid)
    tps = None
    if args.tps_every > 0:
        # The HTTP server runs on rank 0, i.e. this machine
        host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
        tps = TpsReport(host, args.port, args.tps_every)
    start_watcher(os.getpid(), launch.pid, hosts, tag, tps)

    code = launch.wait()
    if received:
        log(f"{signal.Signals(received[0]).name} received: stopping the ranks")
    stop_cluster(launch.pid, hosts, tag)
    return 128 + received[0] if received else code


if __name__ == "__main__":
    sys.exit(main())
