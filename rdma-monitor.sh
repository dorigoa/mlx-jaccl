#!/bin/bash
# Traffic and latency of the RDMA collectives (JACCL) of the MLX rank running on
# this Mac, traced with DTrace in the running process (read-only, needs sudo).
#
#   ./rdma-monitor.sh [-p PID] [SECONDS]     SECONDS: default 60, 0 = until Ctrl-C
#
# Ctrl-C stops early and still prints the totals. See rdma_monitor.d for what
# is measured.
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)

pid=""
if [[ ${1:-} == -p ]]; then
	pid=${2:?"-p needs a PID"}
	shift 2
fi
secs=${1:-60}

if [[ -z $pid ]]; then
	# The rank is the Python started with -X mlx_jaccl_tag=…; mlx.launch and its
	# ssh carry the same command line among their arguments, so match the start
	pids=$(ps -axo pid=,command= |
		awk '$2 == "-X" && $3 ~ /^mlx_jaccl_tag=/ && /mlx_leak_shim[^ \/]*[.]py --model / { print $1 }')
	count=$(echo $pids | wc -w)
	if ((count == 0)); then
		echo "No MLX rank running on this Mac (start the server with start.py, or pass -p PID)" >&2
		exit 1
	elif ((count > 1)); then
		echo "More than one MLX rank on this Mac, choose one with -p:" >&2
		ps -o pid=,command= -p "$(echo $pids | tr ' ' ,)" | cut -c1-150 >&2
		exit 1
	fi
	pid=$pids
fi

if ((secs > 0)); then
	echo "Tracing the JACCL collectives of pid $pid for $secs s (Ctrl-C stops early)"
else
	echo "Tracing the JACCL collectives of pid $pid until Ctrl-C"
fi

# awk ignores Ctrl-C, so it can still print what dtrace writes when stopping
sudo dtrace -Z -s "$DIR/rdma_monitor.d" -p "$pid" "$secs" 2>&1 | (
	trap '' INT
	exec awk '
	BEGIN {
		printf "%7s  %-11s %9s %9s %8s %8s\n", "t(s)", "op", "calls/s", "MB/s", "avg us", "max us"
	}
	/system integrity protection is on/ { next }
	/^[ \t\r]*$/ && secs == 0 { next }
	/^R / { r[++nr] = $0; next }
	/^T / {
		dt = ($2 - prev) / 1000; prev = $2
		if (nr == 0)
			printf "%7.1f  %s\n", $2 / 1000, "no collectives"
		for (i = 1; i <= nr; i++) {
			split(r[i], f, " ")
			printf "%7.1f  %-11s %9.0f %9.1f %8d %8d\n", $2 / 1000, f[2], f[3] / dt, f[4] / 1e6 / dt, f[5], f[6]
		}
		nr = 0; next
	}
	/^E / {
		secs = $2 / 1000
		printf "\nTotals over %.1f s\n", secs
		printf "%-11s %10s %10s %8s %8s %8s %8s\n", "op", "calls", "MB", "MB/s", "avg us", "min us", "max us"
		next
	}
	/^S / {
		total += $4
		printf "%-11s %10d %10.1f %8.1f %8d %8d %8d\n", $2, $3, $4 / 1e6, (secs > 0 ? $4 / 1e6 / secs : 0), $5, $6, $7
		next
	}
	/^Latency per call/ && secs > 0 {
		printf "%-11s %10s %10.1f %8.1f\n", "all", "", total / 1e6, total / 1e6 / secs
	}
	{ print }
	'
)
