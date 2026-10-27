#!/usr/bin/env python3
"""Run mlx.launch with an in-memory fix; no installed file is modified.

    python -P mlx_launch_shim.py /path/to/bin/mlx.launch [mlx.launch arguments]

(start.py runs it so). mlx.launch (mlx 0.32.4, _launch_with_io() in
mlx/_distributed_utils/launch.py) forwards its stdin to the ranks with one
thread per rank, whose select() also waits for the stdin pipe of the rank's
ssh to be writable: it always is, so every thread spins and mlx.launch keeps
Mac1 at ~130% CPU for as long as the server runs. start.py never writes to the
stdin of mlx.launch, so here select() gets an empty write list; the read list
and the timeout are unchanged (each thread wakes up on output of its rank, or
once a second). If mlx has changed that code, the unchanged mlx.launch runs.
"""

import inspect
import os
import select
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.argv = sys.argv[1:]  # as mlx.launch sees them
    try:
        from mlx._distributed_utils import launch
        if (launch.select is not select.select
                or "select(to_read, to_write, " not in inspect.getsource(launch._launch_with_io)):
            raise RuntimeError("code different from mlx 0.32.4")
    except Exception as e:
        print(f"[start] mlx.launch not patched, it can keep 1-2 CPU cores busy: {e}",
              file=sys.stderr, flush=True)
        os.execv(sys.argv[0], sys.argv)
    launch.select = lambda rlist, wlist, xlist, *timeout: select.select(rlist, [], xlist, *timeout)
    return launch.main()


if __name__ == "__main__":
    sys.exit(main())
