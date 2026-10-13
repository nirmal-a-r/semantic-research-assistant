"""Re-run a command when it dies, for unattended jobs on an unstable GPU.

The machine recorded a VIDEO_TDR_ERROR (bugcheck 0x116) during an earlier run, so
every long job is wrapped: each attempt is logged with its exit code, and the
experiment scripts themselves resume from whatever they already wrote.

Usage:
    python scripts/supervise.py --attempts 4 --label sweep -- python scripts/sweep_retrieval.py --stage screen
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--delay", type=int, default=30, help="seconds between attempts")
    ap.add_argument("--label", default="job")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        sys.exit("no command given")

    log_path = ROOT / "results" / "logs" / f"supervise_{args.label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, args.attempts + 1):
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{start}] attempt {attempt}/{args.attempts}: {' '.join(command)}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        t0 = time.time()
        code = subprocess.call(command, cwd=str(ROOT))
        mins = (time.time() - t0) / 60

        result = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] attempt {attempt} exited {code} after {mins:.1f} min"
        print(result, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(result + "\n")

        if code == 0:
            print(f"SUPERVISE_OK {args.label}", flush=True)
            return 0
        if attempt < args.attempts:
            time.sleep(args.delay)

    print(f"SUPERVISE_FAILED {args.label}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
