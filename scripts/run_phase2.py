"""Remaining experiments after the reader fine-tune, ordered by value.

Each stage is retried and skipped if its output already exists, so this can be
restarted after a crash without losing finished work.

Order: end-to-end (rank-1 vs top-K) -> L12 re-ranker sweep -> four-way multi-query
       -> summary.

Usage:
    python scripts/run_phase2.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "results" / "logs" / "overnight.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(stage, command, attempts=3, delay=20):
    stage_log = LOG.parent / f"stage_{stage}.log"
    for attempt in range(1, attempts + 1):
        log(f"{stage}: attempt {attempt}/{attempts}")
        t0 = time.time()
        with stage_log.open("a", encoding="utf-8", buffering=1) as fh:
            fh.write(f"\n===== {stage} attempt {attempt} at {time.strftime('%H:%M:%S')} =====\n")
            proc = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
            for line in proc.stdout:
                fh.write(line)
            code = proc.wait()
        mins = (time.time() - t0) / 60
        if code == 0:
            log(f"{stage}: done in {mins:.1f} min")
            return True
        log(f"{stage}: exit {code} after {mins:.1f} min")
        if attempt < attempts:
            time.sleep(delay)
    log(f"{stage}: GIVING UP")
    return False


def summarise():
    subprocess.call([PY, "-u", "scripts/summarise_runs.py"], cwd=str(ROOT))


def main():
    log("=== phase 2 started ===")

    if not (ROOT / "results/end_to_end_topk.json").exists():
        run("end-to-end", [PY, "-u", "scripts/eval_end_to_end.py",
                           "--checkpoint", "models/roberta-base-squad1", "--n", "200"])
        summarise()
    else:
        log("end-to-end: already present, skipping")

    run("sweep-l12", [PY, "-u", "scripts/sweep_retrieval.py", "--stage", "confirm", "--n", "2000",
                      "--configs", "results/sweep_l12_configs.json", "--out", "results/sweep_l12.csv"])
    summarise()

    if not (ROOT / "results/multiquery_comparison_v2.csv").exists():
        run("multiquery-v2", [PY, "-u", "scripts/eval_multiquery.py", "--n", "2000",
                              "--t5-models", "google/flan-t5-small,humarin/chatgpt_paraphraser_on_T5_base",
                              "--out", "results/multiquery_comparison_v2.csv",
                              "--log", "results/multiquery_run_v2.json"])
        summarise()
    else:
        log("multiquery-v2: already present, skipping")

    summarise()
    log("=== phase 2 finished ===")


if __name__ == "__main__":
    main()
