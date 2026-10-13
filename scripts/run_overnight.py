"""Run the whole overnight plan in order, surviving crashes.

Each stage is retried, stages whose outputs already exist are skipped, and the
summary is rebuilt after every stage so there is always a current report.

Order: prefetch -> multi-query comparison -> retrieval sweep (screen, then confirm)
       -> reader fine-tune -> end-to-end rank-1 vs top-K -> summary.

Usage:
    python scripts/run_overnight.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "results" / "logs" / "overnight.log"

SCREEN_CONFIGS = [
    {"encoder": e, "cross": c, "candidates": 50, "rerank_k": 30, "chunked": False, "mq": False}
    for e in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5",
              "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2"]
    for c in ["cross-encoder/ms-marco-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L12-v2"]
]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(stage, command, attempts=3, delay=30):
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
                print(line, end="", flush=True)
            code = proc.wait()
        mins = (time.time() - t0) / 60
        if code == 0:
            log(f"{stage}: done in {mins:.1f} min")
            return True
        log(f"{stage}: exit {code} after {mins:.1f} min")
        if attempt < attempts:
            time.sleep(delay)
    log(f"{stage}: GIVING UP after {attempts} attempts")
    return False


def summarise():
    subprocess.call([PY, "-u", "scripts/summarise_runs.py"], cwd=str(ROOT))


def main():
    log("=== overnight run started ===")

    run("prefetch", [PY, "-u", "scripts/prefetch_models.py"], attempts=3)

    if not (ROOT / "results/multiquery_comparison.csv").exists():
        run("multiquery", [PY, "-u", "scripts/eval_multiquery.py", "--n", "2000"], attempts=3)
        summarise()
    else:
        log("multiquery: already present, skipping")

    (ROOT / "results/sweep_confirm_configs.json").write_text(json.dumps(SCREEN_CONFIGS, indent=1), encoding="utf-8")
    run("sweep-screen", [PY, "-u", "scripts/sweep_retrieval.py", "--stage", "confirm",
                         "--n", "500", "--out", "results/sweep_screen.csv"], attempts=4)
    summarise()

    # Promote the best screened configuration to the full question set, alongside
    # depth, chunking and multi-query variants of it.
    screen_csv = ROOT / "results/sweep_screen.csv"
    if screen_csv.exists():
        import pandas as pd
        df = pd.read_csv(screen_csv).sort_values("Recall@10", ascending=False)
        best = df.iloc[0]
        base = {"encoder": best["encoder"], "cross": best["cross"],
                "candidates": 50, "rerank_k": 30, "chunked": False, "mq": False}
        confirm = [
            dict(base),
            dict(base, candidates=100, rerank_k=50),
            dict(base, chunked=True),
            dict(base, chunked=True, dedup=True),
            dict(base, mq=True),
            {"encoder": "sentence-transformers/all-MiniLM-L6-v2",
             "cross": "cross-encoder/ms-marco-MiniLM-L6-v2",
             "candidates": 50, "rerank_k": 30, "chunked": False, "mq": False},
        ]
        (ROOT / "results/sweep_confirm_configs.json").write_text(json.dumps(confirm, indent=1), encoding="utf-8")
        log(f"confirm stage: best screened config is {best['config']}")
        run("sweep-confirm", [PY, "-u", "scripts/sweep_retrieval.py", "--stage", "confirm",
                              "--n", "2000", "--out", "results/sweep_confirm.csv"], attempts=4)
        summarise()

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        log("train-reader: no usable GPU in this process, skipping (CPU fine-tuning would take days)")
    elif not (ROOT / "results/roberta-base-squad1_rerun_metrics.json").exists():
        run("train-reader", [PY, "-u", "scripts/train_reader.py", "--model", "roberta-base",
                             "--run-name", "roberta-base-squad1"], attempts=4, delay=60)
        summarise()
    else:
        log("train-reader: already present, skipping")

    if cuda_ok and (ROOT / "models/roberta-base-squad1").exists():
        run("end-to-end", [PY, "-u", "scripts/eval_end_to_end.py", "--checkpoint",
                           "models/roberta-base-squad1", "--n", "200"], attempts=3)
    else:
        log("end-to-end: no checkpoint, skipping")

    summarise()
    log("=== overnight run finished ===")


if __name__ == "__main__":
    main()
