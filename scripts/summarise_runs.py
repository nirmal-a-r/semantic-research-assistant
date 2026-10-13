"""Collect tonight's experiment outputs into one comparison table.

Reads whatever exists in results/ (sweep CSVs, multi-query comparison, reader rerun
metrics, end-to-end JSON) and writes a markdown summary plus a LaTeX fragment that
can be pasted into report/main.tex.

Usage:
    python scripts/summarise_runs.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

PUBLISHED = {
    "BM25 only": (86.15, 89.60, 69.49, 73.39),
    "Dense only": (86.00, 90.55, 63.83, 68.97),
    "Hybrid (RRF)": (91.50, 94.60, 71.70, 76.37),
    "+ Re-ranking": (96.25, 96.65, 85.25, 87.97),
    "+ Adaptive chunking": (91.35, 94.40, 71.56, 75.64),
    "+ Multi-query": (96.25, 96.70, 85.19, 87.91),
    "Complete pipeline": (95.95, 96.45, 85.20, 87.31),
}
BEST_PUBLISHED = {"Recall@10": 96.25, "Recall@20": 96.70, "MRR": 85.25, "NDCG@10": 87.97}
PUBLISHED_READER = {"exact_match": 85.38, "f1": 91.74}
PUBLISHED_E2E = {"exact_match": 71.00, "f1": 72.34}


def load_sweeps():
    """Reads the sweep CSVs tolerantly: early rows were written before the column
    set was fixed, so some files have ragged rows."""
    import csv
    frames = []
    for path in sorted(RESULTS.glob("sweep_*.csv")):
        with path.open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header:
                continue
            records = []
            for raw in reader:
                if len(raw) < len(header):
                    continue
                records.append(dict(zip(header, raw[:len(header)])))
        if not records:
            continue
        df = pd.DataFrame(records)
        for col in ("Recall@10", "Recall@20", "MRR", "NDCG@10", "ms_per_question", "n_questions"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source"] = path.name
        frames.append(df)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    return merged.dropna(subset=["Recall@10"]).drop_duplicates(subset=["config", "n_questions"], keep="last")


def main():
    lines = ["# Overnight experiment summary", ""]

    sweeps = load_sweeps()
    if sweeps is not None:
        cols = ["config", "n_questions", "Recall@10", "Recall@20", "MRR", "NDCG@10", "ms_per_question"]

        # The published ablation used 2,000 development questions. Only rows measured
        # on the same 2,000 can be compared with it; the 500-question rows exist to
        # rank configurations against each other during screening.
        full = sweeps[sweeps["n_questions"] == 2000].sort_values("Recall@10", ascending=False)
        screen = sweeps[sweeps["n_questions"] != 2000].sort_values("MRR", ascending=False)

        if not full.empty:
            lines += ["## Full-scale runs (2,000 questions — comparable with the paper)", "",
                      full[cols].to_markdown(index=False, floatfmt=".2f"), ""]
            for metric in ("Recall@10", "Recall@20", "MRR", "NDCG@10"):
                row = full.loc[full[metric].idxmax()]
                delta = row[metric] - BEST_PUBLISHED[metric]
                verdict = "beats" if delta > 0 else "below"
                lines.append(f"- {metric}: {row[metric]:.2f} ({row['config']}) — {verdict} published "
                             f"best {BEST_PUBLISHED[metric]:.2f} by {abs(delta):.2f}")
            lines.append("")

        if not screen.empty:
            lines += ["## Screening runs (500 questions — not comparable with the paper)", "",
                      screen[cols].to_markdown(index=False, floatfmt=".2f"), ""]

    mq_cols = ["Recall@10", "Recall@20", "MRR", "NDCG@10", "mean_variants",
               "variant_gen_ms_per_question", "ms_per_question"]
    for mq, title in ((RESULTS / "multiquery_comparison_v2.csv",
                       "Multi-query: rules vs transformers (final, filtered generation)"),
                      (RESULTS / "multiquery_comparison.csv",
                       "Multi-query: first run (unfiltered generation, kept for the record)")):
        if mq.exists():
            df = pd.read_csv(mq, index_col=0)
            lines += [f"## {title}", "",
                      df[[c for c in mq_cols if c in df]].to_markdown(floatfmt=".2f"), ""]

    for path in sorted(RESULTS.glob("*_rerun_metrics.json")):
        m = json.loads(path.read_text())
        d_em = m["exact_match"] - PUBLISHED_READER["exact_match"]
        d_f1 = m["f1"] - PUBLISHED_READER["f1"]
        lines += [f"## Reader rerun: {m['run_name']}", "",
                  f"- EM {m['exact_match']:.2f} ({d_em:+.2f} vs published {PUBLISHED_READER['exact_match']})",
                  f"- F1 {m['f1']:.2f} ({d_f1:+.2f} vs published {PUBLISHED_READER['f1']})",
                  f"- {m['train_hours']:.2f} h on {m['device']}", ""]

    e2e = RESULTS / "end_to_end_topk.json"
    if e2e.exists():
        r = json.loads(e2e.read_text())
        lines += ["## End-to-end reading", "",
                  f"- rank-1 only: EM {r['rank1']['exact_match']:.2f}  F1 {r['rank1']['f1']:.2f}",
                  f"- top-K (mean K={r['mean_k']:.1f}): EM {r['topk']['exact_match']:.2f}  F1 {r['topk']['f1']:.2f}",
                  f"- published: EM {PUBLISHED_E2E['exact_match']:.2f}  F1 {PUBLISHED_E2E['f1']:.2f}",
                  f"- answers changed by top-K reading: {r['changed_by_topk']}/{r['n_questions']}", ""]

    out = RESULTS / "overnight_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
