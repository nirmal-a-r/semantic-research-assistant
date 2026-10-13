"""Compare span decoders on gold passages.

qa_infer.qa_predict_single takes the arg-max start and end independently and reads
only the first 384-token window. qa_decode.best_span scores start/end pairs jointly,
caps the span length and covers every window. Both are measured here on the same
questions with the same checkpoint, so the end-to-end gain from reading more passages
can be separated from the gain from decoding better.

Usage:
    python scripts/eval_decoder.py --checkpoint models/roberta-base-squad1 --n 2000
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_squad
from src.qa_decode import best_span
from src.qa_infer import qa_predict_single


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/roberta-base-squad1")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/decoder_comparison.json")
    args = ap.parse_args()

    import evaluate as hf_evaluate
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint).to(device).eval()
    log(f"reader {args.checkpoint} on {device}")

    _, dev = load_squad()
    examples = list(dev.select(range(min(args.n, len(dev)))))
    log(f"{len(examples)} gold-passage questions")

    preds_old, preds_new, refs, changed = [], [], [], 0
    t_old = t_new = 0.0

    for i, ex in enumerate(examples, 1):
        q, ctx = ex["question"], ex["context"]

        t0 = time.time()
        old = qa_predict_single(model, tokenizer, q, ctx, device=device)
        t_old += time.time() - t0

        t0 = time.time()
        new = best_span(model, tokenizer, q, ctx, device=device)["text"]
        t_new += time.time() - t0

        preds_old.append({"id": ex["id"], "prediction_text": old})
        preds_new.append({"id": ex["id"], "prediction_text": new})
        refs.append({"id": ex["id"], "answers": ex["answers"]})
        changed += old != new
        if i % 250 == 0:
            log(f"  {i}/{len(examples)}")

    squad = hf_evaluate.load("squad")
    m_old = squad.compute(predictions=preds_old, references=refs)
    m_new = squad.compute(predictions=preds_new, references=refs)

    log(f"argmax decoder : EM {m_old['exact_match']:.2f}  F1 {m_old['f1']:.2f}  "
        f"{1000*t_old/len(examples):.1f} ms/q")
    log(f"joint decoder  : EM {m_new['exact_match']:.2f}  F1 {m_new['f1']:.2f}  "
        f"{1000*t_new/len(examples):.1f} ms/q")
    log(f"answers changed: {changed}/{len(examples)}")

    (ROOT / args.out).write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": args.checkpoint, "n_questions": len(examples),
        "argmax": {**m_old, "ms_per_question": 1000 * t_old / len(examples)},
        "joint": {**m_new, "ms_per_question": 1000 * t_new / len(examples)},
        "changed": changed,
    }, indent=2), encoding="utf-8")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
