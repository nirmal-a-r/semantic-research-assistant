"""Fine-tune an extractive reader on SQuAD v1.1.

Preprocessing, post-processing and hyperparameters follow Sections 11-12 of
notebooks/Adaptive_Hybrid_Semantic_Research_Assistant.ipynb, so a rerun is
comparable with the published results. Batch size defaults suit an 8 GB GPU.

Usage:
    python scripts/train_reader.py --model roberta-base --run-name roberta-base-squad1
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_squad

MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
SEED = 42
LEARNING_RATES = {
    "roberta-base": 1.5e-5,
    "bert-base-uncased": 3e-5,
    "distilbert-base-uncased": 5e-5,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_preprocess_fn(tokenizer):
    def preprocess_training(examples):
        questions = [q.strip() for q in examples["question"]]
        tokenized = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        start_pos, end_pos = [], []

        for i, offsets in enumerate(offset_mapping):
            sidx = sample_map[i]
            answer = examples["answers"][sidx]
            input_ids = tokenized["input_ids"][i]
            cls_idx = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0

            if len(answer["answer_start"]) == 0:
                start_pos.append(cls_idx); end_pos.append(cls_idx)
                continue

            start_c = answer["answer_start"][0]
            end_c = start_c + len(answer["text"][0])
            seq_ids = tokenized.sequence_ids(i)
            try:
                ctx_start = next(k for k, s in enumerate(seq_ids) if s == 1)
                ctx_end = next(k for k in range(len(seq_ids) - 1, -1, -1) if seq_ids[k] == 1)
            except StopIteration:
                start_pos.append(cls_idx); end_pos.append(cls_idx)
                continue

            if offsets[ctx_start][0] > start_c or offsets[ctx_end][1] < end_c:
                start_pos.append(cls_idx); end_pos.append(cls_idx)
            else:
                idx = ctx_start
                while idx <= ctx_end and offsets[idx][0] <= start_c:
                    idx += 1
                start_pos.append(idx - 1)
                idx = ctx_end
                while idx >= ctx_start and offsets[idx][1] >= end_c:
                    idx -= 1
                end_pos.append(idx + 1)

        tokenized["start_positions"] = start_pos
        tokenized["end_positions"] = end_pos
        return tokenized

    def preprocess_validation(examples):
        questions = [q.strip() for q in examples["question"]]
        tokenized = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized["offset_mapping"]
        example_ids, start_pos, end_pos = [], [], []

        for i in range(len(tokenized["input_ids"])):
            sidx = sample_map[i]
            example_ids.append(examples["id"][sidx])
            seq_ids = tokenized.sequence_ids(i)
            offsets_i = offset_mapping[i]
            answer = examples["answers"][sidx]
            input_ids = tokenized["input_ids"][i]
            cls_idx = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0

            if len(answer["answer_start"]) == 0:
                start_pos.append(cls_idx); end_pos.append(cls_idx)
            else:
                start_c = answer["answer_start"][0]
                end_c = start_c + len(answer["text"][0])
                try:
                    ctx_start = next(k for k, s in enumerate(seq_ids) if s == 1)
                    ctx_end = next(k for k in range(len(seq_ids) - 1, -1, -1) if seq_ids[k] == 1)
                except StopIteration:
                    ctx_start = ctx_end = None

                if ctx_start is None or offsets_i[ctx_start][0] > start_c or offsets_i[ctx_end][1] < end_c:
                    start_pos.append(cls_idx); end_pos.append(cls_idx)
                else:
                    idx = ctx_start
                    while idx <= ctx_end and offsets_i[idx][0] <= start_c:
                        idx += 1
                    start_pos.append(idx - 1)
                    idx = ctx_end
                    while idx >= ctx_start and offsets_i[idx][1] >= end_c:
                        idx -= 1
                    end_pos.append(idx + 1)

            offset_mapping[i] = [(o if seq_ids[k] == 1 else None) for k, o in enumerate(offsets_i)]

        tokenized["example_id"] = example_ids
        tokenized["start_positions"] = start_pos
        tokenized["end_positions"] = end_pos
        return tokenized

    return preprocess_training, preprocess_validation


def postprocess_predictions(examples_list, features, raw_predictions, n_best=20, max_answer_length=30):
    features_per_example = defaultdict(list)
    for i, feat in enumerate(features):
        features_per_example[feat["example_id"]].append(i)

    start_logits, end_logits = raw_predictions
    predictions = {}

    for example in examples_list:
        context = example["context"]
        valid = []
        for feat_idx in features_per_example.get(example["id"], []):
            offsets = features[feat_idx]["offset_mapping"]
            sl, el = start_logits[feat_idx], end_logits[feat_idx]
            start_idxs = sorted(range(len(sl)), key=lambda x: sl[x], reverse=True)[:n_best]
            end_idxs = sorted(range(len(el)), key=lambda x: el[x], reverse=True)[:n_best]
            for s in start_idxs:
                for e in end_idxs:
                    if s >= len(offsets) or e >= len(offsets) or offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or (e - s + 1) > max_answer_length:
                        continue
                    valid.append({"score": float(sl[s] + el[e]),
                                  "text": context[offsets[s][0]:offsets[e][1]]})
        predictions[example["id"]] = (sorted(valid, key=lambda x: x["score"], reverse=True)[0]["text"]
                                      if valid else "")
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--train-subset", type=int, default=None)
    ap.add_argument("--eval-subset", type=int, default=None)
    args = ap.parse_args()

    run_name = args.run_name or f"{args.model}-squad1"
    import evaluate as hf_evaluate
    import torch
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, Trainer,
                              TrainingArguments, default_data_collator)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"fine-tuning {args.model} as {run_name} on {device}")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        log(f"GPU free {free/1024**3:.1f} GB of {total/1024**3:.1f} GB")

    raw_train, raw_eval = load_squad(train_subset=args.train_subset, eval_subset=args.eval_subset)
    log(f"train {len(raw_train)} / eval {len(raw_eval)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    prep_train, prep_val = make_preprocess_fn(tokenizer)

    log("tokenising")
    train_feats = raw_train.map(prep_train, batched=True, remove_columns=raw_train.column_names)
    eval_feats = raw_eval.map(prep_val, batched=True, remove_columns=raw_eval.column_names)
    eval_model = eval_feats.remove_columns(["example_id", "offset_mapping"])
    log(f"features: train {len(train_feats)} / eval {len(eval_feats)}")

    out_dir = ROOT / "models" / run_name

    # transformers 5.x dropped warmup_ratio; keep the original 6% warm-up by
    # converting it to the equivalent number of optimizer steps.
    steps_per_epoch = math.ceil(len(train_feats) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.06 * total_steps)
    log(f"{steps_per_epoch} steps/epoch, {total_steps} total, {warmup_steps} warm-up")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        learning_rate=LEARNING_RATES.get(args.model, 3e-5),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps, weight_decay=0.01, max_grad_norm=1.0,
        lr_scheduler_type="linear", fp16=(device == "cuda"),
        seed=SEED, data_seed=SEED, report_to=[], logging_steps=100,
        label_names=["start_positions", "end_positions"],
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_feats,
                      eval_dataset=eval_model, data_collator=default_data_collator,
                      processing_class=tokenizer)

    resume = None
    if out_dir.exists() and any(out_dir.glob("checkpoint-*")):
        resume = True
        log(f"resuming from the latest checkpoint in {out_dir}")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_hours = (time.time() - t0) / 3600
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log(f"training finished in {train_hours:.2f} h -> {out_dir}")

    (ROOT / "results" / f"{run_name}_log_history_rerun.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    log("predicting on the development set")
    raw_preds = trainer.predict(eval_model)
    preds = postprocess_predictions(list(raw_eval), eval_feats, raw_preds.predictions)
    metrics = hf_evaluate.load("squad").compute(
        predictions=[{"id": k, "prediction_text": v} for k, v in preds.items()],
        references=[{"id": ex["id"], "answers": ex["answers"]} for ex in raw_eval])
    log(f"RESULT {run_name}: EM {metrics['exact_match']:.2f}  F1 {metrics['f1']:.2f}  ({train_hours:.2f} h)")

    (ROOT / "results" / f"{run_name}_rerun_metrics.json").write_text(json.dumps({
        "run_name": run_name, "base_model": args.model, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "epochs": args.epochs, "effective_batch": args.batch_size * args.grad_accum,
        "train_hours": round(train_hours, 3), "exact_match": metrics["exact_match"], "f1": metrics["f1"],
        "n_eval": len(raw_eval), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
