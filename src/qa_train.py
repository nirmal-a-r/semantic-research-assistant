"""
src/qa_train.py
────────────────
Fine-tuning HuggingFace Question Answering models (RoBERTa, BERT, DistilBERT)
on SQuAD v1.1 using the HuggingFace Trainer and evaluate library.
"""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import evaluate as hf_evaluate
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

squad_metric = hf_evaluate.load("squad")


def make_preprocess_fn(tokenizer, max_seq_length=384, doc_stride=128):
    """Create train and validation preprocessing functions with sliding windows."""

    def preprocess_training(examples):
        questions = [q.strip() for q in examples["question"]]
        tokenized = tokenizer(
            questions,
            examples["context"],
            max_length=max_seq_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        start_pos, end_pos = [], []

        for i, offsets in enumerate(offset_mapping):
            sidx = sample_map[i]
            answer = examples["answers"][sidx]
            input_ids = tokenized["input_ids"][i]
            cls_idx = (
                input_ids.index(tokenizer.cls_token_id)
                if tokenizer.cls_token_id in input_ids
                else 0
            )

            if len(answer["answer_start"]) == 0:
                start_pos.append(cls_idx)
                end_pos.append(cls_idx)
                continue

            start_c = answer["answer_start"][0]
            end_c = start_c + len(answer["text"][0])
            seq_ids = tokenized.sequence_ids(i)

            try:
                ctx_start = next(k for k, s in enumerate(seq_ids) if s == 1)
                ctx_end = next(
                    k for k in range(len(seq_ids) - 1, -1, -1) if seq_ids[k] == 1
                )
            except StopIteration:
                start_pos.append(cls_idx)
                end_pos.append(cls_idx)
                continue

            if offsets[ctx_start][0] > start_c or offsets[ctx_end][1] < end_c:
                start_pos.append(cls_idx)
                end_pos.append(cls_idx)
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
            questions,
            examples["context"],
            max_length=max_seq_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        example_ids = []
        for i in range(len(tokenized["input_ids"])):
            sidx = sample_map[i]
            example_ids.append(examples["id"][sidx])
            seq_ids = tokenized.sequence_ids(i)
            tokenized["offset_mapping"][i] = [
                (o if seq_ids[k] == 1 else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        tokenized["example_id"] = example_ids
        return tokenized

    return preprocess_training, preprocess_validation


def postprocess_predictions(
    examples_list, features, raw_predictions, n_best=20, max_answer_length=30
):
    """Convert start/end logits into extracted text predictions for SQuAD evaluation."""
    features_per_example = defaultdict(list)
    for i, feat in enumerate(features):
        eid = feat["example_id"]
        features_per_example[eid].append(i)

    start_logits, end_logits = raw_predictions
    predictions = {}

    for example in examples_list:
        eid = example["id"]
        context = example["context"]
        feat_indices = features_per_example.get(eid, [])
        valid_answers = []

        for feat_idx in feat_indices:
            feat = features[feat_idx]
            offsets = feat["offset_mapping"]
            sl = start_logits[feat_idx]
            el = end_logits[feat_idx]
            start_idxs = sorted(
                range(len(sl)), key=lambda x: sl[x], reverse=True
            )[:n_best]
            end_idxs = sorted(
                range(len(el)), key=lambda x: el[x], reverse=True
            )[:n_best]
            for s in start_idxs:
                for e in end_idxs:
                    if (
                        s >= len(offsets)
                        or e >= len(offsets)
                        or offsets[s] is None
                        or offsets[e] is None
                    ):
                        continue
                    if e < s or (e - s + 1) > max_answer_length:
                        continue
                    valid_answers.append({
                        "score": float(sl[s] + el[e]),
                        "text": context[offsets[s][0] : offsets[e][1]],
                    })

        predictions[eid] = (
            sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0][
                "text"
            ]
            if valid_answers
            else ""
        )
    return predictions


def fine_tune_qa_model(
    model_name: str,
    run_name: str,
    raw_train,
    raw_eval,
    config: dict,
    device: str = "cpu",
):
    """Fine-tune a QA model using HuggingFace Trainer with evaluation on validation set."""
    print(f"\n=== Fine-tuning {run_name} ({model_name}) on {device.upper()} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    prep_train, prep_val = make_preprocess_fn(
        tokenizer,
        max_seq_length=config.get("max_seq_length", 384),
        doc_stride=config.get("doc_stride", 128),
    )

    train_feats = raw_train.map(
        prep_train,
        batched=True,
        remove_columns=raw_train.column_names,
        num_proc=1,
    )
    eval_feats = raw_eval.map(
        prep_val,
        batched=True,
        remove_columns=raw_eval.column_names,
        num_proc=1,
    )
    eval_model = eval_feats.remove_columns(["example_id", "offset_mapping"])

    out_dir = Path(config.get("models_dir", "models")) / run_name
    model_lr = config.get("learning_rates", {}).get(model_name, 3e-5)

    args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=model_lr,
        per_device_train_batch_size=config.get("train_batch_size", 16),
        per_device_eval_batch_size=config.get("eval_batch_size", 16),
        gradient_accumulation_steps=config.get(
            "gradient_accumulation_steps", 2
        ),
        num_train_epochs=config.get("num_train_epochs", 3),
        warmup_ratio=config.get("warmup_ratio", 0.06),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        fp16=(device == "cuda"),
        seed=config.get("seed", 42),
        data_seed=config.get("seed", 42),
        report_to=["tensorboard"],
        logging_steps=config.get("logging_steps", 100),
        label_names=["start_positions", "end_positions"],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_feats,
        eval_dataset=eval_model,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(
        results_dir / f"{run_name}_log_history.json", "w", encoding="utf-8"
    ) as f:
        json.dump(trainer.state.log_history, f, indent=2)

    raw_preds = trainer.predict(eval_model)
    preds = postprocess_predictions(
        list(raw_eval), eval_feats, raw_preds.predictions
    )
    fmt_preds = [{"id": k, "prediction_text": v} for k, v in preds.items()]
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_eval]
    metrics = squad_metric.compute(predictions=fmt_preds, references=refs)
    print(
        f"{run_name} — Exact Match: {metrics['exact_match']:.2f}% | F1:"
        f" {metrics['f1']:.2f}%"
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "predictions": preds,
    }
