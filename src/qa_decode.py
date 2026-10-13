"""Span decoding for the extractive reader.

qa_infer.qa_predict_single takes the arg-max start and end independently, which can
select an invalid span (end before start, or a span longer than any real answer) and
ignores any part of a passage past the 384-token window. The functions here score
start/end pairs jointly over all windows, and can read several candidate chunks and
keep the most confident answer.
"""

import numpy as np
import torch


def _windows(tokenizer, question, context, max_length=384, stride=128):
    return tokenizer(question, context, return_tensors="pt", truncation="only_second",
                     max_length=max_length, stride=stride, padding="max_length",
                     return_overflowing_tokens=True, return_offsets_mapping=True)


def best_span(model, tokenizer, question, context, max_length=384, stride=128,
              n_best=20, max_answer_length=30, device=None):
    """Returns the highest-scoring valid span across every window of the context."""
    device = device or next(model.parameters()).device
    enc = _windows(tokenizer, question, context, max_length, stride)
    offsets_all = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    seq_ids = [enc.sequence_ids(i) for i in range(len(enc["input_ids"]))]
    inputs = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**inputs)
    starts = out.start_logits.float().cpu().numpy()
    ends = out.end_logits.float().cpu().numpy()

    best = {"text": "", "score": -1e9, "start": 0, "end": 0, "start_prob": 0.0, "end_prob": 0.0}
    for w in range(starts.shape[0]):
        sl, el = starts[w], ends[w]
        offsets = offsets_all[w].tolist()
        context_mask = [i for i, s in enumerate(seq_ids[w]) if s == 1]
        if not context_mask:
            continue
        s_probs = torch.softmax(torch.from_numpy(sl), dim=-1).numpy()
        e_probs = torch.softmax(torch.from_numpy(el), dim=-1).numpy()
        top_s = sorted(context_mask, key=lambda i: sl[i], reverse=True)[:n_best]
        top_e = sorted(context_mask, key=lambda i: el[i], reverse=True)[:n_best]
        for s in top_s:
            for e in top_e:
                if e < s or (e - s + 1) > max_answer_length:
                    continue
                score = float(sl[s] + el[e])
                if score > best["score"]:
                    cs, ce = offsets[s][0], offsets[e][1]
                    if ce <= cs:
                        continue
                    best = {"text": context[cs:ce], "score": score, "start": cs, "end": ce,
                            "start_prob": float(s_probs[s]), "end_prob": float(e_probs[e])}
    best["confidence"] = (best["start_prob"] + best["end_prob"]) / 2
    return best


def answer_over_candidates(model, tokenizer, question, contexts, device=None, **kwargs):
    """Reads each candidate chunk and returns the most confident answer.

    contexts: list of (identifier, text), highest-ranked first.
    """
    best, best_idx = None, -1
    for i, (doc_id, text) in enumerate(contexts):
        span = best_span(model, tokenizer, question, text, device=device, **kwargs)
        span["doc_id"] = doc_id
        span["rank"] = i + 1
        if span["text"] and (best is None or span["score"] > best["score"]):
            best, best_idx = span, i
    if best is None:
        return {"text": "", "score": -1e9, "confidence": 0.0, "doc_id": None, "rank": None}
    best["read_candidates"] = len(contexts)
    best["chosen_rank"] = best_idx + 1
    return best
