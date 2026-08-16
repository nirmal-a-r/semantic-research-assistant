"""
build_dashboard_data.py — single source of truth for everything the dashboard shows.

Reads the artefacts produced by the notebook / training run in results/ and writes:

    app/dashboard/public/data/project_data.json   <- every number the dashboard renders
    app/dashboard/public/artifacts/*              <- downloadable copies of results/

Usage
-----
    python scripts/build_dashboard_data.py
        Rebuild project_data.json + refresh the public artefact copies.
        Pure file I/O, no GPU, takes a second.

    python scripts/build_dashboard_data.py --rerun-inference
        Additionally re-run real GPU inference with the fine-tuned checkpoints in
        models/ to regenerate:
            results/detailed_eval.json   per-prediction gold-passage evaluation
            results/demo_bundle.json     precomputed end-to-end pipeline answers
                                         (this is what makes the deployed site's
                                          Live Demo work with no backend running)

Run the plain form after any notebook re-run; run --rerun-inference after retraining.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DASH = ROOT / "app" / "dashboard" / "public"
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
CHUNKED = ROOT / "data" / "processed" / "chunked_corpus.jsonl"

MODELS = {
    # key                        display            checkpoint dir                 params(M)
    "roberta-base":            ("RoBERTa-base",     "roberta-base-squad1",           125),
    "bert-base-uncased":       ("BERT-base",        "bert-base-uncased-squad1",      110),
    "distilbert-base-uncased": ("DistilBERT-base",  "distilbert-base-uncased-squad1", 66),
}
SUMMARY_NAMES = {           # keys as written by the notebook into summary.json
    "roberta-base": "RoBERTa-base",
    "bert-base-uncased": "BERT-base-uncased",
    "distilbert-base-uncased": "DistilBERT-base-uncased",
}
ABLATION_ORDER = ["BM25 only", "Dense only", "Hybrid (RRF)", "+ Re-ranking",
                  "+ Adaptive Chunking", "+ Multi-Query+RRF", "Complete Pipeline"]
COMPLEXITY_TOP_K = {"simple": 5, "comparison": 10, "multistep": 15}
N_DEMO = 24


# ══════════════════════════════════════════════════════════════════════════════
# SQuAD metrics (identical normalisation to the official script)
# ══════════════════════════════════════════════════════════════════════════════
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> bool:
    return _norm(pred) == _norm(gold)


def best(fn, pred, golds):
    return max(fn(pred, g) for g in golds)


# ══════════════════════════════════════════════════════════════════════════════
# GPU pass — per-prediction evaluation + precomputed demo answers
# ══════════════════════════════════════════════════════════════════════════════
def rerun_inference() -> None:
    import random
    import sys

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    sys.path.insert(0, str(ROOT))
    from src.fusion import reciprocal_rank_fusion
    from src.multiquery import estimate_query_complexity, generate_query_variants
    from src.retrieval_dense import DenseRetriever, build_faiss_index
    from src.retrieval_sparse import BM25Retriever

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[inference] device={device} torch={torch.__version__}")

    def qa_with_conf(model, tok, question, context, max_length=384):
        """Span extraction + softmax confidences — mirrors app/api.py exactly."""
        inputs = tok(question, context, return_tensors="pt", truncation="only_second",
                     max_length=max_length, return_offsets_mapping=True).to(device)
        offsets = inputs.pop("offset_mapping")[0].tolist()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000
        si, ei = int(torch.argmax(out.start_logits)), int(torch.argmax(out.end_logits))
        sp = float(torch.softmax(out.start_logits, dim=-1)[0][si])
        ep = float(torch.softmax(out.end_logits, dim=-1)[0][ei])
        if ei < si or not offsets[si] or not offsets[ei]:
            return "", sp, ep, latency
        return context[offsets[si][0]:offsets[ei][1]], sp, ep, latency

    def find_span(answer, context):
        if not answer:
            return 0, 0
        i = context.lower().find(answer.lower())
        return (i, i + len(answer)) if i != -1 else (0, 0)

    docs = [json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip()]
    lookup = {d["doc_id"]: d["text"] for d in docs}
    bm25 = BM25Retriever(docs)
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    index, ids = build_faiss_index(docs, encoder, batch_size=256)
    dense = DenseRetriever(index, ids, encoder)
    print(f"[inference] retrieval ready over {len(docs)} passages")

    loaded = {}
    for key, (disp, ckpt, _) in MODELS.items():
        path = ROOT / "models" / ckpt
        loaded[key] = (
            AutoModelForQuestionAnswering.from_pretrained(str(path)).to(device).eval(),
            AutoTokenizer.from_pretrained(str(path)),
        )
        print(f"[inference] loaded {disp}")

    # Deterministic, topic-diverse sample of dev questions (one per article first,
    # then extra paragraphs) so the demo gallery spans many Wikipedia topics.
    dev = json.loads((ROOT / "data" / "raw" / "dev.json").read_text(encoding="utf-8"))["data"]
    first, extra = [], []
    for art in dev:
        for pi, para in enumerate(art["paragraphs"][:4]):
            qa = para["qas"][0]
            golds = list({a["text"] for a in qa["answers"]})
            if not golds:
                continue
            row = {"title": art["title"], "question": qa["question"],
                   "context": para["context"], "golds": golds}
            (first if pi == 0 else extra).append(row)
    random.Random(42).shuffle(extra)
    sample = (first + extra)[:N_DEMO]

    for mdl, tok in loaded.values():                       # CUDA warm-up
        qa_with_conf(mdl, tok, "warm up?", "This is a warm up context for CUDA kernels.")

    detailed = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "n_sample": len(sample),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {disp: {"key": k, "params": p, "predictions": []}
                   for k, (disp, _, p) in MODELS.items()},
    }
    demo = []

    for i, ex in enumerate(sample, 1):
        q, golds = ex["question"], ex["golds"]

        # (a) gold-passage setting — isolates QA quality from retrieval
        for key, (mdl, tok) in loaded.items():
            pred, sp, ep, lat = qa_with_conf(mdl, tok, q, ex["context"])
            detailed["models"][MODELS[key][0]]["predictions"].append({
                "question": q, "gold": golds[0], "pred": pred,
                "em": bool(best(em_score, pred, golds)),
                "f1": round(best(f1_score, pred, golds) * 100, 1),
                "conf": round((sp + ep) / 2, 3), "lat": round(lat, 1),
            })

        # (b) full pipeline — what the deployed Live Demo replays offline
        complexity = estimate_query_complexity(q)
        top_k = COMPLEXITY_TOP_K[complexity]
        variants = generate_query_variants(q)
        merged = reciprocal_rank_fusion(
            [reciprocal_rank_fusion([bm25.search(v, top_k=top_k * 3),
                                     dense.search(v, top_k=top_k * 3)], top_k=top_k)
             for v in variants], top_k=top_k)
        top_doc, top_score = merged[0]
        context = lookup[top_doc]

        entry = {
            "question": q, "topic": ex["title"].replace("_", " "),
            "gold": golds[0], "gold_answers": golds,
            "query_complexity": complexity, "top_k": top_k, "query_variants": variants,
            "evidence_doc_id": top_doc, "evidence_score": round(top_score, 6),
            "evidence_text": context,
            "retrieval_hit": any(_norm(g) in _norm(context) for g in golds),
            "top_chunks": [{"doc_id": d, "score": round(s, 6), "text": lookup[d], "rank": r + 1}
                           for r, (d, s) in enumerate(merged[:3])],
            "answers": {},
        }
        for key, (mdl, tok) in loaded.items():
            pred, sp, ep, lat = qa_with_conf(mdl, tok, q, context)
            st, en = find_span(pred, context)
            entry["answers"][key] = {
                "answer": pred, "answer_start": st, "answer_end": en,
                "start_confidence": round(sp, 4), "end_confidence": round(ep, 4),
                "avg_confidence": round((sp + ep) / 2, 4), "latency_ms": round(lat, 1),
                "answer_length_words": len(pred.split()) if pred else 0,
                "context_length_words": len(context.split()),
                "exact_match": bool(best(em_score, pred, golds)),
                "token_f1": round(best(f1_score, pred, golds) * 100, 1),
                "model_params_million": MODELS[key][2],
            }
        demo.append(entry)
        print(f"  {i:2d}/{len(sample)} {q[:58]}")

    for m in detailed["models"].values():
        preds = m["predictions"]
        m["exact_match"] = round(100 * sum(p["em"] for p in preds) / len(preds), 1)
        m["f1"] = round(sum(p["f1"] for p in preds) / len(preds), 1)
        m["avg_latency_ms"] = round(sum(p["lat"] for p in preds) / len(preds), 1)
        m["avg_confidence"] = round(sum(p["conf"] for p in preds) / len(preds), 3)

    _write(RESULTS / "detailed_eval.json", detailed)
    _write(RESULTS / "demo_bundle.json", demo)
    print(f"[inference] wrote detailed_eval.json + demo_bundle.json "
          f"({sum(e['retrieval_hit'] for e in demo)}/{len(demo)} retrieval hits)")


# ══════════════════════════════════════════════════════════════════════════════
# Assembly — results/*  ->  project_data.json
# ══════════════════════════════════════════════════════════════════════════════
def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _pct(x):
    """summary.json stores retrieval as 0-1 fractions and EM/F1 as 0-100."""
    return None if x is None else round(x * 100, 2)


def corpus_stats() -> dict:
    docs = [json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip()]
    chunks = sum(1 for l in CHUNKED.open(encoding="utf-8") if l.strip()) if CHUNKED.exists() else 0
    return {
        "passages": len(docs),
        "chunks": chunks,
        "articles": len({d["title"] for d in docs}),
        "avg_words": round(sum(len(d["text"].split()) for d in docs) / len(docs)),
    }


def convergence() -> tuple[dict, dict]:
    """Per-model loss curves + wall-clock training facts from the HF log history."""
    curves, facts = {}, {}
    for key, (_, ckpt, _) in MODELS.items():
        path = RESULTS / f"{ckpt}_log_history.json"
        history = _read_json(path)
        if not history:
            continue
        points = []
        for rec in history:
            if "loss" in rec:
                points.append({"step": rec["step"], "epoch": round(rec["epoch"], 2),
                               "train_loss": round(rec["loss"], 4), "eval_loss": None})
            elif "eval_loss" in rec:
                points.append({"step": rec["step"], "epoch": round(rec["epoch"], 2),
                               "train_loss": None, "eval_loss": round(rec["eval_loss"], 4)})
        curves[key] = sorted(points, key=lambda p: p["step"])
        final = next((r for r in reversed(history) if "train_runtime" in r), {})
        last_eval = next((r for r in reversed(history) if "eval_loss" in r), {})
        facts[key] = {
            "train_runtime_s": round(final.get("train_runtime", 0)),
            "train_runtime_h": round(final.get("train_runtime", 0) / 3600, 2),
            "samples_per_second": round(final.get("train_samples_per_second", 0), 1),
            "total_steps": final.get("step"),
            "final_train_loss": round(final.get("train_loss", 0), 4),
            "final_eval_loss": round(last_eval.get("eval_loss", 0), 4),
            "epochs": round(final.get("epoch", 0)),
        }
    return curves, facts


def qualitative() -> list:
    path = RESULTS / "qa_sample_predictions.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [{
            "question": r["question"],
            "gold": r.get("gold", "?"),
            "retrieved_passage": r["retrieved_passage"],
            "roberta": r["RoBERTa-base"],
            "bert": r["BERT-base-uncased"],
            "distilbert": r["DistilBERT-base-uncased"],
        } for r in csv.DictReader(f)]


def build() -> dict:
    summary = _read_json(RESULTS / "summary.json", {})
    detailed = _read_json(RESULTS / "detailed_eval.json", {})
    demo = _read_json(RESULTS / "demo_bundle.json", [])
    stats = corpus_stats()
    curves, facts = convergence()

    retr = summary.get("retrieval", {})
    gold = summary.get("qa_gold_passage", {})
    e2e = summary.get("e2e", {})

    retrieval_chart = [{
        "name": cfg,
        "Recall@10": _pct(retr.get("Recall@10", {}).get(cfg)),
        "Recall@20": _pct(retr.get("Recall@20", {}).get(cfg)),
        "MRR": _pct(retr.get("MRR", {}).get(cfg)),
        "NDCG@10": _pct(retr.get("NDCG@10", {}).get(cfg)),
    } for cfg in ABLATION_ORDER if cfg in retr.get("Recall@10", {})]

    qa_models = []
    for key, (disp, _, params) in MODELS.items():
        name = SUMMARY_NAMES[key]
        lat = detailed.get("models", {}).get(disp, {}).get("avg_latency_ms")
        qa_models.append({
            "model": name, "key": key, "params_M": params,
            "em_gold": round(gold.get("Exact Match", {}).get(name), 2) if gold else None,
            "f1_gold": round(gold.get("F1", {}).get(name), 2) if gold else None,
            "em_e2e": round(e2e.get("exact_match", {}).get(name), 2) if e2e else None,
            "f1_e2e": round(e2e.get("f1", {}).get(name), 2) if e2e else None,
            "latency_ms": lat,
            "train_runtime_h": facts.get(key, {}).get("train_runtime_h"),
            "pending": False,
        })

    best_row = max(qa_models, key=lambda m: m["f1_gold"] or 0)
    complete = next((r for r in retrieval_chart if r["name"] == "Complete Pipeline"), None)
    best_recall = max((r["Recall@10"] for r in retrieval_chart), default=None)

    training_log = (RESULTS / "training_log.md").read_text(encoding="utf-8") if (RESULTS / "training_log.md").exists() else ""
    gpu = detailed.get("device") or "NVIDIA GeForce RTX 5060 Laptop GPU"

    return {
        "_meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu": gpu,
            "pytorch": detailed.get("pytorch"),
            "cuda": detailed.get("cuda"),
            "smoke_test": "SMOKE_TEST:** False" not in training_log,
            "run_date": time.strftime("%Y-%m-%d %H:%M"),
            "source": "results/ (notebook + full fine-tuning run)",
        },
        "hero": {
            "best_em_gold": best_row["em_gold"],
            "best_f1_gold": best_row["f1_gold"],
            "best_em_e2e": best_row["em_e2e"],
            "best_f1_e2e": best_row["f1_e2e"],
            "retrieval_recall": complete["Recall@10"] if complete else best_recall,
            "retrieval_recall_best": best_recall,
            "retrieval_mrr": complete["MRR"] if complete else None,
            "corpus_passages": stats["passages"],
            "train_examples": 87599,
            "dev_examples": 10570,
            "num_models": len(qa_models),
        },
        "dataset": stats,
        "retrieval_chart": retrieval_chart,
        "qa_models": qa_models,
        "convergence": curves,
        "training_facts": facts,
        "qualitative": qualitative(),
        "detailed": detailed,
        "demo": demo,
        "training_config": {
            "smoke_test": False,
            "train_examples": 87599,
            "eval_examples": 10570,
            "epochs": 3,
            "effective_batch": 32,
            "gpu": gpu,
            "run_date": time.strftime("%Y-%m-%d %H:%M"),
        },
    }


def sync_artifacts() -> None:
    """Refresh the downloadable copies served by the dashboard."""
    dest = DASH / "artifacts"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["retrieval_progressive.csv", "retrieval_progressive.png",
                 "qa_model_comparison.csv", "qa_model_comparison.png",
                 "e2e_evaluation.csv", "qa_sample_predictions.csv",
                 "summary.json", "detailed_eval.json", "demo_bundle.json",
                 "training_log.md"]:
        src = RESULTS / name
        if src.exists():
            shutil.copy2(src, dest / name)
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if notebooks:
        shutil.copy2(notebooks[0], dest / "Adaptive_Hybrid_Semantic_Research_Assistant.ipynb")
    print(f"[artifacts] synced -> {dest.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun-inference", action="store_true",
                    help="re-run GPU inference to regenerate detailed_eval.json + demo_bundle.json")
    args = ap.parse_args()

    if args.rerun_inference:
        rerun_inference()

    data = build()
    out = DASH / "data" / "project_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(out, data)
    sync_artifacts()

    h = data["hero"]
    print(f"[dashboard] {out.relative_to(ROOT)}")
    print(f"            best EM/F1 (gold) {h['best_em_gold']}/{h['best_f1_gold']}  "
          f"E2E {h['best_em_e2e']}/{h['best_f1_e2e']}  "
          f"Recall@10 {h['retrieval_recall']}  corpus {h['corpus_passages']}")
    print(f"            {len(data['demo'])} precomputed demo questions, "
          f"{len(data['retrieval_chart'])} ablation rows, {len(data['qa_models'])} models")


if __name__ == "__main__":
    main()
