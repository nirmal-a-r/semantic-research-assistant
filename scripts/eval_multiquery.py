"""Controlled comparison of rule-based and transformer multi-query expansion.

Runs the same hybrid + re-ranking pipeline three ways on the same questions:
  none  - single query (reproduces the "+ Re-ranking" ablation row)
  rules - rule-based variants from src/multiquery.py
  t5    - generated variants from src/multiquery_t5.py

Usage:
    python scripts/eval_multiquery.py --n 2000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import build_corpus, load_squad
from src.evaluate_retrieval import mrr, ndcg_at_k, recall_at_k
from src.fusion import reciprocal_rank_fusion
from src.multiquery import generate_query_variants
from src.retrieval_dense import DenseRetriever, build_faiss_index
from src.retrieval_sparse import BM25Retriever

CANDIDATE_K = 50
RERANK_K = 30
EVAL_TOP_K = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def search(query, variants, bm25, dense, cross_encoder, doc_lookup, top_k=EVAL_TOP_K):
    rankings = []
    for v in variants:
        rankings.append(bm25.search(v, top_k=CANDIDATE_K))
        rankings.append(dense.search(v, top_k=CANDIDATE_K))
    fused = reciprocal_rank_fusion(rankings, top_k=RERANK_K)
    if not fused:
        return []
    pairs = [(query, doc_lookup[d]) for d, _ in fused]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip([f[0] for f in fused], scores), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_k]]


def evaluate(name, variants_per_query, eval_pairs, bm25, dense, cross_encoder, doc_lookup):
    r10, r20, mrr_s, ndcg_s = [], [], [], []
    t0 = time.time()
    for i, ((question, gold), variants) in enumerate(zip(eval_pairs, variants_per_query), 1):
        hits = search(question, variants, bm25, dense, cross_encoder, doc_lookup)
        ranked = [h[0] for h in hits]
        relevant = {gold}
        r10.append(recall_at_k(ranked, relevant, 10))
        r20.append(recall_at_k(ranked, relevant, 20))
        mrr_s.append(mrr(ranked, relevant))
        ndcg_s.append(ndcg_at_k(ranked, relevant, 10))
        if i % 250 == 0:
            log(f"  {name}: {i}/{len(eval_pairs)} questions")
    elapsed = time.time() - t0
    return {
        "Recall@10": 100 * float(np.mean(r10)),
        "Recall@20": 100 * float(np.mean(r20)),
        "MRR": 100 * float(np.mean(mrr_s)),
        "NDCG@10": 100 * float(np.mean(ndcg_s)),
        "retrieval_ms_per_question": 1000 * elapsed / len(eval_pairs),
        "retrieval_total_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of dev questions")
    ap.add_argument("--variants", type=int, default=3, help="max query variants per question")
    ap.add_argument("--t5-models", default="google/flan-t5-small",
                    help="comma-separated transformer expanders to compare")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--out", default="results/multiquery_comparison.csv")
    ap.add_argument("--log", default="results/multiquery_run.json")
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    log("loading SQuAD v1.1")
    train, dev = load_squad()
    corpus, ctx2docid = build_corpus(train, dev)
    doc_lookup = {d["doc_id"]: d["text"] for d in corpus}
    log(f"corpus: {len(corpus)} passages")

    eval_pairs = []
    for ex in dev.select(range(min(args.n, len(dev)))):
        gold = ctx2docid.get(ex["context"])
        if gold:
            eval_pairs.append((ex["question"], gold))
    questions = [q for q, _ in eval_pairs]
    log(f"evaluation set: {len(eval_pairs)} questions")

    log("building BM25 index")
    bm25 = BM25Retriever(corpus)
    log("encoding passages")
    encoder = SentenceTransformer(args.encoder, device=device)
    index, doc_ids = build_faiss_index(corpus, encoder)
    dense = DenseRetriever(index, doc_ids, encoder)
    log(f"FAISS index: {index.ntotal} vectors, dim={index.d}")
    cross_encoder = CrossEncoder(args.cross_encoder, device=device)

    log("generating rule-based variants")
    t0 = time.time()
    rule_variants = [generate_query_variants(q, n=args.variants) for q in questions]
    rule_gen_ms = 1000 * (time.time() - t0) / len(questions)

    from src.multiquery_t5 import T5QueryExpander

    runs = {
        "+ Re-ranking (no multi-query)": ([[q] for q in questions], 0.0),
        "+ Multi-query (rules)": (rule_variants, rule_gen_ms),
    }
    t5_variant_examples = {}
    for model_name in [m.strip() for m in args.t5_models.split(",") if m.strip()]:
        log(f"generating transformer variants with {model_name}")
        expander = T5QueryExpander(model_name, device=device)
        t0 = time.time()
        variants = expander.expand(questions, n=args.variants)
        gen_ms = 1000 * (time.time() - t0) / len(questions)
        kept = sum(len(v) - 1 for v in variants)
        log(f"  {model_name}: {gen_ms:.1f} ms/question, kept {kept}/{len(questions)*(args.variants-1)} variants")
        runs[f"+ Multi-query ({model_name.split('/')[-1]})"] = (variants, gen_ms)
        t5_variant_examples[model_name] = variants
        del expander

    rows = {}
    for name, (variants, gen_ms) in runs.items():
        log(f"evaluating: {name}")
        metrics = evaluate(name, variants, eval_pairs, bm25, dense, cross_encoder, doc_lookup)
        metrics["variant_gen_ms_per_question"] = gen_ms
        metrics["mean_variants"] = float(np.mean([len(v) for v in variants]))
        metrics["ms_per_question"] = metrics["retrieval_ms_per_question"] + gen_ms
        rows[name] = metrics
        log(f"  {name}: R@10 {metrics['Recall@10']:.2f}  R@20 {metrics['Recall@20']:.2f}  "
            f"MRR {metrics['MRR']:.2f}  NDCG@10 {metrics['NDCG@10']:.2f}")

    df = pd.DataFrame(rows).T
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    log(f"wrote {out}")

    record = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "n_questions": len(eval_pairs),
        "corpus_passages": len(corpus),
        "config": {
            "candidate_k": CANDIDATE_K, "rerank_k": RERANK_K, "eval_top_k": EVAL_TOP_K,
            "max_variants": args.variants, "encoder": args.encoder,
            "cross_encoder": args.cross_encoder, "t5_models": args.t5_models,
        },
        "results": rows,
        "variant_examples": [
            {"question": questions[i], "rules": rule_variants[i],
             **{name.split("/")[-1]: v[i] for name, v in t5_variant_examples.items()}}
            for i in range(min(10, len(questions)))
        ],
    }
    (ROOT / args.log).write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"wrote {ROOT / args.log}")
    print(df.to_string())


if __name__ == "__main__":
    main()
