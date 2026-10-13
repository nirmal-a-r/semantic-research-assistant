"""End-to-end evaluation: retrieval feeds the reader, answers are scored with EM / F1.

Compares reading only the top-ranked chunk (the published setting) with reading the
top-K chunks and keeping the most confident span, which is what makes the
complexity-based K affect the answer.

Usage:
    python scripts/eval_end_to_end.py --checkpoint models/roberta-base-squad1 --n 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import semantic_adaptive_chunk
from src.data import build_corpus, load_squad
from src.fusion import reciprocal_rank_fusion
from src.multiquery import estimate_query_complexity, generate_query_variants
from src.qa_decode import answer_over_candidates, best_span
from src.retrieval_sparse import BM25Retriever

COMPLEXITY_TOP_K = {"simple": 5, "comparison": 10, "multistep": 15}
CANDIDATES = 50
RERANK_K = 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/roberta-base-squad1")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--chunked", action="store_true", default=True)
    ap.add_argument("--max-read", type=int, default=5, help="cap on chunks read per question")
    ap.add_argument("--k-values", default="", help="comma-separated fixed K values to sweep "
                                                  "instead of the complexity-based K")
    ap.add_argument("--candidates", type=int, default=CANDIDATES)
    ap.add_argument("--rerank-k", type=int, default=RERANK_K)
    ap.add_argument("--out", default="results/end_to_end_topk.json")
    args = ap.parse_args()
    k_sweep = [int(k) for k in args.k_values.split(",") if k.strip()]

    import evaluate as hf_evaluate
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from src.retrieval_dense import DenseRetriever, build_faiss_index

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {device}")

    train, dev = load_squad()
    corpus, ctx2docid = build_corpus(train, dev)
    docs, chunk2doc = [], {}
    for d in corpus:
        for i, text in enumerate(semantic_adaptive_chunk(d["text"])):
            cid = f"{d['doc_id']}_c{i}"
            docs.append({"doc_id": cid, "text": text})
            chunk2doc[cid] = d["doc_id"]
    lookup = {d["doc_id"]: d["text"] for d in docs}
    log(f"{len(corpus)} passages -> {len(docs)} chunks")

    bm25 = BM25Retriever(docs)
    encoder = SentenceTransformer(args.encoder, device=device)
    index, doc_ids = build_faiss_index(docs, encoder)
    dense = DenseRetriever(index, doc_ids, encoder)
    cross = CrossEncoder(args.cross_encoder, device=device)
    log(f"indexes ready ({index.ntotal} vectors)")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint).to(device).eval()
    log(f"reader loaded from {args.checkpoint}")

    examples = list(dev.select(range(min(args.n, len(dev)))))
    preds_rank1, preds_topk, refs, rows = [], [], [], []
    preds_by_k = {}
    t0 = time.time()

    for i, ex in enumerate(examples, 1):
        question = ex["question"]
        complexity = estimate_query_complexity(question)
        k = min(COMPLEXITY_TOP_K[complexity], args.max_read)
        variants = generate_query_variants(question, n=3)

        rankings = []
        for v in variants:
            rankings.append(bm25.search(v, top_k=args.candidates))
            rankings.append(dense.search(v, top_k=args.candidates))
        fused = reciprocal_rank_fusion(rankings, top_k=args.rerank_k)
        if not fused:
            continue
        pairs = [(question, lookup[d]) for d, _ in fused]
        scores = cross.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip([f[0] for f in fused], scores), key=lambda x: x[1], reverse=True)

        max_needed = max(k_sweep) if k_sweep else k
        top_chunks = [(d, lookup[d]) for d, _ in ranked[:max_needed]]
        rank1 = best_span(model, tokenizer, question, top_chunks[0][1], device=device)
        topk = answer_over_candidates(model, tokenizer, question, top_chunks[:k], device=device)

        for kk in k_sweep:
            ans = answer_over_candidates(model, tokenizer, question, top_chunks[:kk], device=device)
            preds_by_k.setdefault(kk, []).append({"id": ex["id"], "prediction_text": ans["text"]})

        preds_rank1.append({"id": ex["id"], "prediction_text": rank1["text"]})
        preds_topk.append({"id": ex["id"], "prediction_text": topk["text"]})
        refs.append({"id": ex["id"], "answers": ex["answers"]})
        rows.append({
            "question": question, "complexity": complexity, "k": k,
            "gold": ex["answers"]["text"][0],
            "rank1_answer": rank1["text"], "topk_answer": topk["text"],
            "topk_chosen_rank": topk.get("chosen_rank"),
            "retrieved_gold_rank": next((j + 1 for j, (d, _) in enumerate(ranked)
                                         if chunk2doc.get(d) == ctx2docid.get(ex["context"])), None),
        })
        if i % 25 == 0:
            log(f"  {i}/{len(examples)}")

    squad = hf_evaluate.load("squad")
    m_rank1 = squad.compute(predictions=preds_rank1, references=refs)
    m_topk = squad.compute(predictions=preds_topk, references=refs)
    elapsed = time.time() - t0

    log(f"rank-1 reading : EM {m_rank1['exact_match']:.2f}  F1 {m_rank1['f1']:.2f}")
    log(f"top-K reading  : EM {m_topk['exact_match']:.2f}  F1 {m_topk['f1']:.2f}")
    k_curve = {}
    for kk, preds in sorted(preds_by_k.items()):
        m = squad.compute(predictions=preds, references=refs)
        k_curve[kk] = m
        log(f"  K={kk:<3d}      : EM {m['exact_match']:.2f}  F1 {m['f1']:.2f}")
    log(f"{elapsed/len(examples):.2f} s per question")

    record = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": args.checkpoint, "n_questions": len(refs),
        "encoder": args.encoder, "cross_encoder": args.cross_encoder,
        "max_read": args.max_read,
        "rank1": m_rank1, "topk": m_topk, "k_curve": k_curve,
        "candidates": args.candidates, "rerank_k": args.rerank_k,
        "mean_k": float(np.mean([r["k"] for r in rows])),
        "changed_by_topk": sum(1 for r in rows if r["rank1_answer"] != r["topk_answer"]),
        "rows": rows,
    }
    (ROOT / args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"wrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
