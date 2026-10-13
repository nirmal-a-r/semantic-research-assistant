"""Grid search over retrieval components, reported with the ablation metrics.

Varies the bi-encoder, the cross-encoder, candidate depth, re-rank depth, chunking
and query expansion. Results are appended to the output CSV as each configuration
finishes, so an interrupted sweep can be resumed.

Usage:
    python scripts/sweep_retrieval.py --stage screen --n 500
    python scripts/sweep_retrieval.py --stage confirm --n 2000
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

from src.chunking import semantic_adaptive_chunk
from src.data import build_corpus, load_squad
from src.evaluate_retrieval import mrr, ndcg_at_k, recall_at_k
from src.fusion import reciprocal_rank_fusion
from src.multiquery import generate_query_variants
from src.retrieval_sparse import BM25Retriever

# Instruction prefixes required by the retrieval models that were trained with them.
QUERY_PREFIX = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
}
PASSAGE_PREFIX = {
    "intfloat/e5-base-v2": "passage: ",
}

BASELINE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_CROSS = "cross-encoder/ms-marco-MiniLM-L6-v2"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class DensePassageIndex:
    """FAISS index that applies the model's own query/passage prefixes."""

    def __init__(self, encoder, model_name, docs, batch_size=128):
        import faiss
        self.encoder, self.model_name = encoder, model_name
        texts = [PASSAGE_PREFIX.get(model_name, "") + d["text"] for d in docs]
        emb = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.doc_ids = [d["doc_id"] for d in docs]
        self.dim = emb.shape[1]

    def search(self, query, top_k=10):
        q = self.encoder.encode([QUERY_PREFIX.get(self.model_name, "") + query],
                                convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, top_k)
        return [(self.doc_ids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]


def run_config(cfg, eval_pairs, bm25, dense, cross_encoder, doc_lookup, chunk2doc, variants_fn):
    r10, r20, mrr_s, ndcg_s = [], [], [], []
    t0 = time.time()
    for question, gold in eval_pairs:
        variants = variants_fn(question)
        rankings = []
        for v in variants:
            rankings.append(bm25.search(v, top_k=cfg["candidates"]))
            rankings.append(dense.search(v, top_k=cfg["candidates"]))
        fused = reciprocal_rank_fusion(rankings, top_k=cfg["rerank_k"])
        if cross_encoder is not None and fused:
            pairs = [(question, doc_lookup[d]) for d, _ in fused]
            scores = cross_encoder.predict(pairs, show_progress_bar=False)
            fused = sorted(zip([f[0] for f in fused], scores), key=lambda x: x[1], reverse=True)
        ranked = [d for d, _ in fused[:20]]
        if chunk2doc:
            ranked = [chunk2doc.get(d, d) for d in ranked]
            if cfg.get("dedup"):
                seen, unique = set(), []
                for d in ranked:
                    if d not in seen:
                        unique.append(d)
                        seen.add(d)
                ranked = unique
        relevant = {gold}
        r10.append(recall_at_k(ranked, relevant, 10))
        r20.append(recall_at_k(ranked, relevant, 20))
        mrr_s.append(mrr(ranked, relevant))
        ndcg_s.append(ndcg_at_k(ranked, relevant, 10))
    elapsed = time.time() - t0
    return {
        "Recall@10": 100 * float(np.mean(r10)),
        "Recall@20": 100 * float(np.mean(r20)),
        "MRR": 100 * float(np.mean(mrr_s)),
        "NDCG@10": 100 * float(np.mean(ndcg_s)),
        "ms_per_question": 1000 * elapsed / len(eval_pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "confirm"], default="screen")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--configs", default="results/sweep_confirm_configs.json",
                    help="config list used by the confirm stage")
    args = ap.parse_args()

    out_path = ROOT / (args.out or f"results/sweep_{args.stage}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path)["config"])
        log(f"resuming: {len(done)} configurations already done")

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {device}, {args.stage} stage, n={args.n}")

    train, dev = load_squad()
    corpus, ctx2docid = build_corpus(train, dev)
    eval_pairs = []
    for ex in dev.select(range(min(args.n, len(dev)))):
        gold = ctx2docid.get(ex["context"])
        if gold:
            eval_pairs.append((ex["question"], gold))
    log(f"{len(corpus)} passages, {len(eval_pairs)} questions")

    chunked, chunk2doc = [], {}
    for d in corpus:
        for i, text in enumerate(semantic_adaptive_chunk(d["text"])):
            cid = f"{d['doc_id']}_c{i}"
            chunked.append({"doc_id": cid, "text": text})
            chunk2doc[cid] = d["doc_id"]
    log(f"{len(chunked)} chunks")

    if args.stage == "screen":
        configs = [
            {"encoder": e, "cross": c, "candidates": 50, "rerank_k": 30, "chunked": False, "mq": False}
            for e in [BASELINE_ENCODER, "BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2"]
            for c in [BASELINE_CROSS, "cross-encoder/ms-marco-MiniLM-L12-v2"]
        ]
    else:
        configs = json.loads((ROOT / args.configs).read_text())

    corpora = {False: (corpus, None), True: (chunked, chunk2doc)}
    bm25_cache, dense_cache, cross_cache = {}, {}, {}
    rows = []

    for cfg in configs:
        name = (f"{cfg['encoder'].split('/')[-1]}|{cfg['cross'].split('/')[-1] if cfg['cross'] else 'none'}"
                f"|c{cfg['candidates']}|r{cfg['rerank_k']}|{'chunk' if cfg['chunked'] else 'passage'}"
                f"|{'mq' if cfg['mq'] else 'single'}"
                f"{'|dedup' if cfg.get('dedup') else ''}")
        if name in done:
            log(f"skip {name}")
            continue
        docs, c2d = corpora[cfg["chunked"]]
        lookup = {d["doc_id"]: d["text"] for d in docs}

        if cfg["chunked"] not in bm25_cache:
            log(f"building BM25 ({'chunks' if cfg['chunked'] else 'passages'})")
            bm25_cache[cfg["chunked"]] = BM25Retriever(docs)
        key = (cfg["encoder"], cfg["chunked"])
        if key not in dense_cache:
            log(f"encoding with {cfg['encoder']}")
            t0 = time.time()
            enc = SentenceTransformer(cfg["encoder"], device=device)
            dense_cache[key] = DensePassageIndex(enc, cfg["encoder"], docs)
            log(f"  indexed {dense_cache[key].index.ntotal} vectors (dim {dense_cache[key].dim}) in {time.time()-t0:.0f}s")
        if cfg["cross"] and cfg["cross"] not in cross_cache:
            cross_cache[cfg["cross"]] = CrossEncoder(cfg["cross"], device=device)

        variants_fn = (lambda q: generate_query_variants(q, n=3)) if cfg["mq"] else (lambda q: [q])
        log(f"running {name}")
        metrics = run_config(cfg, eval_pairs, bm25_cache[cfg["chunked"]], dense_cache[key],
                             cross_cache.get(cfg["cross"]), lookup, c2d, variants_fn)
        # fixed column set: configurations differ in which keys they set, and a
        # ragged append makes the CSV unparseable later
        row = {"config": name, "n_questions": len(eval_pairs),
               "encoder": cfg["encoder"], "cross": cfg["cross"] or "",
               "candidates": cfg["candidates"], "rerank_k": cfg["rerank_k"],
               "chunked": cfg["chunked"], "mq": cfg["mq"], "dedup": cfg.get("dedup", False),
               **{k: metrics[k] for k in ("Recall@10", "Recall@20", "MRR", "NDCG@10", "ms_per_question")}}
        rows.append(row)
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        metrics = row
        log(f"  {name}: R@10 {metrics['Recall@10']:.2f}  R@20 {metrics['Recall@20']:.2f}  "
            f"MRR {metrics['MRR']:.2f}  NDCG {metrics['NDCG@10']:.2f}  {metrics['ms_per_question']:.0f} ms/q")

    log(f"sweep complete: {len(rows)} new configurations -> {out_path}")


if __name__ == "__main__":
    main()
