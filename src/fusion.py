from collections import defaultdict


def reciprocal_rank_fusion(rankings, k=60, top_k=10):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            fused[doc_id] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
