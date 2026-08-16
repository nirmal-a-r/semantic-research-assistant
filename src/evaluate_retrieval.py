import math
import numpy as np


def dcg_at_k(rel, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))


def ndcg_at_k(ranked, relevant, k=10):
    rels = [1 if d in relevant else 0 for d in ranked]
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg_at_k(rels, k) / idcg if idcg else 0.0


def mrr(ranked, relevant):
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked, relevant, k):
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0
