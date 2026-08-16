import re
import numpy as np
from rank_bm25 import BM25Okapi


def simple_tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.doc_ids = [d["doc_id"] for d in docs]
        self.bm25 = BM25Okapi([simple_tokenize(d["text"]) for d in docs])

    def search(self, query, top_k=10):
        scores = self.bm25.get_scores(simple_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]
