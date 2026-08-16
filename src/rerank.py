from sentence_transformers import CrossEncoder


class RerankingRetriever:
    def __init__(self, base, ce_name, doc_lookup, candidate_k=30, device="cpu"):
        self.base = base
        self.ce = CrossEncoder(ce_name, device=device)
        self.doc_lookup = doc_lookup
        self.candidate_k = candidate_k

    def search(self, query, top_k=10):
        cands = self.base.search(query, top_k=self.candidate_k)
        pairs = [(query, self.doc_lookup[d]) for d, _ in cands]
        scores = self.ce.predict(pairs)
        ranked = sorted(zip([c[0] for c in cands], scores),
                        key=lambda x: x[1], reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_k]]
