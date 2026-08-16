import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def build_faiss_index(docs, encoder, batch_size=64):
    texts = [d["text"] for d in docs]
    embs = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, [d["doc_id"] for d in docs]


class DenseRetriever:
    def __init__(self, index, doc_ids, encoder):
        self.index, self.doc_ids, self.encoder = index, doc_ids, encoder

    def search(self, query, top_k=10):
        q = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(q, top_k)
        return [(self.doc_ids[i], float(s))
                for s, i in zip(scores[0], idxs[0]) if i != -1]
