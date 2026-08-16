from src.qa_infer import qa_predict_single


def answer_query(query, retriever, model, tokenizer, doc_lookup, device="cpu", top_k=3):
    """End-to-end: retrieve -> extract answer span with evidence."""
    hits = retriever.search(query, top_k=top_k)
    if not hits:
        return {"answer": None, "evidence": None}
    top_doc_id, top_score = hits[0]
    ctx = doc_lookup[top_doc_id]
    return {
        "answer": qa_predict_single(model, tokenizer, query, ctx, device=device),
        "evidence_doc_id": top_doc_id,
        "evidence_score": top_score,
        "evidence_text": ctx,
        "all_hits": hits,
    }
