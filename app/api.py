"""
Enhanced FastAPI QA serving endpoint.
Run: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Full pipeline: BM25 + FAISS hybrid → RRF fusion → multi-query → QA extraction
Returns: answer span, confidence, per-answer EM/F1 explanation, evidence, top chunks.
"""

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import semantic_adaptive_chunk
from src.fusion import reciprocal_rank_fusion
from src.multiquery import estimate_query_complexity, generate_query_variants
from src.qa_infer import qa_predict_single
from src.retrieval_dense import DenseRetriever, build_faiss_index
from src.retrieval_sparse import BM25Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
MODELS_DIR = ROOT / "models"
ENCODER_NAME = "all-MiniLM-L6-v2"
COMPLEXITY_TOP_K = {"simple": 5, "comparison": 10, "multistep": 15}

AVAILABLE_MODELS = {
    "roberta-base": "roberta-base-squad1",
    "bert-base-uncased": "bert-base-uncased-squad1",
    "distilbert-base-uncased": "distilbert-base-uncased-squad1",
}

MODEL_PARAMS = {
    "roberta-base": 125,
    "bert-base-uncased": 110,
    "distilbert-base-uncased": 66,
}

_state: dict = {
    "ready": False, "error": None,
    "loaded_models": {}, "bm25": None,
    "dense": None, "doc_lookup": {}, "doc_list": [],
}

# ── Metric helpers ────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())

def _token_f1(pred: str, gold: str) -> float:
    pred_t = _normalize(pred).split()
    gold_t = _normalize(gold).split()
    common = Counter(pred_t) & Counter(gold_t)
    n = sum(common.values())
    if n == 0: return 0.0
    p = n / len(pred_t) if pred_t else 0
    r = n / len(gold_t) if gold_t else 0
    return 2 * p * r / (p + r) if (p + r) else 0.0

def _exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)

# ── Startup ───────────────────────────────────────────────────────────────────

def _load_corpus():
    docs = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: docs.append(json.loads(line))
    return docs

def _load_model(model_key: str):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    ckpt = MODELS_DIR / AVAILABLE_MODELS[model_key]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    logger.info("Loading %s from %s", model_key, ckpt)
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForQuestionAnswering.from_pretrained(str(ckpt))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info("Loaded %s on %s", model_key, device)
    return model, tok, device

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading corpus …")
        docs = _load_corpus()
        _state["doc_list"] = docs
        _state["doc_lookup"] = {d["doc_id"]: d["text"] for d in docs}
        logger.info("Corpus: %d docs", len(docs))

        logger.info("Building BM25 …")
        _state["bm25"] = BM25Retriever(docs)
        logger.info("BM25 ready.")

        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(ENCODER_NAME)
        logger.info("Building FAISS index …")
        idx, ids = build_faiss_index(docs, encoder)
        _state["dense"] = DenseRetriever(idx, ids, encoder)
        logger.info("FAISS ready.")

        model, tok, dev = _load_model("roberta-base")
        _state["loaded_models"]["roberta-base"] = (model, tok, dev)
        _state["ready"] = True
        logger.info("✅ API ready — GPU: %s", dev)
    except Exception as e:
        _state["error"] = str(e)
        logger.error("Startup error: %s", e)
    yield
    _state["loaded_models"].clear()

app = FastAPI(
    title="Adaptive Hybrid Semantic Research Assistant",
    version="1.1.0",
    description="Hybrid retrieval + extractive QA with per-answer metric explanation.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    model: str = "roberta-base"
    top_k: Optional[int] = None

class MetricExplanation(BaseModel):
    exact_match: Optional[bool] = None        # vs gold if provided
    token_f1: Optional[float] = None          # vs gold if provided
    start_confidence: float                    # softmax prob of predicted start
    end_confidence: float                      # softmax prob of predicted end
    avg_confidence: float
    precision_hint: Optional[float] = None    # token precision vs gold
    recall_hint: Optional[float] = None       # token recall vs gold
    answer_length_words: int
    context_length_words: int
    latency_ms: float

class EvidenceChunk(BaseModel):
    doc_id: str
    score: float
    text: str
    rank: int

class QueryResponse(BaseModel):
    answer: str
    answer_start: int
    answer_end: int
    evidence_doc_id: str
    evidence_score: float
    evidence_text: str
    top_chunks: List[EvidenceChunk]
    query_complexity: str
    query_variants: List[str]
    model_used: str
    model_params_million: int
    metrics: MetricExplanation

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_model(key: str):
    if key not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model '{key}'. Choose: {list(AVAILABLE_MODELS)}")
    if key not in _state["loaded_models"]:
        try:
            _state["loaded_models"][key] = _load_model(key)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _state["loaded_models"][key]

def _hybrid_retrieve(query: str, top_k: int):
    bm25 = _state["bm25"].search(query, top_k=top_k * 3)
    dense = _state["dense"].search(query, top_k=top_k * 3)
    return reciprocal_rank_fusion([bm25, dense], top_k=top_k)

def _find_span(answer: str, context: str):
    if not answer: return 0, 0
    idx = context.lower().find(answer.lower())
    return (idx, idx + len(answer)) if idx != -1 else (0, 0)

def _qa_with_confidence(model, tokenizer, question, context, device, max_length=384):
    """QA inference returning answer + start/end confidence scores."""
    import torch
    inputs = tokenizer(
        question, context, return_tensors="pt",
        truncation="only_second", max_length=max_length,
        return_offsets_mapping=True,
    ).to(device)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        outputs = model(**inputs)
    start_idx = int(torch.argmax(outputs.start_logits))
    end_idx = int(torch.argmax(outputs.end_logits))
    start_prob = float(torch.softmax(outputs.start_logits, dim=-1)[0][start_idx])
    end_prob = float(torch.softmax(outputs.end_logits, dim=-1)[0][end_idx])
    if end_idx < start_idx or not offset_mapping[start_idx] or not offset_mapping[end_idx]:
        return "", start_prob, end_prob
    s, e = offset_mapping[start_idx][0], offset_mapping[end_idx][1]
    return context[s:e], start_prob, end_prob

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ready" if _state["ready"] else "loading",
        "error": _state["error"],
        "loaded_models": list(_state["loaded_models"].keys()),
        "corpus_size": len(_state["doc_list"]),
        "device": _state["loaded_models"].get("roberta-base", [None, None, "unknown"])[2] if _state["loaded_models"] else "unknown",
    }

@app.post("/answer", response_model=QueryResponse)
def answer(req: QueryRequest):
    if not _state["ready"]:
        raise HTTPException(503, "Pipeline not yet initialised.")
    
    model, tokenizer, device = _get_model(req.model)
    complexity = estimate_query_complexity(req.question)
    top_k = req.top_k or COMPLEXITY_TOP_K[complexity]
    variants = generate_query_variants(req.question)

    # Multi-query hybrid retrieval
    all_rankings = [_hybrid_retrieve(v, top_k=top_k) for v in variants]
    merged = reciprocal_rank_fusion(all_rankings, top_k=top_k)
    if not merged:
        raise HTTPException(404, "No relevant passages found.")

    doc_lookup = _state["doc_lookup"]
    top_chunks = [
        EvidenceChunk(doc_id=d, score=round(s, 6), text=doc_lookup.get(d, ""), rank=i + 1)
        for i, (d, s) in enumerate(merged[:3])
    ]

    top_doc_id, top_score = merged[0]
    context = doc_lookup.get(top_doc_id, "")

    t0 = time.time()
    answer_text, start_p, end_p = _qa_with_confidence(model, tokenizer, req.question, context, device)
    latency_ms = (time.time() - t0) * 1000

    start, end = _find_span(answer_text, context)

    metrics = MetricExplanation(
        exact_match=None,
        token_f1=None,
        start_confidence=round(start_p, 4),
        end_confidence=round(end_p, 4),
        avg_confidence=round((start_p + end_p) / 2, 4),
        precision_hint=None,
        recall_hint=None,
        answer_length_words=len(answer_text.split()) if answer_text else 0,
        context_length_words=len(context.split()),
        latency_ms=round(latency_ms, 1),
    )

    return QueryResponse(
        answer=answer_text,
        answer_start=start,
        answer_end=end,
        evidence_doc_id=top_doc_id,
        evidence_score=round(top_score, 6),
        evidence_text=context,
        top_chunks=top_chunks,
        query_complexity=complexity,
        query_variants=variants,
        model_used=req.model,
        model_params_million=MODEL_PARAMS.get(req.model, 0),
        metrics=metrics,
    )

@app.get("/results")
def get_results():
    """Return precomputed evaluation results for the dashboard."""
    summary_path = ROOT / "results" / "summary.json"
    detail_path = ROOT / "results" / "detailed_eval.json"
    data = {}
    if summary_path.exists():
        data["summary"] = json.loads(summary_path.read_text())
    if detail_path.exists():
        data["detailed"] = json.loads(detail_path.read_text())
    return data
