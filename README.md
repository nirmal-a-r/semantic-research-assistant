# Adaptive Hybrid Semantic Research Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.8-EE4C2C?logo=pytorch)](https://pytorch.org)
[![SQuAD v1.1](https://img.shields.io/badge/Dataset-SQuAD_v1.1-orange)](https://rajpurkar.github.io/SQuAD-explorer/)

> **Retrieval-first, evidence-grounded, explainable question answering over SQuAD v1.1**

### 🔗 Live dashboard — https://nirmal-a-r.github.io/semantic-research-assistant/

Fully static: every chart, table and demo answer is precomputed from the real GPU run,
so the site works with no backend running.

**Course:** 23AID472 Text Analytics | Batch A-17  
**Team:** A.R. Nirmal · Gunnala Dheeraj Kumar · Meera S Raj · Vishnu Vardhan N

---

## Problem Statement

Modern research suffers from information overload. Keyword search (BM25 alone) misses paraphrase-level matches; dense semantic search misses exact technical terms; general LLMs hallucinate without traceable evidence. This project builds a hybrid retrieval-augmented QA system that is:

- **Accurate** — extracts exact answer spans directly from retrieved passages
- **Grounded** — every answer is traceable to a specific Wikipedia passage
- **Explainable** — retrieval scores, span confidence, and token-level F1 are returned with each answer
- **Adaptive** — query complexity automatically determines retrieval depth

---

## Architecture

```
User Query
   ↓
Multi-Query Generation (query variants for recall boost)
   ↓
┌─────────────────┐     ┌──────────────────────────┐
│  BM25 Sparse    │     │  SBERT + FAISS Dense     │
│  (BM25Okapi)    │     │  (all-MiniLM-L6-v2)      │
└────────┬────────┘     └────────────┬─────────────┘
         └──────────────┬────────────┘
                        ↓
          Reciprocal Rank Fusion (RRF, k=60)
                        ↓
         Cross-Encoder Re-ranking (ms-marco-MiniLM)
                        ↓
           Dynamic Top-K (simple K=5, comparison K=10, multistep K=15)
                        ↓
         Extractive QA Model (RoBERTa-base [primary] / BERT / DistilBERT)
                        ↓
         Answer Span + Evidence Passage + Confidence + Latency
```

---

## Results

All numbers below are produced by the notebook / training run and live in `results/`.
`scripts/build_dashboard_data.py` is the only path from those files to the dashboard, so
the site can never drift from the artefacts.

### Progressive Retrieval Ablation (7 configurations, 2,000-question eval set)

| Configuration | Recall@10 | Recall@20 | MRR | NDCG@10 |
|---|---|---|---|---|
| BM25 only | 86.15% | 89.60% | 69.49% | 73.39% |
| Dense only | 86.00% | 90.55% | 63.83% | 68.97% |
| Hybrid (RRF) | 91.50% | 94.60% | 71.70% | 76.37% |
| + Re-ranking | **96.25%** | 96.65% | **85.25%** | **87.97%** |
| + Adaptive Chunking | 91.35% | 94.40% | 71.56% | 75.64% |
| + Multi-Query+RRF | **96.25%** | **96.70%** | 85.19% | 87.91% |
| **Complete Pipeline** | 95.95% | 96.45% | 85.20% | 87.31% |

Cross-encoder re-ranking is the single largest contributor: +4.75 Recall@10 and +13.55 MRR
over plain hybrid fusion.

### QA Model Comparison (SQuAD v1.1, fine-tuned 3 epochs on all 87,599 train examples)

| Model | Params | EM (Gold) | F1 (Gold) | EM (E2E) | F1 (E2E) | Latency | Train time |
|---|---|---|---|---|---|---|---|
| **RoBERTa-base ★** | 125M | **85.38%** | **91.74%** | **71.00%** | **72.34%** | 8.3 ms | 2.45 h |
| BERT-base-uncased | 110M | 80.18% | 87.85% | 62.00% | 67.54% | 7.9 ms | 2.33 h |
| DistilBERT-base-uncased | 66M | 76.98% | 85.52% | 60.00% | 62.41% | 4.3 ms | 1.18 h |

> ★ Primary model. Gold = oracle context supplied, full 10,570-question dev set.
> E2E = full retrieval pipeline supplies the context (200-question sample).
> Latency = mean single-question span extraction, measured on the GPU below.
> GPU: NVIDIA GeForce RTX 5060 Laptop (8.5 GB VRAM, sm_120) · PyTorch 2.12.0-nightly+cu128

The ~19-point gold→E2E gap is retrieval-bound, not model-bound: when the retriever returns the
wrong passage the answer span cannot exist in it, so the ceiling is Recall@1, not Recall@10.

---

---

## Tech Stack

| Library | Role in this pipeline |
|---|---|
| `transformers` | Fine-tune + infer with RoBERTa/BERT/DistilBERT for extractive QA |
| `sentence-transformers` | Encode all 20,958 passages with all-MiniLM-L6-v2 for dense retrieval |
| `faiss-cpu` | Exact inner-product kNN search over dense embeddings |
| `rank_bm25` | Sparse BM25Okapi term-frequency retrieval |
| `sentence-transformers` (CrossEncoder) | Re-score top-30 candidates as (query, passage) pairs |
| `datasets` | Load SQuAD v1.1 train/dev splits with offset mapping |
| `evaluate` | Compute official SQuAD EM+F1 metric |
| `FastAPI` | Serve complete inference pipeline at `/answer` |
| `Next.js 16` | Dashboard frontend with all result charts |
| `Recharts` | Animated bar, radar, and scatter charts |
| `Framer Motion` | Section transitions and count-up animations |

---

## Repository Structure

```
semantic-research-assistant/
├── notebooks/
│   └── Adaptive_Hybrid_Semantic_Research_Assistant.ipynb  # Main notebook
├── src/                       # Modular Python pipeline
│   ├── pipeline.py            # end-to-end answer_query()
│   ├── retrieval_sparse.py    # BM25Retriever
│   ├── retrieval_dense.py     # DenseRetriever (FAISS)
│   ├── fusion.py              # reciprocal_rank_fusion()
│   ├── rerank.py              # RerankingRetriever (CrossEncoder)
│   ├── multiquery.py          # estimate_query_complexity(), generate_query_variants()
│   ├── chunking.py            # semantic_adaptive_chunk()
│   ├── qa_infer.py            # qa_predict_single()
│   └── qa_train.py            # fine-tuning wrapper
├── scripts/
│   └── build_dashboard_data.py     # results/ -> dashboard JSON (+ optional GPU re-run)
├── app/
│   ├── api.py                 # FastAPI backend (wired pipeline)
│   └── dashboard/             # Next.js 16 dashboard
│       ├── src/app/           # App Router pages
│       ├── src/components/    # Hero, Charts, LiveDemo, etc.
│       ├── src/lib/           # data.ts (metrics) + api.ts (client)
│       └── public/
│           ├── data/project_data.json   # every number the site renders
│           └── artifacts/               # downloadable copies of results/
├── results/
│   ├── retrieval_progressive.csv   # 7-config ablation
│   ├── qa_model_comparison.csv     # EM/F1 per model (full dev set)
│   ├── e2e_evaluation.csv          # end-to-end EM/F1
│   ├── qa_sample_predictions.csv   # side-by-side model predictions
│   ├── *_log_history.json          # HuggingFace training curves per model
│   ├── detailed_eval.json          # per-prediction GPU results
│   ├── demo_bundle.json            # precomputed pipeline answers (offline demo)
│   ├── training_log.md             # run metadata
│   └── summary.json                # machine-readable consolidation
├── data/processed/
│   ├── corpus.jsonl                # 20,958 retrieval passages
│   └── chunked_corpus.jsonl        # 23,288 adaptive chunks
├── models/                     # Fine-tuned checkpoints (gitignored, ~7.9 GB)
├── requirements.txt
└── README.md
```

---

## Dashboard Data Flow

```
notebook / training run
        ↓
results/*.csv  results/*.json          ← source of truth, committed
        ↓  scripts/build_dashboard_data.py
app/dashboard/public/data/project_data.json
app/dashboard/public/artifacts/*        ← downloadable copies
        ↓  fetch() at runtime
Hero · Ablation · QA Comparison · Deep Dive · Live Demo
```

No metric is hardcoded in a component. After any re-run:

```bash
python scripts/build_dashboard_data.py            # refresh the site from results/
python scripts/build_dashboard_data.py --rerun-inference   # + re-run GPU inference first
```

`--rerun-inference` reloads the three fine-tuned checkpoints, rebuilds BM25 + FAISS over the
corpus and regenerates `results/detailed_eval.json` (per-prediction EM/F1/confidence/latency)
and `results/demo_bundle.json` (the precomputed answers the deployed Live Demo replays).

---

## Setup & Run

### 1. Python environment

```bash
# Clone
git clone https://github.com/nirmal-a-r/semantic-research-assistant
cd semantic-research-assistant

# Create venv (Python 3.10+)
python -m venv nlpenv
.\nlpenv\Scripts\activate     # Windows
source nlpenv/bin/activate    # Linux/Mac

# Install PyTorch with CUDA 12.8 (RTX 50-series / sm_120)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Run the notebook (optional — results already in results/)

```bash
jupyter notebook notebooks/Adaptive_Hybrid_Semantic_Research_Assistant.ipynb
# Set SMOKE_TEST = False in the CONFIG cell for full-scale run
```

### 3. Start the FastAPI backend

```bash
# From repo root, with nlpenv activated:
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# Startup loads BM25 + FAISS + RoBERTa on GPU (~30s first time)
```

### 4. Start the dashboard

```bash
cd app/dashboard
npm install
npm run dev
# Open http://localhost:3000
```

With the backend up, the Live Demo runs real inference over all 20,958 passages.
With it down, the dashboard automatically falls back to precomputed mode — every chart
and 24 fully-worked pipeline answers still render, because they are baked into
`public/data/project_data.json`.

### 5. Refresh the dashboard after a re-run

```bash
python scripts/build_dashboard_data.py
```

Optionally re-run GPU inference (needs the checkpoints in `models/`) before rebuilding:

```bash
python scripts/build_dashboard_data.py --rerun-inference
```

---

## Deployment (permanent URL, no backend)

The dashboard is exported as a fully static site and hosted on GitHub Pages:

**https://nirmal-a-r.github.io/semantic-research-assistant/**

Every push to `main` triggers `.github/workflows/deploy.yml`, which builds the static export
and publishes it — so the live site always matches the committed `results/`.

Build it locally the same way CI does:

```bash
cd app/dashboard
STATIC_EXPORT=true BASE_PATH=/semantic-research-assistant npm run build:static
# -> app/dashboard/out/   (open out/index.html through any static server)
```

The static build has no server component: all metrics, charts, artefacts and demo answers
are read from files in `public/`. The optional FastAPI backend is only needed for free-text
questions during local development.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Backend status, loaded models, corpus size |
| `/answer` | POST | Full pipeline inference — returns answer, evidence, confidence, latency |
| `/results` | GET | Precomputed evaluation results from results/ directory |

**Example request:**
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Who invented the telephone?", "model": "roberta-base"}'
```

---

## License

MIT — see [LICENSE](LICENSE). Dataset: SQuAD v1.1 (CC BY-SA 4.0, Stanford NLP Group).
