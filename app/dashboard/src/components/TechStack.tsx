"use client";
import { motion } from "framer-motion";

const STACK = [
  {
    name: "HuggingFace Transformers",
    badge: "🤗",
    role: "Fine-tunes RoBERTa-base, BERT-base, DistilBERT-base for extractive span extraction on SQuAD v1.1. Also hosts the squad evaluation metric.",
    color: "#F59E0B",
    link: "https://github.com/huggingface/transformers",
  },
  {
    name: "Sentence-Transformers",
    badge: "🔡",
    role: "Encodes all 20,958 corpus passages into dense embeddings using all-MiniLM-L6-v2 for semantic similarity search.",
    color: "#22D3EE",
    link: "https://www.sbert.net",
  },
  {
    name: "FAISS (IndexFlatIP)",
    badge: "⚡",
    role: "Exact inner-product search over dense embedding matrix. Enables sub-millisecond kNN retrieval over the full corpus.",
    color: "#34D399",
    link: "https://github.com/facebookresearch/faiss",
  },
  {
    name: "rank-bm25 (BM25Okapi)",
    badge: "≋",
    role: "Sparse term-frequency retrieval as complement to dense search. BM25 excels at exact keyword matching that SBERT may miss.",
    color: "#8B5CF6",
    link: "https://github.com/dorianbrown/rank_bm25",
  },
  {
    name: "Cross-Encoder (ms-marco-MiniLM)",
    badge: "↕",
    role: "Re-scores top-30 (query, passage) pairs jointly for precision. Provides a final ranking signal more accurate than bi-encoder retrieval alone.",
    color: "#22D3EE",
    link: "https://www.sbert.net/docs/pretrained_cross-encoders.html",
  },
  {
    name: "PyTorch",
    badge: "🔥",
    role: "Tensor computation, GPU acceleration, and gradient computation for all three QA fine-tuning runs (3 epochs each).",
    color: "#EF4444",
    link: "https://pytorch.org",
  },
  {
    name: "HuggingFace Datasets",
    badge: "📚",
    role: "Loads SQuAD v1.1 (train/dev splits) via datasets.load_dataset('squad') and handles batched tokenisation with offset mapping.",
    color: "#F59E0B",
    link: "https://github.com/huggingface/datasets",
  },
  {
    name: "FastAPI",
    badge: "⚡",
    role: "Serves the complete inference pipeline at /answer — query → multi-query → hybrid retrieval → re-rank → QA extraction.",
    color: "#22D3EE",
    link: "https://fastapi.tiangolo.com",
  },
  {
    name: "HuggingFace Evaluate",
    badge: "📊",
    role: "Computes squad metric (Exact Match + F1) for both gold-passage and end-to-end evaluation settings.",
    color: "#34D399",
    link: "https://github.com/huggingface/evaluate",
  },
];

export default function TechStack() {
  return (
    <section id="tech-stack" className="py-20 px-4" style={{ background: "rgba(15,20,32,0.5)" }}>
      <div className="max-w-5xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--accent-cyan)" }}>
            Technology Stack
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-3" style={{ color: "var(--text-primary)" }}>
            Libraries Powering the Pipeline
          </h2>
          <p className="text-base mb-10 max-w-2xl" style={{ color: "var(--text-secondary)" }}>
            Every library shown plays a specific, non-generic role in this pipeline — not just listed for breadth.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {STACK.map((s, i) => (
            <motion.a key={s.name} href={s.link} target="_blank" rel="noopener noreferrer"
              initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.4, delay: i * 0.06 }}
              className="glass-card rounded-xl p-5 flex flex-col gap-3 no-underline group transition-all duration-300"
              style={{ borderColor: "var(--border-subtle)" }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = s.color + "60";
                e.currentTarget.style.boxShadow = `0 0 20px ${s.color}15`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--border-subtle)";
                e.currentTarget.style.boxShadow = "";
              }}>
              <div className="flex items-center gap-3">
                <span className="text-xl">{s.badge}</span>
                <span className="font-semibold text-sm" style={{ color: s.color }}>{s.name}</span>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>{s.role}</p>
              <span className="text-xs font-mono opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ color: s.color }}>
                ↗ {new URL(s.link).hostname}
              </span>
            </motion.a>
          ))}
        </div>
      </div>
    </section>
  );
}
