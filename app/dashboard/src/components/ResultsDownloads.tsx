"use client";
import { motion } from "framer-motion";
import { asset } from "@/lib/data";

// Copies of results/ synced by scripts/build_dashboard_data.py.
const ARTIFACTS = [
  {
    name: "retrieval_progressive.csv",
    desc: "7-configuration progressive ablation — Recall@10/20, MRR, NDCG@10",
    file: "retrieval_progressive.csv",
    icon: "📄",
    size: "499 B",
    color: "#22D3EE",
  },
  {
    name: "retrieval_progressive.png",
    desc: "Bar chart visualization of the ablation study",
    file: "retrieval_progressive.png",
    icon: "🖼",
    size: "59 KB",
    color: "#22D3EE",
  },
  {
    name: "qa_model_comparison.csv",
    desc: "EM/F1 per fine-tuned QA backbone (gold passage setting, full dev set)",
    file: "qa_model_comparison.csv",
    icon: "📄",
    size: "183 B",
    color: "#8B5CF6",
  },
  {
    name: "qa_model_comparison.png",
    desc: "QA model comparison chart",
    file: "qa_model_comparison.png",
    icon: "🖼",
    size: "43 KB",
    color: "#8B5CF6",
  },
  {
    name: "e2e_evaluation.csv",
    desc: "End-to-end EM/F1 (retriever-supplied passage)",
    file: "e2e_evaluation.csv",
    icon: "📄",
    size: "144 B",
    color: "#34D399",
  },
  {
    name: "qa_sample_predictions.csv",
    desc: "Side-by-side predictions of all three models on sample questions",
    file: "qa_sample_predictions.csv",
    icon: "📄",
    size: "2.8 KB",
    color: "#34D399",
  },
  {
    name: "detailed_eval.json",
    desc: "Per-prediction GPU evaluation — EM, F1, span confidence and latency per question",
    file: "detailed_eval.json",
    icon: "📋",
    size: "20 KB",
    color: "#F59E0B",
  },
  {
    name: "demo_bundle.json",
    desc: "Precomputed end-to-end pipeline answers that power the offline Live Demo",
    file: "demo_bundle.json",
    icon: "📋",
    size: "123 KB",
    color: "#F59E0B",
  },
  {
    name: "summary.json",
    desc: "Machine-readable consolidation of all results",
    file: "summary.json",
    icon: "📋",
    size: "1.8 KB",
    color: "#F59E0B",
  },
  {
    name: "training_log.md",
    desc: "Fine-tuning run log — dataset sizes, hyperparameters, GPU",
    file: "training_log.md",
    icon: "📝",
    size: "340 B",
    color: "#22D3EE",
  },
  {
    name: "Adaptive_Hybrid_Semantic_Research_Assistant.ipynb",
    desc: "Complete Jupyter notebook — full pipeline, ablation, training, evaluation",
    file: "Adaptive_Hybrid_Semantic_Research_Assistant.ipynb",
    icon: "📓",
    size: "205 KB",
    color: "#F59E0B",
  },
];

export default function ResultsDownloads() {
  return (
    <section id="downloads" className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--accent-amber)" }}>
            Results & Downloads
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-3" style={{ color: "var(--text-primary)" }}>
            Raw Artifacts
          </h2>
          <p className="text-base mb-10 max-w-2xl" style={{ color: "var(--text-secondary)" }}>
            All charts and metrics in this dashboard are sourced from these files.
            Download them for independent analysis or reproduction.
          </p>
        </motion.div>

        <div className="flex flex-col gap-3">
          {ARTIFACTS.map((a, i) => (
            <motion.a key={a.name} href={asset(`/artifacts/${a.file}`)} download
              initial={{ opacity: 0, x: -16 }} whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.4, delay: i * 0.06 }}
              className="glass-card rounded-xl px-5 py-4 flex items-center gap-4 no-underline group transition-all duration-200"
              style={{ borderColor: "var(--border-subtle)" }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = a.color + "55";
                e.currentTarget.style.boxShadow = `0 0 16px ${a.color}12`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--border-subtle)";
                e.currentTarget.style.boxShadow = "";
              }}>
              <span className="text-2xl">{a.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="font-mono text-sm font-semibold truncate" style={{ color: a.color }}>{a.name}</div>
                <div className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>{a.desc}</div>
              </div>
              <div className="flex items-center gap-3 flex-shrink-0">
                <span className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>{a.size}</span>
                <span className="text-xs px-3 py-1 rounded-full font-mono transition-colors"
                  style={{ background: `${a.color}15`, color: a.color, border: `1px solid ${a.color}30` }}
                  aria-label="Download">
                  ↓ Download
                </span>
              </div>
            </motion.a>
          ))}
        </div>
      </div>
    </section>
  );
}
