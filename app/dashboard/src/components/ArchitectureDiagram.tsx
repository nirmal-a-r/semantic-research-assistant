"use client";
import { motion, useAnimation } from "framer-motion";
import { useEffect, useRef } from "react";

// Pipeline nodes with exact architecture from notebook Section 4
const NODES = [
  { id: "query", label: "User Query", x: 50, y: 50, color: "#22D3EE", icon: "⌨" },
  { id: "multiq", label: "Multi-Query Generation", x: 50, y: 160, color: "#8B5CF6", icon: "⟈" },
  { id: "bm25", label: "BM25 Sparse Retrieval", x: 20, y: 280, color: "#22D3EE", icon: "≋" },
  { id: "dense", label: "SBERT + FAISS Dense", x: 80, y: 280, color: "#22D3EE", icon: "◈" },
  { id: "rrf", label: "RRF Hybrid Fusion", x: 50, y: 390, color: "#8B5CF6", icon: "⊕" },
  { id: "rerank", label: "Cross-Encoder Re-rank", x: 50, y: 490, color: "#22D3EE", icon: "↕" },
  { id: "topk", label: "Dynamic Top-K", x: 50, y: 585, color: "#8B5CF6", icon: "⫧" },
  { id: "qa", label: "Extractive QA Model", x: 50, y: 680, color: "#F59E0B", icon: "◉" },
  { id: "answer", label: "Answer + Evidence", x: 50, y: 775, color: "#22D3EE", icon: "✓" },
];

// Edges connecting nodes (from → to)
const EDGES = [
  { from: "query", to: "multiq" },
  { from: "multiq", to: "bm25" },
  { from: "multiq", to: "dense" },
  { from: "bm25", to: "rrf" },
  { from: "dense", to: "rrf" },
  { from: "rrf", to: "rerank" },
  { from: "rerank", to: "topk" },
  { from: "topk", to: "qa" },
  { from: "qa", to: "answer" },
];

const W = 400; // viewBox width
const NODE_W = 160;
const NODE_H = 36;

function nodeCenter(id: string) {
  const n = NODES.find((n) => n.id === id)!;
  return { x: (n.x / 100) * W, y: n.y + NODE_H / 2 };
}

// Animated particle along an SVG path
function FlowParticle({ d, delay }: { d: string; delay: number }) {
  return (
    <circle r={3} fill="#22D3EE" opacity={0.9}>
      <animateMotion dur="2.5s" repeatCount="indefinite" begin={`${delay}s`} path={d} calcMode="linear" />
    </circle>
  );
}

export default function ArchitectureDiagram() {
  const vbH = 840;

  // Build edge paths
  const paths = EDGES.map((e) => {
    const s = nodeCenter(e.from);
    const t = nodeCenter(e.to);
    const my = (s.y + t.y) / 2;
    // Bezier curve
    const d = `M ${s.x} ${s.y} C ${s.x} ${my}, ${t.x} ${my}, ${t.x} ${t.y}`;
    return { ...e, d };
  });

  const labels = [
    { x: 10, y: 280 + NODE_H / 2, text: "Sparse" },
    { x: 90, y: 280 + NODE_H / 2, text: "Dense" },
  ];

  return (
    <section id="architecture" className="py-20 px-4">
      <div className="max-w-5xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--accent-cyan)" }}>
            Pipeline Architecture
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-3" style={{ color: "var(--text-primary)" }}>
            How the System Works
          </h2>
          <p className="text-base mb-10 max-w-2xl" style={{ color: "var(--text-secondary)" }}>
            End-to-end adaptive retrieval-augmented QA pipeline — data flows from query to grounded answer
            through 7 distinct processing stages, with animated live data flow.
          </p>
        </motion.div>

        <div className="flex flex-col lg:flex-row gap-10 items-start">
          {/* SVG Diagram */}
          <motion.div initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.7, delay: 0.15 }}
            className="glass-card rounded-2xl p-6 flex-shrink-0 w-full lg:w-[380px]"
            style={{ borderColor: "var(--border-subtle)" }}>
            <svg viewBox={`0 0 ${W} ${vbH}`} width="100%" className="overflow-visible">
              {/* Edges */}
              {paths.map((p, i) => (
                <g key={`${p.from}-${p.to}`}>
                  <path d={p.d} fill="none" stroke="rgba(34,211,238,0.2)" strokeWidth={1.5} />
                  {/* Deterministic stagger — a random delay would differ between the
                      server-rendered HTML and the client, breaking hydration. */}
                  <FlowParticle d={p.d} delay={((i * 0.61803) % 1) * 2} />
                </g>
              ))}

              {/* Nodes */}
              {NODES.map((n, i) => {
                const cx = (n.x / 100) * W;
                const cy = n.y;
                const x = cx - NODE_W / 2;
                return (
                  <g key={n.id}>
                    {/* Glow rect */}
                    <rect x={x - 2} y={cy - 2} width={NODE_W + 4} height={NODE_H + 4}
                      rx={10} fill="none" stroke={n.color} strokeWidth={0.5} opacity={0.3} />
                    {/* Main rect */}
                    <rect x={x} y={cy} width={NODE_W} height={NODE_H}
                      rx={8} fill="rgba(15,20,32,0.95)" stroke={n.color} strokeWidth={1.2} />
                    {/* Icon */}
                    <text x={x + 14} y={cy + NODE_H / 2 + 4} fontSize={13} fill={n.color} fontFamily="monospace">{n.icon}</text>
                    {/* Label */}
                    <text x={x + 30} y={cy + NODE_H / 2 + 4.5} fontSize={10.5} fill="#e2e8f0" fontFamily="Inter,sans-serif">
                      {n.label}
                    </text>
                    {/* Animate in */}
                    <rect x={x} y={cy} width={NODE_W} height={NODE_H} rx={8} fill="transparent">
                      <animate attributeName="opacity" values="0;1" dur="0.4s" begin={`${i * 0.15}s`} fill="freeze" />
                    </rect>
                  </g>
                );
              })}

              {/* Branch labels */}
              <text x={57} y={264} fontSize={9} fill="rgba(34,211,238,0.6)" fontFamily="monospace">BM25</text>
              <text x={258} y={264} fontSize={9} fill="rgba(34,211,238,0.6)" fontFamily="monospace">SBERT+FAISS</text>
            </svg>
          </motion.div>

          {/* Stage descriptions */}
          <motion.div initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.7, delay: 0.25 }}
            className="flex flex-col gap-3 flex-1">
            {[
              { icon: "⌨", title: "User Query", desc: "Natural language question from the user.", color: "#22D3EE" },
              { icon: "⟈", title: "Multi-Query Generation", desc: "Expands the query into 3 semantic variants (What-is → definition, Who → Which person, Regarding…) to boost recall across different lexical framings.", color: "#8B5CF6" },
              { icon: "≋ + ◈", title: "Dual Retrieval (BM25 + SBERT+FAISS)", desc: "BM25Okapi exact-term sparse retrieval runs in parallel with Sentence-BERT all-MiniLM-L6-v2 dense retrieval over a FAISS IndexFlatIP index of 20,958 passages.", color: "#22D3EE" },
              { icon: "⊕", title: "Reciprocal Rank Fusion", desc: "Merges sparse + dense rankings using RRF (k=60). Multi-query variants each produce a ranking; all are fused in a second RRF pass.", color: "#8B5CF6" },
              { icon: "↕", title: "Cross-Encoder Re-ranking", desc: "cross-encoder/ms-marco-MiniLM-L6-v2 re-scores the top-30 candidates as (query, passage) pairs for precision-aware ordering.", color: "#22D3EE" },
              { icon: "⫧", title: "Dynamic Top-K", desc: "Query complexity (simple / comparison / multistep) sets retrieval depth: K=5, 10, or 15 respectively, adapting compute to query needs.", color: "#8B5CF6" },
              { icon: "◉", title: "Extractive QA", desc: "Fine-tuned RoBERTa-base (default), BERT-base, or DistilBERT-base extracts the exact answer span from the top retrieved passage.", color: "#F59E0B" },
              { icon: "✓", title: "Answer + Evidence", desc: "Returns the answer span, character offsets for highlighting, evidence doc ID, retrieval score, and top-3 candidate passages.", color: "#22D3EE" },
            ].map((s, i) => (
              <motion.div key={s.title} initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ duration: 0.4, delay: i * 0.07 }}
                className="glass-card rounded-xl p-4 flex gap-3"
                style={{ borderColor: "var(--border-subtle)" }}>
                <span className="text-lg mt-0.5 flex-shrink-0 font-mono" style={{ color: s.color }}>{s.icon}</span>
                <div>
                  <div className="font-semibold text-sm mb-0.5" style={{ color: "var(--text-primary)" }}>{s.title}</div>
                  <div className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>{s.desc}</div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
