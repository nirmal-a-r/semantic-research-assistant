"use client";
import { motion } from "framer-motion";
import { useState } from "react";

// Mirrors the exact heuristic from src/multiquery.py — runs client-side
const COMPARISON_WORDS = new Set(["compare", "versus", "vs", "difference", "better", "than", "both"]);
const MULTISTEP_HINTS = ["and then", "after", "before", "because", "why did", "how did", "led to"];

function estimateComplexity(query: string): "simple" | "comparison" | "multistep" {
  const q = query.toLowerCase();
  const commas = (q.match(/,/g) || []).length;
  if (MULTISTEP_HINTS.some((h) => q.includes(h)) || commas >= 2) return "multistep";
  const words = q.split(/\s+/);
  if (words.some((w) => COMPARISON_WORDS.has(w))) return "comparison";
  return "simple";
}

function generateVariants(query: string): string[] {
  const variants = [query];
  if (query.toLowerCase().startsWith("what is")) {
    variants.push(query.slice(7).trim().replace(/\?$/, "") + " definition?");
  }
  if (query.toLowerCase().startsWith("who")) {
    variants.push(query.replace(/^Who/i, "Which person"));
  }
  variants.push("Regarding " + query.charAt(0).toLowerCase() + query.slice(1));
  const seen = new Set<string>();
  return variants.filter((v) => { if (seen.has(v)) return false; seen.add(v); return true; }).slice(0, 3);
}

const TOP_K: Record<string, number> = { simple: 5, comparison: 10, multistep: 15 };
const COLOR: Record<string, string> = {
  simple: "#22D3EE",
  comparison: "#8B5CF6",
  multistep: "#F59E0B",
};
const COMPLEXITY_DESC: Record<string, string> = {
  simple: "Direct factual query — retrieval depth K=5. Efficient single-hop lookup.",
  comparison: "Comparative query detected — retrieval depth K=10. Wider candidate pool for contrast.",
  multistep: "Multi-hop query — retrieval depth K=15. Deep retrieval for complex reasoning chains.",
};

const DEMO_QUERIES = [
  { q: "What is the speed of light?", expected: "simple" },
  { q: "Compare RoBERTa and BERT in terms of accuracy?", expected: "comparison" },
  { q: "Why did the Roman Empire fall and how did it affect Europe after?", expected: "multistep" },
  { q: "Who invented the telephone?", expected: "simple" },
  { q: "What is better: supervised or unsupervised learning?", expected: "comparison" },
];

export default function AdaptiveExplainer() {
  const [query, setQuery] = useState("What is the speed of light?");
  const complexity = estimateComplexity(query);
  const variants = generateVariants(query);
  const topK = TOP_K[complexity];
  const color = COLOR[complexity];

  return (
    <section id="adaptive" className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--accent-violet)" }}>
            Adaptive Enhancements
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-3" style={{ color: "var(--text-primary)" }}>
            Dynamic Query Intelligence
          </h2>
          <p className="text-base mb-8 max-w-2xl" style={{ color: "var(--text-secondary)" }}>
            The pipeline classifies every query in real-time using{" "}
            <span className="font-mono text-sm" style={{ color: "var(--accent-cyan)" }}>estimate_query_complexity()</span>{" "}
            from <span className="font-mono text-sm" style={{ color: "var(--accent-cyan)" }}>src/multiquery.py</span>{" "}
            and adapts retrieval depth accordingly. Type below to see it live.
          </p>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.1 }}
          className="glass-card rounded-2xl p-6 flex flex-col gap-6" style={{ borderColor: "var(--border-subtle)" }}>

          {/* Input */}
          <div>
            <label className="text-xs font-mono uppercase tracking-widest mb-2 block" style={{ color: "var(--text-muted)" }}>
              Type a query to classify
            </label>
            <input type="text" id="complexity-input"
              value={query} onChange={(e) => setQuery(e.target.value)}
              className="w-full px-4 py-3 rounded-xl text-sm outline-none transition-all"
              style={{
                background: "rgba(255,255,255,0.05)",
                border: `1px solid ${color}55`,
                color: "var(--text-primary)",
                fontFamily: "Inter, sans-serif",
                boxShadow: `0 0 16px ${color}15`,
              }}
            />
          </div>

          {/* Classification result */}
          <motion.div key={complexity} initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className="rounded-xl p-5 flex flex-col md:flex-row gap-5 items-start md:items-center"
            style={{ background: `${color}10`, border: `1px solid ${color}33` }}>
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-1">
                <span className="text-2xl font-mono font-bold uppercase" style={{ color }}>{complexity}</span>
                <span className="text-xs px-2 py-0.5 rounded-full font-mono"
                  style={{ background: `${color}20`, color, border: `1px solid ${color}44` }}>
                  Top-K = {topK}
                </span>
              </div>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{COMPLEXITY_DESC[complexity]}</p>
            </div>
            {/* Visual K indicator */}
            <div className="flex gap-1 flex-shrink-0">
              {Array.from({ length: 15 }).map((_, i) => (
                <div key={i} className="w-2 h-8 rounded-sm transition-all duration-300"
                  style={{ background: i < topK ? color : "rgba(255,255,255,0.06)", opacity: i < topK ? 1 : 0.4 }} />
              ))}
            </div>
          </motion.div>

          {/* Query variants */}
          <div>
            <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>
              Multi-query variants (generate_query_variants)
            </div>
            <div className="flex flex-col gap-2">
              {variants.map((v, i) => (
                <motion.div key={v} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.08 }}
                  className="text-xs font-mono px-4 py-2.5 rounded-lg flex gap-2 items-start"
                  style={{
                    background: "rgba(139,92,246,0.07)",
                    border: "1px solid rgba(139,92,246,0.15)",
                    color: "var(--text-secondary)",
                  }}>
                  <span style={{ color: "var(--accent-violet)", fontWeight: 600, minWidth: 20 }}>→{i + 1}</span>
                  {v}
                </motion.div>
              ))}
            </div>
          </div>

          {/* Quick demo chips */}
          <div>
            <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>
              Test examples
            </div>
            <div className="flex flex-wrap gap-2">
              {DEMO_QUERIES.map((d) => (
                <button key={d.q} onClick={() => setQuery(d.q)}
                  className="px-3 py-1 rounded-full text-xs font-mono transition-all"
                  style={{
                    background: `${COLOR[d.expected]}11`,
                    border: `1px solid ${COLOR[d.expected]}33`,
                    color: COLOR[d.expected],
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = `${COLOR[d.expected]}22`)}
                  onMouseLeave={(e) => (e.currentTarget.style.background = `${COLOR[d.expected]}11`)}>
                  {d.expected} · {d.q.slice(0, 36)}{d.q.length > 36 ? "…" : ""}
                </button>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-4 pt-2 border-t" style={{ borderColor: "var(--border-subtle)" }}>
            {[["simple", "≤10 words, no comparison/multistep keywords → K=5"], ["comparison", "compare/versus/better/than in words → K=10"], ["multistep", "and then/after/because/why did, or ≥2 commas → K=15"]].map(([type, rule]) => (
              <div key={type} className="flex items-start gap-2 text-xs" style={{ color: "var(--text-muted)" }}>
                <span className="font-mono font-semibold" style={{ color: COLOR[type] }}>{type}</span>
                <span>{rule}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
