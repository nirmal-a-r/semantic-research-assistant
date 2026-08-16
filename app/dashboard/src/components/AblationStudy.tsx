"use client";
import { motion } from "framer-motion";
import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
} from "recharts";
import { getAblationRows } from "@/lib/data";
import { useProjectData } from "@/lib/ProjectDataContext";

const METRICS = ["Recall@10", "Recall@20", "MRR", "NDCG@10"] as const;
const COLORS = ["#22D3EE", "#8B5CF6", "#F59E0B", "#34D399"];
const BEST_COLOR = "#F59E0B";

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card rounded-lg p-3 text-xs font-mono" style={{ borderColor: "rgba(34,211,238,0.3)" }}>
      <p className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value.toFixed(2)}%</p>
      ))}
    </div>
  );
};

export default function AblationStudy() {
  const { data } = useProjectData();
  const rows = getAblationRows(data ?? undefined);
  const [activeMetric, setActiveMetric] = useState<(typeof METRICS)[number]>("Recall@10");

  // Grouped bar data
  const barData = rows.map((r) => ({
    name: (r.name || "").length > 14 ? (r.name || "").replace("+ ", "+") : (r.name || ""),
    fullName: r.name,
    ...Object.fromEntries(METRICS.map((m) => [m, r[m]])),
  }));

  // Radar: BM25 only vs Complete Pipeline
  const bm25Row = rows[0] || { "Recall@10": 86.15, "Recall@20": 89.6, MRR: 69.49, "NDCG@10": 73.39 };
  const completeRow = rows[rows.length - 1] || bm25Row;

  const radarData = METRICS.map((m) => ({
    metric: m,
    "BM25 only": bm25Row[m],
    "Complete Pipeline": completeRow[m],
  }));

  // Best value per metric
  const bestPerMetric: Record<string, number> = {};
  METRICS.forEach((m) => {
    bestPerMetric[m] = Math.max(...rows.map((r) => r[m]));
  });

  return (
    <section id="ablation" className="py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--accent-cyan)" }}>
            Progressive Ablation Study
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-3" style={{ color: "var(--text-primary)" }}>
            7-Stage Retrieval Ablation
          </h2>
          <p className="text-base mb-10 max-w-2xl" style={{ color: "var(--text-secondary)" }}>
            Every enhancement is tested incrementally against 4 retrieval metrics to isolate each contribution.
            Real numbers from <span className="font-mono text-sm" style={{ color: "var(--accent-cyan)" }}>results/retrieval_progressive.csv</span>.
          </p>
        </motion.div>

        <div className="flex flex-col gap-10">
          {/* Grouped bar chart */}
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.7, delay: 0.1 }}
            className="glass-card rounded-2xl p-6" style={{ borderColor: "var(--border-subtle)" }}>
            <h3 className="text-base font-semibold mb-5" style={{ color: "var(--text-primary)" }}>All Configurations × 4 Metrics</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={barData} margin={{ top: 5, right: 20, left: -10, bottom: 60 }}
                barCategoryGap="25%" barGap={2}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fill: "rgba(240,244,255,0.5)", fontSize: 10, fontFamily: "monospace" }}
                  angle={-35} textAnchor="end" interval={0} />
                <YAxis domain={[55, 100]} tickFormatter={(v) => `${v.toFixed(0)}%`}
                  tick={{ fill: "rgba(240,244,255,0.4)", fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ paddingTop: 16, fontSize: 12 }} />
                {METRICS.map((m, i) => (
                  <Bar key={m} dataKey={m} fill={COLORS[i]} radius={[3, 3, 0, 0]} maxBarSize={24}
                    animationBegin={i * 120} animationDuration={800} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Radar + table row */}
          <div className="flex flex-col lg:flex-row gap-8">
            {/* Radar chart */}
            <motion.div initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.7, delay: 0.15 }}
              className="glass-card rounded-2xl p-6 flex-1" style={{ borderColor: "var(--border-subtle)" }}>
              <h3 className="text-base font-semibold mb-1" style={{ color: "var(--text-primary)" }}>
                BM25-only vs Complete Pipeline
              </h3>
              <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>Radar comparison showing improvement across all metrics</p>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
                  <PolarGrid stroke="rgba(255,255,255,0.08)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: "rgba(240,244,255,0.6)", fontSize: 11 }} />
                  <Radar name="BM25 only" dataKey="BM25 only" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.12} strokeWidth={1.5} />
                  <Radar name="Complete Pipeline" dataKey="Complete Pipeline" stroke="#22D3EE" fill="#22D3EE" fillOpacity={0.15} strokeWidth={2} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </RadarChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Data table */}
            <motion.div initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.7, delay: 0.2 }}
              className="glass-card rounded-2xl overflow-hidden flex-1" style={{ borderColor: "var(--border-subtle)" }}>
              <div className="p-4 pb-2 border-b" style={{ borderColor: "var(--border-subtle)" }}>
                <h3 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>Full Results Table</h3>
                <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                  <span style={{ color: "var(--accent-amber)" }}>★</span> = best per metric
                </p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full data-table">
                  <thead>
                    <tr>
                      <th className="text-left">Config</th>
                      {METRICS.map((m) => <th key={m} className="text-right">{m}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.name} style={r.name === "Complete Pipeline" ? { background: "rgba(34,211,238,0.05)" } : {}}>
                        <td className="text-left" style={{ color: "var(--text-secondary)", fontFamily: "monospace", fontSize: "0.75rem" }}>
                          {r.name === "Complete Pipeline"
                            ? <span style={{ color: "var(--accent-cyan)" }}>★ {r.name}</span>
                            : r.name}
                        </td>
                        {METRICS.map((m) => (
                          <td key={m} className="text-right"
                            style={{ color: r[m] === bestPerMetric[m] ? BEST_COLOR : "var(--text-secondary)", fontWeight: r[m] === bestPerMetric[m] ? 600 : 400 }}>
                            {r[m].toFixed(2)}%
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
