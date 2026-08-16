"use client";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ScatterChart, Scatter, ZAxis, ResponsiveContainer,
} from "recharts";
import { getQAModelRows } from "@/lib/data";
import { useProjectData } from "@/lib/ProjectDataContext";

const MODEL_COLORS: Record<string, string> = {
  "roberta-base": "#22D3EE",
  "bert-base-uncased": "#8B5CF6",
  "distilbert-base-uncased": "#34D399",
};

const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card rounded-lg p-3 text-xs font-mono" style={{ borderColor: "rgba(34,211,238,0.3)" }}>
      <p className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {(p.value ?? 0).toFixed(2)}%</p>
      ))}
    </div>
  );
};

interface ScatterPayloadItem { name: string; value: number; payload: { model: string; params: number; f1: number; isPrimary: boolean }; }
const ScatterTooltip = ({ active, payload }: { active?: boolean; payload?: ScatterPayloadItem[] }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="glass-card rounded-lg p-3 text-xs font-mono" style={{ borderColor: "rgba(139,92,246,0.4)" }}>
      <p className="font-bold" style={{ color: "var(--text-primary)" }}>{d?.model}</p>
      <p style={{ color: "var(--accent-violet)" }}>Params: {d?.params}M</p>
      <p style={{ color: "var(--accent-cyan)" }}>F1: {(d?.f1 ?? 0).toFixed(2)}%</p>
    </div>
  );
};

// Custom scatter dot — declared at module scope so it is not recreated per render.
const CustomDot = (props: { cx?: number; cy?: number; payload?: { isPrimary: boolean; model: string } }) => {
  const { cx = 0, cy = 0, payload } = props;
  const color = payload?.isPrimary ? "#F59E0B" : "#8B5CF6";
  return (
    <g>
      <circle cx={cx} cy={cy} r={payload?.isPrimary ? 9 : 7} fill={color} opacity={0.9} />
      {payload?.isPrimary && (
        <circle cx={cx} cy={cy} r={14} fill="none" stroke="#F59E0B" strokeWidth={1.5} opacity={0.4} />
      )}
      <text x={cx} y={cy - 14} textAnchor="middle" fontSize={9} fill="rgba(240,244,255,0.7)" fontFamily="monospace">
        {payload?.model?.split("-")[0]}
      </text>
    </g>
  );
};

export default function QAModelComparison() {
  const { data } = useProjectData();
  const rawRows = getQAModelRows(data ?? undefined);

  const rows = rawRows.map((r) => ({
    model: r.key,
    displayName: r.model,
    exactMatch: r.em_gold ?? 0,
    f1: r.f1_gold ?? 0,
    e2eExactMatch: r.em_e2e ?? 0,
    e2eF1: r.f1_e2e ?? 0,
    params: r.params_M ?? 0,
    latency: r.latency_ms ?? null,
    trainHours: r.train_runtime_h ?? null,
    isPrimary: r.key.includes("roberta"),
  }));

  // Gold-passage bar data
  const goldData = rows.map((r) => ({
    name: r.displayName,
    "Exact Match": r.exactMatch,
    F1: r.f1,
    isPrimary: r.isPrimary,
  }));

  // E2E bar data
  const e2eData = rows.map((r) => ({
    name: r.displayName,
    "Exact Match (E2E)": r.e2eExactMatch,
    "F1 (E2E)": r.e2eF1,
    isPrimary: r.isPrimary,
  }));

  // Scatter: Params vs F1
  const scatterData = rows.map((r) => ({
    params: r.params,
    f1: r.f1,
    model: r.displayName,
    isPrimary: r.isPrimary,
  }));

  return (
    <section id="qa-comparison" className="py-20 px-4" style={{ background: "rgba(15,20,32,0.5)" }}>
      <div className="max-w-6xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--accent-violet)" }}>
            QA Model Comparison
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-3" style={{ color: "var(--text-primary)" }}>
            Three Backbones Benchmarked
          </h2>
          <p className="text-base mb-2 max-w-2xl" style={{ color: "var(--text-secondary)" }}>
            RoBERTa-base, BERT-base-uncased, and DistilBERT-base-uncased fine-tuned on SQuAD v1.1
            and evaluated in two settings. Data from{" "}
            <span className="font-mono text-sm" style={{ color: "var(--accent-cyan)" }}>results/qa_model_comparison.csv</span>{" "}
            and{" "}
            <span className="font-mono text-sm" style={{ color: "var(--accent-cyan)" }}>results/e2e_evaluation.csv</span>.
          </p>
          <div className="flex gap-3 mb-8 flex-wrap">
            {rows.map((r) => (
              <span key={r.model}
                className="px-3 py-1 rounded-full text-xs font-mono font-semibold border"
                style={{
                  color: MODEL_COLORS[r.model],
                  borderColor: MODEL_COLORS[r.model] + "55",
                  background: MODEL_COLORS[r.model] + "11",
                  boxShadow: r.isPrimary ? `0 0 12px ${MODEL_COLORS[r.model]}30` : undefined,
                }}>
                {r.isPrimary ? "★ " : ""}{r.displayName} · {r.params}M params
              </span>
            ))}
          </div>
        </motion.div>

        <div className="flex flex-col gap-8">
          {/* Gold passage row */}
          <div className="flex flex-col lg:flex-row gap-8">
            <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.1 }}
              className="glass-card rounded-2xl p-6 flex-1" style={{ borderColor: "var(--border-subtle)" }}>
              <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--text-primary)" }}>Gold Passage Setting</h3>
              <p className="text-xs mb-5" style={{ color: "var(--text-muted)" }}>Oracle context supplied — measures pure QA model quality</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={goldData} margin={{ top: 5, right: 10, left: -15, bottom: 5 }} barCategoryGap="35%">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fill: "rgba(240,244,255,0.5)", fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: "rgba(240,244,255,0.4)", fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="Exact Match" fill="#22D3EE" radius={[4, 4, 0, 0]} maxBarSize={32} animationDuration={900} />
                  <Bar dataKey="F1" fill="#8B5CF6" radius={[4, 4, 0, 0]} maxBarSize={32} animationDuration={900} animationBegin={200} />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.15 }}
              className="glass-card rounded-2xl p-6 flex-1" style={{ borderColor: "var(--border-subtle)" }}>
              <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--text-primary)" }}>End-to-End (Full Pipeline)</h3>
              <p className="text-xs mb-5" style={{ color: "var(--text-muted)" }}>Retriever supplies context — measures complete system quality</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={e2eData} margin={{ top: 5, right: 10, left: -15, bottom: 5 }} barCategoryGap="35%">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fill: "rgba(240,244,255,0.5)", fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: "rgba(240,244,255,0.4)", fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="Exact Match (E2E)" fill="#F59E0B" radius={[4, 4, 0, 0]} maxBarSize={32} animationDuration={900} />
                  <Bar dataKey="F1 (E2E)" fill="#34D399" radius={[4, 4, 0, 0]} maxBarSize={32} animationDuration={900} animationBegin={200} />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          </div>

          {/* Scatter + Summary table */}
          <div className="flex flex-col lg:flex-row gap-8">
            {/* Scatter */}
            <motion.div initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.2 }}
              className="glass-card rounded-2xl p-6 w-full lg:w-80 flex-shrink-0" style={{ borderColor: "var(--border-subtle)" }}>
              <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--text-primary)" }}>Accuracy vs. Efficiency</h3>
              <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
                <span style={{ color: "var(--accent-amber)" }}>★</span> = primary model (RoBERTa-base)
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart margin={{ top: 10, right: 20, left: -20, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="params" name="Params (M)" domain={[55, 135]}
                    label={{ value: "Params (M)", position: "insideBottom", offset: -2, fill: "rgba(240,244,255,0.4)", fontSize: 10 }}
                    tick={{ fill: "rgba(240,244,255,0.4)", fontSize: 10 }} />
                  <YAxis type="number" dataKey="f1" name="F1 (%)" domain={[80, 95]}
                    tickFormatter={(v) => `${v}%`} tick={{ fill: "rgba(240,244,255,0.4)", fontSize: 10 }} />
                  <ZAxis range={[60, 60]} />
                  <Tooltip content={<ScatterTooltip />} />
                  <Scatter data={scatterData} shape={<CustomDot />} />
                </ScatterChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Summary table */}
            <motion.div initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.2 }}
              className="glass-card rounded-2xl overflow-hidden flex-1" style={{ borderColor: "var(--border-subtle)" }}>
              <div className="p-4 border-b" style={{ borderColor: "var(--border-subtle)" }}>
                <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Complete Comparison</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full data-table">
                  <thead>
                    <tr>
                      <th className="text-left">Model</th>
                      <th className="text-right">Params</th>
                      <th className="text-right">EM (Gold)</th>
                      <th className="text-right">F1 (Gold)</th>
                      <th className="text-right">EM (E2E)</th>
                      <th className="text-right">F1 (E2E)</th>
                      <th className="text-right">Latency</th>
                      <th className="text-right">Train Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.model} style={r.isPrimary ? { background: "rgba(34,211,238,0.05)" } : {}}>
                        <td className="text-left font-mono" style={{ color: MODEL_COLORS[r.model], fontSize: "0.75rem" }}>
                          {r.isPrimary && <span style={{ color: "var(--accent-amber)" }}>★ </span>}
                          {r.displayName}
                        </td>
                        <td className="text-right font-mono" style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>{r.params}M</td>
                        <td className="text-right font-mono" style={{ color: r.isPrimary ? "var(--accent-amber)" : "var(--text-secondary)", fontSize: "0.75rem", fontWeight: r.isPrimary ? 600 : 400 }}>
                          {r.exactMatch.toFixed(1)}%
                        </td>
                        <td className="text-right font-mono" style={{ color: r.isPrimary ? "var(--accent-amber)" : "var(--text-secondary)", fontSize: "0.75rem", fontWeight: r.isPrimary ? 600 : 400 }}>
                          {r.f1.toFixed(2)}%
                        </td>
                        <td className="text-right font-mono" style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
                          {r.e2eExactMatch.toFixed(1)}%
                        </td>
                        <td className="text-right font-mono" style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
                          {r.e2eF1.toFixed(2)}%
                        </td>
                        <td className="text-right font-mono" style={{ color: "var(--accent-cyan)", fontSize: "0.75rem" }}>
                          {r.latency != null ? `${r.latency.toFixed(1)}ms` : "—"}
                        </td>
                        <td className="text-right font-mono" style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                          {r.trainHours != null ? `${r.trainHours.toFixed(2)}h` : "—"}
                        </td>
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
