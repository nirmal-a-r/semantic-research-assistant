"use client";
import { motion } from "framer-motion";
import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { getDetailedEval, getHeadlineStats, getQAModelRows } from "@/lib/data";
import { useProjectData } from "@/lib/ProjectDataContext";

const METRIC_EXPLANATIONS = [
  {
    name: "Exact Match (EM)",
    formula: "1 if normalize(pred) == normalize(gold) else 0",
    desc: "Binary — 1 only if the predicted span exactly matches the gold answer after normalization (lower-case, strip articles/punctuation). Strict but interpretable.",
    color: "#22D3EE",
    icon: "≡",
  },
  {
    name: "Token F1",
    formula: "2 × (P × R) / (P + R)",
    desc: "Token-level overlap between prediction and gold. Precision = matching tokens / pred tokens. Recall = matching tokens / gold tokens. Best single metric for extractive QA.",
    color: "#8B5CF6",
    icon: "∑",
  },
  {
    name: "Token Precision",
    formula: "common_tokens / len(pred_tokens)",
    desc: "Fraction of predicted tokens that appear in the gold answer. High precision = concise extraction. Low = over-extraction.",
    color: "#34D399",
    icon: "P",
  },
  {
    name: "Token Recall",
    formula: "common_tokens / len(gold_tokens)",
    desc: "Fraction of gold answer tokens captured by the prediction. High recall = not missing key words. Low = under-extraction.",
    color: "#F59E0B",
    icon: "R",
  },
  {
    name: "Span Confidence",
    formula: "(softmax(start_logits)[i] + softmax(end_logits)[j]) / 2",
    desc: "Model's own probability estimate for the predicted start/end token positions. High confidence ≠ correct, but correlates with answer quality.",
    color: "#22D3EE",
    icon: "σ",
  },
  {
    name: "SQuAD F1 (Macro)",
    formula: "mean(max_F1 over all gold answers) across questions",
    desc: "Official SQuAD metric: for each question take the max F1 over all valid gold answers (some questions have multiple), then macro-average across all questions.",
    color: "#8B5CF6",
    icon: "μ",
  },
];

const MODEL_COLORS: Record<string, string> = {
  "roberta-base": "#22D3EE",
  "bert-base-uncased": "#8B5CF6",
  "distilbert-base-uncased": "#34D399",
};

const ANALYSIS_NOTES: Record<string, string> = {
  "roberta-base":
    "Best performer across every setting. Dynamic masking, a larger pre-training corpus and no NSP objective give it the cleanest span boundaries — and the highest span confidence, so it is also the best-calibrated of the three.",
  "bert-base-uncased":
    "Within ~4 F1 of RoBERTa on gold passages at 12% fewer parameters, but the gap widens end-to-end: it is more sensitive to the noisier retrieved context, so retrieval errors cost it more.",
  "distilbert-base-uncased":
    "Half the parameters and roughly half the inference cost and training time, for about 6 F1 less than RoBERTa on gold passages. The best accuracy-per-millisecond of the three and the obvious pick for a latency budget.",
};

function ConfBar({ value, max = 1 }: { value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = value > 0.5 ? "#22D3EE" : value > 0.25 ? "#F59E0B" : "#f87171";
  return (
    <div className="flex items-center gap-2 w-24">
      <div className="flex-1 h-1.5 rounded-full" style={{ background: "rgba(255,255,255,0.08)" }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-mono w-10 text-right" style={{ color }}>{value.toFixed(3)}</span>
    </div>
  );
}

function F1Bar({ value }: { value: number }) {
  const color = value >= 80 ? "#22D3EE" : value >= 40 ? "#F59E0B" : "#f87171";
  return (
    <div className="flex items-center gap-2 w-20">
      <div className="flex-1 h-1.5 rounded-full" style={{ background: "rgba(255,255,255,0.08)" }}>
        <div className="h-full rounded-full" style={{ width: `${value}%`, background: color }} />
      </div>
      <span className="text-xs font-mono w-10 text-right" style={{ color }}>{value.toFixed(0)}%</span>
    </div>
  );
}

const LossTooltip = ({ active, payload, label }: { active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: number }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card rounded-lg p-3 text-xs font-mono" style={{ borderColor: "rgba(34,211,238,0.3)" }}>
      <p className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>step {label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value?.toFixed(4)}</p>
      ))}
    </div>
  );
};

export default function DetailedResults() {
  const { data } = useProjectData();
  const detailed = getDetailedEval(data ?? undefined);
  const qaRows = getQAModelRows(data ?? undefined);
  const stats = getHeadlineStats(data ?? undefined);
  const facts = data?.training_facts ?? {};
  const convergence = data?.convergence ?? {};

  const modelNames = detailed ? Object.keys(detailed.models) : [];
  const [activeModel, setActiveModel] = useState(0);
  const active = detailed && modelNames.length
    ? detailed.models[modelNames[Math.min(activeModel, modelNames.length - 1)]]
    : null;
  const activeName = modelNames[Math.min(activeModel, modelNames.length - 1)] ?? "";

  // Merge the three loss curves onto a shared step axis for one comparison chart.
  const stepSet = new Set<number>();
  Object.values(convergence).forEach((pts) => pts.forEach((p) => stepSet.add(p.step)));
  const lossData = [...stepSet].sort((a, b) => a - b).map((step) => {
    const row: Record<string, number | null> = { step };
    for (const [key, pts] of Object.entries(convergence)) {
      const hit = pts.find((p) => p.step === step);
      row[`${key}__train`] = hit?.train_loss ?? null;
      row[`${key}__eval`] = hit?.eval_loss ?? null;
    }
    return row;
  });

  return (
    <section id="detailed-results" className="py-20 px-4" style={{ background: "rgba(10,14,23,0.8)" }}>
      <div className="max-w-6xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "#F59E0B" }}>
            Fine-Tuning & GPU Inference Results
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
            Per-Prediction Deep Dive
          </h2>
          <p className="text-sm mb-2" style={{ color: "var(--text-secondary)" }}>
            All three backbones fine-tuned for 3 epochs on the full 87,599-example SQuAD v1.1 train split on{" "}
            <strong className="text-white">{stats.gpu}</strong> ·{" "}
            <span className="font-mono" style={{ color: "var(--accent-cyan)" }}>
              PyTorch {(detailed?.pytorch ?? stats.pytorch).split("+")[0]}
            </span>{" "}
            · CUDA {detailed?.cuda ?? stats.cuda}
          </p>
          <p className="text-xs mb-8 max-w-3xl" style={{ color: "var(--text-muted)" }}>
            Summary row = official SQuAD evaluation over the full {stats.devPairs.toLocaleString()}-example dev set.
            The per-prediction table below is a live GPU re-run over a topic-diverse sample of{" "}
            N={detailed?.n_sample ?? 24} dev questions with the gold passage supplied — small-sample
            numbers run higher than the full-dev figures and are shown for inspection, not as headline results.
          </p>
        </motion.div>

        {/* Metric explanations */}
        <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.5, delay: 0.05 }}
          className="glass-card rounded-2xl p-6 mb-8" style={{ borderColor: "var(--border-subtle)" }}>
          <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>How Metrics Are Calculated</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {METRIC_EXPLANATIONS.map((m) => (
              <div key={m.name} className="rounded-xl p-4" style={{ background: "rgba(255,255,255,0.03)", border: `1px solid ${m.color}22` }}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-mono font-bold text-base" style={{ color: m.color }}>{m.icon}</span>
                  <span className="text-xs font-semibold" style={{ color: m.color }}>{m.name}</span>
                </div>
                <div className="font-mono text-xs mb-2 px-2 py-1 rounded" style={{ background: "rgba(0,0,0,0.3)", color: "#94a3b8" }}>
                  {m.formula}
                </div>
                <p className="text-xs leading-relaxed" style={{ color: "var(--text-muted)" }}>{m.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Full-scale summary from the fine-tuning run */}
        <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.5, delay: 0.1 }}
          className="glass-card rounded-2xl overflow-hidden mb-8" style={{ borderColor: "var(--border-subtle)" }}>
          <div className="p-4 border-b flex items-center justify-between flex-wrap gap-2" style={{ borderColor: "var(--border-subtle)" }}>
            <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Full-Scale Results — SQuAD v1.1 dev ({stats.devPairs.toLocaleString()} questions)
            </h3>
            <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.1)", color: "#F59E0B", border: "1px solid rgba(245,158,11,0.3)" }}>authoritative</span>
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
                  <th className="text-right">Final Eval Loss</th>
                </tr>
              </thead>
              <tbody>
                {qaRows.map((m) => {
                  const isPrimary = m.key === "roberta-base";
                  const f = facts[m.key];
                  return (
                    <tr key={m.key} style={isPrimary ? { background: "rgba(34,211,238,0.05)" } : {}}>
                      <td className="text-left font-mono" style={{ color: isPrimary ? "var(--accent-amber)" : "var(--text-secondary)", fontSize: "0.75rem", fontWeight: isPrimary ? 600 : 400 }}>
                        {isPrimary && "★ "}{m.model}
                      </td>
                      <td className="text-right font-mono" style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>{m.params_M}M</td>
                      <td className="text-right font-mono" style={{ color: isPrimary ? "#F59E0B" : "var(--text-secondary)", fontSize: "0.75rem", fontWeight: isPrimary ? 600 : 400 }}>
                        {m.em_gold?.toFixed(2)}%
                      </td>
                      <td className="text-right font-mono" style={{ color: isPrimary ? "#F59E0B" : "var(--text-secondary)", fontSize: "0.75rem", fontWeight: isPrimary ? 600 : 400 }}>
                        {m.f1_gold?.toFixed(2)}%
                      </td>
                      <td className="text-right font-mono" style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>{m.em_e2e?.toFixed(2)}%</td>
                      <td className="text-right font-mono" style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>{m.f1_e2e?.toFixed(2)}%</td>
                      <td className="text-right font-mono" style={{ color: "var(--accent-cyan)", fontSize: "0.75rem" }}>
                        {m.latency_ms != null ? `${m.latency_ms.toFixed(1)}ms` : "—"}
                      </td>
                      <td className="text-right font-mono" style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                        {f ? `${f.train_runtime_h.toFixed(2)}h` : "—"}
                      </td>
                      <td className="text-right font-mono" style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                        {f ? f.final_eval_loss.toFixed(4) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Training convergence */}
        {lossData.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.5, delay: 0.12 }}
            className="glass-card rounded-2xl p-6 mb-8" style={{ borderColor: "var(--border-subtle)" }}>
            <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--text-primary)" }}>Training Convergence</h3>
            <p className="text-xs mb-5" style={{ color: "var(--text-muted)" }}>
              Training loss (solid, logged every 500 steps) and evaluation loss (dashed, once per epoch) from the
              HuggingFace <span className="font-mono">log_history</span> of each 3-epoch run.
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={lossData} margin={{ top: 5, right: 20, left: -12, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="step" tick={{ fill: "rgba(240,244,255,0.4)", fontSize: 10 }}
                  label={{ value: "training step", position: "insideBottom", offset: -2, fill: "rgba(240,244,255,0.35)", fontSize: 10 }} />
                <YAxis domain={[0.4, 3.4]} tick={{ fill: "rgba(240,244,255,0.4)", fontSize: 10 }} />
                <Tooltip content={<LossTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
                {Object.keys(convergence).map((key) => (
                  <Line key={`${key}-train`} type="monotone" dataKey={`${key}__train`}
                    name={`${key} · train`} stroke={MODEL_COLORS[key]} strokeWidth={2}
                    dot={false} connectNulls animationDuration={900} />
                ))}
                {Object.keys(convergence).map((key) => (
                  <Line key={`${key}-eval`} type="monotone" dataKey={`${key}__eval`}
                    name={`${key} · eval`} stroke={MODEL_COLORS[key]} strokeWidth={1.5}
                    strokeDasharray="5 4" dot={{ r: 3 }} connectNulls animationDuration={900} />
                ))}
              </LineChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
              {Object.entries(facts).map(([key, f]) => (
                <div key={key} className="rounded-xl p-3 text-xs font-mono flex flex-col gap-1"
                  style={{ background: `${MODEL_COLORS[key]}08`, border: `1px solid ${MODEL_COLORS[key]}25` }}>
                  <span style={{ color: MODEL_COLORS[key] }}>{key}</span>
                  <span style={{ color: "var(--text-muted)" }}>
                    {f.total_steps.toLocaleString()} steps · {f.epochs} epochs · {f.train_runtime_h.toFixed(2)}h
                  </span>
                  <span style={{ color: "var(--text-muted)" }}>
                    {f.samples_per_second.toFixed(1)} samples/s · train loss {f.final_train_loss.toFixed(3)} · eval loss {f.final_eval_loss.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Per-prediction table */}
        {active && (
          <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.5, delay: 0.15 }}
            className="glass-card rounded-2xl overflow-hidden mb-6" style={{ borderColor: "var(--border-subtle)" }}>
            <div className="p-4 border-b flex items-center gap-3 flex-wrap" style={{ borderColor: "var(--border-subtle)" }}>
              <div className="flex gap-2 flex-wrap">
                {modelNames.map((name, i) => {
                  const key = detailed!.models[name].key;
                  const on = i === Math.min(activeModel, modelNames.length - 1);
                  return (
                    <button key={name} onClick={() => setActiveModel(i)}
                      className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all"
                      style={{
                        background: on ? `${MODEL_COLORS[key]}22` : "rgba(255,255,255,0.04)",
                        border: `1px solid ${on ? MODEL_COLORS[key] : "var(--border-subtle)"}`,
                        color: on ? MODEL_COLORS[key] : "var(--text-secondary)",
                      }}>
                      {key === "roberta-base" ? "★ " : ""}{name}
                    </button>
                  );
                })}
              </div>
              <span className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
                {active.params}M params · N={detailed!.n_sample} sample · EM {active.exact_match}% · F1 {active.f1}% ·
                avg conf {active.avg_confidence} · {active.avg_latency_ms}ms
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full data-table">
                <thead>
                  <tr>
                    <th className="text-left" style={{ minWidth: 240 }}>Question</th>
                    <th className="text-left">Gold Answer</th>
                    <th className="text-left">Prediction</th>
                    <th className="text-center">EM</th>
                    <th className="text-left">F1</th>
                    <th className="text-left">Confidence</th>
                    <th className="text-right">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {active.predictions.map((p, i) => (
                    <tr key={i}>
                      <td className="text-left" style={{ color: "var(--text-secondary)", fontSize: "0.7rem", maxWidth: 240 }}>
                        {p.question.slice(0, 60)}{p.question.length > 60 ? "…" : ""}
                      </td>
                      <td className="font-mono" style={{ color: "#34D399", fontSize: "0.72rem" }}>
                        {p.gold.slice(0, 28)}{p.gold.length > 28 ? "…" : ""}
                      </td>
                      <td className="font-mono" style={{ color: p.em ? "#22D3EE" : p.f1 > 30 ? "#F59E0B" : "#f87171", fontSize: "0.72rem" }}>
                        {p.pred ? `${p.pred.slice(0, 28)}${p.pred.length > 28 ? "…" : ""}` : "(empty)"}
                      </td>
                      <td className="text-center">
                        <span className="px-1.5 py-0.5 rounded text-xs font-mono font-bold"
                          style={{ background: p.em ? "rgba(34,211,238,0.15)" : "rgba(248,113,113,0.12)", color: p.em ? "#22D3EE" : "#f87171" }}>
                          {p.em ? "YES" : "NO"}
                        </span>
                      </td>
                      <td><F1Bar value={p.f1} /></td>
                      <td><ConfBar value={p.conf} /></td>
                      <td className="text-right font-mono" style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}>{p.lat.toFixed(1)}ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="p-3 border-t text-xs font-mono" style={{ borderColor: "var(--border-subtle)", color: "var(--text-muted)", background: "rgba(34,211,238,0.03)" }}>
              {activeName}: {active.predictions.filter((p) => p.em).length} EM=YES ·{" "}
              {active.predictions.filter((p) => !p.em).length} EM=NO · avg confidence {active.avg_confidence} ·
              avg latency {active.avg_latency_ms}ms on {detailed!.device}
            </div>
          </motion.div>
        )}

        {/* Model-by-model analysis */}
        <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.5, delay: 0.2 }}
          className="glass-card rounded-2xl p-6" style={{ borderColor: "var(--border-subtle)" }}>
          <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Analysis: Where the Differences Come From</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {qaRows.map((m) => {
              const color = MODEL_COLORS[m.key];
              return (
                <div key={m.key} className="rounded-xl p-4 flex flex-col gap-2"
                  style={{ background: `${color}08`, border: `1px solid ${color}25` }}>
                  <div className="font-mono font-semibold text-sm" style={{ color }}>{m.model}</div>
                  <div className="flex gap-3 text-xs font-mono flex-wrap">
                    <span style={{ color: "var(--accent-amber)" }}>EM {m.em_gold?.toFixed(1)}%</span>
                    <span style={{ color: "var(--accent-cyan)" }}>F1 {m.f1_gold?.toFixed(1)}%</span>
                    <span style={{ color: "var(--text-muted)" }}>{m.latency_ms?.toFixed(1)}ms</span>
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-muted)" }}>{ANALYSIS_NOTES[m.key]}</p>
                </div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
