"use client";
import { motion } from "framer-motion";
import { useState, useRef, useEffect, useMemo } from "react";
import { fetchAnswer, fetchHealth, type AnswerResponse, type ModelKey } from "@/lib/api";
import { getDemoEntries, matchDemoEntry, type DemoEntry } from "@/lib/data";
import { useProjectData } from "@/lib/ProjectDataContext";

const MODEL_OPTIONS: { key: ModelKey; label: string; desc: string }[] = [
  { key: "roberta-base", label: "RoBERTa-base", desc: "125M · Best accuracy" },
  { key: "bert-base-uncased", label: "BERT-base", desc: "110M · Balanced" },
  { key: "distilbert-base-uncased", label: "DistilBERT", desc: "66M · Fastest" },
];

/** Precomputed entry + chosen model → the same shape the FastAPI backend returns. */
function toAnswerResponse(entry: DemoEntry, model: ModelKey): AnswerResponse {
  const a = entry.answers[model];
  return {
    answer: a.answer,
    answer_start: a.answer_start,
    answer_end: a.answer_end,
    evidence_doc_id: entry.evidence_doc_id,
    evidence_score: entry.evidence_score,
    evidence_text: entry.evidence_text,
    top_chunks: entry.top_chunks.map((c) => ({ doc_id: c.doc_id, score: c.score, text: c.text })),
    query_complexity: entry.query_complexity,
    query_variants: entry.query_variants,
    model_used: model,
    model_params_million: a.model_params_million,
    metrics: {
      exact_match: a.exact_match,
      token_f1: a.token_f1,
      start_confidence: a.start_confidence,
      end_confidence: a.end_confidence,
      avg_confidence: a.avg_confidence,
      precision_hint: null,
      recall_hint: null,
      answer_length_words: a.answer_length_words,
      context_length_words: a.context_length_words,
      latency_ms: a.latency_ms,
    },
  };
}

function HighlightedText({ text, start, end }: { text: string; start: number; end: number }) {
  if (!text) return null;
  if (start === end || start < 0) return <span style={{ color: "var(--text-secondary)" }}>{text}</span>;
  return (
    <span style={{ color: "var(--text-secondary)", lineHeight: 1.7 }}>
      {text.slice(0, start)}
      <mark style={{
        background: "linear-gradient(90deg,rgba(34,211,238,0.3),rgba(139,92,246,0.3))",
        color: "#fff",
        borderRadius: "3px",
        padding: "0 2px",
        fontWeight: 600,
        boxShadow: "0 0 10px rgba(34,211,238,0.3)",
      }}>
        {text.slice(start, end)}
      </mark>
      {text.slice(end)}
    </span>
  );
}

export default function LiveDemo() {
  const { data } = useProjectData();
  const demoEntries = useMemo(() => getDemoEntries(data ?? undefined), [data]);

  const [question, setQuestion] = useState("");
  const [model, setModel] = useState<ModelKey>("roberta-base");
  const [liveResult, setLiveResult] = useState<AnswerResponse | null>(null);
  const [entry, setEntry] = useState<DemoEntry | null>(null);   // set in precomputed mode
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [expandedChunk, setExpandedChunk] = useState<number | null>(null);
  const [showAllExamples, setShowAllExamples] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchHealth().then((h) => setBackendStatus(h?.status === "ready" ? "online" : "offline"));
  }, []);

  // Derived, not stored: switching the model selector instantly re-renders the
  // stored precomputed entry with that model's answer.
  const result: AnswerResponse | null = entry ? toAnswerResponse(entry, model) : liveResult;

  const visibleExamples = showAllExamples ? demoEntries : demoEntries.slice(0, 8);

  const runPrecomputed = (e: DemoEntry) => {
    setEntry(e);
    setLiveResult(null);
    setError(null);
    setExpandedChunk(null);
    setQuestion(e.question);
  };

  const handleSubmit = async (q?: string) => {
    const finalQ = q ?? question;
    if (!finalQ.trim()) return;
    setExpandedChunk(null);
    if (q) setQuestion(q);

    if (backendStatus !== "online") {
      const match = matchDemoEntry(demoEntries, finalQ);
      if (match) {
        runPrecomputed(match);
      } else {
        setEntry(null);
        setLiveResult(null);
        setError(
          "No precomputed result for that question. This deployed build ships " +
          `${demoEntries.length} questions answered by the real GPU pipeline — pick one below, ` +
          "or run the FastAPI backend locally (uvicorn app.api:app) to query all " +
          "20,958 passages with any question."
        );
      }
      return;
    }

    setLoading(true);
    setError(null);
    setLiveResult(null);
    setEntry(null);
    try {
      setLiveResult(await fetchAnswer(finalQ, model));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const complexityColor = (c: string) =>
    c === "multistep" ? "var(--accent-amber)" : c === "comparison" ? "var(--accent-violet)" : "var(--accent-cyan)";

  const precomputed = backendStatus !== "online";

  return (
    <section id="live-demo" className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--accent-cyan)" }}>
            Inference Demo
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
            Ask a Question
          </h2>
          <div className="flex items-center gap-2 mb-6 flex-wrap">
            <span className={`w-2 h-2 rounded-full ${backendStatus === "online" ? "bg-cyan-400" : backendStatus === "offline" ? "bg-amber-400" : "bg-slate-400"}`}
              style={backendStatus === "online" ? { boxShadow: "0 0 8px #22D3EE" } : {}} />
            <span className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
              {backendStatus === "checking" && "Checking for a local backend…"}
              {backendStatus === "online" && (
                <>Mode: <span style={{ color: "var(--accent-cyan)" }}>live backend</span> — any question, full 20,958-passage corpus</>
              )}
              {backendStatus === "offline" && (
                <>Mode: <span style={{ color: "var(--accent-amber)" }}>precomputed</span> — {demoEntries.length} questions
                  replayed from the real GPU pipeline run (no server needed)</>
              )}
            </span>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.1 }}
          className="glass-card rounded-2xl p-6 flex flex-col gap-5" style={{ borderColor: "var(--border-subtle)" }}>

          {/* Model selector */}
          <div className="flex flex-wrap gap-2">
            {MODEL_OPTIONS.map((m) => (
              <button key={m.key} onClick={() => setModel(m.key)}
                className="px-4 py-2 rounded-lg text-xs font-mono font-semibold transition-all duration-200"
                style={{
                  background: model === m.key ? "rgba(34,211,238,0.15)" : "rgba(255,255,255,0.04)",
                  border: `1px solid ${model === m.key ? "var(--accent-cyan)" : "var(--border-subtle)"}`,
                  color: model === m.key ? "var(--accent-cyan)" : "var(--text-secondary)",
                  boxShadow: model === m.key ? "0 0 12px rgba(34,211,238,0.15)" : "none",
                }}>
                {m.label}
                <span className="ml-1.5 opacity-60 font-normal">{m.desc}</span>
              </button>
            ))}
          </div>

          {/* Search input */}
          <div className="flex gap-3">
            <input ref={inputRef} type="text" id="question-input"
              value={question} onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder={precomputed
                ? "Type one of the precomputed questions, or pick one below…"
                : "Ask a question about any SQuAD article…"}
              className="flex-1 px-4 py-3 rounded-xl text-sm outline-none transition-all"
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid var(--border-subtle)",
                color: "var(--text-primary)",
                fontFamily: "Inter, sans-serif",
              }}
              onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent-cyan)")}
              onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border-subtle)")}
            />
            <button onClick={() => handleSubmit()} disabled={loading || !question.trim()}
              className="px-5 py-3 rounded-xl text-sm font-semibold transition-all"
              style={{
                background: loading || !question.trim() ? "rgba(34,211,238,0.1)" : "linear-gradient(135deg,#22D3EE,#8B5CF6)",
                color: loading || !question.trim() ? "rgba(255,255,255,0.3)" : "#fff",
                cursor: loading || !question.trim() ? "not-allowed" : "pointer",
              }}>
              {loading ? "…" : "Ask →"}
            </button>
          </div>

          {/* Example chips — the precomputed question set */}
          <div className="flex flex-wrap gap-2 items-center">
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>Try:</span>
            {visibleExamples.map((e) => (
              <button key={e.question} onClick={() => runPrecomputed(e)}
                title={`${e.topic} · gold: ${e.gold}`}
                className="px-3 py-1 rounded-full text-xs font-mono transition-all text-left"
                style={{
                  background: entry?.question === e.question ? "rgba(139,92,246,0.25)" : "rgba(139,92,246,0.1)",
                  border: "1px solid rgba(139,92,246,0.25)",
                  color: "var(--accent-violet)",
                }}
                onMouseEnter={(ev) => (ev.currentTarget.style.background = "rgba(139,92,246,0.2)")}
                onMouseLeave={(ev) => (ev.currentTarget.style.background =
                  entry?.question === e.question ? "rgba(139,92,246,0.25)" : "rgba(139,92,246,0.1)")}>
                {e.question.length > 52 ? `${e.question.slice(0, 52)}…` : e.question}
              </button>
            ))}
            {demoEntries.length > 8 && (
              <button onClick={() => setShowAllExamples((v) => !v)}
                className="px-3 py-1 rounded-full text-xs font-mono transition-all"
                style={{ border: "1px solid var(--border-subtle)", color: "var(--text-muted)" }}>
                {showAllExamples ? "− show fewer" : `+ ${demoEntries.length - 8} more`}
              </button>
            )}
          </div>

          {/* Loading skeleton */}
          {loading && (
            <div className="flex flex-col gap-3 animate-pulse">
              <div className="h-3 rounded-full w-32" style={{ background: "rgba(34,211,238,0.15)" }} />
              <div className="h-20 rounded-xl" style={{ background: "rgba(255,255,255,0.04)" }} />
              <div className="h-3 rounded-full w-48" style={{ background: "rgba(255,255,255,0.06)" }} />
            </div>
          )}

          {/* Error / no-match notice */}
          {error && !loading && (
            <div className="rounded-xl p-4 text-sm font-mono leading-relaxed"
              style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.3)", color: "#F59E0B" }}>
              ⚠ {error}
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }} className="flex flex-col gap-4">

              {/* Metrics panel */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { label: "Start Confidence", value: result.metrics?.start_confidence != null ? `${(result.metrics.start_confidence * 100).toFixed(1)}%` : "N/A", color: "#22D3EE" },
                  { label: "End Confidence", value: result.metrics?.end_confidence != null ? `${(result.metrics.end_confidence * 100).toFixed(1)}%` : "N/A", color: "#22D3EE" },
                  { label: "Latency", value: result.metrics?.latency_ms != null ? `${result.metrics.latency_ms.toFixed(1)}ms` : "N/A", color: "#8B5CF6" },
                  { label: "Answer Length", value: result.metrics?.answer_length_words != null ? `${result.metrics.answer_length_words} tokens` : "N/A", color: "#34D399" },
                ].map((m) => (
                  <div key={m.label} className="rounded-xl p-3 flex flex-col gap-1" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}>
                    <div className="text-lg font-mono font-bold" style={{ color: m.color }}>{m.value}</div>
                    <div className="text-xs" style={{ color: "var(--text-muted)" }}>{m.label}</div>
                  </div>
                ))}
              </div>

              {/* Answer + meta */}
              <div className="rounded-xl p-5" style={{ background: "rgba(34,211,238,0.06)", border: "1px solid rgba(34,211,238,0.2)" }}>
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className="text-xs font-mono uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>Answer</span>
                  <span className="px-2 py-0.5 rounded-full text-xs font-mono"
                    style={{ background: "rgba(34,211,238,0.1)", color: "var(--accent-cyan)", border: "1px solid rgba(34,211,238,0.3)" }}>
                    {result.model_used}
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-xs font-mono"
                    style={{ color: complexityColor(result.query_complexity), background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)" }}>
                    {result.query_complexity} · K={entry?.top_k ?? "—"}
                  </span>
                  {entry && (
                    <span className="px-2 py-0.5 rounded-full text-xs font-mono"
                      style={{
                        color: result.metrics.exact_match ? "#34D399" : "#F59E0B",
                        background: result.metrics.exact_match ? "rgba(52,211,153,0.1)" : "rgba(245,158,11,0.1)",
                        border: `1px solid ${result.metrics.exact_match ? "rgba(52,211,153,0.3)" : "rgba(245,158,11,0.3)"}`,
                      }}>
                      EM {result.metrics.exact_match ? "✓" : "✗"} · F1 {result.metrics.token_f1?.toFixed(0)}%
                    </span>
                  )}
                </div>
                <p className="text-2xl font-bold mb-1" style={{ color: "var(--accent-cyan)" }}>
                  {result.answer || <span style={{ color: "var(--text-muted)", fontSize: "1rem" }}>No answer extracted</span>}
                </p>
                {entry && (
                  <p className="text-xs font-mono mb-1" style={{ color: "var(--text-muted)" }}>
                    Gold answer: <span style={{ color: "#34D399" }}>{entry.gold}</span> · topic: {entry.topic}
                  </p>
                )}
                <p className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
                  RRF score: {result.evidence_score.toFixed(6)} · Doc: {result.evidence_doc_id}
                </p>
              </div>

              {/* Evidence with highlighted span */}
              <div>
                <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>Evidence Passage</div>
                <div className="rounded-xl p-5 text-sm leading-relaxed"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-subtle)" }}>
                  <HighlightedText text={result.evidence_text} start={result.answer_start} end={result.answer_end} />
                </div>
              </div>

              {/* Query variants */}
              {result.query_variants.length > 1 && (
                <div>
                  <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>Query Variants Generated</div>
                  <div className="flex flex-col gap-1">
                    {result.query_variants.map((v, i) => (
                      <div key={i} className="text-xs font-mono px-3 py-1.5 rounded-lg"
                        style={{ background: "rgba(139,92,246,0.07)", color: "var(--text-secondary)", border: "1px solid rgba(139,92,246,0.15)" }}>
                        {i === 0 ? "→ original: " : `→ variant ${i}: `}{v}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Top chunks collapsible */}
              <div>
                <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>
                  Top Candidate Passages
                </div>
                {result.top_chunks.map((chunk, i) => (
                  <div key={chunk.doc_id} className="mb-2 rounded-xl overflow-hidden"
                    style={{ border: "1px solid var(--border-subtle)" }}>
                    <button className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-mono text-left transition-all"
                      style={{ background: expandedChunk === i ? "rgba(34,211,238,0.08)" : "rgba(255,255,255,0.02)", color: "var(--text-secondary)" }}
                      onClick={() => setExpandedChunk(expandedChunk === i ? null : i)}>
                      <span>#{i + 1} · {chunk.doc_id} · Score: {chunk.score.toFixed(6)}</span>
                      <span style={{ color: "var(--accent-cyan)" }}>{expandedChunk === i ? "▲" : "▼"}</span>
                    </button>
                    {expandedChunk === i && (
                      <div className="px-4 py-3 text-xs leading-relaxed" style={{ background: "rgba(255,255,255,0.02)", color: "var(--text-secondary)" }}>
                        {chunk.text.slice(0, 400)}{chunk.text.length > 400 ? "…" : ""}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {entry && (
                <p className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
                  Replayed from <span style={{ color: "var(--accent-cyan)" }}>results/demo_bundle.json</span> — captured
                  from a real run of the same pipeline code the backend serves. Start the FastAPI server to query freely.
                </p>
              )}
            </motion.div>
          )}
        </motion.div>
      </div>
    </section>
  );
}
