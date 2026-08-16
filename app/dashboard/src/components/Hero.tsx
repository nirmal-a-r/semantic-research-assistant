"use client";
import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { getHeadlineStats } from "@/lib/data";
import { useProjectData } from "@/lib/ProjectDataContext";
import { fetchHealth, type HealthResponse } from "@/lib/api";


function CountUp({ target, decimals = 0, suffix = "" }: { target: number; decimals?: number; suffix?: string }) {
  const [value, setValue] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        const duration = 1800;
        const start = performance.now();
        const tick = (now: number) => {
          const progress = Math.min((now - start) / duration, 1);
          const ease = 1 - Math.pow(1 - progress, 3);
          setValue(parseFloat((target * ease).toFixed(decimals)));
          if (progress < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.3 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target, decimals]);
  return <span ref={ref}>{value.toFixed(decimals)}{suffix}</span>;
}

function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let animId: number;
    const resize = () => { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; };
    resize();
    window.addEventListener("resize", resize);
    const pts = Array.from({ length: 70 }, () => ({
      x: Math.random() * canvas.width, y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25,
      r: Math.random() * 1.4 + 0.4,
    }));
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      pts.forEach((p) => {
        p.x = (p.x + p.vx + canvas.width) % canvas.width;
        p.y = (p.y + p.vy + canvas.height) % canvas.height;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(34,211,238,0.22)"; ctx.fill();
      });
      for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y, d = Math.hypot(dx, dy);
        if (d < 110) {
          ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y);
          ctx.strokeStyle = `rgba(34,211,238,${0.07 * (1 - d / 110)})`; ctx.lineWidth = 0.5; ctx.stroke();
        }
      }
      animId = requestAnimationFrame(draw);
    };
    draw();
    return () => { window.removeEventListener("resize", resize); cancelAnimationFrame(animId); };
  }, []);
  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden />;
}

export default function Hero() {
  const { data } = useProjectData();
  const stats = getHeadlineStats(data ?? undefined);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth);
    const id = setInterval(() => fetchHealth().then(setHealth), 15000);
    return () => clearInterval(id);
  }, []);

  const backendOnline = health?.status === "ready";
  // With no backend the site still works — every number and demo answer is
  // precomputed from the real GPU run and shipped in project_data.json.
  const backendLabel = health === null ? "Checking…" : backendOnline ? "Live Backend" : "Precomputed Mode";

  // All metrics in project_data.json are already percentages — render as-is.
  const cards = [
    { label: "Recall@10", value: stats.recall10, decimals: 2, suffix: "%" },
    { label: "Best EM", value: stats.bestEM, decimals: 2, suffix: "%" },
    { label: "Best F1", value: stats.bestF1, decimals: 2, suffix: "%" },
    { label: "QA Pairs", value: stats.totalQAPairs, decimals: 0, suffix: "" },
  ];

  return (
    <section id="hero" className="relative min-h-screen flex flex-col items-center justify-center px-4 py-24 overflow-hidden">
      <ParticleCanvas />
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] rounded-full pointer-events-none"
        style={{ background: "radial-gradient(ellipse,rgba(34,211,238,0.06) 0%,transparent 70%)" }} />

      <div className="relative z-10 flex flex-col items-center gap-6 max-w-5xl mx-auto text-center">
        {/* Status pill */}
        <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
          className="flex items-center gap-2 px-4 py-1.5 rounded-full border text-sm font-mono"
          style={{ borderColor: "var(--border-subtle)", background: "rgba(255,255,255,0.04)" }}>
          <span className={`w-2 h-2 rounded-full ${backendOnline ? "bg-cyan-400" : "bg-amber-400"}`}
            style={backendOnline ? { animation: "pulse 2s infinite", boxShadow: "0 0 8px #22D3EE" } : {}} />
          <span style={{ color: "var(--text-secondary)" }}>
            Fine-tuned · 3 epochs · SQuAD v1.1 ·{" "}
            <span style={{ color: backendOnline ? "var(--accent-cyan)" : "var(--accent-amber)" }}>
              {backendLabel}
            </span>
          </span>
        </motion.div>

        {/* Title */}
        <motion.h1 initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.1 }}
          className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.08]"
          style={{ color: "var(--text-primary)" }}>
          Adaptive Hybrid<br />
          <span className="gradient-text">Semantic Research</span><br />
          Assistant
        </motion.h1>

        {/* Thesis */}
        <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.2 }}
          className="text-lg md:text-xl max-w-2xl font-light" style={{ color: "var(--text-secondary)" }}>
          Retrieval-first, evidence-grounded, explainable QA over SQuAD v1.1 using
          BM25 + Dense hybrid retrieval, Reciprocal Rank Fusion, cross-encoder re-ranking,
          and fine-tuned extractive QA with RoBERTa, BERT, and DistilBERT.
        </motion.p>

        {/* Course + team */}
        <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.3 }}
          className="text-sm font-mono" style={{ color: "var(--text-muted)" }}>
          23AID472 Text Analytics · Batch A-17 ·{" "}
          <span style={{ color: "var(--text-secondary)" }}>A.R. Nirmal · Gunnala Dheeraj Kumar · Meera S Raj · Vishnu Vardhan N</span>
        </motion.p>

        {/* Headline stat row */}
        <motion.div initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.4 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full mt-4">
          {cards.map((c) => (
            <div key={c.label} className="glass-card rounded-xl p-5 flex flex-col items-center gap-1 transition-all duration-300
              hover:border-cyan-500/40 cursor-default"
              style={{ borderColor: "var(--border-subtle)" }}
              onMouseEnter={e => (e.currentTarget.style.boxShadow = "0 0 24px rgba(34,211,238,0.14)")}
              onMouseLeave={e => (e.currentTarget.style.boxShadow = "")}>
              <div className="text-3xl md:text-4xl font-mono font-bold" style={{ color: "var(--accent-cyan)" }}>
                <CountUp target={c.value} decimals={c.decimals} suffix={c.suffix} />
              </div>
              <div className="text-xs uppercase tracking-widest font-mono" style={{ color: "var(--text-muted)" }}>{c.label}</div>
            </div>
          ))}
        </motion.div>

        {/* CTAs */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.55 }}
          className="flex flex-wrap gap-4 justify-center mt-2">
          <a href="#live-demo"
            className="px-7 py-3 rounded-lg text-white font-semibold text-sm transition-opacity hover:opacity-90"
            style={{ background: "linear-gradient(135deg,#22D3EE,#8B5CF6)", boxShadow: "0 0 20px rgba(34,211,238,0.3)" }}>
            Try Live Demo →
          </a>
          <a href="#ablation"
            className="px-7 py-3 rounded-lg font-semibold text-sm transition-all hover:border-white/30"
            style={{ border: "1px solid var(--border-subtle)", color: "var(--text-secondary)" }}>
            View Results
          </a>
          <a href="#architecture"
            className="px-7 py-3 rounded-lg font-semibold text-sm transition-all hover:border-white/30"
            style={{ border: "1px solid var(--border-subtle)", color: "var(--text-secondary)" }}>
            Architecture
          </a>
        </motion.div>
      </div>
    </section>
  );
}
