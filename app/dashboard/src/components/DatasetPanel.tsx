"use client";
import { motion } from "framer-motion";
import { getHeadlineStats } from "@/lib/data";
import { useProjectData } from "@/lib/ProjectDataContext";

export default function DatasetPanel() {
  const { data } = useProjectData();
  const s = getHeadlineStats(data ?? undefined);

  const statGrid = [
    { label: "Train QA Pairs", value: s.trainPairs.toLocaleString(), color: "var(--accent-cyan)" },
    { label: "Dev QA Pairs", value: s.devPairs.toLocaleString(), color: "var(--accent-violet)" },
    { label: "Total QA Pairs", value: s.totalQAPairs.toLocaleString(), color: "var(--accent-cyan)" },
    { label: "Retrieval Passages", value: s.uniqueContexts.toLocaleString(), color: "var(--accent-amber)" },
    { label: "Adaptive Chunks", value: s.chunks.toLocaleString(), color: "var(--accent-violet)" },
    { label: "Wikipedia Articles", value: s.articles.toLocaleString(), color: "var(--accent-violet)" },
    { label: "Avg Passage Length", value: `~${s.avgContextLen} words`, color: "var(--accent-cyan)" },
  ];

  const trainFrac = (s.trainPairs / s.totalQAPairs) * 100;

  return (
    <section id="dataset" className="py-20 px-4" style={{ background: "rgba(15,20,32,0.5)" }}>
      <div className="max-w-5xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: "var(--accent-cyan)" }}>
            Dataset
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-3" style={{ color: "var(--text-primary)" }}>
            SQuAD v1.1
          </h2>
          <p className="text-base mb-10 max-w-2xl" style={{ color: "var(--text-secondary)" }}>
            Stanford Question Answering Dataset — crowdsourced extractive QA on Wikipedia passages.
            Loaded via <span className="font-mono text-sm" style={{ color: "var(--accent-cyan)" }}>{`datasets.load_dataset("squad")`}</span>.
            License: CC BY-SA 4.0.
          </p>
        </motion.div>

        <div className="flex flex-col lg:flex-row gap-8">
          {/* Stat grid */}
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.1 }}
            className="grid grid-cols-2 md:grid-cols-3 gap-4 flex-1">
            {statGrid.map((stat) => (
              <div key={stat.label} className="glass-card rounded-xl p-4 flex flex-col gap-1"
                style={{ borderColor: "var(--border-subtle)" }}>
                <div className="text-xl md:text-2xl font-mono font-bold" style={{ color: stat.color }}>
                  {stat.value}
                </div>
                <div className="text-xs uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                  {stat.label}
                </div>
              </div>
            ))}
          </motion.div>

          {/* Split bar + info */}
          <motion.div initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }} transition={{ duration: 0.6, delay: 0.2 }}
            className="glass-card rounded-xl p-6 flex flex-col gap-5 w-full lg:w-72 flex-shrink-0"
            style={{ borderColor: "var(--border-subtle)" }}>
            <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Train / Dev Split</div>

            {/* Stacked horizontal bar */}
            <div className="h-8 rounded-lg overflow-hidden flex w-full" style={{ background: "rgba(255,255,255,0.05)" }}>
              <motion.div initial={{ width: 0 }} whileInView={{ width: `${trainFrac}%` }}
                viewport={{ once: true }} transition={{ duration: 1.2, ease: "easeOut" }}
                className="h-full flex items-center justify-center text-xs font-mono font-bold text-white"
                style={{ background: "linear-gradient(90deg,#22D3EE,#0891b2)" }}>
                Train
              </motion.div>
              <div className="h-full flex-1 flex items-center justify-center text-xs font-mono font-bold"
                style={{ color: "var(--accent-violet)", background: "rgba(139,92,246,0.2)" }}>
                Dev
              </div>
            </div>

            <div className="flex justify-between text-xs font-mono" style={{ color: "var(--text-muted)" }}>
              <span><span style={{ color: "var(--accent-cyan)" }}>●</span> Train: {trainFrac.toFixed(1)}%</span>
              <span><span style={{ color: "var(--accent-violet)" }}>●</span> Dev: {(100 - trainFrac).toFixed(1)}%</span>
            </div>

            <div className="border-t pt-4 flex flex-col gap-2" style={{ borderColor: "var(--border-subtle)" }}>
              {[
                { k: "Source", v: "Wikipedia (English)" },
                { k: "Answer type", v: "Extractive span" },
                { k: "Answerable", v: "100% of pairs" },
                { k: "Version", v: "SQuAD v1.1" },
              ].map((r) => (
                <div key={r.k} className="flex justify-between text-xs">
                  <span style={{ color: "var(--text-muted)" }}>{r.k}</span>
                  <span className="font-mono" style={{ color: "var(--text-secondary)" }}>{r.v}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
