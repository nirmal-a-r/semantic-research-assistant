"use client";
import Hero from "@/components/Hero";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import DatasetPanel from "@/components/DatasetPanel";
import AblationStudy from "@/components/AblationStudy";
import QAModelComparison from "@/components/QAModelComparison";
import LiveDemo from "@/components/LiveDemo";
import DetailedResults from "@/components/DetailedResults";
import AdaptiveExplainer from "@/components/AdaptiveExplainer";
import TechStack from "@/components/TechStack";
import ResultsDownloads from "@/components/ResultsDownloads";
import Footer from "@/components/Footer";

// Sticky nav
function Nav() {
  const links = [
    { href: "#architecture", label: "Architecture" },
    { href: "#dataset", label: "Dataset" },
    { href: "#ablation", label: "Ablation" },
    { href: "#qa-comparison", label: "QA Models" },
    { href: "#detailed-results", label: "GPU Results" },
    { href: "#live-demo", label: "Demo" },
    { href: "#adaptive", label: "Adaptive" },
    { href: "#downloads", label: "Results" },
  ];
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex justify-center py-3 px-4"
      style={{
        background: "rgba(5,7,13,0.85)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}>
      <div className="flex items-center gap-1 flex-wrap justify-center">
        <span className="text-xs font-mono font-bold mr-3" style={{ color: "var(--accent-cyan)" }}>
          AHRSA
        </span>
        {links.map((l) => (
          <a key={l.href} href={l.href}
            className="px-3 py-1 rounded-lg text-xs font-mono transition-all"
            style={{ color: "var(--text-muted)" }}
            onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text-primary)"; e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; e.currentTarget.style.background = "transparent"; }}>
            {l.label}
          </a>
        ))}
      </div>
    </nav>
  );
}

export default function Home() {
  return (
    <>
      <Nav />
      <main className="pt-10">
        {/* 1. Hero */}
        <Hero />

        {/* 2. Architecture */}
        <ArchitectureDiagram />

        {/* 3. Dataset */}
        <DatasetPanel />

        {/* 4. Progressive Ablation */}
        <AblationStudy />

        {/* 5. QA Model Comparison */}
        <QAModelComparison />

        {/* 5b. Detailed GPU Results (per-prediction) */}
        <DetailedResults />

        {/* 6. Live Inference Demo */}
        <LiveDemo />

        {/* 7. Adaptive Enhancements Explainer */}
        <AdaptiveExplainer />

        {/* 8. Tech Stack */}
        <TechStack />

        {/* 9. Results & Downloads */}
        <ResultsDownloads />
      </main>

      {/* 10. Footer */}
      <Footer />
    </>
  );
}
