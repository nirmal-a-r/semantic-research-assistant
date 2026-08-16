export default function Footer() {
  return (
    <footer className="py-12 px-4 border-t" style={{ borderColor: "rgba(255,255,255,0.06)", background: "rgba(5,7,13,0.9)" }}>
      <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex flex-col gap-2">
          <div className="font-bold text-base" style={{ color: "var(--text-primary)" }}>
            Adaptive Hybrid Semantic Research Assistant
          </div>
          <div className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
            23AID472 Text Analytics · Batch A-17 · MIT License
          </div>
          <div className="text-xs" style={{ color: "var(--text-muted)" }}>
            A.R. Nirmal · Gunnala Dheeraj Kumar · Meera S Raj · Vishnu Vardhan N
          </div>
        </div>

        <div className="flex flex-col gap-2 items-center md:items-end">
          <a href="https://github.com/nirmal-a-r/semantic-research-assistant"
            target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 rounded-lg font-mono text-sm transition-all"
            style={{
              border: "1px solid var(--border-subtle)",
              color: "var(--text-secondary)",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent-cyan)"; e.currentTarget.style.color = "var(--accent-cyan)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border-subtle)"; e.currentTarget.style.color = "var(--text-secondary)"; }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            GitHub Repository
          </a>
          <a href="https://semantic-research-assistant-a17.vercel.app"
            target="_blank" rel="noopener noreferrer"
            className="text-xs font-mono no-underline transition-colors"
            style={{ color: "var(--accent-cyan)" }}>
            semantic-research-assistant-a17.vercel.app
          </a>
          <div className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
            Dataset: SQuAD v1.1 · CC BY-SA 4.0
          </div>
          <div className="text-xs font-mono" style={{ color: "var(--text-muted)" }}>
            Metrics sourced from <span style={{ color: "var(--accent-cyan)" }}>results/</span> via
            scripts/build_dashboard_data.py
          </div>
        </div>
      </div>
    </footer>
  );
}
