// FastAPI client typed wrapper
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * True when the configured backend can plausibly be reached from this page.
 * The deployed static build points at localhost:8000, which only exists when
 * someone is running the backend on their own machine — so when the page is
 * served from a real host we skip the request entirely instead of filling the
 * console with connection-refused errors on every poll.
 */
function backendReachable(): boolean {
  if (typeof window === "undefined") return false;
  const apiIsLocal = /^https?:\/\/(localhost|127\.0\.0\.1)/.test(API_BASE);
  const pageIsLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);
  return !apiIsLocal || pageIsLocal;
}

export type ModelKey = "roberta-base" | "bert-base-uncased" | "distilbert-base-uncased";

export interface EvidenceChunk {
  doc_id: string;
  score: number;
  text: string;
}

export interface MetricExplanation {
  exact_match: boolean | null;
  token_f1: number | null;
  start_confidence: number;
  end_confidence: number;
  avg_confidence: number;
  precision_hint: number | null;
  recall_hint: number | null;
  answer_length_words: number;
  context_length_words: number;
  latency_ms: number;
}

export interface AnswerResponse {
  answer: string;
  answer_start: number;
  answer_end: number;
  evidence_doc_id: string;
  evidence_score: number;
  evidence_text: string;
  top_chunks: EvidenceChunk[];
  query_complexity: string;
  query_variants: string[];
  model_used: string;
  model_params_million: number;
  metrics: MetricExplanation;
}

export interface HealthResponse {
  status: "ready" | "loading" | string;
  error: string | null;
  loaded_models: string[];
  corpus_size: number;
}

export async function fetchHealth(): Promise<HealthResponse | null> {
  if (!backendReachable()) return null;
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchAnswer(
  question: string,
  model: ModelKey = "roberta-base",
  topK?: number
): Promise<AnswerResponse> {
  const res = await fetch(`${API_BASE}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, model, top_k: topK ?? null }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(detail?.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}
