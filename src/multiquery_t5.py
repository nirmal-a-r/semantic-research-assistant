"""Transformer-based query variant generation (alternative to the rules in multiquery.py)."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROMPTS = [
    "Rewrite the question below in different words. Do not answer it. "
    "The rewrite must still be a question.\nQuestion: {q}\nRewritten question:",
    "Ask the same question using different wording. Do not answer it.\n"
    "Question: {q}\nDifferent wording:",
]

QUESTION_STARTS = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how",
                   "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
                   "would", "name", "in", "on", "at", "during", "according"}


def _clean(text):
    text = " ".join(text.split())
    if text and not text.endswith("?"):
        text += "?"
    return text


def _is_usable(variant, query):
    """Keeps only variants that are still questions about the same thing.

    Small instruction-tuned models frequently answer the question instead of
    rewriting it, and the invented facts in those answers pollute retrieval, so
    anything declarative or carrying new numbers is discarded.
    """
    words = variant.lower().rstrip("?").split()
    if len(words) < 3:
        return False
    if len(set(words)) / len(words) < 0.6:                      # repeated-token collapse
        return False
    if words[0] not in QUESTION_STARTS:                          # answer, not a question
        return False
    query_words = set(query.lower().rstrip("?").split())
    new_numbers = {w for w in words if any(c.isdigit() for c in w)} - query_words
    if new_numbers:                                              # invented facts
        return False
    content = [w for w in words if w not in {"the", "a", "an", "of", "in", "is", "was", "what", "who"}]
    if content and not (set(content) & query_words):
        return False
    return True


class T5QueryExpander:
    """Generates paraphrase variants with a small seq2seq model."""

    def __init__(self, model_name="google/flan-t5-small", device=None, max_new_tokens=32,
                 num_beams=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        # beam search is the dominant cost on CPU, so halve the beams there
        self.num_beams = num_beams or (4 if self.device == "cuda" else 2)
        self.model_name = model_name
        self.is_paraphraser = "paraphras" in model_name.lower()

    def _generate_paraphraser(self, queries, batch_size, n_variants):
        """Models fine-tuned for paraphrasing take the text directly and return
        several rewrites per input, instead of needing one prompt per variant."""
        out = []
        for i in range(0, len(queries), batch_size):
            batch = [f"paraphrase: {q}" for q in queries[i:i + batch_size]]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                ids = self.model.generate(**enc, max_new_tokens=40, num_beams=max(4, n_variants),
                                          num_return_sequences=n_variants,
                                          repetition_penalty=1.1, no_repeat_ngram_size=3)
            out.extend(self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return out

    def _generate(self, prompts, batch_size):
        out = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                ids = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          num_beams=self.num_beams, do_sample=False,
                                          no_repeat_ngram_size=3, repetition_penalty=1.3,
                                          min_new_tokens=4)
            out.extend(self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return out

    def expand(self, queries, n=3, batch_size=32):
        """Returns, for each query, a list of at most n variants starting with the original."""
        per_query = n - 1
        if self.is_paraphraser:
            generated = self._generate_paraphraser(queries, batch_size, per_query)
        else:
            flat = [p.format(q=q) for q in queries for p in PROMPTS[:per_query]]
            generated = self._generate(flat, batch_size)
        results = []
        for qi, query in enumerate(queries):
            variants, seen = [query], {query.lower()}
            for g in generated[qi * per_query:(qi + 1) * per_query]:
                g = _clean(g)
                if g.lower() not in seen and _is_usable(g, query):
                    variants.append(g)
                    seen.add(g.lower())
            results.append(variants[:n])
        return results

    def expand_one(self, query, n=3):
        return self.expand([query], n=n)[0]
