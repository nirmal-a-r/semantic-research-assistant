import re


def sentence_split(text):
    pattern = r"(?<=[.!?])\s+"
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def semantic_adaptive_chunk(text, max_tokens=180, overlap_tokens=40):
    sentences = sentence_split(text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(sent.split())
        if current and current_len + sent_len > max_tokens:
            chunks.append(" ".join(current))
            overlap_sents, overlap_len = [], 0
            for s in reversed(current):
                if overlap_len >= overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s.split())
            current, current_len = overlap_sents, overlap_len
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text]
