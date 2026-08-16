COMPARISON_WORDS = {"compare", "versus", "vs", "difference", "better", "than", "both"}
MULTISTEP_HINTS = {"and then", "after", "before", "because", "why did", "how did", "led to"}


def estimate_query_complexity(query):
    q = query.lower()
    if any(h in q for h in MULTISTEP_HINTS) or q.count(",") >= 2:
        return "multistep"
    if any(w in q.split() for w in COMPARISON_WORDS):
        return "comparison"
    return "simple"


def generate_query_variants(query, n=3):
    variants = [query]
    if query.lower().startswith("what is"):
        variants.append(query[7:].strip().rstrip("?") + " definition?")
    if query.lower().startswith("who"):
        variants.append(query.replace("Who", "Which person", 1))
    variants.append("Regarding " + query[0].lower() + query[1:])
    seen, uniq = set(), []
    for v in variants:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq[:n]
