import json
from collections import Counter
from typing import Dict, List


with open("data/exemplars.json", "r", encoding="utf-8") as exemplars_file:
    FEW_SHOT_EXEMPLARS = json.load(exemplars_file)


def _to_counter(text: str) -> Counter:
    tokens = [
        tok for tok in "".join(
            ch.lower() if ch.isalnum() else " " for ch in text
        ).split()
        if tok
    ]
    return Counter(tokens)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    denom_a = sum(v * v for v in a.values()) ** 0.5
    denom_b = sum(v * v for v in b.values()) ** 0.5
    return num / (denom_a * denom_b) if denom_a and denom_b else 0.0


def select_top_exemplars(question: str, k: int = 3) -> List[Dict]:
    q_vec = _to_counter(question)
    scored = []
    for ex in FEW_SHOT_EXEMPLARS:
        score = _cosine(q_vec, _to_counter(ex["question"]))
        scored.append((score, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _score, ex in scored[:k]]


def build_question_with_exemplars(
    question: str, pruned_schema: Dict, k: int = 3
) -> str:
    exemplars = select_top_exemplars(question, k=k)
    schema_nodes = ", ".join(n["label"] for n in pruned_schema.get("nodes", []))
    schema_edges = ", ".join(e["label"] for e in pruned_schema.get("edges", []))
    lines: List[str] = [
        "Use the following similar examples to guide Cypher generation.",
    ]
    for idx, ex in enumerate(exemplars, start=1):
        lines.append(
            f"Example {idx} - Question: {ex['question']}\n"
            f"Schema: {ex['schema']}\n"
            f"Cypher: {ex['cypher']}\n"
            "----"
        )
    lines.append(
        "Current task:\n"
        f"- Question: {question}\n"
        f"- Relevant node labels: {schema_nodes or 'n/a'}\n"
        f"- Relevant edge labels: {schema_edges or 'n/a'}\n"
        "Return only the Cypher query."
    )
    return "\n".join(lines)


def tokenize_question(text: str) -> Counter:
    normalized = text.lower().strip()
    
    normalized = normalized.replace("were ", " ").replace("was ", " ").replace("are ", " ").replace("is ", " ")
    normalized = normalized.replace(" won ", " win ").replace(" won?", " win?")
    normalized = normalized.replace(" plus ", " and ").replace(" + ", " and ")
    
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    tokens = [
        tok
        for tok in "".join(
            ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in normalized
        ).split()
        if tok and tok not in stop_words
    ]
    return Counter(tokens)


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    denom_a = sum(v * v for v in a.values()) ** 0.5
    denom_b = sum(v * v for v in b.values()) ** 0.5
    return num / (denom_a * denom_b) if denom_a and denom_b else 0.0


def jaccard_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a.keys())
    set_b = set(b.keys())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def combined_similarity(q1: str, q2: str) -> float:
    vec1 = tokenize_question(q1)
    vec2 = tokenize_question(q2)
    cosine = cosine_counter(vec1, vec2)
    jaccard = jaccard_similarity(vec1, vec2)
    return 0.7 * cosine + 0.3 * jaccard
