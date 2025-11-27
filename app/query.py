import re
import kuzu
from typing import Dict, List, Tuple
from . import models, exemplars, cache


def validate_cypher(conn: kuzu.Connection, query: str) -> Tuple[bool, str | None]:
    """
    Validate a Cypher query using Kuzu's EXPLAIN.
    Returns:
        (True, None)                 if the query is syntactically valid.
        (False, error_message: str)  otherwise.
    """
    print(f"[Validation] Validating query:\n{query}")
    try:
        conn.execute(f"EXPLAIN {query}")
        return True, None
    except RuntimeError as e:
        return False, str(e)


def postprocess_cypher(query: str, pruned_schema: Dict) -> str:
    """
    Deterministic structural cleanup of LLM-generated Cypher.

    Rules:
    - Normalize whitespace
    - Enforce lowercase string comparisons with toLower()
    - Ensure property-level projection (avoid returning whole nodes)
    - Remove trailing semicolons
    """
    print(f"[Postprocessing] Input query:\n{query}")

    q = query.strip()

    q = re.sub(r';\s*$', '', q)

    comparison_ops = ["CONTAINS", "=", "STARTS WITH", "ENDS WITH"]
    for op in comparison_ops:
        regex = rf"(\w+\.\w+)\s+{op}\s+(['\"])([^'\"]+)\2"
        def repl(m):
            left = m.group(1)
            right = m.group(3).lower()
            return f"toLower({left}) {op} '{right}'"
        q = re.sub(regex, repl, q, flags=re.IGNORECASE)

    def extract_properties(label: str) -> List[str]:
        for node in pruned_schema.get("nodes", []):
            if node["label"] == label:
                return [p["name"] for p in node["properties"]]
        return []

    match = re.search(r"RETURN\s+(.+)$", q, flags=re.IGNORECASE)
    if match:
        return_expr = match.group(1).strip()

        m_simple = re.match(r"([a-zA-Z_]\w*)$", return_expr)
        if m_simple:
            var = m_simple.group(1)

            label_match = re.search(
                rf"{var}:\s*(\w+)",
                q, flags=re.IGNORECASE
            )
            if label_match:
                label = label_match.group(1)
                props = extract_properties(label)
                if props:
                    projected = ", ".join(f"{var}.{p}" for p in props)
                    q = re.sub(r"RETURN\s+.+$", f"RETURN {projected}", q, flags=re.IGNORECASE)

    q = re.sub(r"\s+", " ", q).strip()

    print(f"[Postprocessing] Output query:\n{q}")
    return q


def generate_refined_query(
    conn: kuzu.Connection,
    text2cypher,
    repair_module,
    text2cypher_cache: cache.LRUCache,
    question: str,
    pruned_schema: Dict,
    max_attempts: int = 3,
    use_cache: bool = True,
) -> str:
    """Generate and refine a Cypher query with validation and repair."""
    print("\n[Refinement] -----------------------")
    print(f"[Refinement] Question: {question}")

    selected = exemplars.select_top_exemplars(question)
    print("[Refinement] Top exemplars selected:")
    for idx, ex in enumerate(selected, start=1):
        print(f"  {idx}. Q: {ex['question']}")
        print(f"     Schema: {ex['schema']}")
        print(f"     Cypher: {ex['cypher']}")

    question_payload = exemplars.build_question_with_exemplars(question, pruned_schema)

    q_hash = cache.hash_question(question)
    schema_hash = cache.hash_schema(pruned_schema)
    cache_key = f"{q_hash}:{schema_hash}"

    def _extract_query(val):
        if isinstance(val, dict) and "query" in val:
            return val["query"]
        return val

    if use_cache:
        cached_val = text2cypher_cache.get(cache_key)
        if cached_val:
            print(f"[Refinement] Cache hit for key {cache_key}")
            return _extract_query(cached_val)

        q_vec = exemplars.tokenize_question(question)
        best = (0.0, None, None)
        for k, v in text2cypher_cache.items():
            if not k.endswith(schema_hash):
                continue
            q_text = v["question"] if isinstance(v, dict) and "question" in v else None
            if not q_text:
                continue
            score = exemplars.cosine_counter(q_vec, exemplars.tokenize_question(q_text))
            if score > best[0]:
                best = (score, k, v)
        if best[1] and best[0] >= 0.8:
            print(f"[Refinement] Fuzzy cache hit (score={best[0]:.2f}) from key {best[1]}")
            return _extract_query(best[2])

    print("\n[Refinement] Sending augmented question to Text2Cypher.")
    result = text2cypher(question=question_payload, input_schema=pruned_schema)
    print(f"[Refinement] Raw LLM query:\n{result.query.query}")
    query = postprocess_cypher(result.query.query, pruned_schema)
    print(f"[Refinement] Postprocessed query:\n{query}")

    for attempt in range(1, max_attempts + 1):
        print(f"\n[Refinement] Validation attempt {attempt}/{max_attempts}")
        is_valid, error_msg = validate_cypher(conn, query)

        if is_valid:
            print(f"[Refinement] Query validated on attempt {attempt}.")
            if use_cache:
                text2cypher_cache.put(
                    cache_key,
                    {"query": query, "question": question}
                )
            return query

        print(f"[Refinement] Validation failed on attempt {attempt}.")
        print(f"[Refinement] Error: {error_msg}")

        repaired = repair_module(
            question=question,
            pruned_schema=pruned_schema,
            broken_query=query,
            error_message=error_msg
        ).repaired_query.query

        query = postprocess_cypher(repaired, pruned_schema)
        print(f"[Refinement] Repaired + postprocessed query:\n{query}")

    print("[Refinement] Query still invalid after all attempts.")
    if use_cache:
        text2cypher_cache.put(
            cache_key,
            {"query": query, "question": question}
        )
    return query


def run_query_simple(
    conn: kuzu.Connection,
    question: str,
    input_schema: Dict,
    text2cypher,
    timings: List[Tuple[str, float]] | None = None,
) -> Tuple[str, List | None]:
    import time as _time
    
    t0 = None
    if timings is not None:
        t0 = _time.perf_counter()
    
    print("\n[Simple] Generating query without exemplars, postprocessing, or refinement...")
    print(f"[Simple] Question: {question}")
    
    result = text2cypher(question=question, input_schema=input_schema)
    query = result.query.query.strip()
    
    print(f"[Simple] Generated query:\n{query}")
    
    if timings is not None and t0 is not None:
        t1 = _time.perf_counter()
        timings.append(("text2cypher_simple", t1 - t0))
    
    print(f"\nFinal query:\n{query}")
    t2 = None
    if timings is not None:
        t2 = _time.perf_counter()
    
    try:
        result = conn.execute(query)
        results = [item for row in result for item in row]
    except RuntimeError as e:
        print(f"Error running query: {e}")
        results = None
    
    if timings is not None and t2 is not None:
        t3 = _time.perf_counter()
        timings.append(("db_execute", t3 - t2))
    
    print(f"\nResult:\n{results}")
    return query, results


def run_query(
    conn: kuzu.Connection,
    question: str,
    input_schema: Dict,
    text2cypher,
    repair_module,
    text2cypher_cache: cache.LRUCache,
    timings: List[Tuple[str, float]] | None = None,
    use_cache: bool = True,
) -> Tuple[str, List | None]:
    import time as _time
    
    t0 = None
    if timings is not None:
        t0 = _time.perf_counter()
    
    query = generate_refined_query(
        conn=conn,
        text2cypher=text2cypher,
        repair_module=repair_module,
        text2cypher_cache=text2cypher_cache,
        question=question,
        pruned_schema=input_schema,
        use_cache=use_cache,
    )
    
    if timings is not None and t0 is not None:
        t1 = _time.perf_counter()
        timings.append(("text2cypher_refine", t1 - t0))

    print(f"\nFinal query:\n{query}")
    t2 = None
    if timings is not None:
        t2 = _time.perf_counter()
    
    try:
        result = conn.execute(query)
        results = [item for row in result for item in row]
    except RuntimeError as e:
        print(f"Error running query: {e}")
        results = None
    
    if timings is not None and t2 is not None:
        t3 = _time.perf_counter()
        timings.append(("db_execute", t3 - t2))
    
    print(f"\nResult:\n{results}")
    return query, results
