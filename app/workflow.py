import time
import kuzu
import dspy
from typing import Dict, List, Tuple, Optional
from . import models, database, cache, query


def run_full_workflow(
    conn: kuzu.Connection,
    question: str,
    text2cypher_cache: cache.LRUCache,
    prune_schema_cache: cache.LRUCache,
    use_cache: bool = True,
    use_prune_cache: bool = True,
    use_full_schema_cache: bool = True,
    return_timings: bool = False,
) -> Tuple[str, List | None, Optional[object], List[Tuple[str, float]] | None]:
    """
    Run the complete Graph RAG workflow for a given question:
    1. Prune schema
    2. Generate and refine Cypher query
    3. Execute query and get results
    4. Generate natural language answer
    """
    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)

    timings: List[Tuple[str, float]] = []

    t0 = time.perf_counter()
    input_schema = database.get_full_schema(conn, use_cache=use_full_schema_cache)
    t1 = time.perf_counter()
    timings.append(("schema_fetch", t1 - t0))

    prune = dspy.Predict(models.PruneSchema)
    t2 = time.perf_counter()
    prune_cache_key = None
    if use_prune_cache:
        prune_cache_key = f"{cache.hash_question(question)}:{cache.hash_schema(input_schema)}"
        cached_pruned = prune_schema_cache.get(prune_cache_key)
        if cached_pruned:
            print(f"[PruneCache] Hit for key {prune_cache_key}")
            pruned_schema_q = cached_pruned
            t_cached = time.perf_counter()
            timings.append(("schema_prune_cache_hit", t_cached - t2))
        else:
            prune_result = prune(question=question, input_schema=input_schema)
            pruned_schema_q = prune_result.pruned_schema.model_dump()
            prune_schema_cache.put(prune_cache_key, pruned_schema_q)
            t3 = time.perf_counter()
            timings.append(("schema_prune", t3 - t2))
    else:
        prune_result = prune(question=question, input_schema=input_schema)
        pruned_schema_q = prune_result.pruned_schema.model_dump()
        t3 = time.perf_counter()
        timings.append(("schema_prune", t3 - t2))

    text2cypher = dspy.Predict(models.Text2Cypher)
    repair_module = dspy.Predict(models.RepairQuery)
    
    cypher_query, context = query.run_query(
        conn=conn,
        question=question,
        input_schema=pruned_schema_q,
        text2cypher=text2cypher,
        repair_module=repair_module,
        text2cypher_cache=text2cypher_cache,
        timings=timings,
        use_cache=use_cache,
    )

    answer_generator = dspy.ChainOfThought(models.AnswerQuestion)

    t4 = time.perf_counter()
    if context is None:
        print("Empty results obtained from the graph database. Please retry with a different question.")
        answer = None
    else:
        answer = answer_generator(
            question=question,
            cypher_query=cypher_query,
            context=str(context)
        )
        print(f"\nAnswer: {answer.response}\n")
    t5 = time.perf_counter()
    timings.append(("answer_generation", t5 - t4))

    print("=" * 80)
    print()

    if return_timings:
        return cypher_query, context, answer, timings
    return cypher_query, context, answer


def run_full_workflow_simple(
    conn: kuzu.Connection,
    question: str,
    prune_schema_cache: cache.LRUCache,
    use_prune_cache: bool = True,
    use_full_schema_cache: bool = True,
    return_timings: bool = False,
) -> Tuple[str, List | None, Optional[object], List[Tuple[str, float]] | None]:
    """
    Run the complete Graph RAG workflow for a given question WITHOUT:
    - Exemplars
    - Postprocessing
    - Refinement loop
    """
    print("=" * 80)
    print(f"Question: {question} [SIMPLE MODE]")
    print("=" * 80)

    timings: List[Tuple[str, float]] = []

    t0 = time.perf_counter()
    input_schema = database.get_full_schema(conn, use_cache=use_full_schema_cache)
    t1 = time.perf_counter()
    timings.append(("schema_fetch", t1 - t0))

    prune = dspy.Predict(models.PruneSchema)
    t2 = time.perf_counter()
    prune_cache_key = None
    if use_prune_cache:
        prune_cache_key = f"{cache.hash_question(question)}:{cache.hash_schema(input_schema)}"
        cached_pruned = prune_schema_cache.get(prune_cache_key)
        if cached_pruned:
            print(f"[PruneCache] Hit for key {prune_cache_key}")
            pruned_schema_q = cached_pruned
            t_cached = time.perf_counter()
            timings.append(("schema_prune_cache_hit", t_cached - t2))
        else:
            prune_result = prune(question=question, input_schema=input_schema)
            pruned_schema_q = prune_result.pruned_schema.model_dump()
            prune_schema_cache.put(prune_cache_key, pruned_schema_q)
            t3 = time.perf_counter()
            timings.append(("schema_prune", t3 - t2))
    else:
        prune_result = prune(question=question, input_schema=input_schema)
        pruned_schema_q = prune_result.pruned_schema.model_dump()
        t3 = time.perf_counter()
        timings.append(("schema_prune", t3 - t2))

    text2cypher = dspy.Predict(models.Text2Cypher)
    
    cypher_query, context = query.run_query_simple(
        conn=conn,
        question=question,
        input_schema=pruned_schema_q,
        text2cypher=text2cypher,
        timings=timings,
    )

    answer_generator = dspy.ChainOfThought(models.AnswerQuestion)

    t4 = time.perf_counter()
    if context is None:
        print("Empty results obtained from the graph database. Please retry with a different question.")
        answer = None
    else:
        answer = answer_generator(
            question=question,
            cypher_query=cypher_query,
            context=str(context)
        )
        print(f"\nAnswer: {answer.response}\n")
    t5 = time.perf_counter()
    timings.append(("answer_generation", t5 - t4))

    print("=" * 80)
    print()

    if return_timings:
        return cypher_query, context, answer, timings
    return cypher_query, context, answer
