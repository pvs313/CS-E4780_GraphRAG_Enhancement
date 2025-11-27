import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Graph RAG workflow using Kuzu, DSPy and marimo
    In this notebook, we'll show how to build a Graph RAG workflow that leverages a DSPy pipeline on top of Kuzu. The retrieval workflow uses the following steps:

    1. Get the schema (representing the data model) from the graph database
    2. Prune the schema based on the question asked by the user, in alignment with the schema
    3. Run Text2Cypher, where a LM generates a valid Cypher query
    4. Use the LM-generated Cypher query to retieve results from the Kuzu database
    5. Pass the retrieved results as context to another LM that answers the user's question in natural language
    """
    )
    return


@app.cell
def _():
    import re
    from typing import Dict, List
    import hashlib
    import json
    import time
    from collections import Counter, OrderedDict
    return Counter, Dict, List, OrderedDict, hashlib, json, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Walkthrough

    The cells below showcase the methodology and go through the code in more detail.
    First, lets create a Kuzu database connection.
    """
    )
    return


@app.cell
def _(kuzu):
    db_name = "nobel.kuzu"
    db = kuzu.Database(db_name, read_only=True)
    conn = kuzu.Connection(db)
    return (conn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Get graph schema
    To get the LM to write the correct Cypher query that answers the question, it's important to obtain the schema from the graph. The schema informs the LM what nodes and edges exist, and what properties it can query on to answer the question.
    """
    )
    return


@app.cell
def _(kuzu):
    def get_schema_dict(conn:kuzu.Connection) -> dict[str, list[dict]]:
        response = conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
        nodes = [row[1] for row in response]  # type: ignore
        response = conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
        rel_tables = [row[1] for row in response]  # type: ignore
        relationships = []
        for tbl_name in rel_tables:
            response = conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
            for row in response:
                relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})  # type: ignore
        schema = {"nodes": [], "edges": []}

        for node in nodes:
            node_schema = {"label": node, "properties": []}
            node_properties = conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
            for row in node_properties:  # type: ignore
                node_schema["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
            schema["nodes"].append(node_schema)

        for rel in relationships:
            edge = {
                "label": rel["name"],
                "from": rel["from"],
                "to": rel["to"],
                "properties": [],
            }
            rel_properties = conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
            for row in rel_properties:  # type: ignore
                edge["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
            schema["edges"].append(edge)
        return schema
    return (get_schema_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Below is a helper function to display the schema so that it's easier to read. A sample of the full graph schema is shown immediately after.""")
    return


@app.function
def display_schema(schema: dict[str, list[dict]]) -> None:
    for item in schema.items():
        for sub_item in item[1]:
            print(sub_item)


@app.cell
def _(conn, get_schema_dict):
    full_schema = get_schema_dict(conn)
    display_schema(full_schema)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""By default, the full schema can be quite verbose and complex, so pruning it can help narrow down the context for the LM to better interpret in a way that aligns with the question.""")
    return


@app.cell
def _(dspy, load_dotenv, os):
    load_dotenv()

    API_KEY = os.environ.get("API_KEY")

    # Using OpenRouter. Switch to another LLM provider as needed
    # we recommend gemini-2.0-flash for the cost-efficiency
    lm = dspy.LM(
            model="openrouter/google/gemini-2.0-flash-001",
            api_base="https://openrouter.ai/api/v1",
            api_key=API_KEY,
        )
    dspy.configure(lm=lm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define data models
    Let's first define Pydantic data models that represent the graph schema and the generated Cypher query in a structured form with type validation.
    """
    )
    return


@app.cell
def _(BaseModel, Field):
    class Query(BaseModel):
        query: str = Field(description="Valid Cypher query with no newlines")

    class Property(BaseModel):
        name: str
        type: str = Field(description="Data type of the property")

    class Node(BaseModel):
        label: str
        properties: list[Property] | None

    class Edge(BaseModel):
        label: str = Field(description="Relationship label")
        from_: Node = Field(alias="from", description="Source node label")
        to: Node = Field(alias="from", description="Target node label")
        properties: list[Property] | None

    class GraphSchema(BaseModel):
        nodes: list[Node]
        edges: list[Edge]
    return GraphSchema, Query


@app.cell
def _(mo):
    mo.md(
        r"""
    ## DSPy Signature for schema pruning
    The following signature describes the logic for pruning schema by conditioning the output on the given question from the user. The output is a much smaller, context-rich schema for the Cypher-generating LM downstream.
    """
    )
    return


@app.cell
def _(GraphSchema, dspy):
    class PruneSchema(dspy.Signature):
        """
        Understand the given labelled property graph schema and the given user question. Your task
        is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
        relevant to the question.
          - The schema is a list of nodes and edges in a property graph.
          - The nodes are the entities in the graph.
          - The edges are the relationships between the nodes.
          - Properties of nodes and edges are their attributes, which helps answer the question.
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        pruned_schema: GraphSchema = dspy.OutputField()
    return (PruneSchema,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example question
    We can define the following question to demonstrate the sequence of steps.
    """
    )
    return


@app.cell
def _(mo):
    sample_question_ui = mo.ui.text(value="Which scholars won prizes in Physics and were affiliated with University of Cambridge?", full_width=True)
    return (sample_question_ui,)


@app.cell
def _(sample_question_ui):
    sample_question_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Use DSPy's Predict Module
    DSPY's simplest module is `Predict`, which produces a prediction, given a prompt as input. The prompt is auto-generated by DSPy and uses information that we declared in the Signature.
    """
    )
    return


@app.cell
def _(PruneSchema, conn, dspy, get_schema_dict, sample_question_ui):
    # Get input schema
    input_schema = get_schema_dict(conn)
    sample_question = sample_question_ui.value

    # Run Module
    prune = dspy.Predict(PruneSchema)
    r = prune(question=sample_question, input_schema=input_schema)
    pruned_schema = r.pruned_schema.model_dump()

    # Display each item for easier understanding
    display_schema(pruned_schema)
    return pruned_schema, sample_question


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can see that the returned schema is much more concise and useful for the question that was asked. Nice!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## DSPy Signature for Text2Cypher
    The Text2Cypher stage uses the pruned schema from the previous step and a list of domain-specific instructions to generate a valid Cypher query (that as far as possible, respects the schema and correctly retrieves from the graph database).
    """
    )
    return


@app.cell
def _(Query, dspy):
    class Text2Cypher(dspy.Signature):
        """
        Translate the question into a valid Cypher query that respects the graph schema.

        <SYNTAX>
        - When matching on Scholar names, ALWAYS match on the `knownName` property
        - For countries, cities, continents and institutions, you can match on the `name` property
        - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
        - Always strive to respect the relationship direction (FROM/TO) using the schema information.
        - When comparing string properties, ALWAYS do the following:
          - Lowercase the property values before comparison
          - Use the WHERE clause
          - Use the CONTAINS operator to check for presence of one substring in the other
        - DO NOT use APOC as the database does not support it.
        </SYNTAX>

        <RETURN_RESULTS>
        - If the result is an integer, return it as an integer (not a string).
        - When returning results, return property values rather than the entire node or relationship.
        - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
        - NO Cypher keywords should be returned by your query.
        </RETURN_RESULTS>
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        query: Query = dspy.OutputField()
    return (Text2Cypher,)


@app.cell
def _(Counter, json):
    # Load exemplars from external file
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

    def select_top_exemplars(question: str, k: int = 3) -> list[dict]:
        q_vec = _to_counter(question)
        scored = []
        for ex in FEW_SHOT_EXEMPLARS:
            score = _cosine(q_vec, _to_counter(ex["question"]))
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _score, ex in scored[:k]]

    def build_question_with_exemplars(
        question: str, pruned_schema: dict, k: int = 3
    ) -> str:
        exemplars = select_top_exemplars(question, k=k)
        schema_nodes = ", ".join(n["label"] for n in pruned_schema.get("nodes", []))
        schema_edges = ", ".join(e["label"] for e in pruned_schema.get("edges", []))
        lines: list[str] = [
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

    return (
        FEW_SHOT_EXEMPLARS,
        build_question_with_exemplars,
        select_top_exemplars,
    )


@app.cell
def _(kuzu):
    def validate_cypher(conn: kuzu.Connection, query: str) -> tuple[bool, str | None]:
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
    return (validate_cypher,)


@app.cell
def _(get_schema_dict, hash_schema):
    _full_schema_store = {"schema": None, "hash": None}

    def get_full_schema(conn, use_cache: bool = True) -> dict:
        if use_cache and _full_schema_store["schema"] is not None:
            return _full_schema_store["schema"]  # type: ignore
        schema = get_schema_dict(conn)
        if use_cache:
            _full_schema_store["schema"] = schema
            _full_schema_store["hash"] = hash_schema(schema)
        return schema

    def get_full_schema_hash() -> str | None:
        return _full_schema_store["hash"]  # type: ignore

    def clear_full_schema_cache() -> None:
        _full_schema_store["schema"] = None
        _full_schema_store["hash"] = None

    return clear_full_schema_cache, get_full_schema


@app.cell
def _(Query, dspy):
    class RepairQuery(dspy.Signature):
        """
        Attempt to fix a Cypher query that failed validation.

        - Keep the structure and original intent of the query.
        - Only repair syntax errors or invalid label/property names.
        - Do NOT rewrite the entire query unless necessary.
        """

        question: str = dspy.InputField()
        pruned_schema: str = dspy.InputField()
        broken_query: str = dspy.InputField()
        error_message: str = dspy.InputField()

        repaired_query: Query = dspy.OutputField()
    return (RepairQuery,)


@app.cell
def _(OrderedDict, hashlib, json):
    def stable_hash_payload(obj) -> str:
        payload = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def hash_question(question: str) -> str:
        return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()

    def hash_schema(schema: dict) -> str:
        return stable_hash_payload(schema)

    class LRUCache:
        """Simple LRU cache for string-keyed items."""

        def __init__(self, maxsize: int = 128):
            self.maxsize = maxsize
            self._store: OrderedDict[str, object] = OrderedDict()

        def get(self, key: str):
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            return None

        def put(self, key: str, value: object) -> None:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self.maxsize:
                self._store.popitem(last=False)

        def items(self):
            return list(self._store.items())

        def clear(self) -> None:
            """Remove all cached entries."""
            self._store.clear()

    def print_timing_table(timings: list[tuple[str, float]]) -> None:
        if not timings:
            return
        print("\n[Timings] Stage breakdown (ms):")
        for name, duration in timings:
            print(f"  - {name:<24} {duration*1000:.2f} ms")
        total = sum(d for _, d in timings)
        print(f"  - {'total':<24} {total*1000:.2f} ms")
        print()

    return LRUCache, hash_question, hash_schema, print_timing_table


@app.cell
def _(LRUCache):
    # Cache for Text2Cypher results
    text2cypher_cache = LRUCache(maxsize=128)
    # Cache for pruned schema to avoid repeated pruning work
    prune_schema_cache = LRUCache(maxsize=128)
    return prune_schema_cache, text2cypher_cache


@app.cell
def _(Counter):
    def tokenize_question(text: str) -> Counter:
        """Lowercase alnum tokenizer for cache similarity."""
        return Counter(
            tok
            for tok in "".join(
                ch.lower() if ch.isalnum() else " " for ch in text
            ).split()
            if tok
        )

    def cosine_counter(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        denom_a = sum(v * v for v in a.values()) ** 0.5
        denom_b = sum(v * v for v in b.values()) ** 0.5
        return num / (denom_a * denom_b) if denom_a and denom_b else 0.0

    return cosine_counter, tokenize_question


@app.cell
def _(Dict, List, re):
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

        # Remove trailing semicolon
        q = re.sub(r';\s*$', '', q)

        # Enforce lowercase comparisons 
        comparison_ops = ["CONTAINS", "=", "STARTS WITH", "ENDS WITH"]
        for op in comparison_ops:
            regex = rf"(\w+\.\w+)\s+{op}\s+(['\"])([^'\"]+)\2"
            def repl(m):
                left = m.group(1)
                right = m.group(2).lower()
                return f"toLower({left}) {op} '{right}'"
            q = re.sub(regex, repl, q, flags=re.IGNORECASE)

        # Ensure RETURN clause uses property projection 
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
    return (postprocess_cypher,)


@app.cell
def _(postprocess_cypher, validate_cypher):
    def generate_refined_query(
        conn,
        text2cypher,
        repair_module,
        build_question_with_exemplars,
        select_top_exemplars,
        cosine_counter,
        tokenize_question,
        text2cypher_cache,
        hash_question,
        hash_schema,
        question,
        pruned_schema,
        max_attempts=3,
        use_cache: bool = True,
    ):
        print("\n[Refinement] -----------------------")
        print(f"[Refinement] Question: {question}")

        selected = select_top_exemplars(question)
        print("[Refinement] Top exemplars selected:")
        for idx, ex in enumerate(selected, start=1):
            print(f"  {idx}. Q: {ex['question']}")
            print(f"     Schema: {ex['schema']}")
            print(f"     Cypher: {ex['cypher']}")

        question_payload = build_question_with_exemplars(question, pruned_schema)

        q_hash = hash_question(question)
        schema_hash = hash_schema(pruned_schema)
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

            q_vec = tokenize_question(question)
            best = (0.0, None, None) 
            for k, v in text2cypher_cache.items():
                if not k.endswith(schema_hash):
                    continue
                q_text = v["question"] if isinstance(v, dict) and "question" in v else None
                if not q_text:
                    continue
                score = cosine_counter(q_vec, tokenize_question(q_text))
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

    return (generate_refined_query,)


@app.cell
def _(
    RepairQuery,
    Text2Cypher,
    build_question_with_exemplars,
    dspy,
    pruned_schema,
    sample_question,
):
    text2cypher = dspy.Predict(Text2Cypher)
    ####
    repair_module = dspy.Predict(RepairQuery)

    question_payload = build_question_with_exemplars(sample_question, pruned_schema)
    text2cypher_result = text2cypher(question=question_payload, input_schema=pruned_schema)
    cypher_query = text2cypher_result.query.query
    cypher_query
    return repair_module, text2cypher


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Run the Cypher query on the database

    Next, we run the Cypher query on the database.

    Depending on the complexity of the question and the LM's knowledge of Cypher, the query may or may not be correct. What constitutes a "correct" query can be thought of in two ways:
    - Syntax: Does the Cypher query even compile and is it valid?
    - Semantics: Does the query actually retrieve the right data, and has it interpreted the direction of the relationship correctly?

    ### Agent workflows can help!
    In the real world, this sort of Text2Cypher workflow would need fallbacks and some degree of query rewriting to be more robust to failure. However, for this demo, we can see that for even some non-trivial questions, the queries returned are really good in many cases.
    """
    )
    return


@app.cell
def _(
    build_question_with_exemplars,
    cosine_counter,
    generate_refined_query,
    hash_question,
    hash_schema,
    kuzu,
    select_top_exemplars,
    text2cypher,
    text2cypher_cache,
    tokenize_question,
):
    def run_query(
        conn: kuzu.Connection, 
        question: str, 
        input_schema: str,
        repair_module,
        timings: list[tuple[str, float]] | None = None,
        use_cache: bool = True,
    ):
        """
        Run a Cypher query on the Kuzu database and gather the results.
        Uses the refined query generation with validation and repair loop.

        Args:
            conn: Kuzu database connection
            question: User question
            input_schema: Pruned schema for the question
            repair_module: The repair module for fixing queries
        """
        # Use refined query generation with validation and repair
        t0 = None
        if timings is not None:
            import time as _time
            t0 = _time.perf_counter()
        query = generate_refined_query(
            conn=conn,
            text2cypher=text2cypher,
            repair_module=repair_module,
            build_question_with_exemplars=build_question_with_exemplars,
            select_top_exemplars=select_top_exemplars,
            cosine_counter=cosine_counter,
            tokenize_question=tokenize_question,
            text2cypher_cache=text2cypher_cache,
            hash_question=hash_question,
            hash_schema=hash_schema,
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
            # Run the query on the database
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

    def generate_simple_query(
        text2cypher,
        question: str,
        pruned_schema: dict,
    ):
        """
        Simple query generation without exemplars, postprocessing, or refinement loop.
        Just generates the query directly from the question and schema.
        """
        print("\n[Simple] Generating query without exemplars, postprocessing, or refinement...")
        print(f"[Simple] Question: {question}")

        result = text2cypher(question=question, input_schema=pruned_schema)
        query = result.query.query.strip()

        print(f"[Simple] Generated query:\n{query}")
        return query

    def run_query_simple(
        conn: kuzu.Connection,
        question: str,
        input_schema: str,
        text2cypher,
        timings: list[tuple[str, float]] | None = None,
    ):
        """
        Run a Cypher query using simple generation (no exemplars, postprocessing, or refinement).

        Args:
            conn: Kuzu database connection
            question: User question
            input_schema: Pruned schema for the question
            text2cypher: The Text2Cypher module
        """
        t0 = None
        if timings is not None:
            import time as _time
            t0 = _time.perf_counter()

        query = generate_simple_query(
            text2cypher=text2cypher,
            question=question,
            pruned_schema=input_schema,
        )

        if timings is not None and t0 is not None:
            t1 = _time.perf_counter()
            timings.append(("text2cypher_simple", t1 - t0))

        print(f"\nFinal query:\n{query}")
        t2 = None
        if timings is not None:
            t2 = _time.perf_counter()
        try:
            # Run the query on the database
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
    return run_query, run_query_simple


@app.cell
def _(mo):
    mo.md(
        r"""
    ## DSPy Signature for answer generation
    We now need another DSPy signature for generating an answer in natural langauge from the given result as context.
    """
    )
    return


@app.cell
def _(dspy):
    class AnswerQuestion(dspy.Signature):
        """
        - Use the provided question, the generated Cypher query and the context to answer the question.
        - If the context is empty, state that you don't have enough information to answer the question.
        """

        question: str = dspy.InputField()
        cypher_query: str = dspy.InputField()
        context: str = dspy.InputField()
        response: str = dspy.OutputField()
    return (AnswerQuestion,)


@app.cell
def _(
    AnswerQuestion,
    conn,
    dspy,
    pruned_schema,
    repair_module,
    run_query,
    sample_question,
):
    answer_generator = dspy.ChainOfThought(AnswerQuestion)

    # Use run_query with refined workflow
    query, context = run_query(
        conn=conn,
        question=sample_question,
        input_schema=pruned_schema,
        repair_module=repair_module
    )

    if context is None:
        print("Empty results obtained from the graph database. Please retry with a different question.")
    else:
        answer = answer_generator(
            question=sample_question, cypher_query=query, context=str(context)
        )
        print(f"\n{answer}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The query successfully retrieves data from the Kuzu database, and a `ChainOfThought` module in DSPy is called to reason over this context, to generate an answer in natural language. It's also possible to use a simple `Predict` module to achieve the same outcome. The idea behind this example is that it's quite simple and straightforward to begin ideating and testing your ideas in code, using marimo notebooks in this way.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conclusions
    This demo showed how to build a multi-stage AI workflow for Graph RAG using Kuzu and marimo. The self-refinement loop helps ensure that generated Cypher queries are syntactically valid before execution. Give it a try and write your own queries to explore the data further!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Testing with Multiple Questions

    Let's test the Graph RAG workflow with multiple questions to see how well it performs across different query types.
    """
    )
    return


@app.cell
def _(
    AnswerQuestion,
    PruneSchema,
    conn,
    dspy,
    get_full_schema,
    hash_question,
    hash_schema,
    print_timing_table,
    prune_schema_cache,
    repair_module,
    run_query,
    run_query_simple,
    text2cypher,
):
    def run_full_workflow(
        question: str,
        use_cache: bool = True,
        use_prune_cache: bool = True,
        use_full_schema_cache: bool = True,
        return_timings: bool = False,
    ):
        """
        Run the complete Graph RAG workflow for a given question:
        1. Prune schema
        2. Generate and refine Cypher query
        3. Execute query and get results
        4. Generate natural language answer
        """
        import time as _time
        print("=" * 80)
        print(f"Question: {question}")
        print("=" * 80)

        timings: list[tuple[str, float]] = []

        # Step 1: Prune schema
        t0 = _time.perf_counter()
        input_schema = get_full_schema(conn, use_cache=use_full_schema_cache)
        t1 = _time.perf_counter()
        timings.append(("schema_fetch", t1 - t0))

        prune = dspy.Predict(PruneSchema)
        t2 = _time.perf_counter()
        prune_cache_key = None
        if use_prune_cache:
            prune_cache_key = f"{hash_question(question)}:{hash_schema(input_schema)}"
            cached_pruned = prune_schema_cache.get(prune_cache_key)
            if cached_pruned:
                print(f"[PruneCache] Hit for key {prune_cache_key}")
                pruned_schema_q = cached_pruned
                t_cached = _time.perf_counter()
                timings.append(("schema_prune_cache_hit", t_cached - t2))
            else:
                prune_result = prune(question=question, input_schema=input_schema)
                pruned_schema_q = prune_result.pruned_schema.model_dump()
                prune_schema_cache.put(prune_cache_key, pruned_schema_q)
                t3 = _time.perf_counter()
                timings.append(("schema_prune", t3 - t2))
        else:
            prune_result = prune(question=question, input_schema=input_schema)
            pruned_schema_q = prune_result.pruned_schema.model_dump()
            t3 = _time.perf_counter()
            timings.append(("schema_prune", t3 - t2))

        # Step 2 & 3: Generate, refine, and execute query using run_query
        query, context = run_query(
            conn=conn,
            question=question,
            input_schema=pruned_schema_q,
            repair_module=repair_module,
            timings=timings,
            use_cache=use_cache,
        )

        # Step 4: Generate answer
        answer_generator = dspy.ChainOfThought(AnswerQuestion)

        t4 = _time.perf_counter()
        if context is None:
            print("Empty results obtained from the graph database. Please retry with a different question.")
            answer = None
        else:
            answer = answer_generator(
                question=question,
                cypher_query=query,
                context=str(context)
            )
            print(f"\nAnswer: {answer.response}\n")
        t5 = _time.perf_counter()
        timings.append(("answer_generation", t5 - t4))

        print("=" * 80)
        print()

        print_timing_table(timings)

        if return_timings:
            return query, context, answer, timings
        return query, context, answer

    def run_full_workflow_simple(
        question: str,
        use_prune_cache: bool = True,
        use_full_schema_cache: bool = True,
        return_timings: bool = False,
    ):
        """
        Run the complete Graph RAG workflow for a given question WITHOUT:
        - Exemplars
        - Postprocessing
        - Refinement loop

        Steps:
        1. Prune schema
        2. Generate Cypher query (simple, no refinement)
        3. Execute query and get results
        4. Generate natural language answer
        """
        import time as _time
        print("=" * 80)
        print(f"Question: {question} [SIMPLE MODE]")
        print("=" * 80)

        timings: list[tuple[str, float]] = []

        # Step 1: Prune schema
        t0 = _time.perf_counter()
        input_schema = get_full_schema(conn, use_cache=use_full_schema_cache)
        t1 = _time.perf_counter()
        timings.append(("schema_fetch", t1 - t0))

        prune = dspy.Predict(PruneSchema)
        t2 = _time.perf_counter()
        prune_cache_key = None
        if use_prune_cache:
            prune_cache_key = f"{hash_question(question)}:{hash_schema(input_schema)}"
            cached_pruned = prune_schema_cache.get(prune_cache_key)
            if cached_pruned:
                print(f"[PruneCache] Hit for key {prune_cache_key}")
                pruned_schema_q = cached_pruned
                t_cached = _time.perf_counter()
                timings.append(("schema_prune_cache_hit", t_cached - t2))
            else:
                prune_result = prune(question=question, input_schema=input_schema)
                pruned_schema_q = prune_result.pruned_schema.model_dump()
                prune_schema_cache.put(prune_cache_key, pruned_schema_q)
                t3 = _time.perf_counter()
                timings.append(("schema_prune", t3 - t2))
        else:
            prune_result = prune(question=question, input_schema=input_schema)
            pruned_schema_q = prune_result.pruned_schema.model_dump()
            t3 = _time.perf_counter()
            timings.append(("schema_prune", t3 - t2))

        # Step 2 & 3: Generate and execute query using simple method
        query, context = run_query_simple(
            conn=conn,
            question=question,
            input_schema=pruned_schema_q,
            text2cypher=text2cypher,
            timings=timings,
        )

        # Step 4: Generate answer
        answer_generator = dspy.ChainOfThought(AnswerQuestion)

        t4 = _time.perf_counter()
        if context is None:
            print("Empty results obtained from the graph database. Please retry with a different question.")
            answer = None
        else:
            answer = answer_generator(
                question=question,
                cypher_query=query,
                context=str(context)
            )
            print(f"\nAnswer: {answer.response}\n")
        t5 = _time.perf_counter()
        timings.append(("answer_generation", t5 - t4))

        print("=" * 80)
        print()

        print_timing_table(timings)

        if return_timings:
            return query, context, answer, timings
        return query, context, answer

    return run_full_workflow, run_full_workflow_simple


@app.cell
def _(FEW_SHOT_EXEMPLARS, run_full_workflow):
    """
    Run the full workflow against every few-shot exemplar question to sanity-check
    generation, validation, and execution.
    """
    def run_exemplar_regression():
        print("\n=== Running regression over few-shot exemplar questions ===\n")
        regression_results_local = []
        for ex in FEW_SHOT_EXEMPLARS:
            print(f"\n[Exemplar Test] Question: {ex['question']}")
            q, ctx, ans, timings = run_full_workflow(
                ex["question"], return_timings=True
            )
            # Pretty timings summary
            try:
                from pprint import pprint
                pprint(timings)
            except Exception:
                pass
            regression_results_local.append(
                {
                    "question": ex["question"],
                    "query": q,
                    "context": ctx,
                    "answer": getattr(ans, "response", None) if ans else None,
                    "timings": timings,
                }
            )
        print("\n=== Completed exemplar regression ===\n")
        return regression_results_local
    return


@app.cell
def _():
    import os
    import marimo as mo
    import kuzu
    import dspy
    from typing import Any
    from pydantic import BaseModel, Field
    from dotenv import load_dotenv
    return BaseModel, Field, dspy, kuzu, load_dotenv, mo, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Accuracy Evaluation

    Let's evaluate the accuracy of the Graph RAG workflow using a test set of questions with expected answers.
    We'll run each question without caching and verify that the generated answers contain the expected strings.
    """
    )
    return


@app.cell
def _(json, run_full_workflow, run_full_workflow_simple):
    # Load test questions
    with open("data/test_questions.json", "r", encoding="utf-8") as test_file:
        test_questions = json.load(test_file)


    def check_answer_accuracy(answer_text: str, expected_strings: list[str]) -> dict:
        if not answer_text:
            return {
                "contains_any": False,
                "contains_all": False,
                "matched_strings": [],
                "missing_strings": expected_strings,
                "match_count": 0,
                "total_expected": len(expected_strings)
            }

        answer_lower = answer_text.lower()
        matched = []
        missing = []

        for expected in expected_strings:
            # Check if expected string (case-insensitive) appears in answer
            if expected.lower() in answer_lower:
                matched.append(expected)
            else:
                missing.append(expected)

        return {
            "contains_any": len(matched) > 0,
            "contains_all": len(matched) == len(expected_strings),
            "matched_strings": matched,
            "missing_strings": missing,
            "match_count": len(matched),
            "total_expected": len(expected_strings),
            "accuracy": len(matched) / len(expected_strings) if expected_strings else 0.0
        }

    def evaluate_accuracy(use_cache: bool = False, use_prune_cache: bool = False):
        print("\n" + "=" * 80)
        print("ACCURACY EVALUATION")
        print("=" * 80)
        print(f"Running {len(test_questions)} test questions (caching disabled)...\n")

        results = []
        total_accuracy = 0.0

        for i, test_item in enumerate(test_questions, 1):
            question = test_item["question"]
            expected_strings = test_item["expected_strings"]

            print(f"[{i}/{len(test_questions)}] Question: {question}")
            print(f"  Expected strings: {len(expected_strings)} items")

            # Run workflow without caching
            query, context, answer = run_full_workflow(
                question=question,
                use_cache=use_cache,
                use_prune_cache=use_prune_cache,
                use_full_schema_cache=False,
                return_timings=False
            )

            # Extract answer text
            answer_text = getattr(answer, "response", None) if answer else None
            if answer_text is None:
                answer_text = ""

            # Check accuracy
            accuracy_result = check_answer_accuracy(answer_text, expected_strings)
            results.append({
                "question": question,
                "query": query,
                "answer": answer_text,
                "expected_strings": expected_strings,
                "accuracy_result": accuracy_result
            })

            total_accuracy += accuracy_result["accuracy"]

            # Print result
            print(f"  Matched: {accuracy_result['match_count']}/{accuracy_result['total_expected']} strings")
            print(f"  Accuracy: {accuracy_result['accuracy']:.2%}")
            if accuracy_result['matched_strings']:
                print(f"  Found: {', '.join(accuracy_result['matched_strings'][:3])}{'...' if len(accuracy_result['matched_strings']) > 3 else ''}")
            if accuracy_result['missing_strings']:
                print(f"  Missing: {', '.join(accuracy_result['missing_strings'][:3])}{'...' if len(accuracy_result['missing_strings']) > 3 else ''}")
            print()

        # Calculate average accuracy
        avg_accuracy = total_accuracy / len(test_questions) if test_questions else 0.0

        # Summary statistics
        contains_any_count = sum(1 for r in results if r["accuracy_result"]["contains_any"])
        contains_all_count = sum(1 for r in results if r["accuracy_result"]["contains_all"])

        print("=" * 80)
        print("ACCURACY SUMMARY")
        print("=" * 80)
        print(f"Total questions: {len(test_questions)}")
        print(f"Average accuracy: {avg_accuracy:.2%}")
        print(f"Questions with at least one match: {contains_any_count}/{len(test_questions)} ({contains_any_count/len(test_questions)*100:.1f}%)")
        print(f"Questions with all matches: {contains_all_count}/{len(test_questions)} ({contains_all_count/len(test_questions)*100:.1f}%)")
        print("=" * 80)
        print()

        return {
            "results": results,
            "average_accuracy": avg_accuracy,
            "contains_any_count": contains_any_count,
            "contains_all_count": contains_all_count,
            "total_questions": len(test_questions)
        }

    def evaluate_accuracy_comparison(use_cache: bool = False, use_prune_cache: bool = False):
        print("\n" + "=" * 80)
        print("ACCURACY EVALUATION - COMPARISON")
        print("=" * 80)
        print(f"Running {len(test_questions)} test questions in both modes (caching disabled)...\n")

        results_full = []
        results_simple = []
        total_accuracy_full = 0.0
        total_accuracy_simple = 0.0

        for i, test_item in enumerate(test_questions, 1):
            question = test_item["question"]
            expected_strings = test_item["expected_strings"]

            print(f"\n[{i}/{len(test_questions)}] Question: {question}")
            print(f"  Expected strings: {len(expected_strings)} items")
            print("\n" + "-" * 80)
            print("FULL MODE (with exemplars, postprocessing, refinement)")
            print("-" * 80)

            # Run full workflow
            query_full, context_full, answer_full = run_full_workflow(
                question=question,
                use_cache=use_cache,
                use_prune_cache=use_prune_cache,
                use_full_schema_cache=False,
                return_timings=False
            )

            answer_text_full = getattr(answer_full, "response", None) if answer_full else None
            if answer_text_full is None:
                answer_text_full = ""

            accuracy_result_full = check_answer_accuracy(answer_text_full, expected_strings)
            results_full.append({
                "question": question,
                "query": query_full,
                "answer": answer_text_full,
                "expected_strings": expected_strings,
                "accuracy_result": accuracy_result_full
            })

            total_accuracy_full += accuracy_result_full["accuracy"]

            print(f" Matched: {accuracy_result_full['match_count']}/{accuracy_result_full['total_expected']} strings")
            print(f" Accuracy: {accuracy_result_full['accuracy']:.2%}")

            print("\n" + "-" * 80)
            print("SIMPLE MODE (without exemplars, postprocessing, refinement)")
            print("-" * 80)

            # Run simple workflow
            query_simple, context_simple, answer_simple = run_full_workflow_simple(
                question=question,
                use_prune_cache=use_prune_cache,
                use_full_schema_cache=False,
                return_timings=False
            )

            answer_text_simple = getattr(answer_simple, "response", None) if answer_simple else None
            if answer_text_simple is None:
                answer_text_simple = ""

            accuracy_result_simple = check_answer_accuracy(answer_text_simple, expected_strings)
            results_simple.append({
                "question": question,
                "query": query_simple,
                "answer": answer_text_simple,
                "expected_strings": expected_strings,
                "accuracy_result": accuracy_result_simple
            })

            total_accuracy_simple += accuracy_result_simple["accuracy"]

            print(f" Matched: {accuracy_result_simple['match_count']}/{accuracy_result_simple['total_expected']} strings")
            print(f" Accuracy: {accuracy_result_simple['accuracy']:.2%}")

            # Comparison
            print("\n" + "-" * 80)
            print("COMPARISON")
            print("-" * 80)
            diff = accuracy_result_full["accuracy"] - accuracy_result_simple["accuracy"]
            print(f"  Full mode accuracy: {accuracy_result_full['accuracy']:.2%}")
            print(f"  Simple mode accuracy: {accuracy_result_simple['accuracy']:.2%}")
            print(f"  Difference: {diff:+.2%} ({'Full mode better' if diff > 0 else 'Simple mode better' if diff < 0 else 'Equal'})")
            print()

        # Calculate average accuracies
        avg_accuracy_full = total_accuracy_full / len(test_questions) if test_questions else 0.0
        avg_accuracy_simple = total_accuracy_simple / len(test_questions) if test_questions else 0.0

        # Summary statistics
        contains_any_full = sum(1 for r in results_full if r["accuracy_result"]["contains_any"])
        contains_all_full = sum(1 for r in results_full if r["accuracy_result"]["contains_all"])
        contains_any_simple = sum(1 for r in results_simple if r["accuracy_result"]["contains_any"])
        contains_all_simple = sum(1 for r in results_simple if r["accuracy_result"]["contains_all"])

        print("=" * 80)
        print("ACCURACY SUMMARY - COMPARISON")
        print("=" * 80)
        print(f"Total questions: {len(test_questions)}")
        print()
        print("FULL MODE (with exemplars, postprocessing, refinement):")
        print(f"  Average accuracy: {avg_accuracy_full:.2%}")
        print(f"  Questions with at least one match: {contains_any_full}/{len(test_questions)} ({contains_any_full/len(test_questions)*100:.1f}%)")
        print(f"  Questions with all matches: {contains_all_full}/{len(test_questions)} ({contains_all_full/len(test_questions)*100:.1f}%)")
        print()
        print("SIMPLE MODE (without exemplars, postprocessing, refinement):")
        print(f"  Average accuracy: {avg_accuracy_simple:.2%}")
        print(f"  Questions with at least one match: {contains_any_simple}/{len(test_questions)} ({contains_any_simple/len(test_questions)*100:.1f}%)")
        print(f"  Questions with all matches: {contains_all_simple}/{len(test_questions)} ({contains_all_simple/len(test_questions)*100:.1f}%)")
        print()
        print("OVERALL COMPARISON:")
        avg_diff = avg_accuracy_full - avg_accuracy_simple
        print(f"  Average accuracy difference: {avg_diff:+.2%} ({'Full mode better' if avg_diff > 0 else 'Simple mode better' if avg_diff < 0 else 'Equal'})")
        print("=" * 80)
        print()

        return {
            "full_mode": {
                "results": results_full,
                "average_accuracy": avg_accuracy_full,
                "contains_any_count": contains_any_full,
                "contains_all_count": contains_all_full,
            },
            "simple_mode": {
                "results": results_simple,
                "average_accuracy": avg_accuracy_simple,
                "contains_any_count": contains_any_simple,
                "contains_all_count": contains_all_simple,
            },
            "comparison": {
                "average_accuracy_difference": avg_diff,
                "total_questions": len(test_questions)
            }
        }

    accuracy_evaluation = evaluate_accuracy(use_cache=False, use_prune_cache=False)
    accuracy_evaluation_comparison = evaluate_accuracy_comparison(use_cache=False, use_prune_cache=False)
    return (test_questions,)


@app.cell
def _(
    clear_full_schema_cache,
    prune_schema_cache,
    run_full_workflow,
    test_questions,
    text2cypher_cache,
):
    """
    Benchmark comparison: run cache OFF twice (use second run for numbers), then cache ON
    twice (first run warms caches, second run measured). This highlights caching speedups.
    """

    from collections import defaultdict

    def run_round(
        *,
        questions: list[str],
        label: str,
        use_cache: bool,
        use_prune_cache: bool,
        use_full_schema_cache: bool,
    ) -> list[dict]:
        results = []
        for idx, question in enumerate(questions, 1):
            print(f"\n[{idx}/{len(questions)}] {question[:60]}... ({label})")
            q_out, ctx, ans, timings = run_full_workflow(
                question,
                use_cache=use_cache,
                use_prune_cache=use_prune_cache,
                use_full_schema_cache=use_full_schema_cache,
                return_timings=True,
            )
            results.append(
                {
                    "question": question,
                    "query": q_out,
                    "answer": getattr(ans, "response", None) if ans else None,
                    "timings": timings,
                }
            )
        return results

    def summarize_second_run(results: list[dict]):
        stage_samples: defaultdict[str, list[float]] = defaultdict(list)
        for res in results:
            total_ms = 0.0
            for stage_name, duration in res["timings"]:
                ms = duration * 1000
                stage_samples[stage_name].append(ms)
                total_ms += ms
            stage_samples["total"].append(total_ms)
        stage_avg = {
            stage: sum(values) / len(values)
            for stage, values in stage_samples.items()
            if values
        }
        return stage_samples, stage_avg

    def print_stage_summary(label: str, stage_avg: dict[str, float]):
        if not stage_avg:
            return
        print(f"\n=== {label} (second run averages) ===")
        for stage, avg_ms in sorted(stage_avg.items()):
            print(f"  - {stage:<24} {avg_ms:.2f} ms")

    num_questions = min(10, len(test_questions))
    questions = [item["question"] for item in test_questions[:num_questions]]
    if not questions:
        raise ValueError("No test questions available for benchmarking.")

    print(f"\n=== Running benchmark on {len(questions)} questions ===")

    # Cache OFF scenario
    clear_full_schema_cache()
    text2cypher_cache.clear()
    prune_schema_cache.clear()
    print("\n" + "=" * 80)
    print("Cache OFF runs")
    print("=" * 80)
    cache_off_run1 = run_round(
        questions=questions,
        label="Cache OFF run 1",
        use_cache=False,
        use_prune_cache=False,
        use_full_schema_cache=False,
    )
    cache_off_run2 = run_round(
        questions=questions,
        label="Cache OFF run 2",
        use_cache=False,
        use_prune_cache=False,
        use_full_schema_cache=False,
    )
    cache_off_stage_samples, cache_off_stage_avg = summarize_second_run(cache_off_run2)
    print_stage_summary("Cache OFF", cache_off_stage_avg)

    # Cache ON scenario
    clear_full_schema_cache()
    text2cypher_cache.clear()
    prune_schema_cache.clear()
    print("\n" + "=" * 80)
    print("Cache ON runs")
    print("=" * 80)
    cache_on_run1 = run_round(
        questions=questions,
        label="Cache ON run 1",
        use_cache=True,
            use_prune_cache=True,
        use_full_schema_cache=True,
    )
    cache_on_run2 = run_round(
        questions=questions,
        label="Cache ON run 2",
        use_cache=True,
        use_prune_cache=True,
        use_full_schema_cache=True,
    )
    cache_on_stage_samples, cache_on_stage_avg = summarize_second_run(cache_on_run2)
    print_stage_summary("Cache ON", cache_on_stage_avg)

    summary = {
        "questions": questions,
        "runs": {
            "cache_off": {
                "first_run": cache_off_run1,
                "second_run": cache_off_run2,
                "stage_samples_ms": cache_off_stage_samples,
                "stage_avg_ms": cache_off_stage_avg,
            },
            "cache_on": {
                "first_run": cache_on_run1,
                "second_run": cache_on_run2,
                "stage_samples_ms": cache_on_stage_samples,
                "stage_avg_ms": cache_on_stage_avg,
            },
        },
    }

    total_off = cache_off_stage_avg.get("total", 0)
    total_on = cache_on_stage_avg.get("total", 0)
    if total_off and total_on:
        print(f"\nOverall speedup (total): {total_off / total_on:.2f}x")

    print("\n" + "=" * 80)
    print("Benchmark comparison complete")
    print("Numbers reflect second runs only for each mode.")
    print("=" * 80)
    print()

    summary
    return (summary,)


@app.cell
def _():
    import pandas as pd

    def build_benchmark_metrics(summary, verbose=True):
        cache_off_run = summary["runs"]["cache_off"]
        cache_on_run = summary["runs"]["cache_on"]

        cache_off_avg = cache_off_run["stage_avg_ms"]
        cache_on_avg = cache_on_run["stage_avg_ms"]
        cache_off_samples = cache_off_run["stage_samples_ms"]
        cache_on_samples = cache_on_run["stage_samples_ms"]

        total_off = cache_off_avg.get("total", 0.0) or 0.0
        total_on = cache_on_avg.get("total", 0.0) or 0.0

        if verbose:
            if total_off and total_on:
                print(f"Cache OFF avg total: {total_off:.2f} ms")
                print(f"Cache ON avg total:  {total_on:.2f} ms")
                print(f"Overall speedup:     {total_off / total_on:.2f}x")
            else:
                print("Timing data incomplete; skipping aggregate speedup.")

        stage_names = [
            stage for stage in cache_off_avg.keys() if stage != "total"
        ]
        stage_names.sort(key=lambda s: cache_off_avg.get(s, 0.0), reverse=True)

        speedups = {}
        for stage in stage_names:
            off_val = cache_off_avg.get(stage, 0.0)
            on_val = cache_on_avg.get(stage, 0.0)
            speedups[stage] = off_val / on_val if on_val else float("inf")

        time_saved = {
            stage: cache_off_avg.get(stage, 0.0) - cache_on_avg.get(stage, 0.0)
            for stage in stage_names
        }

        if verbose:
            print("\nStage breakdown (second runs)")
            for stage in stage_names + ["total"]:
                off_ms = cache_off_avg.get(stage, 0.0)
                on_ms = cache_on_avg.get(stage, 0.0)
                saved = off_ms - on_ms
                pct = (saved / off_ms * 100) if off_ms else 0.0
                label = stage if stage != "total" else "TOTAL"
                print(
                    f"{label:<24} off={off_ms:8.2f} ms | "
                    f"on={on_ms:8.2f} ms | "
                    f"saved={saved:7.2f} ms ({pct:>5.1f}%)"
                )

        # Build a tidy DataFrame for plotting
        rows = []
        for stage in stage_names:
            off_ms = cache_off_avg.get(stage, 0.0)
            on_ms = cache_on_avg.get(stage, 0.0)
            saved_ms = off_ms - on_ms
            savings_pct = (saved_ms / off_ms * 100) if off_ms else 0.0
            spd = speedups[stage]

            rows.append(
                {
                    "stage": stage,
                    "stage_label": stage.replace("_", " ").title(),
                    "off_ms": off_ms,
                    "on_ms": on_ms,
                    "saved_ms": saved_ms,
                    "savings_pct": savings_pct,
                    "speedup": spd,
                }
            )

        df = pd.DataFrame(rows).sort_values("off_ms", ascending=False).reset_index(drop=True)

        overall = {
            "total_off": total_off,
            "total_on": total_on,
            "total_saved": total_off - total_on,
            "overall_speedup": (total_off / total_on) if total_on else float("inf"),
        }

        if total_off > 0:
            df["off_pct"] = df["off_ms"] / total_off * 100.0
        else:
            df["off_pct"] = 0.0

        if total_on > 0:
            df["on_pct"] = df["on_ms"] / total_on * 100.0
        else:
            df["on_pct"] = 0.0

        return {
            "cache_off_avg": cache_off_avg,
            "cache_on_avg": cache_on_avg,
            "cache_off_samples": cache_off_samples,
            "cache_on_samples": cache_on_samples,
            "stage_names": stage_names,
            "speedups": speedups,
            "time_saved": time_saved,
            "overall": overall,
            "df": df,
        }
    return (build_benchmark_metrics,)


@app.cell
def _():
    import plotly.graph_objects as go
    import numpy as np

    def plot_stage_bars(metrics):
        df = metrics["df"]
        overall = metrics["overall"]

        stage_labels = df["stage_label"].tolist()
        cache_off = df["off_ms"].tolist()
        cache_on = df["on_ms"].tolist()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=stage_labels,
            y=cache_off,
            name="Cache OFF",
            marker_color="#E11818",
            opacity=0.85,
            text=[f"{v:.2f}" if v > 0 else "" for v in cache_off],
            textposition="outside",
            textfont=dict(size=10),
        ))

        fig.add_trace(go.Bar(
            x=stage_labels,
            y=cache_on,
            name="Cache ON",
            marker_color="#16E2D8",
            opacity=0.85,
            text=[f"{v:.2f}" if v > 0 else "" for v in cache_on],
            textposition="outside",
            textfont=dict(size=10),
        ))

        fig.update_layout(
            title=(
                f"Stage Runtime Comparison (Cache OFF vs ON)<br>"
                f"<sup>Total OFF: {overall['total_off']:.2f} ms | "
                f"Total ON: {overall['total_on']:.2f} ms | "
                f"Speedup: {overall['overall_speedup']:.2f}x</sup>"
            ),
            barmode="group",
            yaxis_type="log",
            xaxis_title="Stage",
            yaxis_title="Time (ms, log scale)",
            template="plotly_white",
            width=1000,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
            ),
            hovermode="x unified",
        )

        fig.show()
    return go, plot_stage_bars


@app.cell
def _(go):
    from plotly.subplots import make_subplots

    def plot_flame_breakdown(metrics):
        df = metrics["df"]
        overall = metrics["overall"]

        cache_off_pct = df["off_pct"].tolist()
        cache_on_pct = df["on_pct"].tolist()

        stages = df["stage"].tolist()
        labels = df["stage_label"].tolist()
        cache_off = df["off_ms"].tolist()
        cache_on = df["on_ms"].tolist()

        colors = ["#16E2D8", "#0F70DF", "#8AE415", "#E11818", "#D215E7"]

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Flame Breakdown – Cache OFF", "Flame Breakdown – Cache ON"),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5],
            shared_xaxes=False,
        )

        for i, (stage_label, pct, time_ms) in enumerate(
            zip(labels, cache_off_pct, cache_off)
        ):
            if pct <= 0:
                continue

            hover_text = (
                f"{stage_label}<br>"
                f"Percentage: {pct:.1f}%<br>"
                f"Time: {time_ms:.2f} ms"
            )

            fig.add_trace(
                go.Bar(
                    x=[pct],
                    y=["Cache OFF"],
                    orientation="h",
                    name=stage_label,
                    marker=dict(
                        color=colors[i % len(colors)],
                        line=dict(color="white", width=1),
                    ),
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=True if i == 0 else True,  
                ),
                row=1,
                col=1,
            )

        for i, (stage_label, pct, time_ms) in enumerate(
            zip(labels, cache_on_pct, cache_on)
        ):
            if pct <= 0:
                continue

            hover_text = (
                f"{stage_label}<br>"
                f"Percentage: {pct:.1f}%<br>"
                f"Time: {time_ms:.2f} ms"
            )

            fig.add_trace(
                go.Bar(
                    x=[pct],
                    y=["Cache ON"],
                    orientation="h",
                    name=stage_label,
                    marker=dict(
                        color=colors[i % len(colors)],
                        line=dict(color="white", width=1),
                    ),
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=False,  # keep legend from top row
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title=(
                "Flame Breakdown – Percentage Share of Total Runtime<br>"
                f"<sup>Total OFF: {overall['total_off']:.2f} ms | "
                f"Total ON: {overall['total_on']:.2f} ms | "
                f"Speedup: {overall['overall_speedup']:.2f}x</sup>"
            ),
            barmode="stack",
            xaxis=dict(title="Share of total runtime (%)", range=[0, 100]),
            xaxis2=dict(title="Share of total runtime (%)", range=[0, 100]),
            width=1200,
            height=700,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                title="Pipeline Stages",
            ),
            hovermode="closest",
        )

        fig.update_yaxes(showticklabels=True)
        fig.show()
    return (plot_flame_breakdown,)


@app.cell
def _(build_benchmark_metrics, summary):
    metrics = build_benchmark_metrics(summary)
    return (metrics,)


@app.cell
def _(metrics, plot_stage_bars):
    plot_stage_bars(metrics)
    return


@app.cell
def _(metrics, plot_flame_breakdown):
    plot_flame_breakdown(metrics)
    return


if __name__ == "__main__":
    app.run()
