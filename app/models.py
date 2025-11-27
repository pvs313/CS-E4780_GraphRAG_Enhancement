from pydantic import BaseModel, Field
import dspy


class Query(BaseModel):
    """Cypher query model."""
    query: str = Field(description="Valid Cypher query with no newlines")


class Property(BaseModel):
    """Property model for graph schema."""
    name: str
    type: str = Field(description="Data type of the property")


class Node(BaseModel):
    """Node model for graph schema."""
    label: str
    properties: list[Property] | None


class Edge(BaseModel):
    """Edge model for graph schema."""
    label: str = Field(description="Relationship label")
    from_: Node = Field(alias="from", description="Source node label")
    to: Node = Field(alias="from", description="Target node label")
    properties: list[Property] | None


class GraphSchema(BaseModel):
    """Graph schema model."""
    nodes: list[Node]
    edges: list[Edge]


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


class AnswerQuestion(dspy.Signature):
    """
    - Use the provided question, the generated Cypher query and the context to answer the question.
    - If the context is empty, state that you don't have enough information to answer the question.
    """
    question: str = dspy.InputField()
    cypher_query: str = dspy.InputField()
    context: str = dspy.InputField()
    response: str = dspy.OutputField()

