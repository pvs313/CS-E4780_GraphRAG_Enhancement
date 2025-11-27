import kuzu
from typing import Dict, List


def get_db_connection(db_name: str = "nobel.kuzu", read_only: bool = True) -> kuzu.Connection:
    db = kuzu.Database(db_name, read_only=read_only)
    return kuzu.Connection(db)


def get_schema_dict(conn: kuzu.Connection) -> Dict[str, List[Dict]]:
    response = conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
    nodes = [row[1] for row in response]
    
    response = conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
    rel_tables = [row[1] for row in response]
    
    relationships = []
    for tbl_name in rel_tables:
        response = conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
        for row in response:
            relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})
    
    schema = {"nodes": [], "edges": []}
    
    for node in nodes:
        node_schema = {"label": node, "properties": []}
        node_properties = conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
        for row in node_properties:
            node_schema["properties"].append({"name": row[1], "type": row[2]})
        schema["nodes"].append(node_schema)
    
    for rel in relationships:
        edge = {
            "label": rel["name"],
            "from": rel["from"],
            "to": rel["to"],
            "properties": [],
        }
        rel_properties = conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
        for row in rel_properties:
            edge["properties"].append({"name": row[1], "type": row[2]})
        schema["edges"].append(edge)
    
    return schema

_full_schema_store = {"schema": None, "hash": None}


def get_full_schema(conn: kuzu.Connection, use_cache: bool = True) -> Dict:
    from .cache import hash_schema
    
    if use_cache and _full_schema_store["schema"] is not None:
        return _full_schema_store["schema"]
    
    schema = get_schema_dict(conn)
    if use_cache:
        _full_schema_store["schema"] = schema
        _full_schema_store["hash"] = hash_schema(schema)
    return schema


def get_full_schema_hash() -> str | None:
    return _full_schema_store["hash"]


def clear_full_schema_cache() -> None:
    _full_schema_store["schema"] = None
    _full_schema_store["hash"] = None
