import hashlib
import json
from collections import OrderedDict
from typing import Dict


def stable_hash_payload(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_question(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


def hash_schema(schema: Dict) -> str:
    return stable_hash_payload(schema)


class LRUCache:

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

