"""Append JSONL step traces; thread-safe for optional multi-threaded trainers."""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Mapping


class StepTraceWriter:
    def __init__(self, jsonl_path: str):
        self._path = jsonl_path
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(jsonl_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), sort_keys=False) + "\n"
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
