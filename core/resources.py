"""Process-level registry of live query engines, referenced by id.

Graph state must be serializable for the LangGraph checkpointer
(multi-turn conversations, interrupt/resume), and a database connection
is not. State therefore carries `datasource_id`; nodes resolve the
engine and the source DataFrame here.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Optional

_lock = threading.Lock()
_registry: dict[str, tuple[Any, Optional[Any]]] = {}


def register(engine, df=None, datasource_id: Optional[str] = None) -> str:
    rid = datasource_id or uuid.uuid4().hex[:16]
    with _lock:
        _registry[rid] = (engine, df)
    return rid


def get(datasource_id: str) -> tuple[Any, Optional[Any]]:
    with _lock:
        if datasource_id not in _registry:
            raise KeyError(f"unknown datasource_id {datasource_id!r}")
        return _registry[datasource_id]


def unregister(datasource_id: str) -> None:
    with _lock:
        _registry.pop(datasource_id, None)


def ids() -> list[str]:
    with _lock:
        return list(_registry)
