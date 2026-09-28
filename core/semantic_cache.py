"""Semantic SQL cache: reuse SQL for questions that mean the same thing.

Key: (question embedding, schema version). On a close enough match
(cosine >= threshold) the cached SQL is re-executed on the current data,
so answers stay fresh while the LLM call is skipped. The schema version
hashes table/column names and types only, so data refreshes keep hits
and schema changes invalidate them.

Follow-up questions in a conversation ("now by region") depend on the
previous turn, so callers should not use the cache for them.

Enabled with DATAAGENT_SEMANTIC_CACHE=1. In-process and bounded (LRU);
each Cloud Run instance warms its own.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.datasource import DataSource


def schema_version(ds: DataSource) -> str:
    sig = "|".join(f"{t.name}:" + ",".join(f"{c.name} {c.type}" for c in t.columns) for t in ds.tables)
    return hashlib.sha256(f"{ds.dialect}|{sig}".encode()).hexdigest()[:16]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    usd_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class _Entry:
    question: str
    vector: np.ndarray
    sql: str
    cost_usd: float


class SemanticCache:
    def __init__(self, embedder=None, threshold: float = 0.95, max_entries: int = 5000):
        if embedder is None:
            from retrieval.embed import default_embedder
            embedder = default_embedder()
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries
        self.stats = CacheStats()
        self._by_version: dict[str, OrderedDict[str, _Entry]] = {}
        self._lock = threading.Lock()

    def lookup(self, question: str, version: str) -> Optional[str]:
        q = self.embedder.embed([question])[0]
        with self._lock:
            entries = self._by_version.get(version)
            best, best_sim = None, -1.0
            for key, e in (entries or {}).items():
                sim = float(e.vector @ q)
                if sim > best_sim:
                    best, best_sim = key, sim
            if best is not None and best_sim >= self.threshold:
                entries.move_to_end(best)
                hit = entries[best]
                self.stats.hits += 1
                self.stats.usd_saved = round(self.stats.usd_saved + hit.cost_usd, 6)
                return hit.sql
            self.stats.misses += 1
            return None

    def store(self, question: str, version: str, sql: str, cost_usd: float = 0.0) -> None:
        vec = self.embedder.embed([question])[0]
        with self._lock:
            entries = self._by_version.setdefault(version, OrderedDict())
            entries[question] = _Entry(question, vec, sql, cost_usd)
            entries.move_to_end(question)
            while sum(len(e) for e in self._by_version.values()) > self.max_entries:
                oldest_version = next(v for v in self._by_version if self._by_version[v])
                self._by_version[oldest_version].popitem(last=False)


_default: Optional[SemanticCache] = None
_default_lock = threading.Lock()


def default_cache() -> Optional[SemanticCache]:
    """The process-wide cache, or None when DATAAGENT_SEMANTIC_CACHE is off."""
    global _default
    if os.getenv("DATAAGENT_SEMANTIC_CACHE", "0") in ("0", "", "false"):
        return None
    with _default_lock:
        if _default is None:
            _default = SemanticCache()
        return _default
