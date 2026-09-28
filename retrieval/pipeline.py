"""Retriever: turn a question + database into a focused prompt context.

    retriever = Retriever(RetrievalConfig(schema_k=30, values=True, memory_k=3, semantic=True))
    datasource, context = retriever.prepare(question, db)

Each component is a flag so ablations can switch them one at a time.
Per-database indexes (schema linker, value index) are built once and
cached by database name.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.datasource import DataSource

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RetrievalConfig:
    schema_k: int = 0            # 0 = no schema linking; else keep top-k columns
    values: bool = False         # value hints (+ must-keep columns for linking)
    memory_k: int = 0            # few-shot examples from query memory
    semantic: bool = False       # semantic-layer metrics + column descriptions
    embedder: str = "gemini"     # "gemini" | "hashing" | "none"

    @classmethod
    def from_flags(cls, flags: dict) -> Optional["RetrievalConfig"]:
        known = {k: v for k, v in flags.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        return cfg if cfg.enabled else None

    @property
    def enabled(self) -> bool:
        return bool(self.schema_k or self.values or self.memory_k or self.semantic)


@dataclass
class PromptContext:
    value_hints: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    examples: list[tuple[str, str]] = field(default_factory=list)  # (question, sql)

    def render(self) -> str:
        parts = []
        if self.metrics:
            parts.append("METRIC DEFINITIONS (use these exact formulas):\n"
                         + "\n".join(f"- {m}" for m in self.metrics))
        if self.value_hints:
            parts.append("VALUES FOUND IN THE DATABASE (use the stored spelling in filters):\n"
                         + "\n".join(f"- {h}" for h in self.value_hints))
        if self.examples:
            parts.append("SOLVED EXAMPLES FROM OTHER DATABASES (for SQL patterns only; "
                         "their tables differ):\n"
                         + "\n\n".join(f"Q: {q}\nSQL: {s}" for q, s in self.examples))
        return "\n\n".join(parts)


def _embedder(name: str):
    from retrieval.embed import GeminiEmbedder, HashingEmbedder, default_embedder
    if name == "none":
        return None
    if name == "hashing":
        return HashingEmbedder()
    return default_embedder() if name == "auto" else GeminiEmbedder()


class Retriever:
    def __init__(self, config: RetrievalConfig, memory=None):
        self.config = config
        self._embedder = _embedder(config.embedder)
        self._lock = threading.Lock()
        self._per_db: dict[str, tuple] = {}
        self.memory = memory
        if config.memory_k and memory is None:
            from retrieval.memory import QueryMemory
            self.memory = QueryMemory.from_spider_train(ROOT / "benchmarks" / "spider", self._embedder)

    def _indexes(self, db):
        with self._lock:
            if db.name not in self._per_db:
                from retrieval.schema_linking import SchemaLinker
                from retrieval.values import build
                from semantic_layer import SemanticLayer

                ds = db.datasource()
                layer = SemanticLayer.for_datasource(db.name) if self.config.semantic else None
                if layer is not None:
                    ds = layer.annotate(ds)
                linker = SchemaLinker(ds, self._embedder) if self.config.schema_k else None
                values = build(db, ds) if self.config.values else None
                self._per_db[db.name] = (ds, linker, values, layer)
            return self._per_db[db.name]

    def prepare(self, question: str, db) -> tuple[DataSource, PromptContext]:
        ds, linker, values, layer = self._indexes(db)
        ctx = PromptContext()
        must_keep = set()
        if values is not None:
            matches = values.match(question)
            ctx.value_hints = [m.hint() for m in matches]
            must_keep = {(m.table, m.column) for m in matches}
        if linker is not None:
            ds = linker.link(question, k=self.config.schema_k, must_keep=must_keep)
        if layer is not None:
            ctx.metrics = [m.hint() for m in layer.relevant_metrics(question)]
        if self.memory is not None and self.config.memory_k:
            ctx.examples = [(i.question, i.sql)
                            for i in self.memory.search(question, self.config.memory_k, exclude_db=db.name)]
        return ds, ctx
