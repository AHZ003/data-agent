"""Verified query memory: similar solved questions as few-shot examples.

Items are (question, SQL) pairs known to be correct: Spider *train*
examples, and later user-approved answers (thumbs up). For a new
question we retrieve the k most similar past questions (BM25 ⊕
embeddings, RRF) and show them to the Coder as examples.

Leakage guard: evaluating on an example while its gold SQL sits in the
memory would inflate accuracy for free. `add` refuses ids from any
evaluation split (Spider dev, BIRD mini-dev, the Chinook golden set),
so memory can only be built from train data by construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from retrieval.bm25 import BM25
from retrieval.embed import Embedder
from retrieval.fusion import rrf
from retrieval.text import tokenize

EVAL_ID_PREFIXES = ("spider-dev-", "bird-minidev-", "chk_")


class LeakageError(ValueError):
    """An evaluation example was about to enter the few-shot memory."""


@dataclass(frozen=True)
class MemoryItem:
    id: str
    question: str
    sql: str
    db_id: str
    source: str  # "spider-train" | "user" | ...


class QueryMemory:
    def __init__(self, embedder: Optional[Embedder] = None):
        self.items: list[MemoryItem] = []
        self.embedder = embedder
        self._bm25: Optional[BM25] = None
        self._vecs: Optional[np.ndarray] = None

    def add(self, item: MemoryItem) -> None:
        if item.id.startswith(EVAL_ID_PREFIXES):
            raise LeakageError(f"refusing to add evaluation example {item.id!r} to query memory")
        self.items.append(item)
        self._bm25 = self._vecs = None

    def _index(self) -> None:
        if self._bm25 is None:
            self._bm25 = BM25([tokenize(i.question) for i in self.items])
            if self.embedder is not None and self.items:
                self._vecs = self.embedder.embed([i.question for i in self.items])

    def search(self, question: str, k: int = 3, exclude_db: Optional[str] = None) -> list[MemoryItem]:
        if not self.items:
            return []
        self._index()
        bm = np.array(self._bm25.scores(tokenize(question)))
        rankings = [list(np.argsort(-bm, kind="stable")[:200])]
        if self._vecs is not None:
            q = self.embedder.embed([question])[0]
            rankings.append(list(np.argsort(-(self._vecs @ q), kind="stable")[:200]))
        out = []
        for i in rrf(rankings):
            item = self.items[int(i)]
            if exclude_db and item.db_id == exclude_db:
                continue
            out.append(item)
            if len(out) == k:
                break
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(asdict(i)) + "\n" for i in self.items))

    @classmethod
    def load(cls, path: Path, embedder: Optional[Embedder] = None) -> "QueryMemory":
        mem = cls(embedder)
        for line in path.read_text().splitlines():
            if line.strip():
                mem.add(MemoryItem(**json.loads(line)))
        return mem

    @classmethod
    def from_spider_train(cls, spider_dir: Path, embedder: Optional[Embedder] = None) -> "QueryMemory":
        mem = cls(embedder)
        for fname in ("train_spider.json", "train_others.json"):
            path = spider_dir / fname
            if not path.exists():
                continue
            for i, d in enumerate(json.loads(path.read_text())):
                mem.add(MemoryItem(f"spider-train-{fname.split('_')[1].split('.')[0]}-{i}",
                                   d["question"], d["query"], d["db_id"], "spider-train"))
        return mem
