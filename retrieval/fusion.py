"""Reciprocal rank fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[Hashable]:
    """Fuse ranked lists: score(d) = sum 1 / (k + rank). Robust to score scales,
    which is why it is used to combine BM25 (unbounded) with cosine (-1..1)."""
    score: dict[Hashable, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            score[item] += 1.0 / (k + rank + 1)
    return sorted(score, key=lambda d: -score[d])
