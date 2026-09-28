"""Difficulty router: predict whether the cheap model will fail, before calling it.

Features are free to compute (no extra LLM call): question length and
wording (comparatives, negation, per-group, ratios, superlatives), schema
size, and how many tables the question lexically touches. The label is
"cheap model got it wrong" from cached benchmark results, so training
costs nothing beyond runs we already do.

Evaluation replays cached outcomes: for a threshold t, a question goes to
the strong model when P(cheap fails) > t, and we score it with that
model's recorded result and cost. Sweeping t traces a cost/accuracy curve
with no new LLM calls. The router is trained on one half of the
questions and evaluated on the other half.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from retrieval.text import tokenize

_CUES = {
    "per_group": r"\b(each|per|every|by)\b",
    "compare": r"\b(than|compare|compared|versus|vs\.?|difference|more|less|fewer)\b",
    "negation": r"\b(not|never|no|without|except|excluding|neither|none)\b",
    "superlative": r"\b(most|least|highest|lowest|largest|smallest|top|best|worst|maximum|minimum)\b",
    "ratio": r"\b(ratio|percent|percentage|proportion|share|rate|average|mean)\b",
    "set_op": r"\b(both|either|also|and also|but not|as well as)\b",
    "time": r"\b(year|month|day|date|between|before|after|since|during)\b",
    "multi_answer": r"\b(list|all|which)\b",
}
_CUE_RX = {k: re.compile(v, re.IGNORECASE) for k, v in _CUES.items()}
FEATURES = ["log_len", "n_tables", "log_columns", "tables_mentioned", "has_evidence", *_CUES]


def featurize(question: str, table_names: list[str], n_columns: int, evidence: str = "") -> list[float]:
    q_tokens = set(tokenize(question))
    mentioned = sum(1 for t in table_names if set(tokenize(t)) & q_tokens)
    return [
        math.log1p(len(question.split())),
        float(len(table_names)),
        math.log1p(n_columns),
        float(mentioned),
        1.0 if evidence else 0.0,
        *[1.0 if rx.search(question) else 0.0 for rx in _CUE_RX.values()],
    ]


@dataclass
class Router:
    cheap: str
    strong: str
    threshold: float = 0.5
    coef: list[float] = field(default_factory=list)
    intercept: float = 0.0
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)

    def p_fail(self, x: list[float]) -> float:
        z = (np.asarray(x) - self.mean) / self.scale
        return float(1 / (1 + math.exp(-(z @ self.coef + self.intercept))))

    def pick(self, x: list[float]) -> str:
        return self.strong if self.p_fail(x) > self.threshold else self.cheap

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Router":
        return cls(**json.loads(Path(path).read_text()))


def fit(X: np.ndarray, cheap_failed: np.ndarray, cheap: str, strong: str) -> Router:
    from sklearn.linear_model import LogisticRegression

    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit((X - mean) / scale, cheap_failed)
    return Router(cheap, strong, 0.5, clf.coef_[0].tolist(), float(clf.intercept_[0]), mean.tolist(), scale.tolist())


def simulate(router: Router, X: np.ndarray, cheap_ok, strong_ok, cheap_cost, strong_cost,
             thresholds: Optional[list[float]] = None) -> list[dict]:
    """Replay cached outcomes for each threshold -> accuracy, cost, share routed."""
    p = np.array([router.p_fail(x) for x in X])
    out = []
    for t in thresholds or [i / 20 for i in range(21)]:
        use_strong = p > t
        ok = np.where(use_strong, strong_ok, cheap_ok)
        cost = np.where(use_strong, strong_cost, cheap_cost)
        out.append({"threshold": t, "ex": float(ok.mean()), "cost_mean": float(cost.mean()),
                    "share_strong": float(use_strong.mean()), "matches": ok.tolist()})
    return out
