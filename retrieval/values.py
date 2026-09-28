"""Value retrieval: map phrases in the question to values actually stored.

Many BIRD failures are literal mismatches: the question says
"California" and the column stores "CA", or "Formula 1" vs "Formula One",
or different casing. We index the distinct values of every text column
(capped per column) by character trigrams and match every 1-4 word span
of the question against them (plus quoted strings). A match yields a hint
the Coder sees, e.g.:

    "schools"."County" contains 'Alameda' (question: "alameda")

and marks that column as must-keep for schema linking.

Scoring is Jaccard similarity of trigram sets, with exact
case-insensitive matches scoring 1.0; `min_score` trades precision for
recall. Long free-text columns are skipped (they are not filter values).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

from core.datasource import DataSource
from retrieval.text import STOPWORDS, char_ngrams

MAX_VALUES_PER_COLUMN = 5000
MAX_VALUE_LEN = 60
_TEXT_TYPES = ("CHAR", "TEXT", "CLOB", "VARCHAR", "NVARCHAR", "STRING")
_WORD = re.compile(r"[\w'.&/-]+", re.UNICODE)
_QUOTED = re.compile(r"['\"‘’“”]([^'\"‘’“”]{2,60})['\"‘’“”]")


@dataclass(frozen=True)
class ValueMatch:
    table: str
    column: str
    value: str
    span: str
    score: float

    def hint(self) -> str:
        v = self.value.replace("'", "''")
        return f'"{self.table}"."{self.column}" contains \'{v}\' (question: "{self.span}")'


class ValueIndex:
    def __init__(self, fetch, ds: DataSource):
        """`fetch(sql, params) -> rows` runs trusted SQL on the engine."""
        self.entries: list[tuple[str, str, str, frozenset]] = []  # table, column, value, trigrams
        self.by_gram: dict[str, list[int]] = {}
        self.exact: dict[str, list[int]] = {}
        for t in ds.tables:
            for c in t.columns:
                if c.type and not c.type.upper().startswith(_TEXT_TYPES):
                    continue
                if c.distinct_count is not None and t.row_count and c.distinct_count > MAX_VALUES_PER_COLUMN:
                    continue
                q = (f'SELECT DISTINCT "{c.name}" FROM "{t.name}" WHERE "{c.name}" IS NOT NULL '
                     f"LIMIT {MAX_VALUES_PER_COLUMN + 1}")
                try:
                    rows = fetch(q, ())
                except Exception:
                    continue
                if len(rows) > MAX_VALUES_PER_COLUMN:
                    continue  # identifiers / free text, not filter values
                for (v,) in rows:
                    if not isinstance(v, str) or not (1 < len(v) <= MAX_VALUE_LEN):
                        continue
                    i = len(self.entries)
                    grams = frozenset(char_ngrams(v))
                    self.entries.append((t.name, c.name, v, grams))
                    self.exact.setdefault(v.lower(), []).append(i)
                    for g in grams:
                        self.by_gram.setdefault(g, []).append(i)

    @staticmethod
    def spans(question: str, max_words: int = 4) -> list[str]:
        words = _WORD.findall(question)
        quoted = [m.group(1) for m in _QUOTED.finditer(question)]
        out = []
        for n in range(1, max_words + 1):
            for i in range(len(words) - n + 1):
                span = words[i : i + n]
                # "What", "the list" etc. match stored values by accident.
                if all(w.lower() in STOPWORDS for w in span):
                    continue
                out.append(" ".join(span))
        return [s.strip(".,?") for s in quoted + out if len(s.strip(".,?")) > 2]

    def match(self, question: str, min_score: float = 0.75, limit: int = 8) -> list[ValueMatch]:
        best: dict[int, ValueMatch] = {}
        for span in self.spans(question):
            cands = set(self.exact.get(span.lower(), []))
            exact = set(cands)
            grams = char_ngrams(span)
            if not cands:
                counts: dict[int, int] = {}
                for g in grams:
                    for i in self.by_gram.get(g, ()):
                        counts[i] = counts.get(i, 0) + 1
                # Only entries sharing enough trigrams can reach min_score.
                cands = {i for i, c in counts.items() if c >= min_score * len(grams)}
            for i in cands:
                t, c, v, vg = self.entries[i]
                score = 1.0 if i in exact else len(grams & vg) / len(grams | vg)
                if score >= min_score and (i not in best or score > best[i].score):
                    best[i] = ValueMatch(t, c, v, span, round(score, 3))
        ranked = sorted(best.values(), key=lambda m: (-m.score, -len(m.span)))
        # One hint per (column, value); longest/best span wins.
        seen, out = set(), []
        for m in ranked:
            key = (m.table, m.column, m.value)
            if key not in seen:
                seen.add(key)
                out.append(m)
        return out[:limit]


def build(source, ds: Optional[DataSource] = None) -> ValueIndex:
    """Build from an Engine (uses its trusted `fetch`) or a sqlite3 connection."""
    if isinstance(source, sqlite3.Connection):
        return ValueIndex(lambda q, p: source.execute(q, p).fetchall(),
                          ds or DataSource.from_sqlite_conn(source))
    return ValueIndex(source.fetch, ds or source.datasource())
