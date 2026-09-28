"""Schema linking: show the Coder only the part of the schema a question needs.

Wide databases (BIRD has tables with 100+ columns) bury the relevant
columns in prompt noise and cost tokens. We rank every column against
the question with two retrievers and fuse them:

  - BM25 over a per-column document (table + column name split into
    words, description, sample values) — exact lexical overlap;
  - embedding cosine over the same documents — paraphrase ("earned" ~
    "revenue") when a semantic embedder is used.

Reciprocal rank fusion combines the two rankings without calibrating
their score scales. The top-k columns are kept, then two expansions
protect join correctness, which pruning otherwise breaks:

  1. every kept table keeps its primary- and foreign-key columns;
  2. if two kept tables are not directly connected but a third table
     links both (a bridge such as PlaylistTrack), the bridge is added.

Schemas with at most `min_columns_to_prune` columns are returned whole:
there is nothing to gain from pruning a small schema and recall is at
risk.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from core.datasource import DataSource, Table
from retrieval.bm25 import BM25
from retrieval.embed import Embedder
from retrieval.fusion import rrf
from retrieval.text import tokenize

ColumnKey = tuple[str, str]  # (table, column), original casing


def column_document(table: Table, col) -> str:
    samples = " ".join(str(v) for v in col.sample_values[:3] if isinstance(v, str))
    return f"{table.name} {col.name} {col.description or ''} {samples}".strip()


class SchemaLinker:
    def __init__(self, ds: DataSource, embedder: Optional[Embedder] = None):
        self.ds = ds
        self.keys: list[ColumnKey] = [(t.name, c.name) for t in ds.tables for c in t.columns]
        docs = [column_document(t, c) for t in ds.tables for c in t.columns]
        self.bm25 = BM25([tokenize(d) for d in docs])
        self.embedder = embedder
        self.vectors = embedder.embed(docs) if embedder is not None and docs else None

    def rank(self, question: str) -> list[ColumnKey]:
        """All columns, most relevant first (BM25 ⊕ dense via RRF)."""
        bm = np.array(self.bm25.scores(tokenize(question)))
        rankings = [[self.keys[i] for i in np.argsort(-bm, kind="stable")]]
        if self.vectors is not None:
            q = self.embedder.embed([question])[0]
            rankings.append([self.keys[i] for i in np.argsort(-(self.vectors @ q), kind="stable")])
        return rrf(rankings)

    def link(
        self,
        question: str,
        k: int = 20,
        min_columns_to_prune: int = 30,
        must_keep: Optional[set[ColumnKey]] = None,
    ) -> DataSource:
        """Pruned DataSource with the top-k columns plus join keys and bridges."""
        if len(self.keys) <= min_columns_to_prune:
            return self.ds
        selected = set(self.rank(question)[:k]) | (must_keep or set())
        return prune(self.ds, selected)


def prune(ds: DataSource, selected: set[ColumnKey]) -> DataSource:
    tables = {t for t, _ in selected}
    by_name = {t.name: t for t in ds.tables}

    # Bridge tables: a table with FKs into two kept tables that are not
    # otherwise connected (many-to-many link tables).
    def linked(a: str, b: str) -> bool:
        return any(fk.ref_table == b for fk in by_name[a].foreign_keys) or any(
            fk.ref_table == a for fk in by_name[b].foreign_keys
        )

    for t in ds.tables:
        if t.name in tables:
            continue
        targets = {fk.ref_table for fk in t.foreign_keys} & tables
        if any(not linked(a, b) for a in targets for b in targets if a < b):
            tables.add(t.name)

    new_tables = []
    for t in ds.tables:  # preserve original order
        if t.name not in tables:
            continue
        join_cols = set(t.primary_key) | {fk.column for fk in t.foreign_keys if fk.ref_table in tables}
        # FKs that point at this table from other kept tables need its referenced columns too.
        join_cols |= {fk.ref_column for o in ds.tables if o.name in tables for fk in o.foreign_keys
                      if fk.ref_table == t.name and fk.ref_column}
        cols = [c for c in t.columns if (t.name, c.name) in selected or c.name in join_cols]
        new_tables.append(replace(
            t, columns=cols,
            foreign_keys=[fk for fk in t.foreign_keys if fk.ref_table in tables],
        ))
    return DataSource(name=ds.name, tables=new_tables, dialect=ds.dialect)
