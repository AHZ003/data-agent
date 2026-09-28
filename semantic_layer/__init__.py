"""Semantic layer: business metric definitions and column descriptions.

Real analytics teams live on agreed definitions ("revenue excludes
refunds"). A YAML file per datasource states them once; the retriever
picks the definitions relevant to a question and the Coder is told to
use them verbatim, and column descriptions flow into the schema the
Coder sees and into schema-linking documents.

    datasource: chinook
    metrics:
      - name: revenue
        description: Money received from track sales
        synonyms: [sales, earned, income]
        sql: SUM("InvoiceLine"."UnitPrice" * "InvoiceLine"."Quantity")
    columns:
      Invoice.Total: Invoice amount in USD

`validate` checks every referenced table/column exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import sqlglot
import yaml
from sqlglot import exp

from core.datasource import DataSource
from retrieval.bm25 import BM25
from retrieval.text import tokenize

LAYER_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Metric:
    name: str
    description: str
    sql: str
    synonyms: tuple[str, ...] = ()

    def hint(self) -> str:
        return f"{self.name}: {self.description}. Compute as {self.sql}"


@dataclass
class SemanticLayer:
    datasource: str
    metrics: list[Metric] = field(default_factory=list)
    columns: dict[str, str] = field(default_factory=dict)  # "Table.Column" -> description

    @classmethod
    def load(cls, path: Path) -> "SemanticLayer":
        data = yaml.safe_load(path.read_text())
        return cls(
            datasource=data["datasource"],
            metrics=[Metric(m["name"], m["description"], m["sql"], tuple(m.get("synonyms", [])))
                     for m in data.get("metrics", [])],
            columns=dict(data.get("columns", {})),
        )

    @classmethod
    def for_datasource(cls, name: str) -> Optional["SemanticLayer"]:
        path = LAYER_DIR / f"{name}.yaml"
        return cls.load(path) if path.exists() else None

    def validate(self, ds: DataSource) -> list[str]:
        problems = []
        for key in self.columns:
            table, _, col = key.partition(".")
            t = ds.table(table)
            if t is None or t.column(col) is None:
                problems.append(f"column description for unknown column {key}")
        for m in self.metrics:
            try:
                tree = sqlglot.parse_one(m.sql, read=ds.dialect)
            except Exception as e:
                problems.append(f"metric {m.name}: unparsable SQL ({e})")
                continue
            for col in tree.find_all(exp.Column):
                t = ds.table(col.table) if col.table else None
                if col.table and (t is None or t.column(col.name) is None):
                    problems.append(f"metric {m.name}: unknown column {col.table}.{col.name}")
                elif not col.table and not any(tt.column(col.name) for tt in ds.tables):
                    problems.append(f"metric {m.name}: unknown column {col.name}")
        return problems

    def relevant_metrics(self, question: str, k: int = 3) -> list[Metric]:
        if not self.metrics:
            return []
        docs = [tokenize(f"{m.name} {m.description} {' '.join(m.synonyms)}") for m in self.metrics]
        scores = BM25(docs).scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.metrics[i] for i in ranked[:k] if scores[i] > 0]

    def annotate(self, ds: DataSource) -> DataSource:
        """Copy of `ds` with column descriptions filled in."""
        tables = []
        for t in ds.tables:
            cols = [replace(c, description=self.columns.get(f"{t.name}.{c.name}", c.description))
                    for c in t.columns]
            tables.append(replace(t, columns=cols))
        return replace(ds, tables=tables)
