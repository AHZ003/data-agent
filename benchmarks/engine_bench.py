"""SQLite vs DuckDB latency on the same synthetic table.

    python -m benchmarks.engine_bench --rows 1000000 10000000

Builds an orders-like table (date, region, category, amount, quantity),
loads it into both engines, and times four typical analytics queries
through each engine's guarded `execute_query` (median of 5 runs, after a
warm-up). Writes outputs/engine_bench.md.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERIES = {
    "group-by sum": 'SELECT "region", SUM("amount") AS s FROM "orders" GROUP BY "region" ORDER BY s DESC',
    "filtered count": "SELECT COUNT(*) FROM \"orders\" WHERE \"category\" = 'cat_7' AND \"amount\" > 50",
    "monthly trend": "SELECT substr(\"order_date\", 1, 7) AS m, SUM(\"amount\") FROM \"orders\" GROUP BY m ORDER BY m",
    "top-10 categories": 'SELECT "category", AVG("amount") AS a FROM "orders" GROUP BY "category" ORDER BY a DESC LIMIT 10',
}


def make_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2020-01-01", periods=1461, freq="D").strftime("%Y-%m-%d").to_numpy()
    return pd.DataFrame({
        "order_date": days[rng.integers(0, len(days), n)],
        "region": np.array([f"region_{i}" for i in range(8)])[rng.integers(0, 8, n)],
        "category": np.array([f"cat_{i}" for i in range(50)])[rng.integers(0, 50, n)],
        "amount": rng.gamma(2.0, 30.0, n).round(2),
        "quantity": rng.integers(1, 10, n),
    })


def time_query(engine, sql: str, reps: int = 5) -> float:
    _, err = engine.execute_query(sql)  # warm-up
    if err:
        raise RuntimeError(err)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        engine.execute_query(sql)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def run(n: int) -> dict:
    import sqlite3

    from core.database import Database
    from core.duckdb_engine import DuckDBEngine

    df = make_frame(n)
    out = {"rows": n, "load_s": {}, "query_s": {}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.sqlite"
        t = time.perf_counter()
        with sqlite3.connect(path) as conn:
            df.to_sql("orders", conn, index=False, chunksize=200_000)
        out["load_s"]["sqlite"] = time.perf_counter() - t
        sqlite_db = Database.from_sqlite(str(path), name="bench")
        sqlite_db.timeout_seconds = 600

        t = time.perf_counter()
        duck = DuckDBEngine(name="bench")
        duck.load_dataframe(df, "orders")
        out["load_s"]["duckdb"] = time.perf_counter() - t
        duck.timeout_seconds = 600

        for label, sql in QUERIES.items():
            out["query_s"][label] = {"sqlite": time_query(sqlite_db, sql), "duckdb": time_query(duck, sql)}
        sqlite_db.close()
        duck.close()
    return out


def render(results: list[dict]) -> str:
    import platform
    lines = ["# Engine benchmark: SQLite vs DuckDB", "",
             f"_{platform.machine()} · {platform.system()} · median of 5 runs, guarded execute_query, warm cache_", ""]
    for r in results:
        lines += [f"## {r['rows']:,} rows", "",
                  f"Load: SQLite {r['load_s']['sqlite']:.1f}s, DuckDB {r['load_s']['duckdb']:.1f}s", "",
                  "| Query | SQLite | DuckDB | Speed-up |", "|---|---|---|---|"]
        for label, t in r["query_s"].items():
            lines.append(f"| {label} | {t['sqlite'] * 1000:,.0f} ms | {t['duckdb'] * 1000:,.0f} ms | "
                         f"{t['sqlite'] / t['duckdb']:.0f}× |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1_000_000])
    args = parser.parse_args(argv)
    results = [run(n) for n in args.rows]
    md = render(results)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "engine_bench.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
