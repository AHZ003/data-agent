"""Shared text-to-SQL benchmark harness (Spider, BIRD, internal sets).

Every benchmark runs the *real* DataAgent SQL path —
`coder_agent.generate_and_execute` with the production prompt, the
three-layer guard and the self-repair loop — against a read-only SQLite
database. Only the question source and the match rule differ.

Match rules
-----------
- "spider": the official test-suite `result_eq` — same row count and
  arity, column permutations allowed, order compared only when the gold
  SQL has ORDER BY, rows compared as multisets otherwise.
- "bird": the official BIRD rule — set(pred_rows) == set(gold_rows).

Results are cached per (config hash, example id) in JSONL so a rerun of
the same config is free and an interrupted run resumes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
GOLD_TIMEOUT_S = 60.0
# Predicted-SQL timeout for benchmark runs; matches BIRD's official
# evaluator (meta_time_out=30). The app's interactive default is lower.
PRED_TIMEOUT_S = 30.0


@dataclass
class Example:
    id: str
    db_id: str
    question: str
    gold_sql: str
    difficulty: str = ""
    evidence: str = ""


@dataclass
class ExampleResult:
    id: str
    db_id: str
    difficulty: str
    question: str
    gold_sql: str
    pred_sql: str
    match: bool
    error: Optional[str] = None
    gold_error: Optional[str] = None
    attempts: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "ExampleResult":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass(frozen=True)
class RunConfig:
    """Everything that can change a prediction. Hashed for the cache key."""
    benchmark: str
    model: str
    mode: str = "repair"          # "single" | "repair"
    evidence: bool = False        # BIRD: append the evidence hint
    extra: tuple = field(default_factory=tuple)  # future flags, e.g. retrieval

    @property
    def max_attempts(self) -> int:
        from config import MAX_RETRY_ATTEMPTS
        return 1 if self.mode == "single" else MAX_RETRY_ATTEMPTS

    @property
    def flags(self) -> dict:
        return dict(self.extra)

    def cache_key(self) -> str:
        from agents.coder_agent import PROMPT_BUILDER_VERSION
        from prompts import version_info
        payload = asdict(self) | {"coder_prompt": version_info("coder_agent")["sha"],
                                  "prompt_builder": PROMPT_BUILDER_VERSION}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# ── Result matching ─────────────────────────────────────────────────────

def _unorder_row(row: tuple) -> tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def spider_result_eq(gold: list[tuple], pred: list[tuple], order_matters: bool) -> bool:
    """Official Spider test-suite result equivalence."""
    if len(gold) == 0 and len(pred) == 0:
        return True
    if len(gold) != len(pred):
        return False
    n_cols = len(gold[0])
    if len(pred[0]) != n_cols:
        return False
    # Quick reject: compare rows with values sorted inside each row.
    g_rows = [_unorder_row(r) for r in gold]
    p_rows = [_unorder_row(r) for r in pred]
    if order_matters and g_rows != p_rows:
        return False
    if not order_matters and set(g_rows) != set(p_rows):
        return False
    # Candidate pred columns for each gold column: values must be a subset.
    gold_col_sets = [{r[i] for r in gold} for i in range(n_cols)]
    pred_col_sets = [{r[j] for r in pred} for j in range(n_cols)]
    candidates = [[j for j in range(n_cols) if pred_col_sets[j] <= gold_col_sets[i]] for i in range(n_cols)]
    for perm in product(*candidates):
        if len(set(perm)) != n_cols:
            continue
        permuted = [tuple(r[j] for j in perm) for r in pred]
        if order_matters:
            if permuted == gold:
                return True
        elif Counter(permuted) == Counter(gold):
            return True
    return False


def bird_result_eq(gold: list[tuple], pred: list[tuple]) -> bool:
    """Official BIRD rule: set equality of result rows."""
    return set(pred) == set(gold)


def results_match(rule: str, gold: list[tuple], pred: list[tuple], gold_sql: str) -> bool:
    if rule == "bird":
        return bird_result_eq(gold, pred)
    return spider_result_eq(gold, pred, order_matters="order by" in gold_sql.lower())


# ── Execution ───────────────────────────────────────────────────────────

def execute_gold(db_path: Path, sql: str, timeout_s: float = GOLD_TIMEOUT_S) -> tuple[Optional[list[tuple]], Optional[str]]:
    """Run trusted gold SQL read-only with a timeout; return all rows."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="replace")
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        return [tuple(r) for r in conn.execute(sql).fetchall()], None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def df_to_rows(df) -> list[tuple]:
    """DataFrame back to DB-API rows. pandas turns SQL NULL into NaN, and
    NaN != None, so without this any result containing a NULL would
    never match the gold rows."""
    clean = df.astype(object).where(df.notna(), None)
    return [tuple(r) for r in clean.itertuples(index=False, name=None)]


class _DataSourceCache:
    """Introspect each database once; examples share the DataSource."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict[Path, object] = {}

    def get(self, db_path: Path):
        from core.datasource import DataSource
        with self._lock:
            if db_path not in self._cache:
                self._cache[db_path] = DataSource.from_sqlite(str(db_path), name=db_path.stem)
            return self._cache[db_path]


_DS_CACHE = _DataSourceCache()
_RETRIEVERS: dict[RunConfig, object] = {}
_RETRIEVER_LOCK = threading.Lock()


def retriever_for(cfg: RunConfig):
    """One Retriever per config (its indexes are shared across examples)."""
    from retrieval.pipeline import RetrievalConfig, Retriever

    rc = RetrievalConfig.from_flags(cfg.flags)
    if rc is None:
        return None
    with _RETRIEVER_LOCK:
        if cfg not in _RETRIEVERS:
            _RETRIEVERS[cfg] = Retriever(rc)
        return _RETRIEVERS[cfg]
_GOLD_CACHE: dict[tuple[str, str], tuple[Optional[list[tuple]], Optional[str]]] = {}


def format_question(ex: Example, use_evidence: bool) -> str:
    if use_evidence and ex.evidence:
        return f"{ex.question}\nHint: {ex.evidence}"
    return ex.question


def run_example(ex: Example, db_path: Path, cfg: RunConfig, rule: str) -> ExampleResult:
    """Run one example through the real coder path and score it."""
    from agents.coder_agent import generate_and_execute
    from config import MAX_QUERY_ROWS
    from core import tracing
    from core.database import Database

    key = (str(db_path), ex.gold_sql)
    if key not in _GOLD_CACHE:
        _GOLD_CACHE[key] = execute_gold(db_path, ex.gold_sql)
    gold_rows, gold_error = _GOLD_CACHE[key]

    db = Database.from_sqlite(str(db_path), name=ex.db_id, datasource=_DS_CACHE.get(db_path))
    db.max_rows = max(MAX_QUERY_ROWS, len(gold_rows or []) + 1)  # never truncate below gold
    db.timeout_seconds = PRED_TIMEOUT_S
    start = time.time()
    with tracing.usage_scope() as usage:
        result_df, code = generate_and_execute(
            format_question(ex, cfg.evidence), db, max_attempts=cfg.max_attempts, model=cfg.model,
            retriever=retriever_for(cfg),
        )
    latency = time.time() - start
    db.close()

    match = False
    if result_df is not None and gold_rows is not None:
        match = results_match(rule, gold_rows, df_to_rows(result_df), ex.gold_sql)

    return ExampleResult(
        id=ex.id, db_id=ex.db_id, difficulty=ex.difficulty, question=ex.question,
        gold_sql=ex.gold_sql, pred_sql=code.sql_query or "", match=match,
        error=None if code.success else code.error, gold_error=gold_error,
        attempts=code.attempts, tokens_in=usage.tokens_in, tokens_out=usage.tokens_out,
        cost_usd=usage.cost_usd, latency_s=round(latency, 3),
    )


# ── Batch runner with cache ─────────────────────────────────────────────

def cache_path(cfg: RunConfig, cache_dir: Path) -> Path:
    return cache_dir / cfg.benchmark / f"{cfg.cache_key()}.jsonl"


def load_cached(path: Path) -> dict[str, ExampleResult]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = ExampleResult.from_dict(json.loads(line))
            out[r.id] = r
    return out


def run_examples(
    examples: Iterable[Example],
    db_path_for: Callable[[Example], Path],
    cfg: RunConfig,
    rule: str,
    concurrency: int = 1,
    cache_dir: Optional[Path] = None,
    progress: bool = True,
) -> list[ExampleResult]:
    """Run examples (cached, optionally concurrent) in input order.

    Examples whose result is a terminal provider error (quota, bad key)
    are not cached, so a rerun retries them.
    """
    from agents.coder_agent import is_terminal_error

    examples = list(examples)
    path = cache_path(cfg, cache_dir) if cache_dir else None
    done = load_cached(path) if path else {}
    todo = [ex for ex in examples if ex.id not in done]
    if progress:
        print(f"  {cfg.benchmark}/{cfg.model}/{cfg.mode}: {len(examples)} examples, "
              f"{len(examples) - len(todo)} cached, {len(todo)} to run")

    lock = threading.Lock()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _one(ex: Example) -> ExampleResult:
        r = run_example(ex, db_path_for(ex), cfg, rule)
        with lock:
            done[ex.id] = r
            if path and not (r.error and is_terminal_error(r.error)):
                with open(path, "a") as f:
                    f.write(json.dumps(asdict(r), default=str) + "\n")
            if progress:
                status = "MATCH" if r.match else ("ERROR" if r.error else "miss")
                print(f"  [{len(done)}/{len(examples)}] {r.db_id:24s} {status:5s} {r.latency_s:5.1f}s  {r.question[:60]}")
        return r

    if concurrency <= 1:
        for ex in todo:
            _one(ex)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for fut in as_completed([pool.submit(_one, ex) for ex in todo]):
                fut.result()

    return [done[ex.id] for ex in examples]


def stratified_subset(examples: list[Example], size: int, seed: int = 0) -> list[Example]:
    """Deterministic sample of `size` examples, proportional by difficulty."""
    import random

    if size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    by_diff: dict[str, list[Example]] = {}
    for ex in examples:
        by_diff.setdefault(ex.difficulty, []).append(ex)
    picked: list[Example] = []
    for diff, group in sorted(by_diff.items()):
        k = round(size * len(group) / len(examples))
        picked.extend(rng.sample(group, min(k, len(group))))
    # Rounding can leave us off by one or two; fix up deterministically.
    rest = [ex for ex in examples if ex not in picked]
    rng.shuffle(rest)
    picked = (picked + rest)[:size]
    order = {ex.id: i for i, ex in enumerate(examples)}
    return sorted(picked, key=lambda ex: order[ex.id])


# ── Reporting ───────────────────────────────────────────────────────────

def summarize(results: list[ExampleResult], cfg: RunConfig) -> dict:
    from benchmarks.stats import bootstrap_ci, percentiles

    matches = [r.match for r in results]
    overall = bootstrap_ci(matches)
    by_diff = {}
    for diff in sorted({r.difficulty for r in results}):
        sub = [r.match for r in results if r.difficulty == diff]
        ci = bootstrap_ci(sub)
        by_diff[diff] = {"n": len(sub), "ex": ci.point, "ci": [ci.lo, ci.hi]}
    n = max(len(results), 1)
    return {
        "benchmark": cfg.benchmark,
        "config": asdict(cfg),
        "config_hash": cfg.cache_key(),
        "n": len(results),
        "ex": overall.point,
        "ci": [overall.lo, overall.hi],
        "by_difficulty": by_diff,
        "n_error": sum(1 for r in results if r.error),
        "n_gold_error": sum(1 for r in results if r.gold_error),
        "mean_attempts": sum(r.attempts for r in results) / n,
        "repaired": sum(1 for r in results if r.match and r.attempts > 1),
        "cost_usd_total": round(sum(r.cost_usd for r in results), 4),
        "cost_usd_mean": sum(r.cost_usd for r in results) / n,
        "latency_s": percentiles([r.latency_s for r in results]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def render_report(title: str, results: list[ExampleResult], summary: dict) -> str:
    lo, hi = summary["ci"]
    lines = [
        f"# {title}",
        "",
        f"_Generated {summary['timestamp']} · model `{summary['config']['model']}` · "
        f"mode `{summary['config']['mode']}` · config `{summary['config_hash']}`_",
        "",
        f"**Execution accuracy:** {summary['ex']:.1%} (95% CI {lo:.1%}–{hi:.1%}), n={summary['n']}",
        "",
        f"- Errors (no executable SQL): {summary['n_error']} · gold SQL errors: {summary['n_gold_error']}",
        f"- Fixed by self-repair: {summary['repaired']} · mean attempts: {summary['mean_attempts']:.2f}",
        f"- Cost: ${summary['cost_usd_total']:.4f} total, ${summary['cost_usd_mean']:.5f}/question",
        f"- Latency: p50 {summary['latency_s']['p50']:.1f}s, p95 {summary['latency_s']['p95']:.1f}s",
        "",
        "## By difficulty",
        "",
        "| Difficulty | n | EX | 95% CI |",
        "|---|---|---|---|",
    ]
    for diff, d in summary["by_difficulty"].items():
        lines.append(f"| {diff or '—'} | {d['n']} | {d['ex']:.1%} | {d['ci'][0]:.1%}–{d['ci'][1]:.1%} |")
    misses = [r for r in results if not r.match][:15]
    if misses:
        lines += ["", "## Sample misses", ""]
        for r in misses:
            lines.append(f"**[{r.db_id} · {r.difficulty}]** {r.question}")
            lines.append(f"- gold: `{r.gold_sql[:240]}`")
            lines.append(f"- pred: `{r.pred_sql[:240]}`" + (f" — error: {r.error[:160]}" if r.error else ""))
            lines.append("")
    return "\n".join(lines)


def write_outputs(out_dir: Path, title: str, results: list[ExampleResult], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(render_report(title, results, summary))
