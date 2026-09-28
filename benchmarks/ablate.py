"""Run a set of configs and compare them with CIs and McNemar tests.

    python -m benchmarks.ablate                         # benchmarks/ablations.yaml
    python -m benchmarks.ablate --plan                  # how many LLM calls, no spend
    python -m benchmarks.ablate --only spider-flash-single,spider-flash-repair

Each run in the YAML is one config. Within a benchmark, the first run
is the baseline; every other run is compared to it *on the same
questions* (paired): Δ EX with a paired-bootstrap CI and an exact
McNemar p-value. Per-example results are cached, so re-running after
adding one config only pays for the new config.

Writes outputs/ablations/report.md and summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.stats import bootstrap_ci, mcnemar_exact, paired_delta_ci, percentiles  # noqa: E402
from benchmarks.text2sql import (  # noqa: E402
    ExampleResult,
    RunConfig,
    cache_path,
    load_cached,
    run_examples,
    stratified_subset,
)

DEFAULT_YAML = ROOT / "benchmarks" / "ablations.yaml"
CACHE_DIR = ROOT / "outputs" / "cache"
OUT_DIR = ROOT / "outputs" / "ablations"


def _benchmarks():
    from benchmarks import bird_eval, spider_eval
    return {
        "spider": (spider_eval.load_examples, spider_eval.db_path_for, "spider"),
        "bird": (bird_eval.load_examples, bird_eval.db_path_for, "bird"),
    }


def load_runs(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    defaults = data.get("defaults", {})
    runs = [defaults | r for r in data["runs"]]
    names = [r["name"] for r in runs]
    if len(names) != len(set(names)):
        raise ValueError("run names must be unique")
    return runs


def to_config(run: dict) -> RunConfig:
    from config import MODEL_NAME
    return RunConfig(
        benchmark=run["benchmark"],
        model=run.get("model") or MODEL_NAME,
        mode=run.get("mode", "repair"),
        evidence=bool(run.get("evidence", False)),
        extra=tuple(sorted((run.get("flags") or {}).items())),
    )


def examples_for(run: dict):
    load, db_path_for, rule = _benchmarks()[run["benchmark"]]
    examples = load()
    if run.get("subset"):
        examples = stratified_subset(examples, int(run["subset"]), int(run.get("seed", 0)))
    return examples, db_path_for, rule


def compare(runs: list[dict], results: dict[str, list[ExampleResult]]) -> list[dict]:
    """One row per run: EX + CI, and paired stats vs its benchmark's baseline."""
    rows = []
    baselines: dict[str, str] = {}
    for run in runs:
        baselines.setdefault(run["benchmark"], run["name"])
    for run in runs:
        res = results[run["name"]]
        ex_ci = bootstrap_ci([r.match for r in res])
        row = {
            "name": run["name"],
            "benchmark": run["benchmark"],
            "baseline": baselines[run["benchmark"]],
            "n": len(res),
            "ex": ex_ci.point,
            "ci": [ex_ci.lo, ex_ci.hi],
            "cost_mean": sum(r.cost_usd for r in res) / max(len(res), 1),
            "latency_p95": percentiles([r.latency_s for r in res])["p95"],
        }
        base_name = baselines[run["benchmark"]]
        if base_name != run["name"]:
            base = {r.id: r.match for r in results[base_name]}
            common = [r for r in res if r.id in base]
            a = [base[r.id] for r in common]
            b = [r.match for r in common]
            d = paired_delta_ci(a, b)
            m = mcnemar_exact(a, b)
            row |= {"n_paired": len(common), "delta": d.point, "delta_ci": [d.lo, d.hi],
                    "mcnemar_p": m.p_value, "only_base": m.a_only, "only_this": m.b_only}
        rows.append(row)
    return rows


def render(rows: list[dict]) -> str:
    lines = [
        "# Ablations",
        "",
        "Within each benchmark the first run is the baseline. Δ is paired (same questions); "
        "p is the exact McNemar test on discordant questions (only-baseline-right vs only-this-right).",
        "",
    ]
    for bench in dict.fromkeys(r["benchmark"] for r in rows):
        lines += [f"## {bench}", "",
                  "| Config | n | EX | 95% CI | Δ vs baseline | Δ 95% CI | McNemar p | disc. (base/this) | $/q | p95 s |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["benchmark"] == bench):
            ci = f"{r['ci'][0]:.1%}–{r['ci'][1]:.1%}"
            if "delta" in r:
                delta = f"{r['delta']:+.1%}"
                dci = f"{r['delta_ci'][0]:+.1%} to {r['delta_ci'][1]:+.1%}"
                p = f"{r['mcnemar_p']:.3g}"
                disc = f"{r['only_base']}/{r['only_this']}"
            else:
                delta = dci = p = disc = "baseline"
            lines.append(f"| `{r['name']}` | {r['n']} | {r['ex']:.1%} | {ci} | {delta} | {dci} | {p} | "
                         f"{disc} | {r['cost_mean']:.5f} | {r['latency_p95']:.1f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run ablation configs")
    parser.add_argument("--config", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--only", help="comma-separated run names")
    parser.add_argument("--plan", action="store_true", help="show uncached work, run nothing")
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args(argv)

    os.environ.setdefault("DATAAGENT_LLM_BACKOFF_RETRIES", "5")
    runs = load_runs(args.config)
    if args.only:
        wanted = set(args.only.split(","))
        runs = [r for r in runs if r["name"] in wanted]

    if args.plan:
        total = 0
        for run in runs:
            examples, _, _ = examples_for(run)
            cached = load_cached(cache_path(to_config(run), CACHE_DIR))
            todo = sum(1 for ex in examples if ex.id not in cached)
            total += todo
            print(f"{run['name']:32s} {len(examples):5d} examples, {todo:5d} uncached")
        print(f"{'total':32s} {total:5d} questions to run (~1-3 LLM calls each)")
        return 0

    results: dict[str, list[ExampleResult]] = {}
    for run in runs:
        examples, db_path_for, rule = examples_for(run)
        results[run["name"]] = run_examples(
            examples, db_path_for, to_config(run), rule,
            concurrency=args.concurrency or int(run.get("concurrency", 1)), cache_dir=CACHE_DIR,
        )

    rows = compare(runs, results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"runs": runs, "rows": rows, "configs": {r["name"]: asdict(to_config(r)) for r in runs}},
        indent=2, default=str))
    (OUT_DIR / "report.md").write_text(render(rows))
    print(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
