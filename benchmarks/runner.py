"""Eval harness runner — execute golden cases against the live agent crew.

Usage:
    python -m benchmarks.runner                  # run all cases
    python -m benchmarks.runner --case ss_top_region_by_sales
    python -m benchmarks.runner --tag ranking
    python -m benchmarks.runner --no-judge       # skip LLM-as-judge
    python -m benchmarks.runner --out outputs/eval

The runner writes:
    <out>/results.json   raw scorecards
    <out>/report.md      human-readable scorecard
    <out>/summary.json   aggregate metrics for CI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# Make repo root importable when invoked as `python -m benchmarks.runner`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# NOTE: agents / scorers / core.database are imported lazily inside
# main()/run_one() after the --model flag has a chance to set
# DATAAGENT_MODEL, because config.MODEL_NAME (and every
# `from config import MODEL_NAME`) is captured at import time.


CASES_PATH = ROOT / "benchmarks" / "cases.yaml"


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("cases", [])


def run_one(case: dict, use_llm_judge: bool) -> "CaseScorecard":
    from agents.orchestrator import run_analysis
    from agents.schema_agent import profile_dataframe
    from core.database import Database
    from benchmarks.scorers import score_case

    dataset_path = ROOT / case["dataset"]
    df = pd.read_csv(dataset_path)
    db = Database()
    db.load_dataframe(df, case["table_name"])
    schema = profile_dataframe(df, case["table_name"])

    start = time.time()
    error: Optional[str] = None
    agent_output: dict = {}
    try:
        result = run_analysis(case["question"], schema, db, df)
        agent_output = {
            "sql_query": result.get("sql_query"),
            "result_df": result.get("result_df"),
            "chart_config": result.get("chart_config"),
            "narrative": result.get("narrative"),
            "validation": result.get("validation"),
        }
        if result.get("error"):
            error = str(result["error"])
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    duration = time.time() - start

    return score_case(case, agent_output, duration, error, use_llm_judge=use_llm_judge)


def aggregate(cards: list[CaseScorecard]) -> dict:
    if not cards:
        return {}
    overall = sum(c.weighted_score for c in cards) / len(cards)
    pass_rate = sum(1 for c in cards if c.passed) / len(cards)
    avg_latency = sum(c.duration_seconds for c in cards) / len(cards)

    by_dim: dict[str, list[float]] = {}
    for c in cards:
        for s in c.scores:
            by_dim.setdefault(s.name, []).append(s.score)
    dim_avgs = {k: round(sum(v) / len(v), 3) for k, v in by_dim.items()}

    by_tag: dict[str, list[float]] = {}
    for c in cards:
        for tag in c.tags:
            by_tag.setdefault(tag, []).append(c.weighted_score)
    tag_avgs = {k: round(sum(v) / len(v), 3) for k, v in by_tag.items()}

    return {
        "n_cases": len(cards),
        "overall_score": round(overall, 3),
        "pass_rate": round(pass_rate, 3),
        "avg_latency_seconds": round(avg_latency, 2),
        "by_dimension": dim_avgs,
        "by_tag": tag_avgs,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def render_markdown(cards: list[CaseScorecard], summary: dict) -> str:
    lines = ["# DataAgent — Evaluation Report", ""]
    lines.append(f"_Generated {summary['timestamp']}_")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append(f"- **Overall score:** `{summary['overall_score']:.3f}`  (weighted, 0-1)")
    lines.append(f"- **Pass rate:** `{summary['pass_rate'] * 100:.1f}%`  ({summary['n_cases']} cases)")
    lines.append(f"- **Avg latency:** `{summary['avg_latency_seconds']:.2f}s` per case")
    lines.append("")

    lines.append("## Per-dimension averages")
    lines.append("")
    lines.append("| Dimension | Avg score |")
    lines.append("|---|---|")
    for k, v in sorted(summary["by_dimension"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {v:.3f} |")
    lines.append("")

    if summary.get("by_tag"):
        lines.append("## Per-intent averages")
        lines.append("")
        lines.append("| Tag | Avg score |")
        lines.append("|---|---|")
        for k, v in sorted(summary["by_tag"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{k}` | {v:.3f} |")
        lines.append("")

    lines.append("## Per-case detail")
    lines.append("")
    for c in cards:
        status = "✅" if c.passed else "❌"
        lines.append(f"### {status} `{c.case_id}`  ({c.weighted_score:.3f}, {c.duration_seconds}s)")
        lines.append("")
        if c.error:
            lines.append(f"> **Error:** {c.error}")
            lines.append("")
            continue
        lines.append("| Scorer | Score | Pass | Reason |")
        lines.append("|---|---|---|---|")
        for s in c.scores:
            mark = "✓" if s.passed else "✗"
            reason = s.reason.replace("|", "\\|")
            lines.append(f"| `{s.name}` | {s.score:.2f} | {mark} | {reason} |")
        lines.append("")
    return "\n".join(lines)


def card_to_dict(c: CaseScorecard) -> dict:
    return {
        "case_id": c.case_id,
        "tags": c.tags,
        "duration_seconds": c.duration_seconds,
        "error": c.error,
        "passed": c.passed,
        "weighted_score": round(c.weighted_score, 3),
        "scores": [asdict(s) for s in c.scores],
    }


def _run_comparison(args) -> int:
    """Run the harness once per model in `args.compare` and write a
    comparison report. Each model's run is a subprocess so config
    module-level imports re-bind the new MODEL_NAME cleanly.
    """
    import subprocess

    models = [m.strip() for m in args.compare.split(",") if m.strip()]
    if len(models) < 2:
        print("--compare needs at least 2 comma-separated models", file=sys.stderr)
        return 2

    base_out = ROOT / args.out
    base_out.mkdir(parents=True, exist_ok=True)
    per_model_summaries: dict[str, dict] = {}

    for model in models:
        safe = model.replace("/", "_").replace(":", "_")
        model_out = base_out / safe
        print(f"\n▶▶ Running eval on model: {model}  →  {model_out}")
        sub_cmd = [
            sys.executable, "-m", "benchmarks.runner",
            "--model", model,
            "--out", str(model_out.relative_to(ROOT)),
        ]
        if args.no_judge:
            sub_cmd.append("--no-judge")
        if args.case:
            sub_cmd.extend(["--case", args.case])
        if args.tag:
            sub_cmd.extend(["--tag", args.tag])
        rc = subprocess.call(sub_cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"  ⚠ model {model} exited rc={rc}", file=sys.stderr)
        summary_path = model_out / "summary.json"
        if summary_path.exists():
            per_model_summaries[model] = json.loads(summary_path.read_text())

    if not per_model_summaries:
        print("No per-model summaries produced.", file=sys.stderr)
        return 1

    comp_md = _render_comparison_markdown(per_model_summaries)
    (base_out / "comparison.md").write_text(comp_md)
    (base_out / "comparison.json").write_text(
        json.dumps(per_model_summaries, indent=2)
    )
    print(f"\n✓ Comparison report: {base_out}/comparison.md")
    return 0


def _render_comparison_markdown(summaries: dict[str, dict]) -> str:
    lines = ["# DataAgent — Cross-Model Comparison", ""]
    lines.append("| Model | Overall | Pass rate | Avg latency | n |")
    lines.append("|---|---|---|---|---|")
    ranked = sorted(
        summaries.items(),
        key=lambda kv: -kv[1].get("overall_score", 0),
    )
    for model, s in ranked:
        lines.append(
            f"| `{model}` | {s.get('overall_score', 0):.3f} | "
            f"{s.get('pass_rate', 0) * 100:.1f}% | "
            f"{s.get('avg_latency_seconds', 0):.2f}s | "
            f"{s.get('n_cases', 0)} |"
        )
    lines.append("")

    # Per-dimension matrix
    all_dims: set[str] = set()
    for s in summaries.values():
        all_dims.update((s.get("by_dimension") or {}).keys())
    if all_dims:
        lines.append("## Per-dimension (0–1)")
        lines.append("")
        header = "| Dimension | " + " | ".join(f"`{m}`" for m in ranked_names(ranked)) + " |"
        sep = "|---|" + "---|" * len(ranked)
        lines.append(header)
        lines.append(sep)
        for dim in sorted(all_dims):
            row = [f"`{dim}`"]
            for model, _ in ranked:
                v = (summaries[model].get("by_dimension") or {}).get(dim)
                row.append(f"{v:.3f}" if v is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("_Lower-cost model winning on a dimension is a routing signal: "
                 "consider running only that dimension on the cheaper model in "
                 "production, and reserving the expensive model for the rest._")
    return "\n".join(lines)


def ranked_names(ranked):
    return [m for m, _ in ranked]


def main():
    parser = argparse.ArgumentParser(description="Run DataAgent eval harness")
    parser.add_argument("--case", help="Run a single case by id")
    parser.add_argument("--tag", help="Run only cases with this tag")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge faithfulness scoring")
    parser.add_argument("--out", default="outputs/eval",
                        help="Output directory for reports")
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="Exit non-zero if overall score below this (CI gate)")
    parser.add_argument("--model", default=None,
                        help="Override the agent-crew model for this run "
                             "(sets DATAAGENT_MODEL). e.g. gemini-2.5-pro")
    parser.add_argument("--compare", default=None,
                        help="Comma-separated list of models — runs the harness "
                             "once per model (subprocess) and writes a comparison "
                             "report alongside each per-model report dir.")
    args = parser.parse_args()

    if args.compare:
        sys.exit(_run_comparison(args))

    if args.model:
        os.environ["DATAAGENT_MODEL"] = args.model

    from benchmarks.scorers import CaseScorecard  # noqa: F401

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if args.tag:
        cases = [c for c in cases if args.tag in c.get("tags", [])]
    if not cases:
        print("No cases matched filter.", file=sys.stderr)
        sys.exit(2)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"▶ Running {len(cases)} case(s) "
          f"(LLM judge: {'off' if args.no_judge else 'on'})")
    cards: list[CaseScorecard] = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']:38s}", end=" ", flush=True)
        card = run_one(case, use_llm_judge=not args.no_judge)
        cards.append(card)
        mark = "PASS" if card.passed else "FAIL"
        print(f"{mark}  score={card.weighted_score:.2f}  ({card.duration_seconds}s)")

    summary = aggregate(cards)
    summary["model"] = os.environ.get("DATAAGENT_MODEL", "gemini-flash-latest")
    try:
        from prompts import version_map as _pv
        summary["prompt_versions"] = _pv()
    except Exception as _e:
        summary["prompt_versions"] = {"error": str(_e)}

    (out_dir / "results.json").write_text(
        json.dumps([card_to_dict(c) for c in cards], indent=2, default=str)
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(render_markdown(cards, summary))

    print()
    print(f"✓ Overall score: {summary['overall_score']:.3f}")
    print(f"✓ Pass rate:     {summary['pass_rate'] * 100:.1f}%")
    print(f"✓ Avg latency:   {summary['avg_latency_seconds']:.2f}s")
    print(f"✓ Reports:       {out_dir}/report.md, summary.json, results.json")

    if args.fail_under and summary["overall_score"] < args.fail_under:
        print(f"✗ Below threshold {args.fail_under}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
