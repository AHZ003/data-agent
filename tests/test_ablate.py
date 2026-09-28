"""Ablation comparison: paired deltas and McNemar vs the benchmark baseline."""

import pytest

from benchmarks.ablate import compare, load_runs, render, to_config
from benchmarks.text2sql import ExampleResult


def _res(prefix, matches):
    return [ExampleResult(id=f"q{i}", db_id="d", difficulty="easy", question="q", gold_sql="s",
                          pred_sql="p", match=m, cost_usd=0.001, latency_s=1.0)
            for i, m in enumerate(matches)]


def test_compare_pairs_on_same_questions_against_first_run():
    runs = [{"name": "base", "benchmark": "spider"}, {"name": "new", "benchmark": "spider"},
            {"name": "bird-base", "benchmark": "bird"}]
    base = [True] * 60 + [False] * 40
    new = [True] * 60 + [True] * 12 + [False] * 28  # fixes 12, breaks none
    rows = compare(runs, {"base": _res("b", base), "new": _res("n", new), "bird-base": _res("x", base)})
    by = {r["name"]: r for r in rows}
    assert "delta" not in by["base"] and "delta" not in by["bird-base"]
    assert by["new"]["baseline"] == "base"
    assert by["new"]["delta"] == pytest.approx(0.12)
    assert (by["new"]["only_base"], by["new"]["only_this"]) == (0, 12)
    assert by["new"]["mcnemar_p"] == pytest.approx(2 * 0.5 ** 12)
    md = render(rows)
    assert "## spider" in md and "## bird" in md and "+12.0%" in md


def test_load_runs_applies_defaults_and_rejects_duplicates(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("defaults: {subset: 50}\nruns:\n  - {name: a, benchmark: spider, mode: single}\n")
    [run] = load_runs(p)
    assert run["subset"] == 50
    cfg = to_config(run)
    assert cfg.mode == "single" and cfg.max_attempts == 1
    p.write_text("runs:\n  - {name: a, benchmark: spider}\n  - {name: a, benchmark: bird}\n")
    with pytest.raises(ValueError):
        load_runs(p)


def test_shipped_ablations_yaml_parses():
    from benchmarks.ablate import DEFAULT_YAML
    runs = load_runs(DEFAULT_YAML)
    assert {r["benchmark"] for r in runs} <= {"spider", "bird"}
