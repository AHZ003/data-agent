"""The Chinook golden set is well-formed and the harness scores gold as 100%."""

from benchmarks import golden_sql_eval
from benchmarks.golden_sql_check import check, load


def test_golden_set_has_no_problems():
    assert check() == []


def test_golden_set_size_and_coverage():
    _, cases = load()
    assert len(cases) >= 100
    assert {c["difficulty"] for c in cases} == {"easy", "medium", "hard"}
    assert all(c.get("tags") for c in cases)


def test_oracle_scores_every_case():
    assert golden_sql_eval.main(["--oracle"]) == 0
