"""Every SQL guard-evasion attack is stopped, and no legitimate query is blocked."""

import pytest

from benchmarks.redteam.run import false_positives, guard_case, load_cases

CASES = load_cases()["guard_evasion"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_attack_fails_against_current_defenses(case):
    r = guard_case(case, "after")
    assert not r["succeeded"], r


def test_no_false_positives_on_legitimate_queries():
    assert false_positives()["after"] == 0


def test_pipeline_cases_are_well_formed():
    for c in load_cases()["pipeline"]:
        assert c["canary"] and c["tables"] and c["question"] and c["category"]
