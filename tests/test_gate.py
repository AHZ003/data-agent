"""CI-aware gate CLI."""

import json

from benchmarks import gate


def _write(path, passed):
    path.write_text(json.dumps([{"case_id": str(i), "passed": p} for i, p in enumerate(passed)]))


def test_gate_roundtrip(tmp_path):
    results, baseline = tmp_path / "r.json", tmp_path / "b.json"
    _write(results, [True] * 16 + [False] * 2)
    assert gate.main(["--results", str(results), "--baseline", str(baseline)]) == 0  # no baseline: skip
    assert gate.main(["--results", str(results), "--baseline", str(baseline), "--write-baseline"]) == 0
    assert json.loads(baseline.read_text())["point"] == 16 / 18

    _write(results, [True] * 15 + [False] * 3)   # one case worse: noise
    assert gate.main(["--results", str(results), "--baseline", str(baseline)]) == 0
    _write(results, [True] * 8 + [False] * 10)   # collapse: real regression
    assert gate.main(["--results", str(results), "--baseline", str(baseline)]) == 1


def test_gate_reads_match_field(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"id": "a", "match": True}, {"id": "b", "match": False}]))
    assert gate.outcomes(p) == [True, False]
