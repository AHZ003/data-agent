"""Dashboard history + rendering."""

import json

from benchmarks import dashboard


def _summary(bench, ex, evidence=False):
    return {"benchmark": bench, "config": {"benchmark": bench, "model": "m", "evidence": evidence},
            "config_hash": "h", "n": 200, "ex": ex, "ci": [ex - 0.05, ex + 0.05],
            "cost_usd_mean": 0.0007, "timestamp": "2026-09-28T07:00:00Z"}


def test_append_and_build(tmp_path):
    s1, s2 = tmp_path / "s1.json", tmp_path / "s2.json"
    s1.write_text(json.dumps(_summary("spider", 0.74)))
    s2.write_text(json.dumps(_summary("bird", 0.51, evidence=True)))
    hist, out = tmp_path / "history.jsonl", tmp_path / "index.html"
    assert dashboard.main(["append", str(s1), str(s2), "--history", str(hist), "--commit", "abc1234"]) == 0
    rows = dashboard.load_history(hist)
    assert [r["series"] for r in rows] == ["spider", "bird (evidence)"]
    assert dashboard.main(["build", "--history", str(hist), "--out", str(out)]) == 0
    page = out.read_text()
    assert "<title>DataAgent Evals</title>" in page
    assert "abc1234" in page and "74.0%" in page
    assert "__DATA__" not in page and "__TABLE__" not in page


def test_build_with_empty_history(tmp_path):
    out = tmp_path / "index.html"
    dashboard.main(["build", "--history", str(tmp_path / "none.jsonl"), "--out", str(out)])
    assert "No runs yet." in out.read_text()
