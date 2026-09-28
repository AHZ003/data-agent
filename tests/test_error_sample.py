"""Error-analysis sampling and label summary."""

import csv
import json

from benchmarks import error_sample


def test_sample_then_summarize(tmp_path):
    results = [{"id": f"q{i}", "db_id": "d", "difficulty": "easy" if i % 2 else "hard",
                "question": "q", "gold_sql": "g", "pred_sql": "p", "error": None,
                "match": i % 3 == 0} for i in range(90)]
    rp, out = tmp_path / "r.json", tmp_path / "labels.csv"
    rp.write_text(json.dumps(results))
    assert error_sample.main(["sample", str(rp), "--out", str(out), "-n", "20"]) == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 20 and all(r["category"] == "" for r in rows)
    assert error_sample.main(["sample", str(rp), "--out", str(out)]) == 1  # never overwrite labels

    rows[0]["category"], rows[1]["category"], rows[2]["category"] = "join", "join", "typo"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=error_sample.FIELDS)
        w.writeheader()
        w.writerows(rows)
    text = error_sample.summarize(out)
    assert "Labeled 3/20" in text and "| join | 2 | 67% |" in text and "typo" in text
