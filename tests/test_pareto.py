"""Pareto frontier and SVG rendering."""

from benchmarks.pareto import frontier, svg, table


def _pts():
    return [
        {"name": "flash-single", "ex": 0.70, "ci": [0.64, 0.76], "cost_mean": 0.0004, "kind": "config"},
        {"name": "flash-repair", "ex": 0.74, "ci": [0.68, 0.80], "cost_mean": 0.0006, "kind": "config"},
        {"name": "pro-repair", "ex": 0.80, "ci": [0.74, 0.85], "cost_mean": 0.006, "kind": "config"},
        {"name": "dominated", "ex": 0.69, "ci": [0.63, 0.75], "cost_mean": 0.003, "kind": "config"},
        {"name": "router", "ex": 0.79, "ci": [0.73, 0.84], "cost_mean": 0.0015, "kind": "router"},
    ]


def test_frontier_drops_dominated_points():
    assert [p["name"] for p in frontier(_pts())] == ["flash-single", "flash-repair", "router", "pro-repair"]


def test_svg_and_table_render():
    s = svg(_pts(), "Cost vs accuracy")
    assert s.startswith("<svg") and s.count("<circle") == 5 and "log scale" in s
    t = table(_pts())
    assert "| dominated | 69.0%" in t and t.count("yes") == 4
