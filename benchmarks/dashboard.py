"""Build the static eval dashboard (single HTML file) from run history.

    python -m benchmarks.dashboard append outputs/spider/summary.json --history site/history.jsonl
    python -m benchmarks.dashboard build --history site/history.jsonl --out site/index.html

The nightly workflow appends each benchmark summary (with the commit)
to history.jsonl on the gh-pages branch and rebuilds index.html, so
the dashboard shows execution accuracy over commits with 95% CI bands
and cost per question over commits.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# Categorical slots 1-2 of the reference palette; validated light + dark.
SERIES_COLORS = {
    "light": ["#2a78d6", "#eb6834"],
    "dark": ["#3987e5", "#d95926"],
}


def series_label(summary: dict) -> str:
    cfg = summary.get("config", {})
    label = summary.get("benchmark", cfg.get("benchmark", "?"))
    if label == "bird":
        label += " (evidence)" if cfg.get("evidence") else " (no evidence)"
    return label


def history_record(summary: dict, commit: str) -> dict:
    return {
        "series": series_label(summary),
        "commit": commit,
        "timestamp": summary["timestamp"],
        "n": summary["n"],
        "ex": summary["ex"],
        "ci": summary["ci"],
        "cost_usd_mean": summary["cost_usd_mean"],
        "model": summary["config"]["model"],
        "config_hash": summary["config_hash"],
    }


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _commit() -> str:
    sha = os.environ.get("GITHUB_SHA", "")
    if sha:
        return sha[:7]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return ""


def render(history: list[dict]) -> str:
    rows = sorted(history, key=lambda r: r["timestamp"])
    series = list(dict.fromkeys(r["series"] for r in rows))
    data_json = json.dumps({"rows": rows, "series": series, "colors": SERIES_COLORS})
    table_rows = "\n".join(
        f"<tr><td>{html.escape(r['timestamp'][:10])}</td><td><code>{html.escape(r['commit'])}</code></td>"
        f"<td>{html.escape(r['series'])}</td><td>{r['n']}</td><td>{r['ex']:.1%}</td>"
        f"<td>{r['ci'][0]:.1%}–{r['ci'][1]:.1%}</td><td>${r['cost_usd_mean']:.5f}</td>"
        f"<td>{html.escape(r['model'])}</td></tr>"
        for r in reversed(rows)
    )
    return _TEMPLATE.replace("__DATA__", data_json).replace("__TABLE__", table_rows or
                             '<tr><td colspan="8">No runs yet.</td></tr>')


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DataAgent Evals</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --text: #0b0b0b; --text-2: #52514e; --muted: #8a8984;
    --grid: #e6e5e1; --border: #dcdbd6;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19; --text: #ffffff; --text-2: #c3c2b7; --muted: #8f8e86;
      --grid: #2e2e2c; --border: #3a3a37;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --text: #ffffff; --text-2: #c3c2b7; --muted: #8f8e86;
    --grid: #2e2e2c; --border: #3a3a37;
  }
  body { margin: 0; background: var(--surface); color: var(--text);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 960px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  h2 { font-size: 17px; margin: 40px 0 4px; }
  p.lead, p.note { color: var(--text-2); margin: 0 0 12px; }
  .chart { width: 100%; height: 340px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--text-2); font-weight: 600; }
  .scroll { overflow-x: auto; }
  code { font-size: 12px; }
</style>
</head>
<body>
<main>
  <h1>DataAgent evals</h1>
  <p class="lead">Execution accuracy of the production text-to-SQL path on fixed, difficulty-stratified
  subsets, one point per nightly run. Bands are 95% bootstrap confidence intervals: two points whose
  bands overlap heavily are not a real change.</p>

  <h2>Execution accuracy</h2>
  <div id="ex" class="chart" role="img" aria-label="Execution accuracy over commits with 95% CI bands"></div>

  <h2>Cost per question</h2>
  <p class="note">Mean LLM cost per question in USD, including self-repair attempts.</p>
  <div id="cost" class="chart" role="img" aria-label="Mean cost per question over commits"></div>

  <h2>All runs</h2>
  <div class="scroll"><table>
    <thead><tr><th>Date</th><th>Commit</th><th>Benchmark</th><th>n</th><th>EX</th><th>95% CI</th><th>$/question</th><th>Model</th></tr></thead>
    <tbody>
__TABLE__
    </tbody>
  </table></div>
</main>
<script>
const DATA = __DATA__;
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}
function draw() {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches &&
    document.documentElement.dataset.theme !== "light" || document.documentElement.dataset.theme === "dark";
  const colors = DATA.colors[dark ? "dark" : "light"];
  const axis = { gridcolor: css("--grid"), zeroline: false, linecolor: css("--border"),
                 tickfont: { color: css("--text-2") }, automargin: true };
  const layout = (yTitle, fmt) => ({
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: { color: css("--text"), family: "system-ui, sans-serif", size: 13 },
    margin: { l: 56, r: 16, t: 8, b: 40 }, hovermode: "x unified",
    legend: { orientation: "h", y: 1.12, font: { color: css("--text-2") } },
    xaxis: { ...axis, type: "category", title: { text: "commit", font: { color: css("--muted") } } },
    yaxis: { ...axis, tickformat: fmt, title: { text: yTitle, font: { color: css("--muted") } } },
  });
  const exTraces = [], costTraces = [];
  DATA.series.forEach((name, i) => {
    const rows = DATA.rows.filter(r => r.series === name);
    const x = rows.map(r => r.commit), c = colors[i % colors.length];
    exTraces.push(
      { x, y: rows.map(r => r.ci[1]), mode: "lines", line: { width: 0 }, hoverinfo: "skip",
        showlegend: false, legendgroup: name },
      { x, y: rows.map(r => r.ci[0]), mode: "lines", line: { width: 0 }, fill: "tonexty",
        fillcolor: hexA(c, 0.16), hoverinfo: "skip", showlegend: false, legendgroup: name },
      { x, y: rows.map(r => r.ex), name, legendgroup: name, mode: "lines+markers",
        line: { color: c, width: 2 }, marker: { size: 8, color: c, line: { color: css("--surface"), width: 2 } },
        customdata: rows.map(r => [r.ci[0], r.ci[1], r.n]),
        hovertemplate: "%{y:.1%} [%{customdata[0]:.1%}–%{customdata[1]:.1%}], n=%{customdata[2]}<extra>" + name + "</extra>" });
    costTraces.push({ x, y: rows.map(r => r.cost_usd_mean), name, mode: "lines+markers",
      line: { color: c, width: 2 }, marker: { size: 8, color: c, line: { color: css("--surface"), width: 2 } },
      hovertemplate: "$%{y:.5f}<extra>" + name + "</extra>" });
  });
  const cfg = { displayModeBar: false, responsive: true };
  Plotly.react("ex", exTraces, layout("execution accuracy", ".0%"), cfg);
  Plotly.react("cost", costTraces, layout("USD per question", "$.4f"), cfg);
}
draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
</script>
</body>
</html>
"""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Eval dashboard")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("append", help="append summary.json files to history")
    a.add_argument("summaries", nargs="+", type=Path)
    a.add_argument("--history", type=Path, required=True)
    a.add_argument("--commit", default=None)
    b = sub.add_parser("build", help="render index.html from history")
    b.add_argument("--history", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "append":
        commit = args.commit or _commit()
        args.history.parent.mkdir(parents=True, exist_ok=True)
        with open(args.history, "a") as f:
            for path in args.summaries:
                f.write(json.dumps(history_record(json.loads(path.read_text()), commit)) + "\n")
        print(f"appended {len(args.summaries)} record(s) to {args.history}")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(load_history(args.history)))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
