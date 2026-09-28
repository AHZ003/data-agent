"""Cost/accuracy frontier: one point per config, 95% CI error bars.

    python -m benchmarks.pareto --benchmark spider      # after benchmarks.ablate (+ routing.train)

Reads outputs/ablations/summary.json (and outputs/routing/frontier.json if
present) and writes docs/pareto_<benchmark>.svg plus a markdown table. The
SVG is hand-drawn (no plotting dependency) so it renders in the GitHub
README. x = mean cost per question (log scale), y = execution accuracy;
filled points are on the Pareto frontier (nothing is both cheaper and
more accurate), and the frontier is drawn as a step line.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# Categorical slots 1-2 of the validated reference palette (see dataviz skill).
C_CONFIG, C_ROUTER = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def frontier(points: list[dict]) -> list[dict]:
    """Non-dominated points, sorted by cost."""
    pts = sorted(points, key=lambda p: (p["cost_mean"], -p["ex"]))
    out, best = [], -1.0
    for p in pts:
        if p["ex"] > best:
            out.append(p)
            best = p["ex"]
    return out


def load_points(benchmark: str) -> list[dict]:
    summary = json.loads((ROOT / "outputs" / "ablations" / "summary.json").read_text())
    pts = [{"name": r["name"], "ex": r["ex"], "ci": r["ci"], "cost_mean": r["cost_mean"], "kind": "config"}
           for r in summary["rows"] if r["benchmark"] == benchmark and r["cost_mean"] > 0]
    router = ROOT / "outputs" / "routing" / "frontier.json"
    if router.exists():
        c = json.loads(router.read_text())["chosen"]
        pts.append({"name": f"router (t={c['threshold']:.2f}, {c['share_strong']:.0%} strong)",
                    "ex": c["ex"], "ci": c["ci"], "cost_mean": c["cost_mean"], "kind": "router"})
    return pts


def svg(points: list[dict], title: str, w: int = 720, h: int = 420) -> str:
    left, right, top, bottom = 64, 24, 48, 56
    xs = [math.log10(p["cost_mean"]) for p in points]
    x0, x1 = math.floor(min(xs)), math.ceil(max(xs))
    x1 = x1 if x1 > x0 else x0 + 1
    y0 = max(0.0, math.floor(min(p["ci"][0] for p in points) * 10) / 10)
    y1 = min(1.0, math.ceil(max(p["ci"][1] for p in points) * 10) / 10)
    y1 = y1 if y1 > y0 else y0 + 0.1
    X = lambda c: left + (math.log10(c) - x0) / (x1 - x0) * (w - left - right)  # noqa: E731
    Y = lambda v: top + (1 - (v - y0) / (y1 - y0)) * (h - top - bottom)  # noqa: E731
    on = {id(p) for p in frontier(points)}
    e = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="system-ui, sans-serif" '
             f'role="img" aria-label="{e(title)}"><rect width="{w}" height="{h}" fill="#fcfcfb"/>',
             f'<text x="{left}" y="24" font-size="15" font-weight="600" fill="{INK}">{e(title)}</text>']
    for k in range(int(round((y1 - y0) * 10)) + 1):
        v = y0 + k / 10
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/>'
                     f'<text x="{left - 8}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{INK_2}">{v:.0%}</text>')
    for d in range(x0, x1 + 1):
        parts.append(f'<text x="{X(10 ** d):.1f}" y="{h - bottom + 18}" font-size="11" text-anchor="middle" '
                     f'fill="{INK_2}">${10 ** d:g}</text>')
    parts.append(f'<text x="{(left + w - right) / 2}" y="{h - 12}" font-size="12" text-anchor="middle" fill="{INK_2}">'
                 'mean cost per question (USD, log scale)</text>')
    parts.append(f'<text transform="translate(16 {(top + h - bottom) / 2}) rotate(-90)" font-size="12" '
                 f'text-anchor="middle" fill="{INK_2}">execution accuracy (95% CI)</text>')
    fr = frontier(points)
    if len(fr) > 1:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(p['cost_mean']):.1f},{Y(p['ex']):.1f}" for i, p in enumerate(fr))
        parts.append(f'<path d="{d}" fill="none" stroke="{INK_2}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    for p in points:
        cx, cy = X(p["cost_mean"]), Y(p["ex"])
        color = C_ROUTER if p["kind"] == "router" else C_CONFIG
        fill = color if id(p) in on else "#fcfcfb"
        parts.append(
            f'<g><title>{e(p["name"])}: {p["ex"]:.1%} [{p["ci"][0]:.1%}–{p["ci"][1]:.1%}], ${p["cost_mean"]:.5f}/q</title>'
            f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(p["ci"][0]):.1f}" y2="{Y(p["ci"][1]):.1f}" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{fill}" stroke="{color}" stroke-width="2"/>'
            f'<text x="{cx + 9:.1f}" y="{cy - 7:.1f}" font-size="11" fill="{INK}">{e(p["name"])}</text></g>')
    parts.append("</svg>")
    return "\n".join(parts)


def table(points: list[dict]) -> str:
    on = {id(p) for p in frontier(points)}
    rows = ["| Config | EX | 95% CI | $/question | On frontier |", "|---|---|---|---|---|"]
    for p in sorted(points, key=lambda p: p["cost_mean"]):
        rows.append(f"| {p['name']} | {p['ex']:.1%} | {p['ci'][0]:.1%}–{p['ci'][1]:.1%} | "
                    f"{p['cost_mean']:.5f} | {'yes' if id(p) in on else ''} |")
    return "\n".join(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="spider")
    args = parser.parse_args(argv)
    points = load_points(args.benchmark)
    if not points:
        print("No ablation results with cost > 0 for this benchmark; run benchmarks.ablate first.")
        return 1
    out = ROOT / "docs" / f"pareto_{args.benchmark}.svg"
    out.write_text(svg(points, f"Cost vs accuracy — {args.benchmark}"))
    print(table(points))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
