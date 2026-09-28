"""Train and evaluate the router from cached ablation results.

    python -m routing.train --cheap spider-flash-repair --strong spider-pro-repair

Both runs must exist in benchmarks/ablations.yaml and have been run (their
per-example results are read from the cache; nothing is re-run). Questions
are split in half by a fixed seed: fit on one half, report on the other.
Writes routing/router.json and outputs/routing/frontier.json (one point per
threshold, used by benchmarks/pareto.py).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dataset(cheap_run: dict, strong_run: dict):
    from benchmarks.ablate import CACHE_DIR, examples_for, to_config
    from benchmarks.text2sql import cache_path, load_cached
    from core.datasource import DataSource
    from routing.router import featurize

    examples, db_path_for, _ = examples_for(cheap_run)
    cheap = load_cached(cache_path(to_config(cheap_run), CACHE_DIR))
    strong = load_cached(cache_path(to_config(strong_run), CACHE_DIR))
    common = [ex for ex in examples if ex.id in cheap and ex.id in strong]
    if not common:
        raise SystemExit("no cached results for both runs; run `python -m benchmarks.ablate --only ...` first")
    schemas: dict = {}
    X, rows = [], []
    for ex in common:
        path = db_path_for(ex)
        if path not in schemas:
            schemas[path] = DataSource.from_sqlite(str(path))
        ds = schemas[path]
        X.append(featurize(ex.question, ds.table_names, sum(len(t.columns) for t in ds.tables), ex.evidence))
        c, s = cheap[ex.id], strong[ex.id]
        rows.append((c.match, s.match, c.cost_usd, s.cost_usd))
    return np.array(X), np.array(rows, dtype=float)


def main(argv=None) -> int:
    from benchmarks.ablate import DEFAULT_YAML, load_runs, to_config
    from benchmarks.stats import bootstrap_ci
    from routing.router import fit, simulate

    parser = argparse.ArgumentParser()
    parser.add_argument("--cheap", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    runs = {r["name"]: r for r in load_runs(DEFAULT_YAML)}
    X, R = dataset(runs[args.cheap], runs[args.strong])
    idx = list(range(len(X)))
    random.Random(args.seed).shuffle(idx)
    fit_idx, test_idx = idx[: len(idx) // 2], idx[len(idx) // 2:]
    router = fit(X[fit_idx], 1 - R[fit_idx, 0], to_config(runs[args.cheap]).model, to_config(runs[args.strong]).model)
    curve = simulate(router, X[test_idx], R[test_idx, 0], R[test_idx, 1], R[test_idx, 2], R[test_idx, 3])
    for pt in curve:
        ci = bootstrap_ci(pt.pop("matches"))
        pt["ci"] = [ci.lo, ci.hi]
    strong_ex = R[test_idx, 1].mean()
    # Pick the cheapest threshold within 1 point of the strong model's accuracy.
    ok = [p for p in curve if p["ex"] >= strong_ex - 0.01] or [max(curve, key=lambda p: p["ex"])]
    best = min(ok, key=lambda p: p["cost_mean"])
    router.threshold = best["threshold"]
    router.save(ROOT / "routing" / "router.json")
    out = ROOT / "outputs" / "routing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "frontier.json").write_text(json.dumps({
        "cheap": {"ex": float(R[test_idx, 0].mean()), "cost_mean": float(R[test_idx, 2].mean())},
        "strong": {"ex": float(strong_ex), "cost_mean": float(R[test_idx, 3].mean())},
        "curve": curve, "chosen": best, "n_test": len(test_idx),
    }, indent=2))
    print(f"held-out n={len(test_idx)}: cheap {R[test_idx, 0].mean():.1%}, strong {strong_ex:.1%}; "
          f"router @t={best['threshold']:.2f}: {best['ex']:.1%} at ${best['cost_mean']:.5f}/q "
          f"({best['share_strong']:.0%} routed to strong)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
