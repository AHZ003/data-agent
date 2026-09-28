"""CI-aware regression gate.

    # after an eval run:
    python -m benchmarks.gate --results outputs/eval/results.json --baseline benchmarks/baselines/golden.json
    # promote a run to the new baseline (commit the file):
    python -m benchmarks.gate --results outputs/eval/results.json --baseline benchmarks/baselines/golden.json --write-baseline

A plain `score < baseline` gate fails roughly half of all no-op PRs on
a noisy 18-case eval. This gate fails only when the current run's
*upper* 95% bootstrap bound is below the baseline's point estimate:
the drop is bigger than sampling noise can explain. With few cases the
CI is wide, so only large regressions block a merge; that is the honest
trade-off, and the fix for it is more cases, not a tighter threshold.

Results files are JSON lists of per-item dicts with a boolean field
(`passed` for the golden runner, `match` for text-to-SQL runs).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.stats import ci_gate  # noqa: E402


def outcomes(results_path: Path) -> list[bool]:
    items = json.loads(results_path.read_text())
    key = "passed" if items and "passed" in items[0] else "match"
    return [bool(i[key]) for i in items]


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return ""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CI-aware eval regression gate")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)

    current = outcomes(args.results)
    if args.write_baseline:
        point = sum(current) / len(current)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(
            {"point": point, "n": len(current), "commit": _commit(), "source": str(args.results)}, indent=2
        ) + "\n")
        print(f"Wrote baseline {point:.1%} (n={len(current)}) to {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}; gate skipped. Create one with --write-baseline.")
        return 0

    baseline = json.loads(args.baseline.read_text())
    passed, ci = ci_gate(current, baseline["point"])
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict}: current {ci.fmt()} (n={len(current)}) vs baseline {baseline['point']:.1%} "
          f"(n={baseline['n']}, {baseline.get('commit') or 'unknown commit'})")
    if not passed:
        print("The upper 95% bound is below the baseline: this drop is larger than noise.", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
