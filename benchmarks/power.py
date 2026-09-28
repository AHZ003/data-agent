"""How big must a text-to-SQL improvement be before an eval can see it?

    python -m benchmarks.power

1. Minimum detectable effect (MDE) of a paired comparison (McNemar) at 80%
   power, alpha 0.05, as a function of n and the discordance rate (share
   of questions where exactly one config is right).
2. Simulation of two *identical* configs whose answers differ only by LLM
   non-determinism: how often does one look better by >= 1, 2, 3 points,
   and how often does McNemar (correctly) call it significant?
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy.stats import norm

from benchmarks.stats import mcnemar_exact


def mde_paired(n: int, discordance: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Approximate MDE (in accuracy points) for a paired test: Var(Δ) ≈ p_disc / n."""
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * math.sqrt(discordance / n)


def ci_halfwidth(n: int, p: float = 0.7) -> float:
    return norm.ppf(0.975) * math.sqrt(p * (1 - p) / n)


def simulate_identical(n: int, p: float = 0.7, flip: float = 0.08, trials: int = 2000, seed: int = 0) -> dict:
    """Two runs of the same config; each answer flips vs a shared 'true' outcome
    with probability `flip` (sampling non-determinism)."""
    rng = np.random.default_rng(seed)
    gaps, sig = [], 0
    for _ in range(trials):
        truth = rng.random(n) < p
        a = np.where(rng.random(n) < flip, ~truth, truth)
        b = np.where(rng.random(n) < flip, ~truth, truth)
        gaps.append(abs(b.mean() - a.mean()))
        sig += mcnemar_exact(a, b).p_value < 0.05
    gaps = np.array(gaps)
    return {"n": n, "p_gap_ge_1pt": float((gaps >= 0.01).mean()), "p_gap_ge_2pt": float((gaps >= 0.02).mean()),
            "p_gap_ge_3pt": float((gaps >= 0.03).mean()), "false_positive_rate": sig / trials}


def main() -> int:
    print("Minimum detectable effect (paired, 80% power, alpha .05), accuracy points\n")
    print("| n | disc. 10% | disc. 15% | disc. 20% | unpaired 95% CI half-width at 70% |")
    print("|---|---|---|---|---|")
    for n in (100, 200, 500, 1034):
        row = " | ".join(f"{mde_paired(n, d) * 100:.1f}" for d in (0.10, 0.15, 0.20))
        print(f"| {n} | {row} | ±{ci_halfwidth(n) * 100:.1f} |")
    print("\nTwo identical configs (8% of answers flip run-to-run), 2,000 simulated A/B tests\n")
    print("| n | looks >=1 pt different | >=2 pts | >=3 pts | McNemar p<.05 |")
    print("|---|---|---|---|---|")
    for n in (200, 1034):
        s = simulate_identical(n)
        print(f"| {n} | {s['p_gap_ge_1pt']:.0%} | {s['p_gap_ge_2pt']:.0%} | {s['p_gap_ge_3pt']:.0%} | "
              f"{s['false_positive_rate']:.1%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
