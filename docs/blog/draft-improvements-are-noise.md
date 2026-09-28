# Most text-to-SQL "improvements" are noise

_Draft. Sections marked **[results pending]** need the ablation runs; every
other number here is computed by code in this repo._

You change a prompt, rerun your eval on 200 questions, and accuracy goes from
71% to 73%. Did the change help?

Probably not provably. Two runs of the **same** configuration differ by that
much about half the time.

## How noisy is a 200-question eval?

`benchmarks/power.py` simulates two identical configurations whose answers
differ only through sampling non-determinism (8% of answers flip between runs)
and replays the A/B test 2,000 times:

| Questions | Identical configs look ≥1 pt apart | ≥2 pts | ≥3 pts | McNemar p < .05 |
|---|---|---|---|---|
| 200 | 77% | 51% | 30% | 3.7% |
| 1,034 (Spider dev) | 41% | 12% | 1% | 4.9% |

The first three columns are how often you would see a "gain" that isn't there.
The last column is how often a proper paired test is fooled: close to the 5%
it promises.

## What can an eval of size n detect at all?

For a paired comparison (same questions, two configs), the variance of the
accuracy difference is roughly the discordance rate (share of questions where
exactly one config is right) divided by n. At 80% power and α = .05, the
minimum detectable effect is:

| n | 10% discordant | 15% | 20% | Unpaired 95% CI at 70% accuracy |
|---|---|---|---|---|
| 100 | 8.9 pts | 10.9 | 12.5 | ±9.0 |
| 200 | 6.3 | 7.7 | 8.9 | ±6.4 |
| 500 | 4.0 | 4.9 | 5.6 | ±4.0 |
| 1,034 | 2.8 | 3.4 | 3.9 | ±2.8 |

On a 200-question subset, a prompt tweak has to be worth 6–9 points before an
eval can reliably see it. On the full Spider dev set, ~3 points is the floor.

## What to do instead

1. **Report every accuracy with a CI.** Accuracy is a mean of 0/1 outcomes, so
   its bootstrap distribution is exactly Binomial(n, p̂)/n — cheap to sample
   (`benchmarks/stats.py`).
2. **Compare configs on the same questions, with McNemar's exact test.** Only
   the discordant questions carry information; under "no difference" each is a
   coin flip.
3. **Gate CI on the upper bound, not the point estimate.** A PR fails only when
   the new run's upper 95% bound falls below the baseline — i.e. the drop is
   bigger than noise (`benchmarks/gate.py`). A plain `score < baseline` gate
   fails about half of all no-op PRs.
4. **Measure the harness itself.** Feed gold SQL through the production
   execution path; it must score 100%. Ours didn't at first: pandas turned SQL
   `NULL` into `NaN`, and `NaN != None`, so any answer containing a NULL could
   never match. That bug would have silently depressed every score.

## Results on DataAgent **[results pending]**

- Self-repair loop on vs off, Spider dev subset: Δ = __ pts [__, __], McNemar p = __.
- Each retrieval component on BIRD: Δ, CI, p.
- How many of our "improvements" survive the paired test: __ of __.

## Reproduce

```bash
python -m benchmarks.power                 # the tables above
python -m benchmarks.ablate --plan         # what the ablations will cost
python -m benchmarks.ablate                # paired deltas + McNemar
```
