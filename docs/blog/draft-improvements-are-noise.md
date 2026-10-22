# Most text-to-SQL "improvements" are noise

_Draft. Every number here is computed by code in this repo; DataAgent results are from Gemini 3.8 Flash on 200-question stratified subsets._

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

## Results on DataAgent

Eight comparisons, each a plausible "improvement" someone would ship:

| Change | Δ accuracy | 95% CI | McNemar p |
|---|---|---|---|
| Self-repair loop (Spider) | +0.5 | −1.0, +2.5 | 1.0 |
| Self-repair loop (BIRD) | +0.0 | −3.0, +3.0 | 1.0 |
| Value hints (BIRD) | +1.0 | −1.5, +4.0 | 0.73 |
| Schema linking (BIRD) | −1.0 | −4.5, +2.5 | 0.79 |
| Few-shot memory (BIRD) | +0.0 | −2.5, +2.5 | 1.0 |
| All retrieval combined (BIRD) | +3.5 | −0.5, +7.5 | 0.14 |
| Gemini 3.1 Pro instead of 3.8 Flash (Spider) | +2.0 | −1.5, +5.5 | 0.42 |
| BIRD evidence hints removed | **−17.5** | −23.5, −11.5 | **8×10⁻⁸** |

Read naively, several of these are headlines ("all retrieval combined adds 3.5
points!", "Pro beats Flash by 2!"). None of the first seven survives the paired
test. The one that does — taking away
BIRD's domain hints — is a 17-point effect, the size this eval can actually see.
The self-repair loop, a staple of agent demos, costs 6% more per question and
does nothing measurable here: modern models' SQL almost always runs; their
mistakes are queries that run and return the wrong thing.

## Reproduce

```bash
python -m benchmarks.power                 # the tables above
python -m benchmarks.ablate --plan         # what the ablations will cost
python -m benchmarks.ablate                # paired deltas + McNemar
```
