# When to Use a T-Test

## One-Sample T-Test
Use when comparing a sample mean to a known population mean.
- Requirements: Continuous data, approximately normal distribution, n >= 30 (or normal population for smaller n)
- Example: "Is our average order value different from the industry benchmark of $50?"

## Independent Two-Sample T-Test
Use when comparing means between two independent groups.
- Requirements: Two independent groups, continuous outcome, similar variances (use Welch's t-test if variances differ)
- Example: "Do customers from Region A spend more than customers from Region B?"

## Paired T-Test
Use when comparing means from the same group at two time points.
- Requirements: Paired/matched observations, continuous outcome
- Example: "Did customer satisfaction scores improve after the new policy?"

## When NOT to Use T-Test
- More than 2 groups: Use ANOVA instead
- Non-normal data with small n: Use Mann-Whitney U test
- Categorical outcomes: Use Chi-Square test
- Sample size < 10: Results are unreliable regardless of test choice

## Common Errors
- Using a t-test for proportions (use z-test for proportions or chi-square)
- Ignoring unequal variances (always check with Levene's test first)
- Multiple t-tests instead of ANOVA (inflates Type I error)
