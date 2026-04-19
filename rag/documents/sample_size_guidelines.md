# Sample Size Guidelines

## Minimum Sample Sizes by Analysis Type

### Descriptive Statistics
- Mean/median: n >= 30 for reliable estimates
- Proportions: n >= 100 for stable percentages
- Rare events (< 5%): n >= 500

### Statistical Tests
- T-test: n >= 30 per group (minimum 10 per group for normal data)
- Chi-square: Expected frequency >= 5 in each cell
- ANOVA: n >= 20 per group
- Correlation: n >= 30 for meaningful r value

### Machine Learning
- Regression: 10-20 observations per feature
- Classification: At least 50 per class (ideally 100+)
- Clustering: At least 10x the number of features
- Time series forecasting: At least 2 complete seasonal cycles

## Red Flags
- Reporting percentages with n < 20: "80% of respondents" when n=5 is misleading
- Correlation with n < 10: Almost any pattern looks significant
- Trends from 2-3 data points: Not a trend, just noise
- Predictions from < 12 time points: Unreliable without strong domain knowledge

## Rule of Thumb
When in doubt, apply the "n >= 30" rule. Below 30, explicitly state the sample size
and note that results should be interpreted with caution.
