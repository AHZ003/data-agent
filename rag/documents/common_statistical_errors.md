# Common Statistical Errors

## 1. Correlation vs Causation
Just because two variables move together does not mean one causes the other.
- Always consider confounding variables
- Use language like "associated with" not "caused by"
- Example: Ice cream sales and drowning are correlated (both increase in summer)

## 2. Simpson's Paradox
A trend that appears in several groups reverses when the groups are combined.
- Always check if aggregating hides important subgroup differences
- Example: Treatment A beats B in each age group, but B beats A overall

## 3. Base Rate Neglect
Ignoring the prior probability when interpreting results.
- Example: "Sales doubled!" but from 2 to 4 is not meaningful growth

## 4. Survivorship Bias
Analyzing only the surviving/visible data.
- Example: "Successful companies all did X" ignores failed companies that also did X

## 5. Multiple Comparisons Problem
Testing many hypotheses increases the chance of false positives.
- With 20 tests at p=0.05, expect 1 false positive
- Apply Bonferroni correction: use p = 0.05 / number of tests

## 6. Ecological Fallacy
Assuming what's true for a group is true for individuals.
- Example: States with higher average income don't mean every person is wealthy

## 7. Overfitting
Model fits training data perfectly but fails on new data.
- Signs: Very high R-squared with many predictors, high MAPE on test set
- Fix: Use cross-validation, reduce model complexity

## 8. Cherry-Picking Time Ranges
Selecting time ranges that support a narrative.
- Always show full available data range
- Let the user choose the range, don't select it to prove a point
