# Regression Assumptions Checklist

## Linear Regression Assumptions (must check all)

### 1. Linearity
The relationship between X and Y is linear.
- Check: Scatter plot of residuals vs fitted values should show no pattern
- Fix: Transform variables (log, sqrt) or use polynomial regression

### 2. Independence
Observations are independent of each other.
- Check: Durbin-Watson test (value near 2 = independent)
- Common violation: Time series data (use autocorrelation-aware models)

### 3. Homoscedasticity (Constant Variance)
Residuals have constant variance across all levels of X.
- Check: Residuals vs fitted values plot should have uniform spread
- Fix: Use weighted least squares or transform the dependent variable

### 4. Normality of Residuals
Residuals are approximately normally distributed.
- Check: Q-Q plot, Shapiro-Wilk test
- Note: Less important with large samples (n > 30) due to CLT

### 5. No Multicollinearity (for multiple regression)
Predictors are not highly correlated with each other.
- Check: VIF (Variance Inflation Factor) < 5 for each predictor
- Fix: Remove one of the correlated predictors or use PCA

## Minimum Sample Sizes
- Simple regression: n >= 20 (rule of thumb: 10-20 per predictor)
- Multiple regression: n >= 50 + 8 * number of predictors
- R-squared interpretation: Adjusted R-squared for multiple predictors
