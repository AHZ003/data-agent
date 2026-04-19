# Chart Selection Guide

## Choosing the Right Chart

### Comparison
- **Bar chart**: Compare values across categories (< 15 categories)
- **Horizontal bar**: Categories have long labels or > 7 categories
- **Grouped bar**: Compare multiple measures across categories
- **Bullet chart**: Compare actual vs target

### Trend / Time Series
- **Line chart**: Show change over time (ordered data required)
- **Area chart**: Show cumulative trends or composition over time
- **Sparkline**: Compact trend indicator

### Distribution
- **Histogram**: Show frequency distribution of a single variable
- **Box plot**: Show median, quartiles, and outliers
- **Violin plot**: Show distribution shape

### Composition
- **Pie chart**: Show parts of a whole (MAX 6 slices, use "Other" for rest)
- **Stacked bar**: Show composition across categories
- **Treemap**: Show hierarchical composition

### Relationship
- **Scatter plot**: Show relationship between two numeric variables
- **Bubble chart**: Add a third dimension (size) to scatter
- **Heatmap**: Show correlation matrix or cross-tabulation

### KPI / Single Value
- **Big number / KPI card**: Highlight a single important metric
- **Gauge**: Show progress toward a target

## Anti-Patterns to Avoid
- Pie chart with more than 6 slices (use bar chart instead)
- 3D charts (distort perception)
- Dual y-axes without clear labeling
- Truncated y-axis that exaggerates differences
- Line chart for unordered categorical data
