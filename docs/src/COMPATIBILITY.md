# QuantStats Compatibility

`finance-calcs` uses QuantStats as a reference for report-oriented metric names
and definitions while retaining Polars expressions, explicit missing-value
semantics, and composability in lazy queries.

The following names have compatible definitions:

| QuantStats name       | finance-calcs API                                       |
| --------------------- | ------------------------------------------------------- |
| `best`, `worst`       | Aliases for `best_return`, `worst_return`               |
| `avg_win`, `avg_loss` | Aliases for `average_win`, `average_loss`               |
| `cagr`                | Alias for `annualized_return`                           |
| `expected_shortfall`  | Alias for `conditional_value_at_risk`                   |
| `to_drawdown_series`  | Alias for `drawdown_series`                             |
| `r_squared`           | Squared strategy/benchmark Pearson correlation          |
| `gain_to_pain_ratio`  | Arithmetic net return divided by absolute summed losses |
| `recovery_factor`     | Absolute arithmetic net return divided by max drawdown  |
| `kelly_criterion`     | Active-period win rate and average win/loss payoff      |

Intentional interface differences:

- `expected_shortfall` accepts a lower-tail `cutoff`, so 5% is `cutoff=0.05`,
  rather than a confidence value of 95%.
- `drawdown_details` accepts periodic returns and optional dates directly. It
  returns native Polars date/index values and decimal drawdowns rather than
  formatted date strings and percentage values.
- Missing `NaN` and null returns are excluded from statistical aggregations and
  treated as neutral for compounding.
- Rolling maximum drawdown rebases equity and its initial 1.0 baseline inside
  each trailing window.

Fixed-fixture tests lock these semantics without adding QuantStats as a runtime
or development dependency. See the upstream
[QuantStats statistics reference](https://github.com/ranaroussi/quantstats/blob/main/quantstats/stats.py)
for the comparison implementation.
