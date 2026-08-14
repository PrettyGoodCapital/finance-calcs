# QuantStats Compatibility

`finance-calcs` uses QuantStats as a reference for report-oriented metric names
and definitions while retaining Polars expressions, explicit missing-value
semantics, and composability in lazy queries.

The following concepts have compatible definitions. `finance-calcs` exposes
only its canonical names; it does not export additional QuantStats spellings.

| QuantStats name       | finance-calcs API             |
| --------------------- | ----------------------------- |
| `best`, `worst`       | `best_return`, `worst_return` |
| `avg_win`, `avg_loss` | `average_win`, `average_loss` |
| `cagr`                | `annualized_return`           |
| `expected_shortfall`  | `expected_shortfall`          |
| `to_drawdown_series`  | `drawdown_series`             |
| `r_squared`           | `r_squared`                   |
| `gain_to_pain_ratio`  | `gain_to_pain_ratio`          |
| `recovery_factor`     | `recovery_factor`             |
| `kelly_criterion`     | `kelly_criterion`             |

Intentional interface differences:

- `expected_shortfall` accepts `tail_probability`, so 5% is
  `tail_probability=0.05`,
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
