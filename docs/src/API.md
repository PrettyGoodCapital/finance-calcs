# API

`finance-calcs` exposes calculation functions at the top level and through the
`.finance` namespace on both `polars.Expr` and `polars.Series`. Most functions
return `pl.Expr` so they compose naturally inside `select`, `with_columns`, and
lazy pipelines. A small number of statistical and post-trade helpers take
concrete `pl.Series` or `pl.DataFrame` inputs because they compute sample-level
summaries, extract round trips, or fit extreme-value routines outside the Polars
expression engine.

Use this page as the complete public API map. Function signatures below are
shown in a compact form; the reference blocks at the end of each section are
rendered by yardang/Sphinx from the live docstrings.

______________________________________________________________________

## Namespace and Windowing

Every expression function can be called directly:

```python
import polars as pl
import finance_calcs as fc

out = df.select(fc.sharpe(pl.col("ret")).alias("sharpe"))
```

or through the namespace:

```python
out = df.select(pl.col("ret").finance.sharpe().alias("sharpe"))
```

The same namespace exists on `pl.Series` for eager one-off checks:

```python
value = returns.finance.sharpe()
```

Across return, risk, alpha, factor, and tail metrics, `window=` means a rolling
row-count window. `period=` means bucketed calculations over a calendar or
custom period. `period=` accepts:

- `finance_enums.Frequency`, such as `Frequency.Month`
- aliases accepted by `finance_enums.to_frequency()`, such as `"monthly"`
- Polars `dt.truncate()` duration strings, such as `"1q"` or `"2w"`
- a precomputed bucket expression, such as `pl.col("fiscal_period")`

When `period` is a frequency or duration string, pass `date=pl.col("date")` so
`finance-calcs` can build the bucket expression.

______________________________________________________________________

## Returns and Periods

Return functions turn prices into returns, compound return paths, or terminal
period returns. They are the base layer for most risk and factor metrics.

| Function                                                                                       | Use it for                                | Notes                                                                            |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------- |
| `period_bucket(date, period)`                                                                  | Build a reusable period bucket from dates | Accepts `Frequency`, aliases, Polars durations, or an existing bucket expression |
| `simple_returns(price)`                                                                        | Arithmetic price returns                  | Computes `price / price.shift(1) - 1`                                            |
| `log_returns(price)`                                                                           | Log price returns                         | Computes `log(price / price.shift(1))`                                           |
| `cum_returns(returns, starting_value=0.0, *, window=None, period=None, date=None)`             | Compounded return path                    | Resets inside each rolling window or period bucket                               |
| `cum_returns_final(returns, *, window=None, period=None, date=None)`                           | Terminal compounded return                | Produces the final compound return for the sample, window, or bucket             |
| `returns(returns, *, window=None, period=None, date=None)`                                     | Alias-style aggregate return              | Same aggregation as `cum_returns_final`                                          |
| `aggregate_returns(returns, date, period)`                                                     | Calendar/custom period compound return    | Convenience wrapper around `returns(..., period=..., date=...)`                  |
| `annualized_return(returns, periods_per_year=252, *, window=None, period=None, date=None)`     | Annualized geometric return               | Uses compound return and observation count                                       |
| `annualized_volatility(returns, periods_per_year=252, *, window=None, period=None, date=None)` | Annualized standard deviation             | `std * sqrt(periods_per_year)`                                                   |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: period_bucket
.. autofunction:: simple_returns
.. autofunction:: log_returns
.. autofunction:: cum_returns
.. autofunction:: cum_returns_final
.. autofunction:: returns
.. autofunction:: aggregate_returns
.. autofunction:: annualized_return
.. autofunction:: annualized_volatility
```

______________________________________________________________________

## Risk and Drawdown

Risk metrics operate on return expressions. Scalar `risk_free` inputs are annual
rates and are converted to per-period rates where appropriate; expression
`risk_free` inputs are treated as already per-period.

| Function                                                                                                         | Use it for                            | Notes                                                            |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------- |
| `volatility(returns, periods_per_year=252, *, window=None, period=None, date=None)`                              | Annualized volatility                 | Alias for `annualized_volatility`                                |
| `sharpe(returns, risk_free=0.0, periods_per_year=252, *, window=None, period=None, date=None)`                   | Annualized Sharpe ratio               | Supports scalar annual risk-free rates or per-period expressions |
| `sortino(returns, required_return=0.0, periods_per_year=252, *, window=None, period=None, date=None)`            | Annualized Sortino ratio              | Uses downside deviation below `required_return`                  |
| `calmar(returns, periods_per_year=252, *, window=None, period=None, date=None)`                                  | Annualized return / abs(max drawdown) | Uses the same sample/window/period controls                      |
| `downside_deviation(returns, required_return=0.0, periods_per_year=252, *, window=None, period=None, date=None)` | Annualized semi-deviation             | Squares only observations below the threshold                    |
| `downside_risk(...)`                                                                                             | Naming alias for downside deviation   | Same arguments and result as `downside_deviation`                |
| `drawdown_series(returns, *, period=None, date=None)`                                                            | Running drawdown path                 | Equity curve divided by running peak minus one                   |
| `underwater_series(returns, *, period=None, date=None)`                                                          | Drawdown path alias                   | Same result as `drawdown_series`                                 |
| `max_drawdown(returns, *, window=None, period=None, date=None)`                                                  | Most negative drawdown                | Supports lifetime, rolling, or period-bucketed drawdown          |
| `value_at_risk(returns, cutoff=0.05, *, window=None, period=None, date=None)`                                    | Historical VaR quantile               | Returns the lower-tail return quantile                           |
| `conditional_value_at_risk(returns, cutoff=0.05, *, window=None, period=None, date=None)`                        | Expected shortfall                    | Mean return of observations at or below VaR                      |
| `parametric_var(returns, cutoff=0.05, *, period=None, date=None)`                                                | Gaussian VaR                          | Supports common cutoffs from the built-in z-score table          |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: volatility
.. autofunction:: sharpe
.. autofunction:: sortino
.. autofunction:: calmar
.. autofunction:: downside_deviation
.. autofunction:: downside_risk
.. autofunction:: drawdown_series
.. autofunction:: underwater_series
.. autofunction:: max_drawdown
.. autofunction:: value_at_risk
.. autofunction:: conditional_value_at_risk
.. autofunction:: parametric_var
```

______________________________________________________________________

## Overlap and Price Channels

Overlap studies smooth prices or build price channels from high/low/close data.
The `period` argument in this section is an indicator lookback length, not a
calendar bucket.

| Function                                       | Use it for                         | Notes                                             |
| ---------------------------------------------- | ---------------------------------- | ------------------------------------------------- |
| `sma(close, period=20)`                        | Simple moving average              | Rolling mean                                      |
| `ema(close, period=20)`                        | Exponential moving average         | Uses Polars EWM mean with `span=period`           |
| `wma(close, period=20)`                        | Weighted moving average            | Recent observations receive larger linear weights |
| `dema(close, period=20)`                       | Double EMA                         | `2 * EMA - EMA(EMA)`                              |
| `tema(close, period=20)`                       | Triple EMA                         | `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))`              |
| `midpoint(close, period=14)`                   | Midpoint of rolling high/low close | Uses close-only rolling max/min                   |
| `midprice(high, low, period=14)`               | Midpoint of high/low channel       | Uses rolling high max and low min                 |
| `bbands_upper(close, period=20, nbdev_up=2.0)` | Bollinger upper band               | Middle plus standard-deviation multiple           |
| `bbands_middle(close, period=20)`              | Bollinger middle band              | SMA                                               |
| `bbands_lower(close, period=20, nbdev_dn=2.0)` | Bollinger lower band               | Middle minus standard-deviation multiple          |
| `donchian_upper(high, period=20)`              | Donchian upper channel             | Rolling high maximum                              |
| `donchian_lower(low, period=20)`               | Donchian lower channel             | Rolling low minimum                               |
| `donchian_middle(high, low, period=20)`        | Donchian midline                   | Average of upper and lower channels               |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: sma
.. autofunction:: ema
.. autofunction:: wma
.. autofunction:: dema
.. autofunction:: tema
.. autofunction:: midpoint
.. autofunction:: midprice
.. autofunction:: bbands_upper
.. autofunction:: bbands_middle
.. autofunction:: bbands_lower
.. autofunction:: donchian_upper
.. autofunction:: donchian_lower
.. autofunction:: donchian_middle
```

______________________________________________________________________

## Momentum

Momentum functions consume close or OHLC expressions and return oscillator,
rate-of-change, or directional-movement expressions. The `period` argument is an
indicator lookback length.

| Function                                           | Use it for                 | Notes                                        |
| -------------------------------------------------- | -------------------------- | -------------------------------------------- |
| `rsi(close, period=14)`                            | Relative Strength Index    | Wilder smoothing                             |
| `macd_line(close, fast=12, slow=26)`               | MACD line                  | Fast EMA minus slow EMA                      |
| `macd_signal(close, fast=12, slow=26, signal=9)`   | MACD signal line           | EMA of `macd_line`                           |
| `macd_hist(close, fast=12, slow=26, signal=9)`     | MACD histogram             | MACD line minus signal line                  |
| `mom(close, period=10)`                            | Price momentum             | Difference from `period` bars ago            |
| `roc(close, period=10)`                            | Percent rate of change     | `100 * (close / close.shift(period) - 1)`    |
| `rocp(close, period=10)`                           | Decimal rate of change     | `(close - prior) / prior`                    |
| `rocr(close, period=10)`                           | Price ratio                | `close / prior`                              |
| `rocr100(close, period=10)`                        | Price ratio scaled by 100  | `100 * rocr`                                 |
| `willr(high, low, close, period=14)`               | Williams %R                | Close location within rolling high/low range |
| `stoch_k(high, low, close, period=14)`             | Fast stochastic %K         | Range-normalized close                       |
| `stoch_d(high, low, close, period=14, d_period=3)` | Stochastic %D              | SMA of `%K`                                  |
| `cci(high, low, close, period=20)`                 | Commodity Channel Index    | Typical-price deviation oscillator           |
| `cmo(close, period=14)`                            | Chande Momentum Oscillator | Up/down movement balance                     |
| `trix(close, period=15)`                           | TRIX                       | One-bar ROC of triple-smoothed log price     |
| `plus_dm(high, low)`                               | Raw +DM                    | Wilder directional movement                  |
| `minus_dm(high, low)`                              | Raw -DM                    | Wilder directional movement                  |
| `plus_di(high, low, close, period=14)`             | +DI                        | Smoothed +DM divided by true range           |
| `minus_di(high, low, close, period=14)`            | -DI                        | Smoothed -DM divided by true range           |
| `adx(high, low, close, period=14)`                 | Average Directional Index  | Trend-strength measure from +DI and -DI      |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: rsi
.. autofunction:: macd_line
.. autofunction:: macd_signal
.. autofunction:: macd_hist
.. autofunction:: mom
.. autofunction:: roc
.. autofunction:: rocp
.. autofunction:: rocr
.. autofunction:: rocr100
.. autofunction:: willr
.. autofunction:: stoch_k
.. autofunction:: stoch_d
.. autofunction:: cci
.. autofunction:: cmo
.. autofunction:: trix
.. autofunction:: plus_dm
.. autofunction:: minus_dm
.. autofunction:: plus_di
.. autofunction:: minus_di
.. autofunction:: adx
```

______________________________________________________________________

## Volatility Indicators

These functions estimate realized or range-based volatility from returns or
OHLC bars. The `period` argument is an indicator lookback length.

| Function                                                     | Use it for                                | Notes                                              |
| ------------------------------------------------------------ | ----------------------------------------- | -------------------------------------------------- |
| `true_range(high, low, close)`                               | Wilder true range                         | Max of high-low, high-prior-close, low-prior-close |
| `atr(high, low, close, period=14)`                           | Average True Range                        | Wilder-smoothed true range                         |
| `natr(high, low, close, period=14)`                          | Normalized ATR                            | `100 * ATR / close`                                |
| `parkinson_vol(high, low, period=20)`                        | High-low volatility                       | Range-based estimator                              |
| `garman_klass_vol(open_, high, low, close, period=20)`       | OHLC volatility                           | Uses open/high/low/close within each bar           |
| `rogers_satchell_vol(open_, high, low, close, period=20)`    | Drift-independent OHLC volatility         | Works better when drift is nonzero                 |
| `yang_zhang_vol(open_, high, low, close, period=20, k=None)` | Overnight + open-close + range volatility | Combines several OHLC variance components          |
| `ewma_vol(returns, span=20)`                                 | Exponentially weighted volatility         | EWM standard deviation                             |
| `realized_vol(returns, period=20)`                           | Rolling realized volatility               | Rolling sample standard deviation                  |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: true_range
.. autofunction:: atr
.. autofunction:: natr
.. autofunction:: parkinson_vol
.. autofunction:: garman_klass_vol
.. autofunction:: rogers_satchell_vol
.. autofunction:: yang_zhang_vol
.. autofunction:: ewma_vol
.. autofunction:: realized_vol
```

______________________________________________________________________

## Volume Indicators

Volume indicators combine close movement, intrabar range, and volume into flow
or accumulation measures.

| Function                                           | Use it for                             | Notes                                             |
| -------------------------------------------------- | -------------------------------------- | ------------------------------------------------- |
| `obv(close, volume)`                               | On-Balance Volume                      | Cumulative signed volume based on close direction |
| `ad(high, low, close, volume)`                     | Chaikin Accumulation/Distribution line | Cumulative money-flow volume                      |
| `adosc(high, low, close, volume, fast=3, slow=10)` | Chaikin A/D Oscillator                 | Fast EMA of AD minus slow EMA of AD               |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: obv
.. autofunction:: ad
.. autofunction:: adosc
```

______________________________________________________________________

## Alpha and Information Coefficient

Alpha helpers are designed for cross-sectional signal panels. Compute forward
returns, per-date IC values, and IC summary statistics from generated or real
`date, symbol, signal, fwd_returns` data.

| Function                                                    | Use it for                     | Notes                                                       |
| ----------------------------------------------------------- | ------------------------------ | ----------------------------------------------------------- |
| `forward_returns(price, periods=1)`                         | Future simple returns          | `price.shift(-periods) / price - 1`                         |
| `pearson_ic(signal, fwd)`                                   | Linear information coefficient | Pearson correlation                                         |
| `spearman_ic(signal, fwd)`                                  | Rank information coefficient   | Spearman correlation through ranks                          |
| `information_coefficient(signal, fwd)`                      | Default IC alias               | Alias for `spearman_ic`                                     |
| `conditional_ic(signal, fwd, condition, method="spearman")` | Conditional IC                 | Correlation after filtering observations by a condition     |
| `horizon_ic(signal, fwd, method="spearman")`                | One-horizon IC                 | IC against one forward-return horizon                       |
| `ic_decay(signal, forward_returns_by_horizon)`              | IC decay expressions           | Builds one aliased IC expression per horizon                |
| `ic_ir(ic, *, window=None, period=None, date=None)`         | IC information ratio           | Mean IC divided by IC standard deviation                    |
| `hit_rate(signal, fwd)`                                     | Directional hit rate           | Fraction where signal and forward return signs agree        |
| `ic_summary_stats(ic)`                                      | Series-level IC summary        | Returns count, mean, std, IR, t-stat, and positive-IC share |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: forward_returns
.. autofunction:: pearson_ic
.. autofunction:: spearman_ic
.. autofunction:: information_coefficient
.. autofunction:: conditional_ic
.. autofunction:: horizon_ic
.. autofunction:: ic_decay
.. autofunction:: ic_ir
.. autofunction:: hit_rate
.. autofunction:: ic_summary_stats
```

______________________________________________________________________

## Quantile and Signal Transforms

These functions prepare cross-sectional signals for portfolio construction or
quantile spread analytics.

| Function                                             | Use it for                      | Notes                                              |
| ---------------------------------------------------- | ------------------------------- | -------------------------------------------------- |
| `assign_quantile(signal, n_quantiles=5)`             | Cross-sectional quantile labels | Produces integer labels `0..n_quantiles-1`         |
| `rank_normalize(signal)`                             | Rank-normalized signal          | Scales ranks to `[-0.5, 0.5]`                      |
| `zscore(signal)`                                     | Cross-sectional z-score         | Centers and scales by sample standard deviation    |
| `winsorize(signal, cutoff=3.0)`                      | Outlier clipping                | Clips to `mean +/- cutoff * std`                   |
| `long_short_spread(returns, quantile, upper, lower)` | Quantile spread return          | Mean return of upper quantile minus lower quantile |
| `mean_return_by_quantile(returns, quantile)`         | Quantile return expressions     | Builds one mean-return expression per quantile     |
| `quantile_changed(quantile)`                         | Turnover signal                 | True when quantile label changed from previous row |
| `quantile_turnover(changed)`                         | Quantile turnover               | Mean of quantile-change flags                      |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: assign_quantile
.. autofunction:: rank_normalize
.. autofunction:: zscore
.. autofunction:: winsorize
.. autofunction:: long_short_spread
.. autofunction:: mean_return_by_quantile
.. autofunction:: quantile_changed
.. autofunction:: quantile_turnover
```

______________________________________________________________________

## Factor and Benchmark Metrics

Factor metrics compare strategy returns against a benchmark return series.
They support lifetime, rolling, and period-bucketed calculations where the
signature includes `window`, `period`, and `date`.

| Function                                                                                                 | Use it for                               | Notes                                                                            |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------- |
| `alpha(returns, benchmark, risk_free=0.0, periods_per_year=252, *, window=None, period=None, date=None)` | Annualized Jensen alpha                  | Return unexplained by benchmark beta                                             |
| `beta(returns, benchmark, *, window=None, period=None, date=None)`                                       | Market beta                              | `cov(returns, benchmark) / var(benchmark)`                                       |
| `up_alpha(...)`                                                                                          | Alpha in up markets                      | Restricts observations to `benchmark > 0`                                        |
| `down_alpha(...)`                                                                                        | Alpha in down markets                    | Restricts observations to `benchmark < 0`                                        |
| `up_beta(...)`                                                                                           | Beta in up markets                       | Restricts observations to `benchmark > 0`                                        |
| `down_beta(...)`                                                                                         | Beta in down markets                     | Restricts observations to `benchmark < 0`                                        |
| `up_capture(returns, benchmark, *, window=None, period=None, date=None)`                                 | Up-market capture                        | Mean strategy return divided by mean benchmark return when benchmark is positive |
| `down_capture(returns, benchmark, *, window=None, period=None, date=None)`                               | Down-market capture                      | Mean strategy return divided by mean benchmark return when benchmark is negative |
| `up_down_capture(returns, benchmark, *, window=None, period=None, date=None)`                            | Capture balance                          | Up capture divided by down capture                                               |
| `batting_average(returns, benchmark, *, window=None, period=None, date=None)`                            | Fraction of outperformance observations  | `returns > benchmark` mean                                                       |
| `tracking_error(returns, benchmark, periods_per_year=252, *, window=None, period=None, date=None)`       | Annualized active risk                   | Standard deviation of active return                                              |
| `information_ratio(returns, benchmark, periods_per_year=252, *, window=None, period=None, date=None)`    | Annualized active return per active risk | Mean active return divided by active standard deviation, scaled                  |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: alpha
.. autofunction:: beta
.. autofunction:: up_alpha
.. autofunction:: down_alpha
.. autofunction:: up_beta
.. autofunction:: down_beta
.. autofunction:: up_capture
.. autofunction:: down_capture
.. autofunction:: up_down_capture
.. autofunction:: batting_average
.. autofunction:: tracking_error
.. autofunction:: information_ratio
```

______________________________________________________________________

## Distribution and Sharpe Statistics

The first five functions are expression metrics. The Sharpe significance and
confidence-interval helpers consume a concrete `pl.Series` because they perform
sample-level statistical calculations outside the Polars expression engine.

| Function                                                                                           | Use it for                                    | Notes                                       |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------- |
| `skewness(returns)`                                                                                | Sample skewness                               | Expression metric                           |
| `kurtosis(returns)`                                                                                | Excess kurtosis                               | Fisher definition                           |
| `higher_moments(returns)`                                                                          | Bundled skew/kurt struct                      | Returns a Polars struct expression          |
| `stability_of_timeseries(returns)`                                                                 | Trend stability of cumulative log returns     | R-squared of cumulative log returns vs time |
| `common_sense_ratio(returns)`                                                                      | Tail-ratio-adjusted total return sanity check | `tail_ratio * (1 + cumulative_return)`      |
| `probabilistic_sharpe(returns, benchmark_sr=0.0, periods_per_year=252)`                            | Probability Sharpe exceeds benchmark          | Lopez de Prado PSR                          |
| `deflated_sharpe(returns, n_trials, sr_variance=None, periods_per_year=252)`                       | Multiple-testing-adjusted Sharpe probability  | Bailey and Lopez de Prado DSR               |
| `minimum_track_record_length(returns, benchmark_sr=0.0, alpha=0.05, periods_per_year=252)`         | Required sample length                        | Observations needed for Sharpe confidence   |
| `sharpe_ci_bootstrap(returns, n_bootstrap=1000, confidence=0.95, periods_per_year=252, seed=None)` | Bootstrap Sharpe CI                           | Returns point estimate, lower, upper        |
| `sharpe_with_ci(returns, risk_free=0.0, periods_per_year=252, confidence=0.95)`                    | HAC-style Sharpe CI                           | Returns point estimate, lower, upper        |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: skewness
.. autofunction:: kurtosis
.. autofunction:: higher_moments
.. autofunction:: stability_of_timeseries
.. autofunction:: common_sense_ratio
.. autofunction:: probabilistic_sharpe
.. autofunction:: deflated_sharpe
.. autofunction:: minimum_track_record_length
.. autofunction:: sharpe_ci_bootstrap
.. autofunction:: sharpe_with_ci
```

______________________________________________________________________

## Tail Risk

Tail-risk expression metrics support lifetime, rolling, and period-bucketed
calculations. The GPD helpers consume `pl.Series` and fit a Peaks-over-Threshold
model to tail losses.

| Function                                                                            | Use it for                         | Notes                                    |
| ----------------------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------- |
| `tail_ratio(returns, *, window=None, period=None, date=None)`                       | Right-tail / left-tail balance     | `abs(p95) / abs(p05)`                    |
| `ulcer_index(returns, *, window=None, period=None, date=None)`                      | Drawdown depth persistence         | RMS of drawdown sequence                 |
| `omega_ratio(returns, required_return=0.0, *, window=None, period=None, date=None)` | Gain/loss balance around threshold | Sum gains divided by absolute sum losses |
| `gpd_var(returns, var_p=0.01, threshold_p=0.10)`                                    | Extreme VaR from GPD fit           | Returns positive loss magnitude          |
| `gpd_cvar(returns, var_p=0.01, threshold_p=0.10)`                                   | Extreme CVaR from GPD fit          | Expected shortfall beyond `var_p`        |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: tail_ratio
.. autofunction:: ulcer_index
.. autofunction:: omega_ratio
.. autofunction:: gpd_var
.. autofunction:: gpd_cvar
```

______________________________________________________________________

## Portfolio

Portfolio metrics aggregate position weights. They are most useful inside a
`group_by("date")` aggregation over a long-form position panel.

| Function                                   | Use it for                | Notes                                           |
| ------------------------------------------ | ------------------------- | ----------------------------------------------- |
| `gross_leverage(weights)`                  | Total absolute exposure   | Sum of absolute weights                         |
| `gross_exposure(weights)`                  | Long plus short notional  | Alias for `gross_leverage`                      |
| `net_exposure(weights)`                    | Signed net exposure       | Sum of weights                                  |
| `long_exposure(weights)`                   | Long exposure             | Sum of positive weights                         |
| `short_exposure(weights)`                  | Short exposure            | Sum of negative weights, returned as negative   |
| `concentration(weights)`                   | Herfindahl concentration  | Sum of squared normalized absolute weights      |
| `top_n_concentration(weights, n=10)`       | Top-name exposure share   | Gross exposure held by top `n` absolute weights |
| `active_share(weights, benchmark_weights)` | Active share vs benchmark | `0.5 * sum(abs(weights - benchmark_weights))`   |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: gross_leverage
.. autofunction:: gross_exposure
.. autofunction:: net_exposure
.. autofunction:: long_exposure
.. autofunction:: short_exposure
.. autofunction:: concentration
.. autofunction:: top_n_concentration
.. autofunction:: active_share
```

______________________________________________________________________

## Post-Trade

Post-trade utilities consume transaction, round-trip, or execution data. Cost,
slippage, turnover, and trade-quality metrics are expression kernels. Round-trip
extraction and summary helpers take concrete `pl.DataFrame` inputs because they
need ordered trade sequences.

| Function                                                                  | Use it for                      | Notes                                                |
| ------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------- |
| `transaction_notional(quantity, price)`                                   | Absolute traded notional        | `abs(quantity) * price`                              |
| `transaction_cost(quantity, price, *, commission=0.0, fees=0.0, bps=0.0)` | Explicit plus basis-point costs | Adds commission, fees, and bps cost on notional      |
| `transaction_volume(quantity, price, *, period=None, date=None)`          | Traded notional volume          | Sums notional over the full sample or period bucket  |
| `slippage_bps(execution_price, benchmark_price, *, side=None)`            | Execution slippage              | Side-aware when a side expression is provided        |
| `implementation_shortfall(execution_price, decision_price, *, side=None)` | Decision-price slippage         | Side-aware implementation shortfall in bps           |
| `vwap_slippage(execution_price, vwap, *, side=None)`                      | VWAP slippage                   | Side-aware execution vs. VWAP in bps                 |
| `turnover(weights, *, window=None)`                                       | Position-weight turnover        | Absolute weight change; optional rolling sum         |
| `cost_per_trade(...)`                                                     | Per-trade cost alias            | Same calculation as `transaction_cost`               |
| `cost_attribution(transactions)`                                          | Cost decomposition              | Returns component totals and percentages             |
| `extract_round_trips(transactions)`                                       | FIFO round-trip extraction      | Builds entry/exit trade rows from signed quantities  |
| `round_trip_stats(round_trips)`                                           | Trade-quality summary           | Count, win rate, average PnL, total PnL, PF, payoff  |
| `long_short_round_trip_stats(round_trips)`                                | Long/short trade summary        | Aggregates round trips by side                       |
| `sector_round_trip_stats(round_trips, sector_map)`                        | Sector trade summary            | Aggregates round trips by mapped sector              |
| `win_rate(pnl)`                                                           | Profitable-trade fraction       | Expression metric                                    |
| `profit_factor(pnl)`                                                      | Gross profit / gross loss       | Expression metric                                    |
| `payoff_ratio(pnl)`                                                       | Average win / average loss      | Expression metric                                    |
| `avg_trade_pnl(pnl)`                                                      | Mean trade PnL                  | Expression metric                                    |
| `trade_duration_stats(duration)`                                          | Holding-period summary          | Returns mean, median, and max duration               |
| `mae_mfe(trades, prices)`                                                 | Maximum adverse/favorable move  | Adds `mae` and `mfe` to round trips                  |
| `consecutive_wins_losses(pnl)`                                            | Win/loss streaks                | Returns max consecutive wins and losses              |
| `exit_reason_stats(trades)`                                               | PnL by exit reason              | Groups counts and PnL by exit-reason label           |
| `trade_size_return_correlation(size, returns)`                            | Size/return relationship        | Correlation of absolute trade size with trade return |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: transaction_notional
.. autofunction:: transaction_cost
.. autofunction:: transaction_volume
.. autofunction:: slippage_bps
.. autofunction:: implementation_shortfall
.. autofunction:: vwap_slippage
.. autofunction:: turnover
.. autofunction:: cost_per_trade
.. autofunction:: cost_attribution
.. autofunction:: extract_round_trips
.. autofunction:: round_trip_stats
.. autofunction:: long_short_round_trip_stats
.. autofunction:: sector_round_trip_stats
.. autofunction:: win_rate
.. autofunction:: profit_factor
.. autofunction:: payoff_ratio
.. autofunction:: avg_trade_pnl
.. autofunction:: trade_duration_stats
.. autofunction:: mae_mfe
.. autofunction:: consecutive_wins_losses
.. autofunction:: exit_reason_stats
.. autofunction:: trade_size_return_correlation
```
