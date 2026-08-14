# API

`finance-calcs` exposes calculation functions at the top level and through the
`.fcalcs` namespace on both `polars.Expr` and `polars.Series`. Most functions
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
out = df.select(pl.col("ret").fcalcs.sharpe().alias("sharpe"))
```

The same namespace exists on `pl.Series` for eager one-off checks:

```python
value = returns.fcalcs.sharpe()
```

## Execution and Input Contracts

Expression metrics compose inside `select`, `with_columns`, and lazy queries.
Return, risk, and tail metrics consume **periodic returns**, not price levels;
use `simple_returns` or `log_returns` to derive them from prices. In these
metrics, floating-point `NaN` and Polars null values are equivalent missing
observations. Compounding treats missing returns as neutral and statistical
aggregations exclude them.

APIs that require sample-level algorithms are deliberately eager:

- `native_adx`, `native_parabolic_sar`, and `native_garch11_variance` accept
  numeric sequences and return NumPy arrays. They are native bridges, not
  Polars expression plugins.
- GPD fits and bootstrap/regime helpers accept concrete `pl.Series` values and
  return scalars, tuples, or materialized series.
- `neutralize`, `orthogonalize`, round-trip extraction, and related post-trade
  helpers accept concrete `pl.DataFrame` values.

Keep eager helpers outside lazy-query plans. A future expression-plugin layer
must preserve existing results and null semantics before replacing the native
bridges.

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

Annualized metrics use `frequency`. It accepts a
`finance_enums.Frequency`, a standard alias such as `"daily"` or
`"monthly"`, or a positive raw number of observations per year. Raw values
support intraday and custom trading schedules without assuming market hours.

| Function                                                                                    | Use it for                                | Notes                                                                            |
| ------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------- |
| `period_bucket(date, period)`                                                               | Build a reusable period bucket from dates | Accepts `Frequency`, aliases, Polars durations, or an existing bucket expression |
| `simple_returns(price)`                                                                     | Arithmetic price returns                  | Computes `price / price.shift(1) - 1`                                            |
| `log_returns(price)`                                                                        | Log price returns                         | Computes `log(price / price.shift(1))`                                           |
| `cumulative_returns(returns, starting_value=0.0, *, window=None, period=None, date=None)`   | Compounded return path                    | Resets inside each rolling window or period bucket                               |
| `cumulative_return(returns, *, window=None, period=None, date=None)`                        | Terminal compounded return                | Produces the final compound return for the sample, window, or bucket             |
| `annualized_return(returns, frequency="daily", *, window=None, period=None, date=None)`     | Annualized geometric return               | Uses compound return and non-missing observation count, not elapsed dates        |
| `annualized_volatility(returns, frequency="daily", *, window=None, period=None, date=None)` | Annualized standard deviation             | Scales by observations per year implied by `frequency`                           |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: period_bucket
.. autofunction:: simple_returns
.. autofunction:: log_returns
.. autofunction:: cumulative_returns
.. autofunction:: cumulative_return
.. autofunction:: annualized_return
.. autofunction:: annualized_volatility
```

______________________________________________________________________

## Risk and Drawdown

Risk metrics operate on return expressions. Scalar `risk_free` and
`required_return` inputs are annual rates converted geometrically to
per-observation rates. Expression inputs are treated as already per-observation.

| Function                                                                                                      | Use it for                            | Notes                                                                 |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------- |
| `sharpe(returns, *, risk_free=0.0, frequency="daily", window=None, period=None, date=None)`                   | Annualized Sharpe ratio               | Supports scalar annual risk-free rates or per-observation expressions |
| `sortino(returns, *, required_return=0.0, frequency="daily", window=None, period=None, date=None)`            | Annualized Sortino ratio              | Uses downside deviation below the annual hurdle                       |
| `calmar(returns, *, frequency="daily", window=None, period=None, date=None)`                                  | Annualized return / abs(max drawdown) | Uses the same sample/window/period controls                           |
| `downside_deviation(returns, *, required_return=0.0, frequency="daily", window=None, period=None, date=None)` | Annualized semi-deviation             | Squares only observations below the annual hurdle                     |
| `drawdown_series(returns, *, period=None, date=None)`                                                         | Running drawdown path                 | Equity curve divided by running peak, including initial 1.0 baseline  |
| `max_drawdown(returns, *, window=None, period=None, date=None)`                                               | Most negative drawdown                | Rolling windows rebase their equity and peak inside each window       |
| `value_at_risk(returns, *, tail_probability=0.05, window=None, period=None, date=None)`                       | Historical VaR quantile               | Returns the lower-tail return quantile                                |
| `conditional_value_at_risk(returns, *, tail_probability=0.05, window=None, period=None, date=None)`           | Historical conditional VaR            | Mean return of observations at or below VaR                           |
| `expected_shortfall(returns, *, tail_probability=0.05, window=None, period=None, date=None)`                  | Historical expected shortfall         | Industry synonym for conditional VaR                                  |
| `value_at_risk_parametric(returns, *, tail_probability=0.05, window=None, period=None, date=None)`            | Gaussian VaR                          | Supports common probabilities from the built-in z-score table         |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: sharpe
.. autofunction:: sortino
.. autofunction:: calmar
.. autofunction:: downside_deviation
.. autofunction:: drawdown_series
.. autofunction:: max_drawdown
.. autofunction:: value_at_risk
.. autofunction:: conditional_value_at_risk
.. autofunction:: expected_shortfall
.. autofunction:: value_at_risk_parametric
```

______________________________________________________________________

## Report Metrics

These metrics provide the calculation layer for static, terminal, and notebook
reports. Expression metrics remain lazy-query compatible. `drawdown_details`
is eager because it returns one row per variable-length drawdown episode.

| Function                                           | Use it for                              | Notes                                                     |
| -------------------------------------------------- | --------------------------------------- | --------------------------------------------------------- |
| `best_return(returns, *, period=None, date=None)`  | Highest raw or compounded period return | Canonical best-return metric                              |
| `worst_return(returns, *, period=None, date=None)` | Lowest raw or compounded period return  | Canonical worst-return metric                             |
| `average_win(returns)`                             | Mean positive return                    | Positive observations only                                |
| `average_loss(returns)`                            | Mean negative return                    | Negative observations only                                |
| `gain_to_pain_ratio(returns)`                      | Net return per unit of summed loss      | Uses arithmetic return sums                               |
| `recovery_factor(returns)`                         | Net return relative to maximum drawdown | Uses absolute arithmetic net return and drawdown          |
| `kelly_criterion(returns)`                         | Estimated Kelly allocation fraction     | Zero returns are excluded from active-period win rate     |
| `drawdown_details(returns, *, date=None)`          | Materialized drawdown episodes          | Returns start, valley, end, duration, depth, and recovery |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: best_return
.. autofunction:: worst_return
.. autofunction:: average_win
.. autofunction:: average_loss
.. autofunction:: gain_to_pain_ratio
.. autofunction:: recovery_factor
.. autofunction:: kelly_criterion
.. autofunction:: drawdown_details
```

______________________________________________________________________

## Overlap and Price Channels

Overlap studies smooth prices or build price channels from high/low/close data.
`window` is an observation lookback, not a calendar bucket.

| Function                                               | Use it for                         | Notes                                             |
| ------------------------------------------------------ | ---------------------------------- | ------------------------------------------------- |
| `sma(close, window=20)`                                | Simple moving average              | Rolling mean                                      |
| `ema(close, window=20)`                                | Exponential moving average         | Uses Polars EWM mean with `span=window`           |
| `wma(close, window=20)`                                | Weighted moving average            | Recent observations receive larger linear weights |
| `dema(close, window=20)`                               | Double EMA                         | `2 * EMA - EMA(EMA)`                              |
| `tema(close, window=20)`                               | Triple EMA                         | `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))`              |
| `midpoint(close, window=14)`                           | Midpoint of rolling high/low close | Uses close-only rolling max/min                   |
| `midprice(high, low, window=14)`                       | Midpoint of high/low channel       | Uses rolling high max and low min                 |
| `bbands_upper(close, window=20, upper_deviations=2.0)` | Bollinger upper band               | Middle plus standard-deviation multiple           |
| `bbands_middle(close, window=20)`                      | Bollinger middle band              | SMA                                               |
| `bbands_lower(close, window=20, lower_deviations=2.0)` | Bollinger lower band               | Middle minus standard-deviation multiple          |
| `donchian_upper(high, window=20)`                      | Donchian upper channel             | Rolling high maximum                              |
| `donchian_lower(low, window=20)`                       | Donchian lower channel             | Rolling low minimum                               |
| `donchian_middle(high, low, window=20)`                | Donchian midline                   | Average of upper and lower channels               |

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
rate-of-change, or directional-movement expressions. `window` is the observation
lookback length.

| Function                                                              | Use it for                 | Notes                                        |
| --------------------------------------------------------------------- | -------------------------- | -------------------------------------------- |
| `rsi(close, window=14)`                                               | Relative Strength Index    | Wilder smoothing                             |
| `macd_line(close, fast_window=12, slow_window=26)`                    | MACD line                  | Fast EMA minus slow EMA                      |
| `macd_signal(close, fast_window=12, slow_window=26, signal_window=9)` | MACD signal line           | EMA of `macd_line`                           |
| `macd_hist(close, fast_window=12, slow_window=26, signal_window=9)`   | MACD histogram             | MACD line minus signal line                  |
| `mom(close, window=10)`                                               | Price momentum             | Difference from `window` observations ago    |
| `roc(close, window=10)`                                               | Percent rate of change     | `100 * (close / close.shift(window) - 1)`    |
| `rocp(close, window=10)`                                              | Decimal rate of change     | `(close - prior) / prior`                    |
| `rocr(close, window=10)`                                              | Price ratio                | `close / prior`                              |
| `rocr100(close, window=10)`                                           | Price ratio scaled by 100  | `100 * rocr`                                 |
| `willr(high, low, close, window=14)`                                  | Williams %R                | Close location within rolling high/low range |
| `stoch_k(high, low, close, window=14)`                                | Fast stochastic %K         | Range-normalized close                       |
| `stoch_d(high, low, close, window=14, signal_window=3)`               | Stochastic %D              | SMA of `%K`                                  |
| `cci(high, low, close, window=20)`                                    | Commodity Channel Index    | Typical-price deviation oscillator           |
| `cmo(close, window=14)`                                               | Chande Momentum Oscillator | Up/down movement balance                     |
| `trix(close, window=15)`                                              | TRIX                       | One-bar ROC of triple-smoothed log price     |
| `plus_dm(high, low)`                                                  | Raw +DM                    | Wilder directional movement                  |
| `minus_dm(high, low)`                                                 | Raw -DM                    | Wilder directional movement                  |
| `plus_di(high, low, close, window=14)`                                | +DI                        | Smoothed +DM divided by true range           |
| `minus_di(high, low, close, window=14)`                               | -DI                        | Smoothed -DM divided by true range           |
| `adx(high, low, close, window=14)`                                    | Average Directional Index  | Trend-strength measure from +DI and -DI      |

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
OHLC bars. `window` is the observation lookback length.

| Function                                                                                       | Use it for                                | Notes                                              |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------- |
| `true_range(high, low, close)`                                                                 | Wilder true range                         | Max of high-low, high-prior-close, low-prior-close |
| `atr(high, low, close, window=14)`                                                             | Average True Range                        | Wilder-smoothed true range                         |
| `natr(high, low, close, window=14)`                                                            | Normalized ATR                            | `100 * ATR / close`                                |
| `parkinson_volatility(high, low, window=20, *, frequency="daily")`                             | High-low volatility                       | Annualized range-based estimator                   |
| `garman_klass_volatility(open_, high, low, close, window=20, *, frequency="daily")`            | OHLC volatility                           | Annualized OHLC estimator                          |
| `rogers_satchell_volatility(open_, high, low, close, window=20, *, frequency="daily")`         | Drift-independent OHLC volatility         | Annualized; works when drift is nonzero            |
| `yang_zhang_volatility(open_, high, low, close, window=20, *, weight=None, frequency="daily")` | Overnight + open-close + range volatility | Annualized combination of OHLC variance components |
| `exponentially_weighted_volatility(returns, window=20, *, frequency="daily")`                  | Exponentially weighted volatility         | Annualized EWM standard deviation                  |
| `realized_volatility(returns, window=20, *, frequency="daily")`                                | Rolling realized volatility               | Annualized rolling sample standard deviation       |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: true_range
.. autofunction:: atr
.. autofunction:: natr
.. autofunction:: parkinson_volatility
.. autofunction:: garman_klass_volatility
.. autofunction:: rogers_satchell_volatility
.. autofunction:: yang_zhang_volatility
.. autofunction:: exponentially_weighted_volatility
.. autofunction:: realized_volatility
```

______________________________________________________________________

## Volume Indicators

Volume indicators combine close movement, intrabar range, and volume into flow
or accumulation measures.

| Function                                                         | Use it for                             | Notes                                             |
| ---------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------- |
| `obv(close, volume)`                                             | On-Balance Volume                      | Cumulative signed volume based on close direction |
| `ad(high, low, close, volume)`                                   | Chaikin Accumulation/Distribution line | Cumulative money-flow volume                      |
| `adosc(high, low, close, volume, fast_window=3, slow_window=10)` | Chaikin A/D Oscillator                 | Fast EMA of AD minus slow EMA of AD               |

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

| Function                                                                                         | Use it for                      | Notes                                                   |
| ------------------------------------------------------------------------------------------------ | ------------------------------- | ------------------------------------------------------- |
| `forward_returns(price, horizon=1)`                                                              | Future simple returns           | `price.shift(-horizon) / price - 1`                     |
| `information_coefficient(signal, forward_returns, *, method="spearman")`                         | General information coefficient | Selects correlation method explicitly                   |
| `information_coefficient_pearson(signal, forward_returns)`                                       | Linear information coefficient  | Pearson correlation                                     |
| `information_coefficient_spearman(signal, forward_returns)`                                      | Rank information coefficient    | Spearman correlation through ranks                      |
| `information_coefficient_conditional(signal, forward_returns, condition, *, method="spearman")`  | Conditional IC                  | Correlation after filtering observations by a condition |
| `information_coefficient_by_horizon(signal, forward_returns, *, method="spearman")`              | One-horizon IC                  | IC against one forward-return horizon                   |
| `information_coefficient_decay(signal, forward_returns_by_horizon)`                              | IC decay expressions            | Builds one aliased IC expression per horizon            |
| `information_coefficient_ratio(information_coefficient, *, window=None, period=None, date=None)` | IC information ratio            | Mean IC divided by IC standard deviation                |
| `hit_rate(signal, forward_returns)`                                                              | Directional hit rate            | Fraction where signal and forward-return signs agree    |
| `information_coefficient_statistics(information_coefficient)`                                    | Series-level IC summary         | Returns fully named summary fields                      |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: forward_returns
.. autofunction:: information_coefficient
.. autofunction:: information_coefficient_pearson
.. autofunction:: information_coefficient_spearman
.. autofunction:: information_coefficient_conditional
.. autofunction:: information_coefficient_by_horizon
.. autofunction:: information_coefficient_decay
.. autofunction:: information_coefficient_ratio
.. autofunction:: hit_rate
.. autofunction:: information_coefficient_statistics
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

| Function                                                                                              | Use it for                               | Notes                                                                            |
| ----------------------------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------- |
| `alpha(returns, benchmark, risk_free=0.0, frequency="daily", *, window=None, period=None, date=None)` | Annualized Jensen alpha                  | Return unexplained by benchmark beta                                             |
| `beta(returns, benchmark, *, window=None, period=None, date=None)`                                    | Market beta                              | `cov(returns, benchmark) / var(benchmark)`                                       |
| `r_squared(returns, benchmark, *, window=None, period=None, date=None)`                               | Benchmark coefficient of determination   | Squared Pearson correlation                                                      |
| `up_alpha(...)`                                                                                       | Alpha in up markets                      | Restricts observations to `benchmark > 0`                                        |
| `down_alpha(...)`                                                                                     | Alpha in down markets                    | Restricts observations to `benchmark < 0`                                        |
| `up_beta(...)`                                                                                        | Beta in up markets                       | Restricts observations to `benchmark > 0`                                        |
| `down_beta(...)`                                                                                      | Beta in down markets                     | Restricts observations to `benchmark < 0`                                        |
| `up_capture(returns, benchmark, *, window=None, period=None, date=None)`                              | Up-market capture                        | Mean strategy return divided by mean benchmark return when benchmark is positive |
| `down_capture(returns, benchmark, *, window=None, period=None, date=None)`                            | Down-market capture                      | Mean strategy return divided by mean benchmark return when benchmark is negative |
| `up_down_capture(returns, benchmark, *, window=None, period=None, date=None)`                         | Capture balance                          | Up capture divided by down capture                                               |
| `batting_average(returns, benchmark, *, window=None, period=None, date=None)`                         | Fraction of outperformance observations  | `returns > benchmark` mean                                                       |
| `tracking_error(returns, benchmark, frequency="daily", *, window=None, period=None, date=None)`       | Annualized active risk                   | Standard deviation of active return                                              |
| `information_ratio(returns, benchmark, frequency="daily", *, window=None, period=None, date=None)`    | Annualized active return per active risk | Mean active return divided by active standard deviation, scaled                  |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: alpha
.. autofunction:: beta
.. autofunction:: r_squared
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

| Function                                                                                                                     | Use it for                                    | Notes                                       |
| ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------- |
| `skewness(returns)`                                                                                                          | Sample skewness                               | Expression metric                           |
| `kurtosis(returns)`                                                                                                          | Excess kurtosis                               | Fisher definition                           |
| `higher_moments(returns)`                                                                                                    | Bundled higher-moment struct                  | Fields are `skewness` and `kurtosis`        |
| `stability_of_timeseries(returns)`                                                                                           | Trend stability of cumulative log returns     | R-squared of cumulative log returns vs time |
| `common_sense_ratio(returns)`                                                                                                | Tail-ratio-adjusted total return sanity check | `tail_ratio * (1 + cumulative_return)`      |
| `sharpe_probability(returns, benchmark_sharpe=0.0, frequency="daily")`                                                       | Probability Sharpe exceeds benchmark          | Probabilistic Sharpe                        |
| `sharpe_deflated_probability(returns, trial_count, sharpe_variance=None, frequency="daily")`                                 | Multiple-testing-adjusted Sharpe probability  | Deflated Sharpe probability                 |
| `sharpe_minimum_track_record_length(returns, benchmark_sharpe=0.0, significance_level=0.05, frequency="daily")`              | Required sample length                        | Observations needed for Sharpe confidence   |
| `sharpe_bootstrap_confidence_interval(returns, bootstrap_samples=1000, confidence_level=0.95, frequency="daily", seed=None)` | Bootstrap Sharpe confidence interval          | Returns point estimate, lower, upper        |
| `sharpe_confidence_interval(returns, risk_free=0.0, frequency="daily", confidence_level=0.95)`                               | Asymptotic Sharpe confidence interval         | Returns point estimate, lower, upper        |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: skewness
.. autofunction:: kurtosis
.. autofunction:: higher_moments
.. autofunction:: stability_of_timeseries
.. autofunction:: common_sense_ratio
.. autofunction:: sharpe_probability
.. autofunction:: sharpe_deflated_probability
.. autofunction:: sharpe_minimum_track_record_length
.. autofunction:: sharpe_bootstrap_confidence_interval
.. autofunction:: sharpe_confidence_interval
```

______________________________________________________________________

## Tail Risk

Tail-risk expression metrics support lifetime, rolling, and period-bucketed
calculations. The GPD helpers consume `pl.Series` and fit a Peaks-over-Threshold
model to tail losses.

| Function                                                                                                      | Use it for                           | Notes                                        |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------- |
| `tail_ratio(returns, *, window=None, period=None, date=None)`                                                 | Right-tail / left-tail balance       | `abs(p95) / abs(p05)`                        |
| `ulcer_index(returns, *, window=None, period=None, date=None)`                                                | Drawdown depth persistence           | Decimal RMS from initial 1.0 equity baseline |
| `omega_ratio(returns, *, required_return=0.0, frequency="daily", window=None, period=None, date=None)`        | Gain/loss balance around threshold   | Scalar hurdle is annual                      |
| `value_at_risk_generalized_pareto(returns, *, tail_probability=0.01, threshold_probability=0.10)`             | Extreme VaR from GPD fit             | Returns a negative lower-tail return         |
| `conditional_value_at_risk_generalized_pareto(returns, *, tail_probability=0.01, threshold_probability=0.10)` | Extreme conditional VaR from GPD fit | Returns a negative lower-tail return         |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: tail_ratio
.. autofunction:: ulcer_index
.. autofunction:: omega_ratio
.. autofunction:: value_at_risk_generalized_pareto
.. autofunction:: conditional_value_at_risk_generalized_pareto
```

______________________________________________________________________

## Market Microstructure

These expression metrics quantify spreads, liquidity, and price impact.

| Function                                                            | Use it for                            |
| ------------------------------------------------------------------- | ------------------------------------- |
| `quoted_spread_bps(bid, ask, mid=None)`                             | Quoted bid-ask spread in basis points |
| `effective_spread_bps(execution_price, mid_price, side=None)`       | Execution spread in basis points      |
| `realized_spread_bps(execution_price, future_mid_price, side=None)` | Post-trade realized spread            |
| `order_imbalance(buy_volume, sell_volume)`                          | Normalized buy/sell volume imbalance  |
| `amihud_illiquidity(returns, traded_notional)`                      | Absolute return per traded notional   |
| `kyle_lambda(returns, signed_volume)`                               | Price impact per signed volume        |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: quoted_spread_bps
.. autofunction:: effective_spread_bps
.. autofunction:: realized_spread_bps
.. autofunction:: order_imbalance
.. autofunction:: amihud_illiquidity
.. autofunction:: kyle_lambda
```

______________________________________________________________________

## Regime and Persistence

| Function                                               | Use it for                                |
| ------------------------------------------------------ | ----------------------------------------- |
| `regime_signal(returns, window=63, threshold=1.0)`     | Rolling volatility-regime classification  |
| `hurst_exponent(values, max_lag=None)`                 | Long-memory and mean-reversion estimate   |
| `fractional_difference(values, order, threshold=1e-5)` | Memory-preserving fractional differencing |

```{eval-rst}
.. currentmodule:: finance_calcs

.. autofunction:: regime_signal
.. autofunction:: hurst_exponent
.. autofunction:: fractional_difference
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
| `cost_attribution(transactions)`                                          | Cost decomposition              | Returns component totals and percentages             |
| `extract_round_trips(transactions)`                                       | FIFO round-trip extraction      | Builds entry/exit trade rows from signed quantities  |
| `round_trip_stats(round_trips)`                                           | Trade-quality summary           | Count, win rate, average PnL, total PnL, PF, payoff  |
| `long_short_round_trip_stats(round_trips)`                                | Long/short trade summary        | Aggregates round trips by side                       |
| `sector_round_trip_stats(round_trips, sector_map)`                        | Sector trade summary            | Aggregates round trips by mapped sector              |
| `win_rate(pnl)`                                                           | Profitable-trade fraction       | Expression metric                                    |
| `profit_factor(pnl)`                                                      | Gross profit / gross loss       | Expression metric                                    |
| `payoff_ratio(pnl)`                                                       | Average win / average loss      | Expression metric                                    |
| `average_trade_pnl(pnl)`                                                  | Mean trade PnL                  | Expression metric                                    |
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
.. autofunction:: cost_attribution
.. autofunction:: extract_round_trips
.. autofunction:: round_trip_stats
.. autofunction:: long_short_round_trip_stats
.. autofunction:: sector_round_trip_stats
.. autofunction:: win_rate
.. autofunction:: profit_factor
.. autofunction:: payoff_ratio
.. autofunction:: average_trade_pnl
.. autofunction:: trade_duration_stats
.. autofunction:: mae_mfe
.. autofunction:: consecutive_wins_losses
.. autofunction:: exit_reason_stats
.. autofunction:: trade_size_return_correlation
```
