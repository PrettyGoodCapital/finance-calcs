# Examples

The examples use `finance-datagen` so the inputs look like real financial
fixtures: generated price paths, OHLCV bars, signal panels, position panels,
and transaction logs. All generators are deterministic when `seed` is set, so
the snippets are suitable for documentation, tests, and notebooks.

______________________________________________________________________

## Price Path to Return and Risk Metrics

Use `generate_prices` for a realistic close series, then derive returns and risk
metrics with expression calls.

```python
import polars as pl
from finance_datagen import generate_prices

import finance_calcs as fc

prices = generate_prices(symbol="ACME", seed=7)

metrics = prices.with_columns(
    pl.col("price").finance.simple_returns().alias("ret"),
).select(
    fc.returns(pl.col("ret")).alias("total_return"),
    fc.annualized_return(pl.col("ret")).alias("ann_return"),
    fc.volatility(pl.col("ret")).alias("ann_vol"),
    fc.sharpe(pl.col("ret")).alias("sharpe"),
    fc.max_drawdown(pl.col("ret")).alias("max_drawdown"),
)
```

The direct function style and namespace style are equivalent for expression
metrics. Prefer the namespace when it improves pipeline readability, and use the
top-level functions when several input columns are involved.

______________________________________________________________________

## Calendar and Custom Period Slices

`period=` creates per-bucket metrics while preserving row-level output. Use
`finance_enums.Frequency` or a Polars duration string when the bucket can be
derived from a timestamp column.

```python
import polars as pl
from finance_datagen import generate_prices
from finance_enums import Frequency

import finance_calcs as fc

prices = generate_prices(symbol="ACME", seed=11)
returns = prices.with_columns(
    pl.col("price").finance.simple_returns().alias("ret"),
)

monthly = returns.with_columns(
    fc.period_bucket(pl.col("timestamp"), Frequency.Month).alias("month"),
    pl.col("ret").finance.returns(period=Frequency.Month, date=pl.col("timestamp")).alias("month_return"),
    pl.col("ret").finance.sharpe(period="1q", date=pl.col("timestamp")).alias("quarter_sharpe"),
)
```

For fiscal calendars, strategy regimes, or exchange-calendar grids generated
upstream, pass the bucket expression directly. In that case `date=` is not
required.

```python
regime_metrics = returns.with_columns(
    pl.when(pl.col("price") > pl.col("price").rolling_mean(63))
    .then(pl.lit("above_trend"))
    .otherwise(pl.lit("below_trend"))
    .alias("regime"),
).with_columns(
    fc.tail_ratio(pl.col("ret"), period=pl.col("regime")).alias("regime_tail_ratio"),
)
```

______________________________________________________________________

## OHLCV Indicators

Use `ohlc_from_close` to turn any generated close path into bars. That gives the
high, low, open, close, and volume inputs needed by overlap, momentum,
volatility, and volume indicators.

```python
import polars as pl
from finance_datagen import generate_prices, ohlc_from_close

prices = generate_prices(symbol="ACME", seed=5)
bars = ohlc_from_close(prices["price"], symbol="ACME", seed=5)

features = bars.with_columns(
    pl.col("close").finance.sma(20).alias("sma_20"),
    pl.col("close").finance.ema(20).alias("ema_20"),
    pl.col("close").finance.rsi(14).alias("rsi_14"),
    pl.col("close").finance.macd_line().alias("macd"),
    fc.atr(pl.col("high"), pl.col("low"), pl.col("close"), period=14).alias("atr_14"),
    fc.natr(pl.col("high"), pl.col("low"), pl.col("close"), period=14).alias("natr_14"),
    fc.obv(pl.col("close"), pl.col("volume")).alias("obv"),
    fc.adosc(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("adosc"),
)
```

For OHLC volatility estimators, use the same bars:

```python
vol_features = bars.with_columns(
    fc.parkinson_vol(pl.col("high"), pl.col("low"), period=20).alias("parkinson"),
    fc.garman_klass_vol(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"), period=20).alias("gk"),
    fc.yang_zhang_vol(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"), period=20).alias("yz"),
)
```

______________________________________________________________________

## Cross-Sectional Alpha and Quantiles

`generate_signal` produces a long-form `date, symbol, signal, fwd_returns` panel
with a controlled information coefficient. Use `over("date")` for per-date
signal transforms and `group_by("date")` for per-date IC or spread metrics.

```python
from datetime import date

import polars as pl
from finance_datagen import generate_signal

import finance_calcs as fc

signals = generate_signal(n_dates=60, n_assets=100, ic=0.06, start=date(2024, 1, 2), seed=3)

ranked = signals.with_columns(
    fc.zscore(pl.col("signal")).over("date").alias("signal_z"),
    fc.rank_normalize(pl.col("signal")).over("date").alias("signal_rank"),
    fc.assign_quantile(pl.col("signal"), n_quantiles=5).over("date").alias("quantile"),
)

ic = ranked.group_by("date").agg(
    fc.spearman_ic(pl.col("signal"), pl.col("fwd_returns")).alias("ic"),
    fc.hit_rate(pl.col("signal"), pl.col("fwd_returns")).alias("hit_rate"),
    fc.long_short_spread(pl.col("fwd_returns"), pl.col("quantile"), upper=4, lower=0).alias("q5_q1_spread"),
)

ic_summary = fc.ic_summary_stats(ic["ic"])
```

To monitor quantile turnover, evaluate changes within each symbol and aggregate
by date.

```python
turnover = ranked.with_columns(
    fc.quantile_changed(pl.col("quantile")).over("symbol").fill_null(False).alias("changed"),
).group_by("date").agg(
    pl.col("changed").mean().alias("quantile_turnover"),
)
```

______________________________________________________________________

## Benchmark and Factor Metrics

`generate_multi_asset_gbm` gives a correlated multi-asset panel. A simple market
benchmark can be built as the cross-sectional average return for each timestamp.
Then factor metrics can be computed by symbol.

```python
import polars as pl
from finance_datagen import generate_multi_asset_gbm

import finance_calcs as fc

panel = generate_multi_asset_gbm(
    n_steps=252,
    n_assets=5,
    symbols=["AAA", "BBB", "CCC", "DDD", "EEE"],
    rho=0.35,
    seed=9,
).with_columns(
    pl.col("price").finance.simple_returns().over("symbol").alias("ret"),
).with_columns(
    pl.col("ret").mean().over("timestamp").alias("benchmark"),
)

factor_metrics = panel.group_by("symbol").agg(
    fc.beta(pl.col("ret"), pl.col("benchmark")).alias("beta"),
    fc.alpha(pl.col("ret"), pl.col("benchmark")).alias("alpha"),
    fc.tracking_error(pl.col("ret"), pl.col("benchmark")).alias("tracking_error"),
    fc.information_ratio(pl.col("ret"), pl.col("benchmark")).alias("information_ratio"),
    fc.up_capture(pl.col("ret"), pl.col("benchmark")).alias("up_capture"),
    fc.down_capture(pl.col("ret"), pl.col("benchmark")).alias("down_capture"),
)
```

The same functions can be run with `window=` for rolling estimates or with
`period=` and `date=` for period-bucketed estimates.

______________________________________________________________________

## Portfolio Exposures

`generate_positions` creates long-form position weights. Portfolio metrics are
intended for `group_by("date")` aggregations.

```python
from datetime import date

import polars as pl
from finance_datagen import generate_positions

import finance_calcs as fc

positions = generate_positions(
    n_dates=20,
    n_assets=40,
    gross_exposure=1.25,
    start=date(2024, 1, 2),
    exchange="XNYS",
    currency="USD",
    seed=4,
)

exposures = positions.group_by("date").agg(
    fc.gross_leverage(pl.col("weight")).alias("gross"),
    fc.net_exposure(pl.col("weight")).alias("net"),
    fc.long_exposure(pl.col("weight")).alias("long"),
    fc.short_exposure(pl.col("weight")).alias("short"),
    fc.concentration(pl.col("weight")).alias("hhi"),
    fc.top_n_concentration(pl.col("weight"), n=5).alias("top_5"),
)
```

To compute active share, join benchmark weights aligned by date and symbol, then
aggregate.

```python
with_benchmark = positions.with_columns(
    (1.0 / pl.len().over("date")).alias("benchmark_weight"),
)

active = with_benchmark.group_by("date").agg(
    fc.active_share(pl.col("weight"), pl.col("benchmark_weight")).alias("active_share"),
)
```

______________________________________________________________________

## Post-Trade Costs and Turnover

`generate_transactions` produces side-aware transaction logs with prices,
quantities, commissions, fees, and slippage bps. Use row-wise expressions for
costs, then aggregate by date, symbol, or venue.

```python
from datetime import date

import polars as pl
from finance_datagen import generate_transactions

import finance_calcs as fc

transactions = generate_transactions(
    n_dates=10,
    n_assets=20,
    trades_per_day=30,
    start=date(2024, 1, 2),
    exchange="XNYS",
    currency="USD",
    seed=6,
)

costed = transactions.with_columns(
    pl.col("timestamp").dt.date().alias("date"),
    fc.transaction_notional(pl.col("amount"), pl.col("price")).alias("notional"),
    fc.transaction_cost(
        pl.col("amount"),
        pl.col("price"),
        commission=pl.col("commission"),
        fees=pl.col("fees"),
        bps=pl.col("bps"),
    ).alias("cost"),
)

daily_costs = costed.group_by("date").agg(
    pl.col("notional").sum().alias("gross_notional"),
    pl.col("cost").sum().alias("total_cost"),
)
```

Turnover starts from a position panel because it measures changes in weights:

```python
turnover = positions.sort(["symbol", "date"]).with_columns(
    fc.turnover(pl.col("weight")).over("symbol").alias("turnover_contribution"),
).group_by("date").agg(
    pl.col("turnover_contribution").sum().alias("turnover"),
)
```

______________________________________________________________________

## Series-Level Statistics

Some statistical helpers consume an eager `pl.Series`. Use generated return
series from the same price fixtures.

```python
import polars as pl
from finance_datagen import generate_prices

import finance_calcs as fc

prices = generate_prices(n_steps=756, sigma=0.25, seed=12)
returns = prices.select(pl.col("price").finance.simple_returns().alias("ret"))["ret"].drop_nulls()

psr = fc.probabilistic_sharpe(returns, benchmark_sr=0.0)
ds = fc.deflated_sharpe(returns, n_trials=20)
min_obs = fc.minimum_track_record_length(returns, benchmark_sr=0.5)
sharpe, lower, upper = fc.sharpe_ci_bootstrap(returns, seed=12)
gpd_var = fc.gpd_var(returns, var_p=0.01)
gpd_cvar = fc.gpd_cvar(returns, var_p=0.01)
```
