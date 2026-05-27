# finance calcs

Standard financial calculations

[![Build Status](https://github.com/prettygoodcapital/finance-calcs/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/prettygoodcapital/finance-calcs/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/prettygoodcapital/finance-calcs/branch/main/graph/badge.svg)](https://codecov.io/gh/prettygoodcapital/finance-calcs)
[![License](https://img.shields.io/github/license/prettygoodcapital/finance-calcs)](https://github.com/prettygoodcapital/finance-calcs)
[![PyPI](https://img.shields.io/pypi/v/finance-calcs.svg)](https://pypi.python.org/pypi/finance-calcs)

## Overview

`finance-calcs` provides financial calculations as composable Polars
expressions. It is designed for lazy execution, namespace-style ergonomics, and
direct interoperability with the rest of the `finance-*` stack.

The public API follows a few rules:

- every expression metric accepts and returns `pl.Expr`
- metrics are exposed once, with optional `window=` and `period=` controls
  rather than separate rolling, monthly, and annual variants
- functions are also available through the `.finance` namespace on both
  `pl.Expr` and `pl.Series`
- examples use synthetic but realistic fixtures from `finance-datagen`

## Implemented coverage

| Topic                        | Functions                                                                                                                                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Returns and periods          | `period_bucket`, `simple_returns`, `log_returns`, `cum_returns`, `cum_returns_final`, `returns`, `aggregate_returns`, `annualized_return`, `annualized_volatility`                                         |
| Risk and drawdown            | `volatility`, `sharpe`, `sortino`, `calmar`, `downside_deviation`, `downside_risk`, `drawdown_series`, `underwater_series`, `max_drawdown`, `value_at_risk`, `conditional_value_at_risk`, `parametric_var` |
| Technical indicators         | Moving averages, Bollinger/Donchian channels, momentum oscillators, range volatility, and volume indicators                                                                                                |
| Alpha and quantiles          | Forward returns, conditional/horizon IC, IC decay, IC summaries, quantile assignment, signal normalization, quantile returns, turnover, and long/short spreads                                             |
| Factor and benchmark metrics | Alpha, beta, up/down capture, batting average, tracking error, and information ratio                                                                                                                       |
| Distribution and tail risk   | Higher moments, Sharpe significance helpers, tail ratio, ulcer index, omega ratio, GPD VaR, and GPD CVaR                                                                                                   |
| Portfolio and post-trade     | Exposure, concentration, active share, transaction costs/volume/attribution, slippage, turnover, round trips, MAE/MFE, and trade-quality metrics                                                           |

See the [Examples](docs/src/EXAMPLES.md) page for workflows with generated data
and the [API](docs/src/API.md) page for a complete grouped reference for every
public function.

## Quick start

Generate a deterministic daily equity path with `finance-datagen`, then compute
return and risk metrics as Polars expressions.

```python
import polars as pl
from finance_datagen import generate_prices

import finance_calcs as fc

prices = generate_prices(symbol="ACME", seed=7)

out = prices.with_columns(
    pl.col("price").finance.simple_returns().alias("ret"),
).select(
    fc.returns(pl.col("ret")).alias("total_return"),
    pl.col("ret").finance.annualized_return().alias("ann_return"),
    pl.col("ret").finance.volatility().alias("ann_vol"),
    pl.col("ret").finance.sharpe().alias("sharpe"),
    pl.col("ret").finance.max_drawdown().alias("max_drawdown"),
)
```

Use `finance-datagen.ohlc_from_close` when calculations need OHLCV bars:

```python
from finance_datagen import ohlc_from_close

bars = ohlc_from_close(prices["price"], symbol="ACME", seed=7)

features = bars.with_columns(
    pl.col("close").finance.sma(20).alias("sma_20"),
    pl.col("close").finance.rsi(14).alias("rsi_14"),
    fc.atr(pl.col("high"), pl.col("low"), pl.col("close")).alias("atr_14"),
    fc.obv(pl.col("close"), pl.col("volume")).alias("obv"),
)
```

## Period and frequency slices

Use `period=` for calendar-style slices and keep `window=` for rolling row-count
windows. A `period` can be a `finance_enums.Frequency`, any alias accepted by
`finance_enums.to_frequency()`, any Polars `dt.truncate()` duration string, or a
precomputed bucket expression.

```python
import polars as pl
from finance_enums import Frequency

monthly = prices.with_columns(
    pl.col("price").finance.simple_returns().alias("ret"),
).with_columns(
    fc.period_bucket(pl.col("timestamp"), Frequency.Month).alias("month"),
    pl.col("ret").finance.returns(period="month", date=pl.col("timestamp")).alias("month_return"),
    pl.col("ret").finance.sharpe(period="1q", date=pl.col("timestamp")).alias("quarter_sharpe"),
)
```

For fiscal periods, strategy regimes, or exchange-calendar grids built upstream,
pass the bucket expression directly:

```python
bucketed = prices.with_columns(
    pl.col("price").finance.simple_returns().alias("ret"),
    pl.col("timestamp").dt.year().alias("fiscal_year"),
).with_columns(
    fc.returns(pl.col("ret"), period=pl.col("fiscal_year")).alias("fiscal_return"),
)
```

## Stack integration

`finance-calcs` is intended to pair with:

- `finance-datagen` for synthetic fixtures and test inputs
- `finance-dates` for calendar-aware date handling upstream
- `finance-enums` for shared enum-backed trading semantics upstream

That keeps calculations focused on typed expressions instead of schema cleanup,
string parsing, or calendar repair.
