"""Core return calculations as Polars expressions.

Every public function accepts and returns ``pl.Expr``. Floating-point NaN and
Polars null values are both treated as missing observations. Functions with a natural
rolling form take a ``window=None`` keyword: ``None`` means full-sample
(a scalar), an integer means a trailing rolling window of that many
observations. Calendar or custom slices use ``period=`` with either a
``date=`` expression or a precomputed bucket expression.

Per the workspace rule, there are no separate ``rolling_*`` or
``periodic_*`` siblings — temporal granularity is controlled by
``window=`` or ``period=``.
"""

from __future__ import annotations

import polars as pl

from ._periods import PeriodLike, _bucket_or_none, _check_window_period, period_bucket

__all__ = [
    "aggregate_returns",
    "annualized_return",
    "annualized_volatility",
    "cum_returns",
    "cum_returns_final",
    "log_returns",
    "period_bucket",
    "returns",
    "simple_returns",
]


def _clean_returns(returns: pl.Expr) -> pl.Expr:
    """Treat floating-point NaN values as missing observations."""
    return returns.fill_nan(None)


def _rolling_product(values: pl.Expr, window: int) -> pl.Expr:
    """Compute a rolling product using native Polars expressions."""
    zero_count = (values == 0.0).cast(pl.UInt32).rolling_sum(window)
    negative_count = (values < 0.0).cast(pl.UInt32).rolling_sum(window)
    log_sum = pl.when(values == 0.0).then(0.0).otherwise(values.abs().log()).rolling_sum(window)
    sign = pl.when((negative_count % 2) == 0).then(1.0).otherwise(-1.0)
    return pl.when(zero_count > 0).then(0.0).otherwise(log_sum.exp() * sign)


def simple_returns(price: pl.Expr) -> pl.Expr:
    r"""Per-period simple return :math:`p_t / p_{t-1} - 1`."""
    return (price / price.shift(1)) - 1.0


def log_returns(price: pl.Expr) -> pl.Expr:
    r"""Per-period log return :math:`\log(p_t / p_{t-1})`."""
    return (price / price.shift(1)).log()


def cum_returns(
    returns: pl.Expr,
    starting_value: float = 0.0,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Cumulative compounded return.

    With ``window=None`` returns the cumulative path
    ``(1 + r).cumprod() - 1``. With ``window=N`` returns the compounded
    return over each trailing ``N``-bar window. With ``period=...``, the
    cumulative path resets inside each period bucket. Missing observations are
    neutral for compounding.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    one_plus = 1.0 + _clean_returns(returns).fill_null(0.0)
    if bucket is not None:
        growth = one_plus.cum_prod().over(bucket)
    elif window is None:
        growth = one_plus.cum_prod()
    else:
        growth = _rolling_product(one_plus, window)
    if starting_value == 0.0:
        return growth - 1.0
    return growth * starting_value


def cum_returns_final(
    returns: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Total compounded return.

    ``window=None`` → scalar terminal compounded return. ``window=N`` →
    rolling compounded return over each trailing ``N``-bar window.
    ``period=...`` → terminal compounded return for each period bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    one_plus = 1.0 + _clean_returns(returns).fill_null(0.0)
    if bucket is not None:
        return (one_plus.product() - 1.0).over(bucket)
    if window is None:
        return one_plus.product() - 1.0
    return _rolling_product(one_plus, window) - 1.0


def returns(
    returns: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Compound return over a trailing window or full sample.

    ``window=None, period=None`` returns the full-sample compound return.
    ``window=N`` returns trailing compounded returns over ``N`` rows.
    ``period=...`` returns the compounded return for each period bucket.
    """
    return cum_returns_final(returns, window=window, period=period, date=date)


def aggregate_returns(returns: pl.Expr, date: pl.Expr, period: PeriodLike) -> pl.Expr:
    """Compound returns by a calendar or custom period bucket."""
    return cum_returns_final(returns, period=period, date=date)


def annualized_return(
    returns: pl.Expr,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised geometric return.

    ``window=None`` → scalar lifetime CAGR. ``window=N`` → rolling
    CAGR annualised by ``periods_per_year / window``. ``period=...`` →
    CAGR for each period bucket. Annualisation uses the count of non-missing
    observations, not elapsed calendar time; ``date`` is used only to build a
    period bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    clean_returns = _clean_returns(returns)
    one_plus = 1.0 + clean_returns.fill_null(0.0)
    if bucket is not None:
        observation_count = clean_returns.is_not_null().sum().over(bucket)
        total_growth = one_plus.product().over(bucket)
        return total_growth.pow(pl.lit(periods_per_year) / observation_count) - 1.0
    if window is None:
        n = clean_returns.is_not_null().sum()
        total_growth = one_plus.product()
        return total_growth.pow(pl.lit(periods_per_year) / n) - 1.0
    growth = _rolling_product(one_plus, window)
    observation_count = clean_returns.is_not_null().cast(pl.UInt32).rolling_sum(window)
    return growth.pow(periods_per_year / observation_count) - 1.0


def annualized_volatility(
    returns: pl.Expr,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    r"""Annualised standard deviation of returns.

    ``window=None`` → scalar lifetime volatility; ``window=N`` →
    rolling annualised volatility; ``period=...`` → volatility for each
    period bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    clean_returns = _clean_returns(returns)
    if bucket is not None:
        return clean_returns.std().over(bucket) * (periods_per_year**0.5)
    if window is None:
        return clean_returns.std() * (periods_per_year**0.5)
    return clean_returns.rolling_std(window) * (periods_per_year**0.5)
