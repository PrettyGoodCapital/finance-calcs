"""Core return calculations as polars expressions.

Every function accepts and returns ``pl.Expr``. Functions with a natural
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
    "period_bucket",
    "simple_returns",
    "log_returns",
    "cum_returns",
    "cum_returns_final",
    "returns",
    "aggregate_returns",
    "annualized_return",
    "annualized_volatility",
]


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
    cumulative path resets inside each period bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    one_plus = 1.0 + returns.fill_null(0.0)
    if bucket is not None:
        growth = one_plus.cum_prod().over(bucket)
    elif window is None:
        growth = one_plus.cum_prod()
    else:
        growth = one_plus.rolling_map(lambda s: s.product(), window_size=window)
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
    one_plus = 1.0 + returns.fill_null(0.0)
    if bucket is not None:
        return (one_plus.product() - 1.0).over(bucket)
    if window is None:
        return one_plus.product() - 1.0
    return one_plus.rolling_map(lambda s: s.product() - 1.0, window_size=window)


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
    """Annualised geometric return (CAGR).

    ``window=None`` → scalar lifetime CAGR. ``window=N`` → rolling
    CAGR annualised by ``periods_per_year / window``. ``period=...`` →
    CAGR for each period bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    one_plus = 1.0 + returns.fill_null(0.0)
    if bucket is not None:
        observation_count = returns.is_not_null().sum().over(bucket)
        total_growth = one_plus.product().over(bucket)
        return total_growth.pow(pl.lit(periods_per_year) / observation_count) - 1.0
    if window is None:
        n = returns.is_not_null().sum()
        total_growth = one_plus.product()
        return total_growth.pow(pl.lit(periods_per_year) / n) - 1.0
    growth = one_plus.rolling_map(lambda s: s.product(), window_size=window)
    return growth.pow(periods_per_year / window) - 1.0


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
    if bucket is not None:
        return returns.std().over(bucket) * (periods_per_year**0.5)
    if window is None:
        return returns.std() * (periods_per_year**0.5)
    return returns.rolling_std(window) * (periods_per_year**0.5)
