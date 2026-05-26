"""Alpha / beta and capture-ratio calculations.

Two-input expressions: ``returns`` is the asset/portfolio series and
``benchmark`` is the corresponding benchmark series, sampled at the
same frequency.

Per the workspace rule, every metric is a single function with an
optional ``window=None`` keyword: ``None`` collapses to a scalar
lifetime value, an integer ``N`` produces a rolling expression, and
``period=`` computes inside each period bucket. There are no
``rolling_*`` siblings.
"""

from __future__ import annotations

import math

import polars as pl

from ._periods import PeriodLike, _bucket_or_none, _check_window_period

__all__ = [
    "beta",
    "alpha",
    "up_beta",
    "down_beta",
    "up_alpha",
    "down_alpha",
    "up_capture",
    "down_capture",
    "up_down_capture",
    "batting_average",
    "tracking_error",
    "information_ratio",
]


def _rf_per_period(risk_free: float | pl.Expr, periods_per_year: int) -> float | pl.Expr:
    """Convert annual scalar ``risk_free`` to per-period.

    A :class:`pl.Expr` is assumed to already be a per-period rate
    column and is returned unchanged.
    """
    if isinstance(risk_free, pl.Expr):
        return risk_free
    return risk_free / periods_per_year


def beta(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """OLS market beta — ``cov(r, b) / var(b)``.

    ``window=None`` → scalar; ``window=N`` → rolling beta;
    ``period=...`` → per-bucket beta.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    if bucket is not None:
        return pl.cov(returns, benchmark).over(bucket) / benchmark.var().over(bucket)
    if window is None:
        return pl.cov(returns, benchmark) / benchmark.var()
    cov = pl.rolling_cov(returns, benchmark, window_size=window)
    return cov / benchmark.rolling_var(window)


def alpha(
    returns: pl.Expr,
    benchmark: pl.Expr,
    risk_free: float | pl.Expr = 0.0,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised Jensen's alpha.

    ``risk_free`` may be a scalar annual rate (divided to per-period)
    or a :class:`pl.Expr` per-period rate column for a time-varying
    risk-free rate. ``window=None`` → scalar; ``window=N`` → rolling
    annualised alpha; ``period=...`` → per-bucket alpha.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    rf = _rf_per_period(risk_free, periods_per_year)
    b = beta(returns, benchmark, window=window, period=period, date=date)
    if bucket is not None:
        return ((returns - rf).mean().over(bucket) - b * (benchmark - rf).mean().over(bucket)) * periods_per_year
    if window is None:
        return ((returns - rf).mean() - b * (benchmark - rf).mean()) * periods_per_year
    mean_r = (returns - rf).rolling_mean(window)
    mean_b = (benchmark - rf).rolling_mean(window)
    return (mean_r - b * mean_b) * periods_per_year


def _filter_capture(returns, benchmark, mask, window):
    if window is None:
        return returns.filter(mask), benchmark.filter(mask)
    r_m = pl.when(mask).then(returns).otherwise(None)
    b_m = pl.when(mask).then(benchmark).otherwise(None)
    return r_m, b_m


def up_beta(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Beta restricted to bars where ``benchmark > 0``."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    r_up, b_up = _filter_capture(returns, benchmark, benchmark > 0, window)
    if bucket is not None:
        return pl.cov(r_up, b_up).over(bucket) / b_up.var().over(bucket)
    if window is None:
        return pl.cov(r_up, b_up) / b_up.var()
    cov = pl.rolling_cov(r_up, b_up, window_size=window)
    return cov / b_up.rolling_var(window)


def down_beta(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Beta restricted to bars where ``benchmark < 0``."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    r_dn, b_dn = _filter_capture(returns, benchmark, benchmark < 0, window)
    if bucket is not None:
        return pl.cov(r_dn, b_dn).over(bucket) / b_dn.var().over(bucket)
    if window is None:
        return pl.cov(r_dn, b_dn) / b_dn.var()
    cov = pl.rolling_cov(r_dn, b_dn, window_size=window)
    return cov / b_dn.rolling_var(window)


def up_alpha(
    returns: pl.Expr,
    benchmark: pl.Expr,
    risk_free: float | pl.Expr = 0.0,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised alpha on up-market bars only."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    rf = _rf_per_period(risk_free, periods_per_year)
    r_up, b_up = _filter_capture(returns, benchmark, benchmark > 0, window)
    if bucket is not None:
        bu = pl.cov(r_up, b_up).over(bucket) / b_up.var().over(bucket)
        return ((r_up - rf).mean().over(bucket) - bu * (b_up - rf).mean().over(bucket)) * periods_per_year
    if window is None:
        bu = pl.cov(r_up, b_up) / b_up.var()
        return ((r_up - rf).mean() - bu * (b_up - rf).mean()) * periods_per_year
    bu = pl.rolling_cov(r_up, b_up, window_size=window) / b_up.rolling_var(window)
    return ((r_up - rf).rolling_mean(window) - bu * (b_up - rf).rolling_mean(window)) * periods_per_year


def down_alpha(
    returns: pl.Expr,
    benchmark: pl.Expr,
    risk_free: float | pl.Expr = 0.0,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised alpha on down-market bars only."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    rf = _rf_per_period(risk_free, periods_per_year)
    r_dn, b_dn = _filter_capture(returns, benchmark, benchmark < 0, window)
    if bucket is not None:
        bd = pl.cov(r_dn, b_dn).over(bucket) / b_dn.var().over(bucket)
        return ((r_dn - rf).mean().over(bucket) - bd * (b_dn - rf).mean().over(bucket)) * periods_per_year
    if window is None:
        bd = pl.cov(r_dn, b_dn) / b_dn.var()
        return ((r_dn - rf).mean() - bd * (b_dn - rf).mean()) * periods_per_year
    bd = pl.rolling_cov(r_dn, b_dn, window_size=window) / b_dn.rolling_var(window)
    return ((r_dn - rf).rolling_mean(window) - bd * (b_dn - rf).rolling_mean(window)) * periods_per_year


def up_capture(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Mean asset return / mean benchmark return on up-market bars."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    mask = benchmark > 0
    if bucket is not None:
        return returns.filter(mask).mean().over(bucket) / benchmark.filter(mask).mean().over(bucket)
    if window is None:
        return returns.filter(mask).mean() / benchmark.filter(mask).mean()
    r_up = pl.when(mask).then(returns).otherwise(None)
    b_up = pl.when(mask).then(benchmark).otherwise(None)
    return r_up.rolling_mean(window, min_samples=1) / b_up.rolling_mean(window, min_samples=1)


def down_capture(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Mean asset return / mean benchmark return on down-market bars."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    mask = benchmark < 0
    if bucket is not None:
        return returns.filter(mask).mean().over(bucket) / benchmark.filter(mask).mean().over(bucket)
    if window is None:
        return returns.filter(mask).mean() / benchmark.filter(mask).mean()
    r_dn = pl.when(mask).then(returns).otherwise(None)
    b_dn = pl.when(mask).then(benchmark).otherwise(None)
    return r_dn.rolling_mean(window, min_samples=1) / b_dn.rolling_mean(window, min_samples=1)


def up_down_capture(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """``up_capture / down_capture``."""
    return up_capture(returns, benchmark, window=window, period=period, date=date) / down_capture(
        returns, benchmark, window=window, period=period, date=date
    )


def batting_average(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Fraction of periods where ``returns > benchmark``."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    hit = (returns > benchmark).cast(pl.Float64)
    if bucket is not None:
        return hit.mean().over(bucket)
    if window is None:
        return hit.mean()
    return hit.rolling_mean(window)


def tracking_error(
    returns: pl.Expr,
    benchmark: pl.Expr,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised tracking error — ``std(r - b) * sqrt(ppy)``."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    active = returns - benchmark
    scale = math.sqrt(periods_per_year)
    if bucket is not None:
        return active.std().over(bucket) * scale
    if window is None:
        return active.std() * scale
    return active.rolling_std(window) * scale


def information_ratio(
    returns: pl.Expr,
    benchmark: pl.Expr,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised information ratio — ``mean(r-b)/std(r-b) * sqrt(ppy)``."""
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    active = returns - benchmark
    scale = math.sqrt(periods_per_year)
    if bucket is not None:
        return active.mean().over(bucket) / active.std().over(bucket) * scale
    if window is None:
        return active.mean() / active.std() * scale
    return active.rolling_mean(window) / active.rolling_std(window) * scale
