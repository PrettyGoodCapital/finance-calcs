"""Overlap studies / moving averages as polars expressions."""

from __future__ import annotations

import polars as pl

__all__ = [
    "sma",
    "ema",
    "wma",
    "dema",
    "tema",
    "midpoint",
    "midprice",
    "bbands_upper",
    "bbands_middle",
    "bbands_lower",
    "donchian_upper",
    "donchian_lower",
    "donchian_middle",
]


def sma(close: pl.Expr, period: int = 20) -> pl.Expr:
    """Simple moving average over ``period`` observations.

    Args:
        close: Price (or any series) to average.
        period: Window length.

    Returns:
        Rolling mean expression.
    """
    return close.rolling_mean(period)


def ema(close: pl.Expr, period: int = 20) -> pl.Expr:
    """Exponential moving average with ``span = period``.

    Args:
        close: Series to smooth.
        period: Span. The smoothing factor is ``2 / (period + 1)``.

    Returns:
        EWMA expression.
    """
    return close.ewm_mean(span=period, adjust=False, ignore_nulls=True)


def wma(close: pl.Expr, period: int = 20) -> pl.Expr:
    """Linearly-weighted moving average.

    Args:
        close: Series to smooth.
        period: Window length.

    Returns:
        Expression yielding the WMA. Recent observations have higher
        weight: weight ``i`` = ``i + 1`` for ``i in 0..period-1``.
    """
    weights = list(range(1, period + 1))
    return close.rolling_mean(window_size=period, weights=weights)


def dema(close: pl.Expr, period: int = 20) -> pl.Expr:
    """Double exponential moving average: ``2 * EMA - EMA(EMA)``.

    Args:
        close: Series to smooth.
        period: Span.

    Returns:
        DEMA expression.
    """
    e1 = ema(close, period)
    e2 = e1.ewm_mean(span=period, adjust=False, ignore_nulls=True)
    return 2.0 * e1 - e2


def tema(close: pl.Expr, period: int = 20) -> pl.Expr:
    """Triple exponential moving average ``3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))``.

    Args:
        close: Series to smooth.
        period: Span.

    Returns:
        TEMA expression.
    """
    e1 = ema(close, period)
    e2 = e1.ewm_mean(span=period, adjust=False, ignore_nulls=True)
    e3 = e2.ewm_mean(span=period, adjust=False, ignore_nulls=True)
    return 3.0 * e1 - 3.0 * e2 + e3


def midpoint(close: pl.Expr, period: int = 14) -> pl.Expr:
    """``(rolling_max(close) + rolling_min(close)) / 2``.

    Args:
        close: Price series.
        period: Window length.

    Returns:
        Midpoint expression.
    """
    return (close.rolling_max(period) + close.rolling_min(period)) / 2.0


def midprice(high: pl.Expr, low: pl.Expr, period: int = 14) -> pl.Expr:
    """``(rolling_max(high) + rolling_min(low)) / 2``.

    Args:
        high: Bar high.
        low: Bar low.
        period: Window length.

    Returns:
        Midprice expression.
    """
    return (high.rolling_max(period) + low.rolling_min(period)) / 2.0


def bbands_middle(close: pl.Expr, period: int = 20) -> pl.Expr:
    """Bollinger middle band — SMA of close.

    Args:
        close: Price series.
        period: Window length.

    Returns:
        Rolling mean expression.
    """
    return sma(close, period)


def bbands_upper(
    close: pl.Expr,
    period: int = 20,
    nbdev_up: float = 2.0,
) -> pl.Expr:
    """Bollinger upper band ``SMA + nbdev_up * std``.

    Args:
        close: Price series.
        period: Window length.
        nbdev_up: Number of standard deviations above the SMA.

    Returns:
        Upper-band expression.
    """
    return sma(close, period) + nbdev_up * close.rolling_std(period)


def bbands_lower(
    close: pl.Expr,
    period: int = 20,
    nbdev_dn: float = 2.0,
) -> pl.Expr:
    """Bollinger lower band ``SMA - nbdev_dn * std``.

    Args:
        close: Price series.
        period: Window length.
        nbdev_dn: Number of standard deviations below the SMA.

    Returns:
        Lower-band expression.
    """
    return sma(close, period) - nbdev_dn * close.rolling_std(period)


def donchian_upper(high: pl.Expr, period: int = 20) -> pl.Expr:
    """Donchian upper channel — rolling maximum of ``high``.

    Args:
        high: Bar high.
        period: Window length.

    Returns:
        Rolling max expression.
    """
    return high.rolling_max(period)


def donchian_lower(low: pl.Expr, period: int = 20) -> pl.Expr:
    """Donchian lower channel — rolling minimum of ``low``.

    Args:
        low: Bar low.
        period: Window length.

    Returns:
        Rolling min expression.
    """
    return low.rolling_min(period)


def donchian_middle(high: pl.Expr, low: pl.Expr, period: int = 20) -> pl.Expr:
    """Donchian midline.

    Args:
        high: Bar high.
        low: Bar low.
        period: Window length.

    Returns:
        Average of the upper and lower Donchian channels.
    """
    return (donchian_upper(high, period) + donchian_lower(low, period)) / 2.0
