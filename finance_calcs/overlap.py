"""Overlap studies / moving averages as polars expressions."""

from __future__ import annotations

import polars as pl

__all__ = [
    "bbands_lower",
    "bbands_middle",
    "bbands_upper",
    "dema",
    "donchian_lower",
    "donchian_middle",
    "donchian_upper",
    "ema",
    "midpoint",
    "midprice",
    "sma",
    "tema",
    "wma",
]


def sma(close: pl.Expr, window: int = 20) -> pl.Expr:
    """Simple moving average over ``window`` observations.

    Args:
        close: Price (or any series) to average.
        window: Window length.

    Returns:
        Rolling mean expression.
    """
    return close.rolling_mean(window)


def ema(close: pl.Expr, window: int = 20) -> pl.Expr:
    """Exponential moving average with ``span = window``.

    Args:
        close: Series to smooth.
        window: Span. The smoothing factor is ``2 / (window + 1)``.

    Returns:
        EWMA expression.
    """
    return close.ewm_mean(span=window, adjust=False, ignore_nulls=True)


def wma(close: pl.Expr, window: int = 20) -> pl.Expr:
    """Linearly-weighted moving average.

    Args:
        close: Series to smooth.
        window: Window length.

    Returns:
        Expression yielding the WMA. Recent observations have higher
        weight: weight ``i`` = ``i + 1`` for ``i in 0..window-1``.
    """
    weights = list(range(1, window + 1))
    return close.rolling_mean(window_size=window, weights=weights)


def dema(close: pl.Expr, window: int = 20) -> pl.Expr:
    """Double exponential moving average: ``2 * EMA - EMA(EMA)``.

    Args:
        close: Series to smooth.
        window: Span.

    Returns:
        DEMA expression.
    """
    e1 = ema(close, window)
    e2 = e1.ewm_mean(span=window, adjust=False, ignore_nulls=True)
    return 2.0 * e1 - e2


def tema(close: pl.Expr, window: int = 20) -> pl.Expr:
    """Triple exponential moving average ``3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))``.

    Args:
        close: Series to smooth.
        window: Span.

    Returns:
        TEMA expression.
    """
    e1 = ema(close, window)
    e2 = e1.ewm_mean(span=window, adjust=False, ignore_nulls=True)
    e3 = e2.ewm_mean(span=window, adjust=False, ignore_nulls=True)
    return 3.0 * e1 - 3.0 * e2 + e3


def midpoint(close: pl.Expr, window: int = 14) -> pl.Expr:
    """``(rolling_max(close) + rolling_min(close)) / 2``.

    Args:
        close: Price series.
        window: Window length.

    Returns:
        Midpoint expression.
    """
    return (close.rolling_max(window) + close.rolling_min(window)) / 2.0


def midprice(high: pl.Expr, low: pl.Expr, window: int = 14) -> pl.Expr:
    """``(rolling_max(high) + rolling_min(low)) / 2``.

    Args:
        high: Bar high.
        low: Bar low.
        window: Window length.

    Returns:
        Midprice expression.
    """
    return (high.rolling_max(window) + low.rolling_min(window)) / 2.0


def bbands_middle(close: pl.Expr, window: int = 20) -> pl.Expr:
    """Bollinger middle band — SMA of close.

    Args:
        close: Price series.
        window: Window length.

    Returns:
        Rolling mean expression.
    """
    return sma(close, window)


def bbands_upper(
    close: pl.Expr,
    window: int = 20,
    upper_deviations: float = 2.0,
) -> pl.Expr:
    """Bollinger upper band ``SMA + upper_deviations * std``.

    Args:
        close: Price series.
        window: Window length.
        upper_deviations: Number of standard deviations above the SMA.

    Returns:
        Upper-band expression.
    """
    return sma(close, window) + upper_deviations * close.rolling_std(window)


def bbands_lower(
    close: pl.Expr,
    window: int = 20,
    lower_deviations: float = 2.0,
) -> pl.Expr:
    """Bollinger lower band ``SMA - lower_deviations * std``.

    Args:
        close: Price series.
        window: Window length.
        lower_deviations: Number of standard deviations below the SMA.

    Returns:
        Lower-band expression.
    """
    return sma(close, window) - lower_deviations * close.rolling_std(window)


def donchian_upper(high: pl.Expr, window: int = 20) -> pl.Expr:
    """Donchian upper channel — rolling maximum of ``high``.

    Args:
        high: Bar high.
        window: Window length.

    Returns:
        Rolling max expression.
    """
    return high.rolling_max(window)


def donchian_lower(low: pl.Expr, window: int = 20) -> pl.Expr:
    """Donchian lower channel — rolling minimum of ``low``.

    Args:
        low: Bar low.
        window: Window length.

    Returns:
        Rolling min expression.
    """
    return low.rolling_min(window)


def donchian_middle(high: pl.Expr, low: pl.Expr, window: int = 20) -> pl.Expr:
    """Donchian midline.

    Args:
        high: Bar high.
        low: Bar low.
        window: Window length.

    Returns:
        Average of the upper and lower Donchian channels.
    """
    return (donchian_upper(high, window) + donchian_lower(low, window)) / 2.0
