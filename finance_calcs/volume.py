"""Volume indicators as polars expressions."""

from __future__ import annotations

import polars as pl

from .overlap import ema

__all__ = ["obv", "ad", "adosc"]


def obv(close: pl.Expr, volume: pl.Expr) -> pl.Expr:
    """On-Balance Volume.

    Args:
        close: Price series.
        volume: Volume series.

    Returns:
        Running cumulative signed volume. The first bar contributes zero
        because the prior close is unknown.
    """
    diff = close.diff()
    direction = pl.when(diff > 0).then(1.0).when(diff < 0).then(-1.0).otherwise(0.0)
    return (direction * volume).fill_null(0.0).cum_sum()


def ad(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
) -> pl.Expr:
    """Chaikin Accumulation/Distribution line.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        volume: Bar volume.

    Returns:
        Running A/D line. Bars with zero range contribute zero flow.
    """
    rng = high - low
    mfm = pl.when(rng > 0).then(((close - low) - (high - close)) / rng).otherwise(0.0)
    return (mfm * volume).cum_sum()


def adosc(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    fast: int = 3,
    slow: int = 10,
) -> pl.Expr:
    """Chaikin A/D Oscillator — ``EMA(AD, fast) - EMA(AD, slow)``.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        volume: Bar volume.
        fast: Fast EMA span.
        slow: Slow EMA span.

    Returns:
        ADOSC expression.
    """
    line = ad(high, low, close, volume)
    return ema(line, fast) - ema(line, slow)
