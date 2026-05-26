"""Momentum technical indicators as polars expressions."""

from __future__ import annotations

import polars as pl

from .overlap import ema

__all__ = [
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "mom",
    "roc",
    "rocp",
    "rocr",
    "rocr100",
    "willr",
    "stoch_k",
    "stoch_d",
    "cci",
    "cmo",
    "trix",
    "plus_dm",
    "minus_dm",
    "plus_di",
    "minus_di",
    "adx",
]


def _wilder(expr: pl.Expr, period: int) -> pl.Expr:
    """Wilder smoothing — EMA with ``alpha = 1/period``."""
    return expr.ewm_mean(alpha=1.0 / period, adjust=False, ignore_nulls=True)


def rsi(close: pl.Expr, period: int = 14) -> pl.Expr:
    """Relative Strength Index (Wilder).

    Args:
        close: Price series.
        period: Smoothing period.

    Returns:
        Expression yielding RSI in ``[0, 100]``.
    """
    diff = close.diff()
    gain = pl.when(diff > 0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd_line(
    close: pl.Expr,
    fast: int = 12,
    slow: int = 26,
) -> pl.Expr:
    """MACD line — ``EMA(fast) - EMA(slow)``.

    Args:
        close: Price series.
        fast: Fast EMA span.
        slow: Slow EMA span.

    Returns:
        Expression yielding the MACD line.
    """
    return ema(close, fast) - ema(close, slow)


def macd_signal(
    close: pl.Expr,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.Expr:
    """MACD signal line — EMA of :func:`macd_line`.

    Args:
        close: Price series.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal EMA span.

    Returns:
        Expression yielding the MACD signal line.
    """
    return ema(macd_line(close, fast, slow), signal)


def macd_hist(
    close: pl.Expr,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.Expr:
    """MACD histogram — ``MACD - signal``.

    Args:
        close: Price series.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal EMA span.

    Returns:
        Expression yielding the MACD histogram.
    """
    return macd_line(close, fast, slow) - macd_signal(close, fast, slow, signal)


def mom(close: pl.Expr, period: int = 10) -> pl.Expr:
    """Momentum — ``close - close[period]``.

    Args:
        close: Price series.
        period: Look-back length.

    Returns:
        Difference expression.
    """
    return close - close.shift(period)


def roc(close: pl.Expr, period: int = 10) -> pl.Expr:
    """Rate-of-change in percent — ``100 * (close / close[period] - 1)``.

    Args:
        close: Price series.
        period: Look-back length.

    Returns:
        ROC expression.
    """
    return (close / close.shift(period) - 1.0) * 100.0


def rocp(close: pl.Expr, period: int = 10) -> pl.Expr:
    """ROC percentage (TA-Lib): ``(close - close[period]) / close[period]``.

    Args:
        close: Price series.
        period: Look-back length.

    Returns:
        ROCP expression.
    """
    prev = close.shift(period)
    return (close - prev) / prev


def rocr(close: pl.Expr, period: int = 10) -> pl.Expr:
    """ROC ratio: ``close / close[period]``.

    Args:
        close: Price series.
        period: Look-back length.

    Returns:
        ROCR expression.
    """
    return close / close.shift(period)


def rocr100(close: pl.Expr, period: int = 10) -> pl.Expr:
    """ROC ratio scaled by 100.

    Args:
        close: Price series.
        period: Look-back length.

    Returns:
        ROCR100 expression.
    """
    return rocr(close, period) * 100.0


def willr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 14,
) -> pl.Expr:
    """Williams %R.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Window length.

    Returns:
        Expression in ``[-100, 0]``.
    """
    hh = high.rolling_max(period)
    ll = low.rolling_min(period)
    return -100.0 * (hh - close) / (hh - ll)


def stoch_k(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 14,
) -> pl.Expr:
    """Fast stochastic %K.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Window length.

    Returns:
        Expression in ``[0, 100]``.
    """
    ll = low.rolling_min(period)
    hh = high.rolling_max(period)
    return 100.0 * (close - ll) / (hh - ll)


def stoch_d(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 14,
    d_period: int = 3,
) -> pl.Expr:
    """Stochastic %D — SMA of :func:`stoch_k`.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: %K window length.
        d_period: Smoothing window for %D.

    Returns:
        Expression yielding %D.
    """
    return stoch_k(high, low, close, period).rolling_mean(d_period)


def cci(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 20,
) -> pl.Expr:
    """Commodity Channel Index.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Window length.

    Returns:
        CCI expression. Uses mean absolute deviation in the denominator.
    """
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling_mean(period)
    mad = (tp - sma_tp).abs().rolling_mean(period)
    return (tp - sma_tp) / (0.015 * mad)


def cmo(close: pl.Expr, period: int = 14) -> pl.Expr:
    """Chande Momentum Oscillator.

    Args:
        close: Price series.
        period: Window length.

    Returns:
        CMO expression in ``[-100, 100]``.
    """
    diff = close.diff()
    gain = pl.when(diff > 0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
    g = gain.rolling_sum(period)
    loss = loss.rolling_sum(period)
    denom = g + loss
    return pl.when(denom > 0).then(100.0 * (g - loss) / denom).otherwise(0.0)


def trix(close: pl.Expr, period: int = 15) -> pl.Expr:
    """TRIX — 1-day ROC of triple-smoothed log price.

    Args:
        close: Price series.
        period: EMA span.

    Returns:
        TRIX expression in percent.
    """
    e1 = ema(close, period)
    e2 = e1.ewm_mean(span=period, adjust=False, ignore_nulls=True)
    e3 = e2.ewm_mean(span=period, adjust=False, ignore_nulls=True)
    return (e3 / e3.shift(1) - 1.0) * 100.0


def plus_dm(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    """Wilder's +DM raw (un-smoothed).

    Args:
        high: Bar high.
        low: Bar low.

    Returns:
        Per-bar +DM expression. Zero when down-move dominates.
    """
    up = high - high.shift(1)
    down = low.shift(1) - low
    return pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)


def minus_dm(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    """Wilder's -DM raw (un-smoothed).

    Args:
        high: Bar high.
        low: Bar low.

    Returns:
        Per-bar -DM expression. Zero when up-move dominates.
    """
    up = high - high.shift(1)
    down = low.shift(1) - low
    return pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)


def _true_range(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    prev_close = close.shift(1)
    return pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )


def plus_di(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 14,
) -> pl.Expr:
    """Wilder's +DI.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Smoothing period.

    Returns:
        +DI expression in percent.
    """
    tr = _wilder(_true_range(high, low, close), period)
    return 100.0 * _wilder(plus_dm(high, low), period) / tr


def minus_di(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 14,
) -> pl.Expr:
    """Wilder's -DI.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Smoothing period.

    Returns:
        -DI expression in percent.
    """
    tr = _wilder(_true_range(high, low, close), period)
    return 100.0 * _wilder(minus_dm(high, low), period) / tr


def adx(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 14,
) -> pl.Expr:
    """Average Directional Index (Wilder).

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Smoothing period.

    Returns:
        ADX expression in ``[0, 100]``.
    """
    p_di = plus_di(high, low, close, period)
    m_di = minus_di(high, low, close, period)
    denom = p_di + m_di
    dx = pl.when(denom > 0).then(100.0 * (p_di - m_di).abs() / denom).otherwise(0.0)
    return _wilder(dx, period)
