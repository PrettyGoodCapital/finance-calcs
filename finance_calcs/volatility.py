"""Volatility indicators as polars expressions."""

from __future__ import annotations

import math

import polars as pl

__all__ = [
    "true_range",
    "atr",
    "natr",
    "parkinson_vol",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "yang_zhang_vol",
    "ewma_vol",
    "realized_vol",
]


def true_range(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """Wilder's true range.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.

    Returns:
        Per-bar TR expression
        ``max(H-L, |H - C[-1]|, |L - C[-1]|)``.
    """
    prev_close = close.shift(1)
    return pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )


def atr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 14,
) -> pl.Expr:
    """Average True Range using Wilder smoothing.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Smoothing period.

    Returns:
        ATR expression.
    """
    return true_range(high, low, close).ewm_mean(alpha=1.0 / period, adjust=False, ignore_nulls=True)


def natr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 14,
) -> pl.Expr:
    """Normalised ATR — ``100 * ATR / close``.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Smoothing period.

    Returns:
        NATR expression in percent.
    """
    return 100.0 * atr(high, low, close, period) / close


def parkinson_vol(high: pl.Expr, low: pl.Expr, period: int = 20) -> pl.Expr:
    r"""Parkinson high-low range volatility estimator.

    .. math::
        \\hat{\\sigma}^2 = \\frac{1}{4 \\ln 2} \\cdot \\overline{\\left(\\ln(H/L)\\right)^2}

    Args:
        high: Bar high.
        low: Bar low.
        period: Window length.

    Returns:
        Per-period volatility expression (rolling).
    """
    log_hl = (high / low).log()
    return (log_hl.pow(2).rolling_mean(period) / (4.0 * math.log(2.0))).sqrt()


def garman_klass_vol(
    open_: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 20,
) -> pl.Expr:
    r"""Garman-Klass OHLC volatility estimator.

    .. math::
        \\hat{\\sigma}^2 = \\overline{\\tfrac{1}{2}(\\ln H/L)^2 - (2\\ln 2 - 1)(\\ln C/O)^2}

    Args:
        open_: Bar open.
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Window length.

    Returns:
        Per-period GK volatility expression (rolling).
    """
    log_hl = (high / low).log()
    log_co = (close / open_).log()
    term = 0.5 * log_hl.pow(2) - (2.0 * math.log(2.0) - 1.0) * log_co.pow(2)
    return term.rolling_mean(period).sqrt()


def rogers_satchell_vol(
    open_: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 20,
) -> pl.Expr:
    r"""Rogers-Satchell drift-independent volatility.

    .. math::
        \\hat{\\sigma}^2 = \\overline{\\ln(H/C)\\ln(H/O) + \\ln(L/C)\\ln(L/O)}

    Args:
        open_: Bar open.
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Window length.

    Returns:
        RS volatility expression (rolling).
    """
    log_hc = (high / close).log()
    log_ho = (high / open_).log()
    log_lc = (low / close).log()
    log_lo = (low / open_).log()
    return (log_hc * log_ho + log_lc * log_lo).rolling_mean(period).sqrt()


def yang_zhang_vol(
    open_: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    period: int = 20,
    k: float | None = None,
) -> pl.Expr:
    r"""Yang-Zhang volatility — minimum-variance combination of overnight,
    open-to-close, and Rogers-Satchell drift-independent components.

    Args:
        open_: Bar open.
        high: Bar high.
        low: Bar low.
        close: Bar close.
        period: Window length.
        k: Weight on open-to-close variance. Defaults to
            ``0.34 / (1.34 + (period+1)/(period-1))``.

    Returns:
        YZ volatility expression (rolling).
    """
    if k is None:
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
    prev_close = close.shift(1)
    overnight = (open_ / prev_close).log()
    oc = (close / open_).log()
    sigma_on = overnight.rolling_var(period)
    sigma_oc = oc.rolling_var(period)
    sigma_rs = rogers_satchell_vol(open_, high, low, close, period).pow(2)
    return (sigma_on + k * sigma_oc + (1.0 - k) * sigma_rs).sqrt()


def ewma_vol(returns: pl.Expr, span: int = 20) -> pl.Expr:
    """Exponentially weighted standard deviation.

    Args:
        returns: Return series.
        span: EWMA span.

    Returns:
        Square root of the EWMA variance of ``returns``.
    """
    return returns.ewm_std(span=span, adjust=False, ignore_nulls=True)


def realized_vol(returns: pl.Expr, period: int = 20) -> pl.Expr:
    """Rolling realised volatility (sample standard deviation).

    Args:
        returns: Return series.
        period: Window length.

    Returns:
        Rolling standard deviation expression.
    """
    return returns.rolling_std(period)
