"""Volatility indicators as polars expressions."""

from __future__ import annotations

import math

import polars as pl
from finance_enums import Frequency

from ._periods import FrequencyLike, _observations_per_year

__all__ = [
    "atr",
    "exponentially_weighted_volatility",
    "garman_klass_volatility",
    "natr",
    "parkinson_volatility",
    "realized_volatility",
    "rogers_satchell_volatility",
    "true_range",
    "yang_zhang_volatility",
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
    window: int = 14,
) -> pl.Expr:
    """Average True Range using Wilder smoothing.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        window: Smoothing window.

    Returns:
        ATR expression.
    """
    return true_range(high, low, close).ewm_mean(alpha=1.0 / window, adjust=False, ignore_nulls=True)


def natr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int = 14,
) -> pl.Expr:
    """Normalised ATR — ``100 * ATR / close``.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.
        window: Smoothing window.

    Returns:
        NATR expression in percent.
    """
    return 100.0 * atr(high, low, close, window) / close


def parkinson_volatility(
    high: pl.Expr,
    low: pl.Expr,
    window: int = 20,
    *,
    frequency: FrequencyLike = Frequency.Day,
) -> pl.Expr:
    r"""Parkinson high-low range volatility estimator.

    .. math::
        \\hat{\\sigma}^2 = \\frac{1}{4 \\ln 2} \\cdot \\overline{\\left(\\ln(H/L)\\right)^2}

    Args:
        high: Bar high.
        low: Bar low.
        window: Window length.

    Returns:
        Annualized rolling volatility expression.
    """
    log_hl = (high / low).log()
    return (log_hl.pow(2).rolling_mean(window) / (4.0 * math.log(2.0))).sqrt() * math.sqrt(_observations_per_year(frequency))


def garman_klass_volatility(
    open_: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int = 20,
    *,
    frequency: FrequencyLike = Frequency.Day,
) -> pl.Expr:
    r"""Garman-Klass OHLC volatility estimator.

    .. math::
        \\hat{\\sigma}^2 = \\overline{\\tfrac{1}{2}(\\ln H/L)^2 - (2\\ln 2 - 1)(\\ln C/O)^2}

    Args:
        open_: Bar open.
        high: Bar high.
        low: Bar low.
        close: Bar close.
        window: Window length.

    Returns:
        Annualized rolling Garman-Klass volatility expression.
    """
    log_hl = (high / low).log()
    log_co = (close / open_).log()
    term = 0.5 * log_hl.pow(2) - (2.0 * math.log(2.0) - 1.0) * log_co.pow(2)
    return term.rolling_mean(window).sqrt() * math.sqrt(_observations_per_year(frequency))


def _rogers_satchell_variance(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    log_hc = (high / close).log()
    log_ho = (high / open_).log()
    log_lc = (low / close).log()
    log_lo = (low / open_).log()
    return (log_hc * log_ho + log_lc * log_lo).rolling_mean(window)


def rogers_satchell_volatility(
    open_: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int = 20,
    *,
    frequency: FrequencyLike = Frequency.Day,
) -> pl.Expr:
    r"""Rogers-Satchell drift-independent volatility.

    .. math::
        \\hat{\\sigma}^2 = \\overline{\\ln(H/C)\\ln(H/O) + \\ln(L/C)\\ln(L/O)}

    Args:
        open_: Bar open.
        high: Bar high.
        low: Bar low.
        close: Bar close.
        window: Window length.

    Returns:
        Annualized rolling Rogers-Satchell volatility expression.
    """
    return _rogers_satchell_variance(open_, high, low, close, window).sqrt() * math.sqrt(_observations_per_year(frequency))


def yang_zhang_volatility(
    open_: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int = 20,
    *,
    weight: float | None = None,
    frequency: FrequencyLike = Frequency.Day,
) -> pl.Expr:
    r"""Yang-Zhang volatility — minimum-variance combination of overnight,
    open-to-close, and Rogers-Satchell drift-independent components.

    Args:
        open_: Bar open.
        high: Bar high.
        low: Bar low.
        close: Bar close.
        window: Window length.
        weight: Weight on open-to-close variance. Defaults to
            ``0.34 / (1.34 + (window+1)/(window-1))``.

    Returns:
        YZ volatility expression (rolling).
    """
    if weight is None:
        weight = 0.34 / (1.34 + (window + 1) / (window - 1))
    prev_close = close.shift(1)
    overnight = (open_ / prev_close).log()
    oc = (close / open_).log()
    sigma_on = overnight.rolling_var(window)
    sigma_oc = oc.rolling_var(window)
    sigma_rs = _rogers_satchell_variance(open_, high, low, close, window)
    return (sigma_on + weight * sigma_oc + (1.0 - weight) * sigma_rs).sqrt() * math.sqrt(_observations_per_year(frequency))


def exponentially_weighted_volatility(
    returns: pl.Expr,
    window: int = 20,
    *,
    frequency: FrequencyLike = Frequency.Day,
) -> pl.Expr:
    """Exponentially weighted standard deviation.

    Args:
        returns: Return series.
        window: EWMA window.

    Returns:
        Square root of the EWMA variance of ``returns``.
    """
    return returns.ewm_std(span=window, adjust=False, ignore_nulls=True) * math.sqrt(_observations_per_year(frequency))


def realized_volatility(
    returns: pl.Expr,
    window: int = 20,
    *,
    frequency: FrequencyLike = Frequency.Day,
) -> pl.Expr:
    """Rolling realised volatility (sample standard deviation).

    Args:
        returns: Return series.
        window: Window length.

    Returns:
        Annualized rolling standard deviation expression.
    """
    return returns.rolling_std(window) * math.sqrt(_observations_per_year(frequency))
