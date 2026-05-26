"""Tail-risk metrics.

Lightweight expression-level helpers for tail-shape statistics
(:func:`tail_ratio`, :func:`ulcer_index`, :func:`omega_ratio`), plus a
``pl.Series``-level GPD peak-over-threshold fit for extreme VaR/CVaR
estimates that need an MLE.

Rolling forms of historical VaR / CVaR live on
:func:`finance_calcs.value_at_risk` / :func:`finance_calcs.conditional_value_at_risk`
via their ``window=`` keyword — there are no separate ``rolling_*``
siblings.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ._periods import PeriodLike, _bucket_or_none, _check_window_period

__all__ = [
    "tail_ratio",
    "ulcer_index",
    "omega_ratio",
    "gpd_var",
    "gpd_cvar",
]


def tail_ratio(
    returns: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Right tail / left tail ratio — ``|p95| / |p05|``.

    ``window=None`` → scalar; ``window=N`` → rolling;
    ``period=...`` → per-bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    if bucket is not None:
        return returns.quantile(0.95).abs().over(bucket) / returns.quantile(0.05).abs().over(bucket)
    if window is None:
        return returns.quantile(0.95).abs() / returns.quantile(0.05).abs()
    p95 = returns.rolling_quantile(quantile=0.95, window_size=window).abs()
    p05 = returns.rolling_quantile(quantile=0.05, window_size=window).abs()
    return p95 / p05


def ulcer_index(
    returns: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """RMS of the drawdown sequence.

    ``UI = sqrt(mean(dd_t^2))`` where ``dd_t`` is the percentage
    drawdown at time ``t``. ``window=None`` → scalar; ``window=N`` →
    rolling RMS over each trailing ``N``-bar window. ``period=...`` →
    per-bucket RMS drawdown.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    equity = (1.0 + returns).cum_prod()
    if bucket is not None:
        equity = equity.over(bucket)
        peak = equity.cum_max().over(bucket)
        dd = (equity / peak) - 1.0
        return dd.pow(2).mean().over(bucket).sqrt()
    peak = equity.cum_max()
    dd = (equity / peak) - 1.0
    if window is None:
        return dd.pow(2).mean().sqrt()
    return dd.pow(2).rolling_mean(window).sqrt()


def omega_ratio(
    returns: pl.Expr,
    required_return: float | pl.Expr = 0.0,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Omega ratio — gain/loss probability-weighted ratio.

    ``required_return`` may be a scalar per-period threshold or a
    :class:`pl.Expr` per-period column for a time-varying threshold.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    excess = returns - required_return
    gains = pl.when(excess > 0).then(excess).otherwise(0.0)
    losses = pl.when(excess < 0).then(-excess).otherwise(0.0)
    if bucket is not None:
        return gains.sum().over(bucket) / losses.sum().over(bucket)
    if window is None:
        return gains.sum() / losses.sum()
    return gains.rolling_sum(window) / losses.rolling_sum(window)


def _fit_gpd(excess: np.ndarray) -> tuple[float, float]:
    """Fit GPD shape (xi) and scale (beta) by method of moments.

    For a GPD ``X ~ GPD(xi, beta)``:
        ``mean = beta / (1 - xi)`` (xi < 1)
        ``var  = beta^2 / ((1 - xi)^2 (1 - 2*xi))`` (xi < 0.5)

    Solve for xi, beta from the sample mean/variance of the excesses.
    Falls back to the exponential case (xi=0) if moments are degenerate.
    """
    m = float(excess.mean())
    v = float(excess.var(ddof=1)) if excess.size > 1 else 0.0
    if m <= 0 or v <= 0:
        # Exponential fallback.
        return (0.0, max(m, 1e-12))
    ratio = (m * m) / v
    xi = 0.5 * (1.0 - ratio)
    # clamp to the GPD-valid region.
    xi = max(min(xi, 0.45), -0.5)
    beta = m * (1.0 - xi)
    if beta <= 0:
        return (0.0, m)
    return (xi, beta)


def gpd_var(returns: pl.Series, var_p: float = 0.01, threshold_p: float = 0.10) -> float:
    r"""GPD-fitted extreme VaR (positive number, magnitude of loss).

    Fits a Generalized Pareto Distribution to the excess of losses
    over a threshold (peak-over-threshold) and inverts to obtain the
    ``var_p`` quantile.

    Closed form:
        :math:`VaR_p = u + \frac{\beta}{\xi}\left(\left(\frac{n}{n_u} p\right)^{-\xi} - 1\right)`

    Args:
        returns: Periodic returns (``pl.Series``).
        var_p: Tail probability (``0.01`` → 1% VaR).
        threshold_p: Probability mass beyond the threshold ``u``
            used for the GPD fit (``0.10`` → top-10% of losses).

    Returns:
        VaR magnitude as a positive float.
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    if arr.size < 20:
        return float("nan")
    losses = -arr
    u = float(np.quantile(losses, 1.0 - threshold_p))
    excess = losses[losses > u] - u
    if excess.size < 5:
        return float(np.quantile(losses, 1.0 - var_p))
    xi, beta = _fit_gpd(excess)
    n = arr.size
    nu = excess.size
    ratio = (n / nu) * var_p
    if abs(xi) < 1e-8:
        var = u + beta * (-math.log(ratio))
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
    return float(var)


def gpd_cvar(returns: pl.Series, var_p: float = 0.01, threshold_p: float = 0.10) -> float:
    r"""GPD-fitted extreme CVaR (expected shortfall beyond ``var_p``).

    Closed form for the GPD tail (xi < 1):
        :math:`CVaR_p = \frac{VaR_p}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}`

    Args:
        returns: Periodic returns.
        var_p: Tail probability.
        threshold_p: Mass beyond the threshold used for the fit.

    Returns:
        CVaR magnitude as a positive float.
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    if arr.size < 20:
        return float("nan")
    losses = -arr
    u = float(np.quantile(losses, 1.0 - threshold_p))
    excess = losses[losses > u] - u
    if excess.size < 5:
        var_fallback = float(np.quantile(losses, 1.0 - var_p))
        tail = losses[losses >= var_fallback]
        return float(tail.mean()) if tail.size else var_fallback
    xi, beta = _fit_gpd(excess)
    n = arr.size
    nu = excess.size
    ratio = (n / nu) * var_p
    if abs(xi) < 1e-8:
        var = u + beta * (-math.log(ratio))
        cvar = var + beta
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
        if xi >= 1.0:
            return float("inf")
        cvar = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
    return float(cvar)
