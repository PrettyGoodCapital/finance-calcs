"""Tail-risk metrics.

Lightweight expression-level helpers for tail-shape statistics
(:func:`tail_ratio`, :func:`ulcer_index`, :func:`omega_ratio`), plus a
``pl.Series``-level GPD peak-over-threshold fit for extreme VaR/CVaR
estimates that need an eager distribution fit.

Rolling forms of historical VaR / CVaR live on
:func:`finance_calcs.value_at_risk` / :func:`finance_calcs.conditional_value_at_risk`
via their ``window=`` keyword — there are no separate ``rolling_*``
siblings.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from finance_enums import Frequency

from ._periods import FrequencyLike, PeriodLike, _annual_rate_to_observation_rate, _bucket_or_none, _check_window_period, _observations_per_year
from ._validation import _validate_probability
from .returns import _clean_returns

__all__ = [
    "conditional_value_at_risk_generalized_pareto",
    "omega_ratio",
    "tail_ratio",
    "ulcer_index",
    "value_at_risk_generalized_pareto",
]

__finance_namespace__ = [
    "omega_ratio",
    "tail_ratio",
    "ulcer_index",
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
    clean_returns = _clean_returns(returns)
    if bucket is not None:
        return clean_returns.quantile(0.95).abs().over(bucket) / clean_returns.quantile(0.05).abs().over(bucket)
    if window is None:
        return clean_returns.quantile(0.95).abs() / clean_returns.quantile(0.05).abs()
    p95 = clean_returns.rolling_quantile(quantile=0.95, window_size=window).abs()
    p05 = clean_returns.rolling_quantile(quantile=0.05, window_size=window).abs()
    return p95 / p05


def ulcer_index(
    returns: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """RMS of the drawdown sequence, expressed as a decimal.

    ``UI = sqrt(mean(dd_t^2))`` where ``dd_t`` is the percentage
    drawdown at time ``t``. ``window=None`` → scalar; ``window=N`` →
    rolling RMS over each trailing ``N``-bar window. ``period=...`` →
    per-bucket RMS drawdown. The equity path starts from a ``1.0`` baseline;
    multiply the result by 100 for percentage-point units.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    equity = (1.0 + _clean_returns(returns).fill_null(0.0)).cum_prod()
    if bucket is not None:
        equity = equity.over(bucket)
        peak = equity.cum_max().over(bucket).clip(lower_bound=1.0)
        dd = (equity / peak) - 1.0
        return dd.pow(2).mean().over(bucket).sqrt()
    peak = equity.cum_max().clip(lower_bound=1.0)
    dd = (equity / peak) - 1.0
    if window is None:
        return dd.pow(2).mean().sqrt()
    return dd.pow(2).rolling_mean(window).sqrt()


def omega_ratio(
    returns: pl.Expr,
    *,
    required_return: float | pl.Expr = 0.0,
    frequency: FrequencyLike = Frequency.Day,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Omega ratio — gain/loss probability-weighted ratio.

    ``required_return`` may be a scalar annual threshold or a
    :class:`pl.Expr` per-observation column for a time-varying threshold.
    """
    observations_per_year = _observations_per_year(frequency)
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    threshold = _annual_rate_to_observation_rate(required_return, observations_per_year)
    excess = (_clean_returns(returns) - threshold).fill_nan(None)
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


def value_at_risk_generalized_pareto(
    returns: pl.Series,
    *,
    tail_probability: float = 0.01,
    threshold_probability: float = 0.10,
) -> float:
    r"""GPD-fitted extreme VaR as a lower-tail return.

    Fits a Generalized Pareto Distribution to the excess of losses
    over a threshold (peak-over-threshold) and inverts to obtain the
    ``tail_probability`` quantile.

    Closed form:
        :math:`VaR_p = u + \frac{\beta}{\xi}\left(\left(\frac{n}{n_u} p\right)^{-\xi} - 1\right)`

    Args:
        returns: Periodic returns (``pl.Series``).
        tail_probability: Tail probability (``0.01`` → 1% VaR).
        threshold_probability: Probability mass beyond the threshold ``u``
            used for the GPD fit (``0.10`` → top-10% of losses).

    Returns:
        VaR as a negative return.
    """
    _validate_probability(tail_probability, name="tail_probability")
    _validate_probability(threshold_probability, name="threshold_probability")
    if tail_probability >= threshold_probability:
        raise ValueError("tail_probability must be less than threshold_probability")
    arr = returns.drop_nulls().to_numpy().astype(float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        return float("nan")
    losses = -arr
    u = float(np.quantile(losses, 1.0 - threshold_probability))
    excess = losses[losses > u] - u
    if excess.size < 5:
        return -float(np.quantile(losses, 1.0 - tail_probability))
    xi, beta = _fit_gpd(excess)
    n = arr.size
    nu = excess.size
    ratio = (n / nu) * tail_probability
    if abs(xi) < 1e-8:
        var = u + beta * (-math.log(ratio))
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
    return -float(var)


def conditional_value_at_risk_generalized_pareto(
    returns: pl.Series,
    *,
    tail_probability: float = 0.01,
    threshold_probability: float = 0.10,
) -> float:
    r"""GPD-fitted extreme CVaR as a lower-tail return.

    Closed form for the GPD tail (xi < 1):
        :math:`CVaR_p = \frac{VaR_p}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}`

    Args:
        returns: Periodic returns.
        tail_probability: Tail probability.
        threshold_probability: Mass beyond the threshold used for the fit.

    Returns:
        CVaR as a negative return.
    """
    _validate_probability(tail_probability, name="tail_probability")
    _validate_probability(threshold_probability, name="threshold_probability")
    if tail_probability >= threshold_probability:
        raise ValueError("tail_probability must be less than threshold_probability")
    arr = returns.drop_nulls().to_numpy().astype(float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        return float("nan")
    losses = -arr
    u = float(np.quantile(losses, 1.0 - threshold_probability))
    excess = losses[losses > u] - u
    if excess.size < 5:
        var_fallback = float(np.quantile(losses, 1.0 - tail_probability))
        tail = losses[losses >= var_fallback]
        return -float(tail.mean()) if tail.size else -var_fallback
    xi, beta = _fit_gpd(excess)
    n = arr.size
    nu = excess.size
    ratio = (n / nu) * tail_probability
    if abs(xi) < 1e-8:
        var = u + beta * (-math.log(ratio))
        cvar = var + beta
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
        if xi >= 1.0:
            return float("-inf")
        cvar = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
    return -float(cvar)
