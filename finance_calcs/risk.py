"""Basic risk metrics as polars expressions.

Every function returns a ``pl.Expr``. Risk-adjusted-return metrics
(:func:`sharpe`, :func:`sortino`, :func:`calmar`) and tail-statistic
metrics (:func:`value_at_risk`, :func:`conditional_value_at_risk`)
take a ``window=None`` keyword: ``None`` collapses to a scalar lifetime
value, an integer ``N`` produces a rolling expression over each trailing
``N``-bar window. ``period=`` computes the metric inside each period
bucket when paired with ``date=`` or a precomputed bucket expression.

Per the workspace rule, there are no ``rolling_*`` / ``periodic_*``
siblings; one function per metric.
"""

from __future__ import annotations

import polars as pl

from ._periods import PeriodLike, _bucket_or_none, _check_window_period
from .returns import annualized_return, annualized_volatility

__all__ = [
    "volatility",
    "sharpe",
    "sortino",
    "calmar",
    "downside_risk",
    "downside_deviation",
    "drawdown_series",
    "underwater_series",
    "max_drawdown",
    "value_at_risk",
    "conditional_value_at_risk",
    "parametric_var",
]


_Z_TABLE = {
    0.01: -2.3263478740408408,
    0.025: -1.9599639845400545,
    0.05: -1.6448536269514722,
    0.1: -1.2815515655446004,
}


def volatility(
    returns: pl.Expr,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised volatility alias for :func:`annualized_volatility`."""
    return annualized_volatility(returns, periods_per_year, window=window, period=period, date=date)


def _rf_per_period(risk_free: float | pl.Expr, periods_per_year: int) -> float | pl.Expr:
    """Convert an annual scalar ``risk_free`` to a per-period rate.

    If ``risk_free`` is a :class:`pl.Expr` it is assumed to already be a
    per-period rate column (sampled at the same frequency as returns)
    and is returned unchanged.
    """
    if isinstance(risk_free, pl.Expr):
        return risk_free
    if risk_free == 0.0:
        return 0.0
    return (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0


def sharpe(
    returns: pl.Expr,
    risk_free: float | pl.Expr = 0.0,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    r"""Annualised Sharpe ratio.

    :math:`\sqrt{\mathrm{ppy}}\,\mathrm{mean}(r - r_f) / \mathrm{std}(r - r_f)`.

    ``risk_free`` may be a scalar annual rate (converted to per-period
    geometrically) or a :class:`pl.Expr` per-period rate column for a
    time-varying risk-free rate. ``window=None`` → scalar lifetime
    Sharpe; ``window=N`` → rolling; ``period=...`` → per-bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    excess = returns - _rf_per_period(risk_free, periods_per_year)
    scale = periods_per_year**0.5
    if bucket is not None:
        return excess.mean().over(bucket) / excess.std().over(bucket) * scale
    if window is None:
        return excess.mean() / excess.std() * scale
    return excess.rolling_mean(window) / excess.rolling_std(window) * scale


def downside_deviation(
    returns: pl.Expr,
    required_return: float | pl.Expr = 0.0,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised semi-deviation below ``required_return``.

    ``required_return`` may be a scalar per-period threshold or a
    :class:`pl.Expr` per-period column for a time-varying threshold.
    ``window=None`` → scalar; ``window=N`` → rolling;
    ``period=...`` → per-bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    diff = returns - required_return
    neg_sq = pl.when(diff < 0).then(diff.pow(2)).otherwise(0.0)
    scale = periods_per_year**0.5
    if bucket is not None:
        observation_count = returns.is_not_null().sum().over(bucket)
        return (neg_sq.sum().over(bucket) / observation_count).sqrt() * scale
    if window is None:
        n = returns.is_not_null().sum()
        return (neg_sq.sum() / n).sqrt() * scale
    return neg_sq.rolling_mean(window).sqrt() * scale


def downside_risk(
    returns: pl.Expr,
    required_return: float | pl.Expr = 0.0,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised downside-risk alias for :func:`downside_deviation`."""
    return downside_deviation(returns, required_return, periods_per_year, window=window, period=period, date=date)


def sortino(
    returns: pl.Expr,
    required_return: float | pl.Expr = 0.0,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised Sortino ratio.

    ``required_return`` may be a scalar per-period threshold or a
    :class:`pl.Expr` per-period column. ``window=None`` → scalar;
    ``window=N`` → rolling; ``period=...`` → per-bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    excess = returns - required_return
    dd = downside_deviation(returns, required_return, periods_per_year, window=window, period=period, date=date)
    if bucket is not None:
        return excess.mean().over(bucket) * periods_per_year / dd
    if window is None:
        return excess.mean() * periods_per_year / dd
    return excess.rolling_mean(window) * periods_per_year / dd


def drawdown_series(
    returns: pl.Expr,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Per-period drawdown series ``equity / running_peak - 1``."""
    bucket = _bucket_or_none(date, period)
    equity = (1.0 + returns.fill_null(0.0)).cum_prod()
    if bucket is not None:
        equity = equity.over(bucket)
        return equity / equity.cum_max().over(bucket) - 1.0
    return equity / equity.cum_max() - 1.0


def underwater_series(
    returns: pl.Expr,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Alias of :func:`drawdown_series`."""
    return drawdown_series(returns, period=period, date=date)


def max_drawdown(
    returns: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Maximum (most negative) drawdown.

    ``window=None`` → lifetime; ``window=N`` → rolling minimum of the
    drawdown series over each trailing ``N``-bar window. ``period=...``
    → maximum drawdown inside each period bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    dd = drawdown_series(returns, period=period, date=date)
    if bucket is not None:
        return dd.min().over(bucket)
    if window is None:
        return dd.min()
    return dd.rolling_min(window)


def calmar(
    returns: pl.Expr,
    periods_per_year: int = 252,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised return divided by the absolute max drawdown.

    ``window=None`` → scalar; ``window=N`` → rolling;
    ``period=...`` → per-bucket.
    """
    ar = annualized_return(returns, periods_per_year, window=window, period=period, date=date)
    mdd = max_drawdown(returns, window=window, period=period, date=date)
    return ar / mdd.abs()


def value_at_risk(
    returns: pl.Expr,
    cutoff: float = 0.05,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Historical Value-at-Risk.

    ``window=None`` → scalar lower-tail quantile; ``window=N`` →
    rolling historical VaR. ``period=...`` → per-bucket VaR.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    if bucket is not None:
        return returns.quantile(cutoff).over(bucket)
    if window is None:
        return returns.quantile(cutoff)
    return returns.rolling_quantile(quantile=cutoff, window_size=window)


def conditional_value_at_risk(
    returns: pl.Expr,
    cutoff: float = 0.05,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Historical CVaR / Expected Shortfall.

    ``window=None`` → scalar; ``window=N`` → rolling mean of returns at
    or below the rolling VaR. ``period=...`` → per-bucket CVaR.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    if bucket is not None:
        threshold = returns.quantile(cutoff).over(bucket)
        tail = pl.when(returns <= threshold).then(returns).otherwise(None)
        return tail.mean().over(bucket)
    if window is None:
        threshold = returns.quantile(cutoff)
        tail = pl.when(returns <= threshold).then(returns).otherwise(None)
        return tail.mean()
    var = returns.rolling_quantile(quantile=cutoff, window_size=window)
    masked = pl.when(returns <= var).then(returns).otherwise(None)
    return masked.rolling_mean(window_size=window, min_samples=1)


def parametric_var(
    returns: pl.Expr,
    cutoff: float = 0.05,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    r"""Gaussian (parametric) VaR :math:`\mu + \sigma \Phi^{-1}(p)`.

    ``cutoff`` must be one of ``{0.01, 0.025, 0.05, 0.1}``.
    """
    if cutoff not in _Z_TABLE:
        raise ValueError(f"parametric_var: cutoff={cutoff} not in {sorted(_Z_TABLE)}")
    z = _Z_TABLE[cutoff]
    bucket = _bucket_or_none(date, period)
    if bucket is not None:
        return returns.mean().over(bucket) + returns.std().over(bucket) * z
    return returns.mean() + returns.std() * z
