"""Basic risk metrics as Polars expressions.

Every public function returns a ``pl.Expr``. Floating-point NaN and Polars null
values are both treated as missing observations. Risk-adjusted-return metrics
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
from finance_enums import Frequency

from ._periods import FrequencyLike, PeriodLike, _annual_rate_to_observation_rate, _bucket_or_none, _check_window_period, _observations_per_year
from ._validation import _validate_probability
from .returns import _clean_returns, annualized_return

__all__ = [
    "calmar",
    "conditional_value_at_risk",
    "downside_deviation",
    "drawdown_series",
    "expected_shortfall",
    "max_drawdown",
    "sharpe",
    "sortino",
    "value_at_risk",
    "value_at_risk_parametric",
]


_Z_TABLE = {
    0.01: -2.3263478740408408,
    0.025: -1.9599639845400545,
    0.05: -1.6448536269514722,
    0.1: -1.2815515655446004,
}


def _safe_scalar_ratio(numerator: float, denominator: float) -> float:
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return float("nan")
    return float("inf") if numerator > 0.0 else float("-inf")


def sharpe(
    returns: pl.Expr,
    *,
    risk_free: float | pl.Expr = 0.0,
    frequency: FrequencyLike = Frequency.Day,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    r"""Annualised Sharpe ratio.

    Mean excess return divided by its standard deviation, scaled by the
    square root of observations per year implied by ``frequency``.

    ``risk_free`` may be a scalar annual rate (converted to per-period
    geometrically) or a :class:`pl.Expr` per-period rate column for a
    time-varying risk-free rate. ``window=None`` → scalar lifetime
    Sharpe; ``window=N`` → rolling; ``period=...`` → per-bucket.
    """
    observations_per_year = _observations_per_year(frequency)
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    excess = _clean_returns(returns) - _annual_rate_to_observation_rate(risk_free, observations_per_year)
    excess = excess.fill_nan(None)
    scale = observations_per_year**0.5
    if bucket is not None:
        return excess.mean().over(bucket) / excess.std().over(bucket) * scale
    if window is None:
        mean = excess.mean()
        std = excess.std()
        if isinstance(mean, pl.Expr) or isinstance(std, pl.Expr):
            return mean / std * scale
        return _safe_scalar_ratio(mean, std) * scale
    return excess.rolling_mean(window) / excess.rolling_std(window) * scale


def downside_deviation(
    returns: pl.Expr,
    *,
    required_return: float | pl.Expr = 0.0,
    frequency: FrequencyLike = Frequency.Day,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised semi-deviation below ``required_return``.

    ``required_return`` may be a scalar annual threshold (converted to
    per-observation geometrically) or a :class:`pl.Expr` per-observation
    column for a time-varying threshold.
    ``window=None`` → scalar; ``window=N`` → rolling;
    ``period=...`` → per-bucket.
    """
    observations_per_year = _observations_per_year(frequency)
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    clean_returns = _clean_returns(returns)
    threshold = _annual_rate_to_observation_rate(required_return, observations_per_year)
    diff = (clean_returns - threshold).fill_nan(None)
    neg_sq = pl.when(diff.is_null()).then(None).when(diff < 0).then(diff.pow(2)).otherwise(0.0)
    scale = observations_per_year**0.5
    if bucket is not None:
        observation_count = clean_returns.is_not_null().sum().over(bucket)
        return (neg_sq.sum().over(bucket) / observation_count).sqrt() * scale
    if window is None:
        n = clean_returns.is_not_null().sum()
        return (neg_sq.sum() / n).sqrt() * scale
    return neg_sq.rolling_mean(window).sqrt() * scale


def sortino(
    returns: pl.Expr,
    *,
    required_return: float | pl.Expr = 0.0,
    frequency: FrequencyLike = Frequency.Day,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised Sortino ratio.

    ``required_return`` may be a scalar annual threshold or a
    :class:`pl.Expr` per-observation column. ``window=None`` → scalar;
    ``window=N`` → rolling; ``period=...`` → per-bucket.
    """
    observations_per_year = _observations_per_year(frequency)
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    clean_returns = _clean_returns(returns)
    threshold = _annual_rate_to_observation_rate(required_return, observations_per_year)
    excess = (clean_returns - threshold).fill_nan(None)
    dd = downside_deviation(clean_returns, required_return=required_return, frequency=frequency, window=window, period=period, date=date)
    if bucket is not None:
        return excess.mean().over(bucket) * observations_per_year / dd
    if window is None:
        return excess.mean() * observations_per_year / dd
    return excess.rolling_mean(window) * observations_per_year / dd


def drawdown_series(
    returns: pl.Expr,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Per-period drawdown series ``equity / running_peak - 1``.

    The running peak includes an initial equity baseline of ``1.0``.
    """
    bucket = _bucket_or_none(date, period)
    equity = (1.0 + _clean_returns(returns).fill_null(0.0)).cum_prod()
    if bucket is not None:
        equity = equity.over(bucket)
        peak = equity.cum_max().over(bucket).clip(lower_bound=1.0)
        return equity / peak - 1.0
    return equity / equity.cum_max().clip(lower_bound=1.0) - 1.0


def _rolling_max_drawdown(returns: pl.Expr, window: int) -> pl.Expr:
    if window < 1:
        raise ValueError("window must be positive")
    clean_returns = _clean_returns(returns).fill_null(0.0)
    windows = pl.concat_list([clean_returns.shift(offset) for offset in reversed(range(window))])
    equity = (1.0 + pl.element()).cum_prod()
    drawdown = equity / equity.cum_max().clip(lower_bound=1.0) - 1.0
    result = windows.list.eval(drawdown).list.min()
    return pl.when(pl.int_range(0, pl.len()) >= window - 1).then(result)


def max_drawdown(
    returns: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Maximum (most negative) drawdown.

    ``window=None`` → lifetime; ``window=N`` → maximum drawdown rebased
    inside each trailing ``N``-bar window. ``period=...`` → maximum
    drawdown inside each period bucket.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    if bucket is not None:
        return drawdown_series(returns, period=period, date=date).min().over(bucket)
    if window is None:
        return drawdown_series(returns).min()
    return _rolling_max_drawdown(returns, window)


def calmar(
    returns: pl.Expr,
    *,
    frequency: FrequencyLike = Frequency.Day,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Annualised return divided by the absolute max drawdown.

    ``window=None`` → scalar; ``window=N`` → rolling;
    ``period=...`` → per-bucket.
    """
    ar = annualized_return(returns, frequency=frequency, window=window, period=period, date=date)
    mdd = max_drawdown(returns, window=window, period=period, date=date)
    return ar / mdd.abs()


def value_at_risk(
    returns: pl.Expr,
    *,
    tail_probability: float = 0.05,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Historical Value-at-Risk.

    ``window=None`` → scalar lower-tail quantile; ``window=N`` →
    rolling historical VaR. ``period=...`` → per-bucket VaR.
    """
    _validate_probability(tail_probability, name="tail_probability")
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    clean_returns = _clean_returns(returns)
    if bucket is not None:
        return clean_returns.quantile(tail_probability).over(bucket)
    if window is None:
        return clean_returns.quantile(tail_probability)
    return clean_returns.rolling_quantile(quantile=tail_probability, window_size=window)


def conditional_value_at_risk(
    returns: pl.Expr,
    *,
    tail_probability: float = 0.05,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Historical CVaR / Expected Shortfall.

    ``window=None`` → scalar; ``window=N`` → rolling mean of returns at
    or below the rolling VaR. ``period=...`` → per-bucket CVaR.
    """
    _validate_probability(tail_probability, name="tail_probability")
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    clean_returns = _clean_returns(returns)
    if bucket is not None:
        threshold = clean_returns.quantile(tail_probability).over(bucket)
        tail = pl.when(clean_returns <= threshold).then(clean_returns).otherwise(None)
        return tail.mean().over(bucket)
    if window is None:
        threshold = clean_returns.quantile(tail_probability)
        tail = pl.when(clean_returns <= threshold).then(clean_returns).otherwise(None)
        return tail.mean()
    var = clean_returns.rolling_quantile(quantile=tail_probability, window_size=window)
    masked = pl.when(clean_returns <= var).then(clean_returns).otherwise(None)
    return masked.rolling_mean(window_size=window, min_samples=1)


def expected_shortfall(
    returns: pl.Expr,
    *,
    tail_probability: float = 0.05,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Historical expected shortfall, also known as conditional VaR."""
    return conditional_value_at_risk(returns, tail_probability=tail_probability, window=window, period=period, date=date)


def value_at_risk_parametric(
    returns: pl.Expr,
    *,
    tail_probability: float = 0.05,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    r"""Gaussian (parametric) VaR :math:`\mu + \sigma \Phi^{-1}(p)`.

    ``tail_probability`` must be one of ``{0.01, 0.025, 0.05, 0.1}``.
    """
    _validate_probability(tail_probability, name="tail_probability")
    if tail_probability not in _Z_TABLE:
        raise ValueError(f"value_at_risk_parametric: tail_probability={tail_probability} not in {sorted(_Z_TABLE)}")
    z = _Z_TABLE[tail_probability]
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    clean_returns = _clean_returns(returns)
    if bucket is not None:
        return clean_returns.mean().over(bucket) + clean_returns.std().over(bucket) * z
    if window is None:
        return clean_returns.mean() + clean_returns.std() * z
    return clean_returns.rolling_mean(window) + clean_returns.rolling_std(window) * z
