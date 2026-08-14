"""Statistical-validity calculations.

Combines polars-expression metrics (skew/kurtosis/stability) with
``pl.Series``-level helpers for bootstrap and probabilistic-Sharpe
analyses that need numerical work outside the expression engine.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from finance_enums import Frequency

from ._periods import FrequencyLike, _observations_per_year

__all__ = [
    "common_sense_ratio",
    "higher_moments",
    "kurtosis",
    "skewness",
    "stability_of_timeseries",
]


def skewness(returns: pl.Expr) -> pl.Expr:
    """Sample skewness of ``returns``.

    Args:
        returns: Returns expression.

    Returns:
        Scalar skewness.
    """
    return returns.skew()


def kurtosis(returns: pl.Expr) -> pl.Expr:
    """Excess kurtosis of ``returns`` (Fisher definition).

    Args:
        returns: Returns expression.

    Returns:
        Scalar excess kurtosis.
    """
    return returns.kurtosis()


def higher_moments(returns: pl.Expr) -> pl.Expr:
    """Bundled struct of skewness and kurtosis for ``returns``.

    Args:
        returns: Returns expression.

    Returns:
        Struct expression with fields ``skewness`` and ``kurtosis``.
    """
    return pl.struct(
        skewness=returns.skew(),
        kurtosis=returns.kurtosis(),
    )


def stability_of_timeseries(returns: pl.Expr) -> pl.Expr:
    r"""Coefficient of determination of cumulative log returns vs time.

    Implements pyfolio's ``stability_of_timeseries`` — fit
    :math:`y_t = a + b \cdot t` to the log-equity curve and return
    ``R^2``. Closer to 1 means more linear (steady) growth.

    Args:
        returns: Periodic returns (not log).

    Returns:
        Scalar ``R^2`` expression.
    """
    log_eq = (1.0 + returns).log().cum_sum()
    n = log_eq.count().cast(pl.Float64)
    t = pl.int_range(0, log_eq.len()).cast(pl.Float64)
    # Pearson correlation squared between t and log_eq.
    mean_t = t.mean()
    mean_y = log_eq.mean()
    num = ((t - mean_t) * (log_eq - mean_y)).sum()
    den = ((t - mean_t).pow(2).sum() * (log_eq - mean_y).pow(2).sum()).sqrt()
    r = num / den
    _ = n  # not needed for R^2
    return r.pow(2)


def common_sense_ratio(returns: pl.Expr) -> pl.Expr:
    """``tail_ratio * (1 + cumulative_return)`` — sanity sniff test.

    Args:
        returns: Periodic returns expression.

    Returns:
        Scalar expression.
    """
    p95 = returns.quantile(0.95).abs()
    p05 = returns.quantile(0.05).abs()
    tail = p95 / p05
    cum = (1.0 + returns).product() - 1.0
    return tail * (1.0 + cum)


def _sharpe(arr: np.ndarray, observations_per_year: float) -> float:
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return 0.0
    return float(arr.mean() / arr.std(ddof=1) * math.sqrt(observations_per_year))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Beasley-Springer-Moro inverse normal CDF, sufficient for our needs.
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def sharpe_probability(
    returns: pl.Series,
    *,
    benchmark_sharpe: float = 0.0,
    frequency: FrequencyLike = Frequency.Day,
) -> float:
    """Lopez de Prado probabilistic Sharpe ratio.

    Probability that the observed Sharpe is greater than
    ``benchmark_sharpe``, accounting for sample skew and kurtosis.

    Args:
        returns: Periodic returns.
        benchmark_sharpe: Annualised threshold Sharpe.
        frequency: Observation frequency alias, enum, or observations per year.

    Returns:
        ``Pr(SR_true > benchmark_sharpe)`` in ``[0, 1]``.
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    n = arr.size
    if n < 3:
        return float("nan")
    observations_per_year = _observations_per_year(frequency)
    sr_hat = _sharpe(arr, observations_per_year)
    sr_per = sr_hat / math.sqrt(observations_per_year)
    bench_per = benchmark_sharpe / math.sqrt(observations_per_year)
    skew = float(((arr - arr.mean()) ** 3).mean() / (arr.std(ddof=0) ** 3 + 1e-30))
    kurt = float(((arr - arr.mean()) ** 4).mean() / (arr.std(ddof=0) ** 4 + 1e-30)) - 3.0
    num = (sr_per - bench_per) * math.sqrt(n - 1)
    den = math.sqrt(max(1.0 - skew * sr_per + (kurt / 4.0) * sr_per**2, 1e-12))
    return _norm_cdf(num / den)


def sharpe_deflated_probability(
    returns: pl.Series,
    *,
    trial_count: int,
    sharpe_variance: float | None = None,
    frequency: FrequencyLike = Frequency.Day,
) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado).

    Adjusts the probabilistic Sharpe for multiple-testing across
    ``trial_count`` candidate strategies.

    Args:
        returns: Periodic returns.
        trial_count: Number of independent strategies tried.
        sharpe_variance: Variance of the trial Sharpes. If ``None`` a
            conservative default of ``1.0`` is used (worst case).
        frequency: Observation frequency alias, enum, or observations per year.

    Returns:
        ``Pr(SR_true > expected_max_SR_under_null)`` in ``[0, 1]``.
    """
    if trial_count < 2:
        raise ValueError("trial_count must be >= 2")
    if sharpe_variance is None:
        sharpe_variance = 1.0
    observations_per_year = _observations_per_year(frequency)
    euler_mascheroni = 0.5772156649015329
    expected_max_z = (1.0 - euler_mascheroni) * _norm_ppf(1.0 - 1.0 / trial_count) + euler_mascheroni * _norm_ppf(1.0 - 1.0 / (trial_count * math.e))
    threshold_sharpe = expected_max_z * math.sqrt(sharpe_variance)
    return sharpe_probability(returns, benchmark_sharpe=threshold_sharpe, frequency=observations_per_year)


def sharpe_minimum_track_record_length(
    returns: pl.Series,
    *,
    benchmark_sharpe: float = 0.0,
    significance_level: float = 0.05,
    frequency: FrequencyLike = Frequency.Day,
) -> float:
    """Minimum observations for Sharpe above benchmark at requested confidence.

    Args:
        returns: Periodic returns.
        benchmark_sharpe: Annualised threshold Sharpe.
        significance_level: Significance level (``0.05`` → 95% confidence).
        frequency: Observation frequency alias, enum, or observations per year.

    Returns:
        Minimum number of observations (float; round up in practice).
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    if arr.size < 3:
        return float("nan")
    observations_per_year = _observations_per_year(frequency)
    sr_hat = _sharpe(arr, observations_per_year)
    sr_per = sr_hat / math.sqrt(observations_per_year)
    bench_per = benchmark_sharpe / math.sqrt(observations_per_year)
    if sr_per <= bench_per:
        return float("inf")
    skew = float(((arr - arr.mean()) ** 3).mean() / (arr.std(ddof=0) ** 3 + 1e-30))
    kurt = float(((arr - arr.mean()) ** 4).mean() / (arr.std(ddof=0) ** 4 + 1e-30)) - 3.0
    z = _norm_ppf(1.0 - significance_level)
    num = z**2 * (1.0 - skew * sr_per + (kurt / 4.0) * sr_per**2)
    den = (sr_per - bench_per) ** 2
    return 1.0 + num / den


def sharpe_bootstrap_confidence_interval(
    returns: pl.Series,
    *,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    frequency: FrequencyLike = Frequency.Day,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the Sharpe ratio.

    Args:
        returns: Periodic returns.
        bootstrap_samples: Number of bootstrap resamples.
        confidence_level: Two-sided confidence level.
        frequency: Observation frequency alias, enum, or observations per year.
        seed: RNG seed.

    Returns:
        Tuple ``(sharpe, lower, upper)``.
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    n = arr.size
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    observations_per_year = _observations_per_year(frequency)
    samples = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        idx = rng.integers(0, n, size=n)
        samples[i] = _sharpe(arr[idx], observations_per_year)
    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return (_sharpe(arr, observations_per_year), float(lo), float(hi))


def sharpe_confidence_interval(
    returns: pl.Series,
    *,
    risk_free: float | pl.Series | np.ndarray = 0.0,
    frequency: FrequencyLike = Frequency.Day,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Sharpe with a Mertens-style asymptotic confidence interval.

    Args:
        returns: Periodic returns.
        risk_free: Annual risk-free rate (subtracted period-wise) as a
            scalar, or a per-period rate series (``pl.Series`` /
            ``np.ndarray``) aligned to ``returns`` for a time-varying
            risk-free rate.
        frequency: Observation frequency alias, enum, or observations per year.
        confidence_level: Two-sided confidence level.

    Returns:
        Tuple ``(sharpe, lower, upper)`` where the bounds are derived
        from the Mertens (2002) asymptotic variance of the Sharpe.
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    n = arr.size
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))
    observations_per_year = _observations_per_year(frequency)
    if isinstance(risk_free, pl.Series):
        rf = risk_free.to_numpy().astype(float)
    elif isinstance(risk_free, np.ndarray):
        rf = risk_free.astype(float)
    else:
        rf = (1.0 + risk_free) ** (1.0 / observations_per_year) - 1.0
    excess = arr - rf
    sr_per = excess.mean() / (excess.std(ddof=1) + 1e-30)
    sr_ann = sr_per * math.sqrt(observations_per_year)
    skew = float(((excess - excess.mean()) ** 3).mean() / (excess.std(ddof=0) ** 3 + 1e-30))
    kurt = float(((excess - excess.mean()) ** 4).mean() / (excess.std(ddof=0) ** 4 + 1e-30)) - 3.0
    var_sr = (1.0 + 0.5 * sr_per**2 - skew * sr_per + (kurt / 4.0) * sr_per**2) / n
    se = math.sqrt(max(var_sr, 0.0)) * math.sqrt(observations_per_year)
    z = _norm_ppf(1.0 - (1.0 - confidence_level) / 2.0)
    return (sr_ann, sr_ann - z * se, sr_ann + z * se)
