"""Statistical-validity calculations.

Combines polars-expression metrics (skew/kurtosis/stability) with
``pl.Series``-level helpers for bootstrap and probabilistic-Sharpe
analyses that need numerical work outside the expression engine.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import polars as pl

__all__ = [
    "skewness",
    "kurtosis",
    "higher_moments",
    "stability_of_timeseries",
    "common_sense_ratio",
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
    """Bundled struct of ``{skew, kurt}`` for ``returns``.

    Args:
        returns: Returns expression.

    Returns:
        Struct expression with fields ``skew`` and ``kurt``.
    """
    return pl.struct(
        skew=returns.skew(),
        kurt=returns.kurtosis(),
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


def _sharpe(arr: np.ndarray, periods_per_year: int = 252) -> float:
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return 0.0
    return float(arr.mean() / arr.std(ddof=1) * math.sqrt(periods_per_year))


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


def probabilistic_sharpe(
    returns: pl.Series,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Lopez de Prado probabilistic Sharpe ratio.

    Probability that the observed Sharpe is greater than
    ``benchmark_sr``, accounting for sample skew and kurtosis.

    Args:
        returns: Periodic returns.
        benchmark_sr: Annualised threshold Sharpe.
        periods_per_year: Periods per year.

    Returns:
        ``Pr(SR_true > benchmark_sr)`` in ``[0, 1]``.
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    n = arr.size
    if n < 3:
        return float("nan")
    sr_hat = _sharpe(arr, periods_per_year)
    sr_per = sr_hat / math.sqrt(periods_per_year)
    bench_per = benchmark_sr / math.sqrt(periods_per_year)
    skew = float(((arr - arr.mean()) ** 3).mean() / (arr.std(ddof=0) ** 3 + 1e-30))
    kurt = float(((arr - arr.mean()) ** 4).mean() / (arr.std(ddof=0) ** 4 + 1e-30)) - 3.0
    num = (sr_per - bench_per) * math.sqrt(n - 1)
    den = math.sqrt(max(1.0 - skew * sr_per + (kurt / 4.0) * sr_per**2, 1e-12))
    return _norm_cdf(num / den)


def deflated_sharpe(
    returns: pl.Series,
    n_trials: int,
    sr_variance: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado).

    Adjusts the probabilistic Sharpe for multiple-testing across
    ``n_trials`` candidate strategies.

    Args:
        returns: Periodic returns.
        n_trials: Number of independent strategies tried.
        sr_variance: Variance of the trial Sharpes. If ``None`` a
            conservative default of ``1.0`` is used (worst case).
        periods_per_year: Periods per year.

    Returns:
        ``Pr(SR_true > expected_max_SR_under_null)`` in ``[0, 1]``.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_variance is None:
        sr_variance = 1.0
    euler_mascheroni = 0.5772156649015329
    expected_max_z = (1.0 - euler_mascheroni) * _norm_ppf(1.0 - 1.0 / n_trials) + euler_mascheroni * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    threshold_sr_per = expected_max_z * math.sqrt(sr_variance) / math.sqrt(periods_per_year)
    threshold_sr_ann = threshold_sr_per * math.sqrt(periods_per_year)
    return probabilistic_sharpe(returns, threshold_sr_ann, periods_per_year)


def minimum_track_record_length(
    returns: pl.Series,
    benchmark_sr: float = 0.0,
    alpha: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """Minimum number of observations for ``SR > benchmark_sr`` at confidence ``1-alpha``.

    Args:
        returns: Periodic returns.
        benchmark_sr: Annualised threshold Sharpe.
        alpha: Significance level (``0.05`` → 95% confidence).
        periods_per_year: Periods per year.

    Returns:
        Minimum number of observations (float; round up in practice).
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    if arr.size < 3:
        return float("nan")
    sr_hat = _sharpe(arr, periods_per_year)
    sr_per = sr_hat / math.sqrt(periods_per_year)
    bench_per = benchmark_sr / math.sqrt(periods_per_year)
    if sr_per <= bench_per:
        return float("inf")
    skew = float(((arr - arr.mean()) ** 3).mean() / (arr.std(ddof=0) ** 3 + 1e-30))
    kurt = float(((arr - arr.mean()) ** 4).mean() / (arr.std(ddof=0) ** 4 + 1e-30)) - 3.0
    z = _norm_ppf(1.0 - alpha)
    num = z**2 * (1.0 - skew * sr_per + (kurt / 4.0) * sr_per**2)
    den = (sr_per - bench_per) ** 2
    return 1.0 + num / den


def sharpe_ci_bootstrap(
    returns: pl.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the Sharpe ratio.

    Args:
        returns: Periodic returns.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Two-sided confidence level.
        periods_per_year: Periods per year.
        seed: RNG seed.

    Returns:
        Tuple ``(sharpe, lower, upper)``.
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    n = arr.size
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = _sharpe(arr[idx], periods_per_year)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return (_sharpe(arr, periods_per_year), float(lo), float(hi))


def sharpe_with_ci(
    returns: pl.Series,
    risk_free: float | pl.Series | np.ndarray = 0.0,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Sharpe with HAC-style asymptotic confidence interval.

    Args:
        returns: Periodic returns.
        risk_free: Annual risk-free rate (subtracted period-wise) as a
            scalar, or a per-period rate series (``pl.Series`` /
            ``np.ndarray``) aligned to ``returns`` for a time-varying
            risk-free rate.
        periods_per_year: Periods per year.
        confidence: Two-sided confidence level.

    Returns:
        Tuple ``(sharpe, lower, upper)`` where the bounds are derived
        from the Mertens (2002) asymptotic variance of the Sharpe.
    """
    arr = returns.drop_nulls().to_numpy().astype(float)
    n = arr.size
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))
    if isinstance(risk_free, pl.Series):
        rf = risk_free.to_numpy().astype(float)
    elif isinstance(risk_free, np.ndarray):
        rf = risk_free.astype(float)
    else:
        rf = risk_free / periods_per_year
    excess = arr - rf
    sr_per = excess.mean() / (excess.std(ddof=1) + 1e-30)
    sr_ann = sr_per * math.sqrt(periods_per_year)
    skew = float(((excess - excess.mean()) ** 3).mean() / (excess.std(ddof=0) ** 3 + 1e-30))
    kurt = float(((excess - excess.mean()) ** 4).mean() / (excess.std(ddof=0) ** 4 + 1e-30)) - 3.0
    var_sr = (1.0 + 0.5 * sr_per**2 - skew * sr_per + (kurt / 4.0) * sr_per**2) / n
    se = math.sqrt(max(var_sr, 0.0)) * math.sqrt(periods_per_year)
    z = _norm_ppf(1.0 - (1.0 - confidence) / 2.0)
    return (sr_ann, sr_ann - z * se, sr_ann + z * se)
