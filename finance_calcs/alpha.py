"""Alpha / signal evaluation as polars expressions.

These functions are designed to be composed inside ``group_by("date").agg(...)``
to produce a cross-sectional information-coefficient time series, then
aggregated across time with :func:`information_coefficient_ratio` and friends.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from ._periods import PeriodLike, _bucket_or_none, _check_window_period

__all__ = [
    "forward_returns",
    "hit_rate",
    "information_coefficient",
    "information_coefficient_by_horizon",
    "information_coefficient_conditional",
    "information_coefficient_decay",
    "information_coefficient_pearson",
    "information_coefficient_ratio",
    "information_coefficient_spearman",
    "information_coefficient_statistics",
]

__finance_namespace__ = [
    "forward_returns",
    "hit_rate",
    "information_coefficient",
    "information_coefficient_by_horizon",
    "information_coefficient_conditional",
    "information_coefficient_decay",
    "information_coefficient_pearson",
    "information_coefficient_ratio",
    "information_coefficient_spearman",
]


def forward_returns(price: pl.Expr, horizon: int = 1) -> pl.Expr:
    """Forward simple return over ``horizon`` observations.

    Args:
        price: Price series.
        horizon: Look-ahead horizon in observations.

    Returns:
        Expression yielding ``price.shift(-horizon) / price - 1``.
    """
    return price.shift(-horizon) / price - 1.0


def information_coefficient_pearson(signal: pl.Expr, forward_returns: pl.Expr) -> pl.Expr:
    """Pearson information coefficient.

    Args:
        signal: Signal / alpha series.
        forward_returns: Forward-return series of the same length.

    Returns:
        Scalar correlation expression.
    """
    return pl.corr(signal, forward_returns, method="pearson")


def information_coefficient_spearman(signal: pl.Expr, forward_returns: pl.Expr) -> pl.Expr:
    """Spearman rank information coefficient.

    Args:
        signal: Signal / alpha series.
        forward_returns: Forward-return series of the same length.

    Returns:
        Scalar rank-correlation expression.
    """
    return pl.corr(signal, forward_returns, method="spearman")


def information_coefficient(signal: pl.Expr, forward_returns: pl.Expr, *, method: str = "spearman") -> pl.Expr:
    """Information coefficient using the requested correlation method."""
    return pl.corr(signal, forward_returns, method=method)


def information_coefficient_conditional(
    signal: pl.Expr,
    forward_returns: pl.Expr,
    condition: pl.Expr,
    *,
    method: str = "spearman",
) -> pl.Expr:
    """Information coefficient on observations matching ``condition``."""
    return pl.corr(signal.filter(condition), forward_returns.filter(condition), method=method)


def information_coefficient_by_horizon(
    signal: pl.Expr,
    forward_returns: pl.Expr,
    *,
    method: str = "spearman",
) -> pl.Expr:
    """Information coefficient for one forward-return horizon."""
    return pl.corr(signal, forward_returns, method=method)


def information_coefficient_decay(
    signal: pl.Expr,
    forward_returns_by_horizon: Mapping[int, pl.Expr],
    *,
    method: str = "spearman",
    prefix: str = "information_coefficient_",
) -> list[pl.Expr]:
    """Build one horizon IC expression per forward-return horizon."""
    return [
        information_coefficient_by_horizon(signal, forward_return, method=method).alias(f"{prefix}{horizon}")
        for horizon, forward_return in sorted(forward_returns_by_horizon.items())
    ]


def information_coefficient_ratio(
    information_coefficient: pl.Expr,
    *,
    window: int | None = None,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """IC information ratio — ``mean(ic) / std(ic)``.

    ``window=None`` → scalar; ``window=N`` → rolling IR over each
    trailing ``N``-observation window; ``period=...`` → per-bucket IR.
    """
    _check_window_period(window, period)
    bucket = _bucket_or_none(date, period)
    if bucket is not None:
        return information_coefficient.mean().over(bucket) / information_coefficient.std().over(bucket)
    if window is None:
        return information_coefficient.mean() / information_coefficient.std()
    return information_coefficient.rolling_mean(window) / information_coefficient.rolling_std(window)


def hit_rate(signal: pl.Expr, forward_returns: pl.Expr) -> pl.Expr:
    """Fraction of observations where signal and forward-return signs agree.

    Args:
        signal: Signal series.
        forward_returns: Forward return series.

    Returns:
        Scalar mean expression in ``[0, 1]``.
    """
    same = (signal.sign() == forward_returns.sign()).cast(pl.Float64)
    return same.mean()


def information_coefficient_statistics(information_coefficient: pl.Series) -> dict[str, float | int]:
    """Summary statistics of an IC time series.

    Args:
        information_coefficient: Information-coefficient time series.

    Returns:
        Dict with mean, standard deviation, information ratio, t-statistic,
        positive fraction, and observation count.
    """
    arr = information_coefficient.drop_nulls().drop_nans() if hasattr(information_coefficient, "drop_nans") else information_coefficient.drop_nulls()
    n = arr.len()
    if n == 0:
        return {
            "mean": float("nan"),
            "standard_deviation": float("nan"),
            "information_ratio": float("nan"),
            "t_statistic": float("nan"),
            "positive_fraction": float("nan"),
            "observation_count": 0,
        }
    mean = float(arr.mean())
    standard_deviation = float(arr.std()) if n > 1 else 0.0
    information_ratio = mean / standard_deviation if standard_deviation > 0 else float("nan")
    t_statistic = information_ratio * (n**0.5) if standard_deviation > 0 else float("nan")
    positive_fraction = float((arr > 0).cast(pl.Float64).mean())
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "information_ratio": information_ratio,
        "t_statistic": t_statistic,
        "positive_fraction": positive_fraction,
        "observation_count": int(n),
    }
