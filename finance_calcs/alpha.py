"""Alpha / signal evaluation as polars expressions.

These functions are designed to be composed inside ``group_by("date").agg(...)``
to produce a cross-sectional information-coefficient time series, then
aggregated across time with :func:`ic_ir` and friends.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from ._periods import PeriodLike, _bucket_or_none, _check_window_period

__all__ = [
    "forward_returns",
    "pearson_ic",
    "spearman_ic",
    "information_coefficient",
    "conditional_ic",
    "horizon_ic",
    "ic_decay",
    "ic_ir",
    "hit_rate",
    "ic_summary_stats",
]


def forward_returns(price: pl.Expr, periods: int = 1) -> pl.Expr:
    """Forward simple return over ``periods`` bars.

    Args:
        price: Price series.
        periods: Look-ahead horizon in bars.

    Returns:
        Expression yielding ``price.shift(-periods) / price - 1``.
    """
    return price.shift(-periods) / price - 1.0


def pearson_ic(signal: pl.Expr, fwd: pl.Expr) -> pl.Expr:
    """Pearson information coefficient.

    Args:
        signal: Signal / alpha series.
        fwd: Forward-return series of the same length.

    Returns:
        Scalar correlation expression.
    """
    return pl.corr(signal, fwd, method="pearson")


def spearman_ic(signal: pl.Expr, fwd: pl.Expr) -> pl.Expr:
    """Spearman rank information coefficient.

    Args:
        signal: Signal / alpha series.
        fwd: Forward-return series of the same length.

    Returns:
        Scalar rank-correlation expression.
    """
    return pl.corr(signal, fwd, method="spearman")


information_coefficient = spearman_ic


def conditional_ic(
    signal: pl.Expr,
    fwd: pl.Expr,
    condition: pl.Expr,
    *,
    method: str = "spearman",
) -> pl.Expr:
    """Information coefficient on observations matching ``condition``."""
    return pl.corr(signal.filter(condition), fwd.filter(condition), method=method)


def horizon_ic(
    signal: pl.Expr,
    fwd: pl.Expr,
    *,
    method: str = "spearman",
) -> pl.Expr:
    """Information coefficient for one forward-return horizon."""
    return pl.corr(signal, fwd, method=method)


def ic_decay(
    signal: pl.Expr,
    forward_returns_by_horizon: Mapping[int, pl.Expr],
    *,
    method: str = "spearman",
    prefix: str = "ic_",
) -> list[pl.Expr]:
    """Build one horizon IC expression per forward-return horizon."""
    return [horizon_ic(signal, fwd, method=method).alias(f"{prefix}{horizon}") for horizon, fwd in sorted(forward_returns_by_horizon.items())]


def ic_ir(
    ic: pl.Expr,
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
        return ic.mean().over(bucket) / ic.std().over(bucket)
    if window is None:
        return ic.mean() / ic.std()
    return ic.rolling_mean(window) / ic.rolling_std(window)


def hit_rate(signal: pl.Expr, fwd: pl.Expr) -> pl.Expr:
    """Fraction of observations where ``sign(signal) == sign(fwd)``.

    Args:
        signal: Signal series.
        fwd: Forward return series.

    Returns:
        Scalar mean expression in ``[0, 1]``.
    """
    same = (signal.sign() == fwd.sign()).cast(pl.Float64)
    return same.mean()


def ic_summary_stats(ic: pl.Series) -> dict[str, float]:
    """Summary statistics of an IC time series.

    Args:
        ic: IC time series as a polars Series.

    Returns:
        Dict with ``mean``, ``std``, ``ir``, ``t_stat``, ``pct_positive``,
        ``n``. ``t_stat`` is ``ir * sqrt(n)``.
    """
    arr = ic.drop_nulls().drop_nans() if hasattr(ic, "drop_nans") else ic.drop_nulls()
    n = arr.len()
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "t_stat": float("nan"),
            "pct_positive": float("nan"),
            "n": 0,
        }
    mean = float(arr.mean())
    std = float(arr.std()) if n > 1 else 0.0
    ir = mean / std if std > 0 else float("nan")
    t_stat = ir * (n**0.5) if std > 0 else float("nan")
    pct_pos = float((arr > 0).cast(pl.Float64).mean())
    return {
        "mean": mean,
        "std": std,
        "ir": ir,
        "t_stat": t_stat,
        "pct_positive": pct_pos,
        "n": int(n),
    }
