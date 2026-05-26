"""Quantile and signal-quantile analysis as polars expressions.

Most functions are intended to be evaluated cross-sectionally inside
``group_by("date").agg(...)`` or ``with_columns(... .over("date"))``.
"""

from __future__ import annotations

import polars as pl

__all__ = [
    "assign_quantile",
    "rank_normalize",
    "zscore",
    "winsorize",
    "long_short_spread",
    "quantile_changed",
]


def assign_quantile(signal: pl.Expr, n_quantiles: int = 5) -> pl.Expr:
    """Assign integer quantile labels ``0..n_quantiles-1`` to ``signal``.

    Args:
        signal: Signal series. Nulls produce null labels.
        n_quantiles: Number of quantile buckets.

    Returns:
        Integer expression in ``[0, n_quantiles - 1]``. Higher signal
        values map to higher labels.
    """
    rank = signal.rank(method="ordinal")
    n = signal.count()
    q = ((rank - 1) * n_quantiles / n).floor().cast(pl.Int32)
    return q.clip(0, n_quantiles - 1)


def rank_normalize(signal: pl.Expr) -> pl.Expr:
    """Cross-sectional rank scaled to ``[-0.5, 0.5]``.

    Args:
        signal: Signal series.

    Returns:
        Expression with mean zero and bounded support.
    """
    rank = signal.rank(method="average")
    return (rank - 0.5) / signal.count() - 0.5


def zscore(signal: pl.Expr) -> pl.Expr:
    """Cross-sectional z-score: ``(x - mean) / std``.

    Args:
        signal: Signal series.

    Returns:
        Z-score expression.
    """
    return (signal - signal.mean()) / signal.std()


def winsorize(signal: pl.Expr, cutoff: float = 3.0) -> pl.Expr:
    """Clip values to ``mean ± cutoff * std``.

    Args:
        signal: Signal series.
        cutoff: Number of standard deviations. Must be positive.

    Returns:
        Clipped expression.
    """
    mu = signal.mean()
    sd = signal.std()
    return signal.clip(mu - cutoff * sd, mu + cutoff * sd)


def long_short_spread(
    returns: pl.Expr,
    quantile: pl.Expr,
    upper: int,
    lower: int,
) -> pl.Expr:
    """Top-quantile mean return minus bottom-quantile mean return.

    Use inside ``group_by("date").agg(...)``::

        df.group_by("date").agg(
            long_short_spread(pl.col("ret"), pl.col("q"), upper=4, lower=0)
            .alias("ls"),
        )

    Args:
        returns: Forward return series.
        quantile: Integer quantile label series.
        upper: Long quantile label.
        lower: Short quantile label.

    Returns:
        Scalar expression.
    """
    long_leg = returns.filter(quantile == upper).mean()
    short_leg = returns.filter(quantile == lower).mean()
    return long_leg - short_leg


def quantile_changed(quantile: pl.Expr) -> pl.Expr:
    """Boolean expression: ``quantile != quantile.shift(1)``.

    Use inside ``... .over("asset")`` to compute per-asset turnover
    flags. Aggregate by date to get the fraction of names that changed
    quantile.

    Args:
        quantile: Integer quantile label series.

    Returns:
        Boolean expression. The first observation is null/false.
    """
    return quantile != quantile.shift(1)
