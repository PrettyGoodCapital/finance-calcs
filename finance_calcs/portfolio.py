"""Portfolio-level exposure and concentration metrics.

All inputs are expressions over a column of position weights or
notional dollar exposures, evaluated within the appropriate group
(typically a ``group_by("date")``). Weights need not sum to one — the
expressions operate on whatever is supplied.
"""

from __future__ import annotations

import polars as pl

__all__ = [
    "gross_leverage",
    "gross_exposure",
    "net_exposure",
    "long_exposure",
    "short_exposure",
    "concentration",
    "top_n_concentration",
    "active_share",
]


def gross_leverage(weights: pl.Expr) -> pl.Expr:
    """Sum of absolute weights — total notional / equity.

    Args:
        weights: Position weight expression.

    Returns:
        Scalar gross-leverage expression.
    """
    return weights.abs().sum()


def gross_exposure(weights: pl.Expr) -> pl.Expr:
    """Alias for ``gross_leverage`` — long + short notional.

    Args:
        weights: Position weight expression.

    Returns:
        Scalar gross-exposure expression.
    """
    return weights.abs().sum()


def net_exposure(weights: pl.Expr) -> pl.Expr:
    """Long minus short notional — signed sum of weights.

    Args:
        weights: Position weight expression.

    Returns:
        Scalar net-exposure expression.
    """
    return weights.sum()


def long_exposure(weights: pl.Expr) -> pl.Expr:
    """Sum of positive weights.

    Args:
        weights: Position weight expression.

    Returns:
        Scalar long-exposure expression.
    """
    return pl.when(weights > 0).then(weights).otherwise(0.0).sum()


def short_exposure(weights: pl.Expr) -> pl.Expr:
    """Sum of negative weights (returned as a negative number).

    Args:
        weights: Position weight expression.

    Returns:
        Scalar short-exposure expression.
    """
    return pl.when(weights < 0).then(weights).otherwise(0.0).sum()


def concentration(weights: pl.Expr) -> pl.Expr:
    """Herfindahl-Hirschman index of normalised absolute weights.

    Computed on absolute weights normalised to sum to 1 — yields
    ``1/N`` for an equal-weight portfolio of ``N`` names and ``1.0``
    for a single-name portfolio.

    Args:
        weights: Position weight expression.

    Returns:
        Scalar HHI expression in ``(0, 1]``.
    """
    abs_w = weights.abs()
    norm = abs_w / abs_w.sum()
    return norm.pow(2).sum()


def top_n_concentration(weights: pl.Expr, n: int = 10) -> pl.Expr:
    """Fraction of gross exposure held by the top ``n`` absolute weights.

    Args:
        weights: Position weight expression.
        n: Number of top positions.

    Returns:
        Scalar expression in ``[0, 1]``.
    """
    abs_w = weights.abs()
    return abs_w.top_k(n).sum() / abs_w.sum()


def active_share(weights: pl.Expr, benchmark_weights: pl.Expr) -> pl.Expr:
    """Active share — ``0.5 * sum(|w - b|)``.

    Args:
        weights: Portfolio weight expression.
        benchmark_weights: Benchmark weight expression aligned to ``weights``.

    Returns:
        Scalar active-share expression in ``[0, 1]``.
    """
    return 0.5 * (weights - benchmark_weights).abs().sum()
