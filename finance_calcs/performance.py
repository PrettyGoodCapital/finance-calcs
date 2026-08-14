"""Report-oriented performance metrics."""

from __future__ import annotations

import numpy as np
import polars as pl

from ._periods import PeriodLike
from .returns import _clean_returns, cum_returns_final
from .risk import max_drawdown

__all__ = [
    "average_loss",
    "average_win",
    "avg_loss",
    "avg_win",
    "best",
    "best_return",
    "drawdown_details",
    "gain_to_pain_ratio",
    "kelly_criterion",
    "recovery_factor",
    "worst",
    "worst_return",
]

__finance_namespace__ = [
    "average_loss",
    "average_win",
    "avg_loss",
    "avg_win",
    "best",
    "best_return",
    "gain_to_pain_ratio",
    "kelly_criterion",
    "recovery_factor",
    "worst",
    "worst_return",
]


def best_return(
    returns: pl.Expr,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Highest raw or compounded period return."""
    if period is None:
        return _clean_returns(returns).max()
    return cum_returns_final(returns, period=period, date=date).max()


def best(
    returns: pl.Expr,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Compatibility alias for :func:`best_return`."""
    return best_return(returns, period=period, date=date)


def worst_return(
    returns: pl.Expr,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Lowest raw or compounded period return."""
    if period is None:
        return _clean_returns(returns).min()
    return cum_returns_final(returns, period=period, date=date).min()


def worst(
    returns: pl.Expr,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Compatibility alias for :func:`worst_return`."""
    return worst_return(returns, period=period, date=date)


def average_win(returns: pl.Expr) -> pl.Expr:
    """Mean positive return."""
    clean_returns = _clean_returns(returns)
    return clean_returns.filter(clean_returns > 0.0).mean()


def avg_win(returns: pl.Expr) -> pl.Expr:
    """Compatibility alias for :func:`average_win`."""
    return average_win(returns)


def average_loss(returns: pl.Expr) -> pl.Expr:
    """Mean negative return, expressed as a negative value."""
    clean_returns = _clean_returns(returns)
    return clean_returns.filter(clean_returns < 0.0).mean()


def avg_loss(returns: pl.Expr) -> pl.Expr:
    """Compatibility alias for :func:`average_loss`."""
    return average_loss(returns)


def gain_to_pain_ratio(returns: pl.Expr) -> pl.Expr:
    """Arithmetic net return divided by absolute summed losses."""
    clean_returns = _clean_returns(returns)
    pain = clean_returns.filter(clean_returns < 0.0).sum().abs()
    return clean_returns.sum() / pain


def recovery_factor(returns: pl.Expr) -> pl.Expr:
    """Absolute arithmetic net return divided by absolute maximum drawdown."""
    clean_returns = _clean_returns(returns)
    return clean_returns.sum().abs() / max_drawdown(clean_returns).abs()


def kelly_criterion(returns: pl.Expr) -> pl.Expr:
    """Kelly fraction estimated from active-period win rate and payoff ratio."""
    clean_returns = _clean_returns(returns)
    wins = clean_returns.filter(clean_returns > 0.0)
    losses = clean_returns.filter(clean_returns < 0.0)
    win_probability = wins.count() / (wins.count() + losses.count())
    payoff = wins.mean() / losses.mean().abs()
    return (payoff * win_probability - (1.0 - win_probability)) / payoff


def drawdown_details(returns: pl.Series, *, date: pl.Series | None = None) -> pl.DataFrame:
    """Return peak, valley, recovery, duration, and depth for each drawdown.

    ``end`` is null and ``recovered`` is false for an open drawdown. Without a
    date series, integer row positions identify each point.
    """
    if date is not None and len(date) != len(returns):
        raise ValueError("date and returns lengths must match")

    index = date.rename("index") if date is not None else pl.Series("index", range(len(returns)), dtype=pl.Int64)
    schema = {
        "start": index.dtype,
        "valley": index.dtype,
        "end": index.dtype,
        "duration": pl.Int64,
        "max_drawdown": pl.Float64,
        "recovered": pl.Boolean,
    }
    if returns.is_empty():
        return pl.DataFrame(schema=schema)

    values = returns.cast(pl.Float64, strict=False).fill_nan(None).fill_null(0.0).to_numpy()
    equity = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    drawdown = equity / peak - 1.0
    index_values = index.to_list()

    rows: list[dict[str, object]] = []
    start: int | None = None
    valley = 0
    for position, value in enumerate(drawdown):
        if value < 0.0 and start is None:
            start = max(position - 1, 0)
            valley = position
        if start is not None and value < drawdown[valley]:
            valley = position
        recovered = start is not None and value >= 0.0
        open_at_end = start is not None and position == len(drawdown) - 1
        if recovered or open_at_end:
            rows.append(
                {
                    "start": index_values[start],
                    "valley": index_values[valley],
                    "end": index_values[position] if recovered else None,
                    "duration": position - start,
                    "max_drawdown": float(drawdown[valley]),
                    "recovered": recovered,
                }
            )
            start = None

    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
