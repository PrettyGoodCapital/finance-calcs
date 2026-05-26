"""Post-trade transaction cost metrics as polars expressions."""

from __future__ import annotations

import polars as pl

__all__ = [
    "transaction_notional",
    "transaction_cost",
    "slippage_bps",
    "turnover",
]


def _expr(value: float | pl.Expr) -> pl.Expr:
    if isinstance(value, pl.Expr):
        return value
    return pl.lit(value)


def transaction_notional(quantity: pl.Expr, price: pl.Expr) -> pl.Expr:
    """Absolute traded notional, ``abs(quantity) * price``."""
    return quantity.abs() * price


def transaction_cost(
    quantity: pl.Expr,
    price: pl.Expr,
    *,
    commission: float | pl.Expr = 0.0,
    fees: float | pl.Expr = 0.0,
    bps: float | pl.Expr = 0.0,
) -> pl.Expr:
    """Per-trade cost from explicit charges plus basis-point slippage.

    ``bps`` is applied to absolute traded notional. ``commission`` and
    ``fees`` may be scalars or expressions aligned to the transaction rows.
    """
    notional = transaction_notional(quantity, price)
    return notional * (_expr(bps) / 10_000.0) + _expr(commission) + _expr(fees)


def slippage_bps(
    execution_price: pl.Expr,
    benchmark_price: pl.Expr,
    *,
    side: pl.Expr | None = None,
) -> pl.Expr:
    """Execution slippage in basis points.

    Without ``side``, the result is signed price difference versus the
    benchmark. With ``side``, positive values mean adverse execution cost
    for buy/cover and sell/short transactions.
    """
    raw = (execution_price - benchmark_price) / benchmark_price * 10_000.0
    if side is None:
        return raw

    side_label = side.cast(pl.Utf8).str.to_lowercase()
    return pl.when(side_label.is_in(["buy", "cover"])).then(raw).when(side_label.is_in(["sell", "short"])).then(-raw).otherwise(None)


def turnover(weights: pl.Expr, *, window: int | None = None) -> pl.Expr:
    """Portfolio turnover contribution from position-weight changes.

    Apply over a symbol/security partition, then aggregate by rebalance
    date. The contribution is ``0.5 * abs(weight - prior_weight)``.
    """
    contribution = 0.5 * weights.diff().abs()
    if window is None:
        return contribution
    return contribution.rolling_sum(window)
