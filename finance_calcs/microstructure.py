"""Market microstructure metrics as polars expressions."""

from __future__ import annotations

import polars as pl

__all__ = [
    "quoted_spread_bps",
    "effective_spread_bps",
    "realized_spread_bps",
    "order_imbalance",
    "amihud_illiquidity",
    "kyle_lambda",
]


def quoted_spread_bps(bid: pl.Expr, ask: pl.Expr, *, mid: pl.Expr | None = None) -> pl.Expr:
    reference = mid if mid is not None else (bid + ask) / 2.0
    return (ask - bid) / reference * 10_000.0


def effective_spread_bps(execution_price: pl.Expr, mid_price: pl.Expr, *, side: pl.Expr | None = None) -> pl.Expr:
    if side is None:
        return 2.0 * (execution_price - mid_price).abs() / mid_price * 10_000.0
    return 2.0 * side * (execution_price - mid_price) / mid_price * 10_000.0


def realized_spread_bps(execution_price: pl.Expr, future_mid_price: pl.Expr, *, side: pl.Expr | None = None) -> pl.Expr:
    if side is None:
        return 2.0 * (execution_price - future_mid_price).abs() / future_mid_price * 10_000.0
    return 2.0 * side * (execution_price - future_mid_price) / future_mid_price * 10_000.0


def order_imbalance(buy_volume: pl.Expr, sell_volume: pl.Expr) -> pl.Expr:
    total = buy_volume + sell_volume
    return (buy_volume - sell_volume) / total


def amihud_illiquidity(returns: pl.Expr, traded_notional: pl.Expr) -> pl.Expr:
    return returns.abs() / traded_notional


def kyle_lambda(returns: pl.Expr, signed_volume: pl.Expr) -> pl.Expr:
    return returns / signed_volume
