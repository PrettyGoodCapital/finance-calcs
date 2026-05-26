"""Tests for post-trade transaction cost metrics."""

from __future__ import annotations

import polars as pl
import pytest

import finance_calcs as fc


def test_transaction_notional_uses_absolute_quantity() -> None:
    trades = pl.DataFrame({"qty": [100.0, -50.0], "price": [10.0, 20.0]})

    out = trades.select(fc.transaction_notional(pl.col("qty"), pl.col("price")).alias("notional"))

    assert out["notional"].to_list() == [1000.0, 1000.0]


def test_transaction_cost_combines_bps_and_explicit_costs() -> None:
    trades = pl.DataFrame(
        {
            "qty": [100.0, -50.0],
            "price": [10.0, 20.0],
            "commission": [1.0, 1.5],
            "fees": [0.25, 0.50],
        }
    )

    out = trades.select(
        fc.transaction_cost(
            pl.col("qty"),
            pl.col("price"),
            commission=pl.col("commission"),
            fees=pl.col("fees"),
            bps=5.0,
        ).alias("cost")
    )

    assert out["cost"].to_list() == pytest.approx([1.75, 2.50])


def test_slippage_bps_is_side_aware_cost() -> None:
    trades = pl.DataFrame(
        {
            "side": ["Buy", "Sell", "Short", "Cover"],
            "exec": [101.0, 99.0, 99.0, 101.0],
            "arrival": [100.0, 100.0, 100.0, 100.0],
        }
    )

    out = trades.select(
        fc.slippage_bps(
            pl.col("exec"),
            pl.col("arrival"),
            side=pl.col("side"),
        ).alias("slip")
    )

    assert out["slip"].to_list() == pytest.approx([100.0, 100.0, 100.0, 100.0])


def test_turnover_aggregates_position_weight_changes() -> None:
    weights = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["A", "B", "A", "B", "A", "B"],
            "weight": [0.2, 0.8, 0.5, 0.5, 0.4, 0.6],
        }
    )

    out = (
        weights.sort("symbol", "date")
        .with_columns(fc.turnover(pl.col("weight")).over("symbol").fill_null(0.0).alias("turnover"))
        .group_by("date")
        .agg(pl.col("turnover").sum())
        .sort("date")
    )

    assert out["turnover"].to_list() == pytest.approx([0.0, 0.3, 0.1])


def test_namespace_post_trade_metrics() -> None:
    trades = pl.DataFrame({"qty": [100.0], "price": [10.0], "arrival": [9.95], "side": ["Buy"]})

    out = trades.select(
        pl.col("qty").finance.transaction_notional(pl.col("price")).alias("notional"),
        pl.col("price").finance.slippage_bps(pl.col("arrival"), side=pl.col("side")).alias("slip"),
    )

    assert out["notional"][0] == pytest.approx(1000.0)
    assert out["slip"][0] == pytest.approx((10.0 - 9.95) / 9.95 * 10_000.0)
