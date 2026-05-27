"""Tests for post-trade transaction cost metrics."""

from __future__ import annotations

from datetime import date

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


def test_transaction_volume_can_repeat_period_volume() -> None:
    trades = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1)],
            "qty": [100.0, -50.0, 25.0],
            "price": [10.0, 20.0, 40.0],
        }
    )

    out = trades.with_columns(
        fc.transaction_volume(
            pl.col("qty"),
            pl.col("price"),
            period="month",
            date=pl.col("date"),
        ).alias("volume")
    )

    assert out["volume"].to_list() == pytest.approx([2000.0, 2000.0, 1000.0])


def test_cost_attribution_summarizes_components() -> None:
    trades = pl.DataFrame(
        {
            "amount": [100.0, -50.0],
            "price": [10.0, 20.0],
            "commission": [1.0, 1.5],
            "fees": [0.25, 0.50],
            "bps": [5.0, 10.0],
        }
    )

    out = fc.cost_attribution(trades)

    totals = dict(zip(out["component"], out["total"]))
    assert totals["commission"] == pytest.approx(2.5)
    assert totals["fees"] == pytest.approx(0.75)
    assert totals["slippage"] == pytest.approx(1.5)
    assert sum(totals.values()) == pytest.approx(out["total"].sum())


def test_extract_round_trips_and_trade_quality_stats() -> None:
    trades = pl.DataFrame(
        {
            "timestamp": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 8),
            ],
            "symbol": ["A", "A", "A", "B", "B"],
            "amount": [10.0, -4.0, -6.0, -5.0, 5.0],
            "price": [100.0, 110.0, 90.0, 50.0, 40.0],
        }
    )

    round_trips = fc.extract_round_trips(trades)
    stats = fc.round_trip_stats(round_trips)
    long_short = fc.long_short_round_trip_stats(round_trips)
    sector = fc.sector_round_trip_stats(round_trips, {"A": "Tech", "B": "Energy"})

    assert round_trips.height == 3
    assert round_trips["pnl"].to_list() == pytest.approx([40.0, -60.0, 50.0])
    assert stats["n_trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["total_pnl"] == pytest.approx(30.0)
    assert set(long_short["side"]) == {"long", "short"}
    assert set(sector["sector"]) == {"Tech", "Energy"}


def test_mae_mfe_and_execution_quality_helpers() -> None:
    trades = pl.DataFrame(
        {
            "symbol": ["A"],
            "side": ["long"],
            "entry_timestamp": [date(2024, 1, 2)],
            "exit_timestamp": [date(2024, 1, 5)],
            "entry_price": [100.0],
            "quantity": [10.0],
            "pnl": [100.0],
        }
    )
    prices = pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
            "symbol": ["A", "A", "A", "A"],
            "price": [100.0, 95.0, 112.0, 110.0],
        }
    )

    excursions = fc.mae_mfe(trades, prices)
    assert excursions["mae"].to_list() == pytest.approx([-0.05])
    assert excursions["mfe"].to_list() == pytest.approx([0.12])

    quality = pl.DataFrame(
        {
            "side": ["Buy", "Sell"],
            "exec": [101.0, 99.0],
            "decision": [100.0, 100.0],
            "vwap": [100.5, 99.5],
        }
    ).select(
        fc.implementation_shortfall(pl.col("exec"), pl.col("decision"), side=pl.col("side")).alias("is_bps"),
        fc.vwap_slippage(pl.col("exec"), pl.col("vwap"), side=pl.col("side")).alias("vwap_bps"),
    )

    assert quality["is_bps"].to_list() == pytest.approx([100.0, 100.0])
    assert quality["vwap_bps"].to_list() == pytest.approx([(101.0 - 100.5) / 100.5 * 10_000.0, (99.5 - 99.0) / 99.5 * 10_000.0])


def test_trade_quality_expression_metrics() -> None:
    trades = pl.DataFrame(
        {
            "pnl": [10.0, -5.0, 20.0, -10.0, -1.0, 3.0],
            "size": [100.0, 100.0, 200.0, 150.0, 80.0, 120.0],
            "ret": [0.10, -0.05, 0.10, -0.0667, -0.0125, 0.025],
        }
    )

    out = trades.select(
        fc.win_rate(pl.col("pnl")).alias("win_rate"),
        fc.profit_factor(pl.col("pnl")).alias("profit_factor"),
        fc.payoff_ratio(pl.col("pnl")).alias("payoff_ratio"),
        fc.avg_trade_pnl(pl.col("pnl")).alias("avg_trade_pnl"),
        fc.trade_size_return_correlation(pl.col("size"), pl.col("ret")).alias("size_corr"),
    )

    assert out["win_rate"][0] == pytest.approx(0.5)
    assert out["profit_factor"][0] == pytest.approx(33.0 / 16.0)
    assert out["payoff_ratio"][0] == pytest.approx(11.0 / (16.0 / 3.0))
    assert out["avg_trade_pnl"][0] == pytest.approx(17.0 / 6.0)
    assert out["size_corr"][0] is not None


def test_sequence_trade_quality_helpers() -> None:
    trades = pl.DataFrame(
        {
            "pnl": [10.0, 5.0, -2.0, -3.0, -1.0, 4.0],
            "duration": [1, 2, 3, 4, 5, 6],
            "exit_reason": ["target", "target", "stop", "stop", "stop", "time"],
        }
    )

    runs = fc.consecutive_wins_losses(trades["pnl"])
    duration = fc.trade_duration_stats(trades["duration"])
    exits = fc.exit_reason_stats(trades)

    assert runs == {"max_consecutive_wins": 2, "max_consecutive_losses": 3}
    assert duration["mean"] == pytest.approx(3.5)
    assert duration["max"] == 6
    assert dict(zip(exits["exit_reason"], exits["count"])) == {"stop": 3, "target": 2, "time": 1}


def test_namespace_post_trade_metrics() -> None:
    trades = pl.DataFrame({"qty": [100.0], "price": [10.0], "arrival": [9.95], "side": ["Buy"]})

    out = trades.select(
        pl.col("qty").finance.transaction_notional(pl.col("price")).alias("notional"),
        pl.col("price").finance.slippage_bps(pl.col("arrival"), side=pl.col("side")).alias("slip"),
    )

    assert out["notional"][0] == pytest.approx(1000.0)
    assert out["slip"][0] == pytest.approx((10.0 - 9.95) / 9.95 * 10_000.0)
