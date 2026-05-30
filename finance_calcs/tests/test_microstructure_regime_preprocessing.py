from __future__ import annotations

import math

import polars as pl
import pytest

import finance_calcs as fc


def test_microstructure_expression_metrics() -> None:
    df = pl.DataFrame(
        {
            "bid": [99.9, 100.0],
            "ask": [100.1, 100.4],
            "exec": [100.15, 99.95],
            "mid": [100.0, 100.2],
            "future_mid": [100.05, 100.05],
            "side": [1, -1],
            "buy_volume": [60.0, 40.0],
            "sell_volume": [40.0, 60.0],
            "returns": [0.01, -0.02],
            "notional": [1_000_000.0, 2_000_000.0],
        }
    )

    out = df.with_columns(
        fc.quoted_spread_bps(pl.col("bid"), pl.col("ask")).alias("quoted"),
        fc.effective_spread_bps(pl.col("exec"), pl.col("mid"), side=pl.col("side")).alias("effective"),
        fc.realized_spread_bps(pl.col("exec"), pl.col("future_mid"), side=pl.col("side")).alias("realized"),
        fc.order_imbalance(pl.col("buy_volume"), pl.col("sell_volume")).alias("imbalance"),
        fc.amihud_illiquidity(pl.col("returns"), pl.col("notional")).alias("amihud"),
    )

    assert out["quoted"].to_list() == pytest.approx([20.0, 39.92015968])
    assert out["effective"][0] > 0.0
    assert out["realized"][1] > 0.0
    assert out["imbalance"].to_list() == pytest.approx([0.2, -0.2])
    assert out["amihud"][0] == pytest.approx(1e-8)


def test_regime_and_fractional_difference_helpers() -> None:
    trending = pl.Series("x", [1.0, 1.2, 1.4, 1.7, 2.1, 2.7, 3.4, 4.2])
    mean_reverting = pl.Series("x", [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    assert fc.hurst_exponent(trending) > fc.hurst_exponent(mean_reverting)

    diffed = fc.fractional_difference(trending, order=0.5, threshold=0.05)
    assert isinstance(diffed, pl.Series)
    assert diffed.len() == trending.len()
    assert diffed.null_count() > 0

    regimes = pl.DataFrame({"r": [-0.03, -0.02, -0.01, 0.01, 0.02, 0.04]}).with_columns(
        fc.regime_signal(pl.col("r"), window=3, threshold=0.5).alias("regime")
    )
    assert set(regimes["regime"].drop_nulls().to_list()) <= {-1, 0, 1}


def test_signal_preprocessing_helpers() -> None:
    frame = pl.DataFrame(
        {
            "date": [1, 1, 1, 1],
            "sector": ["Tech", "Tech", "Energy", "Energy"],
            "signal": [1.0, 3.0, 10.0, 12.0],
            "beta": [0.8, 1.0, 1.2, 1.4],
        }
    )

    neutral = fc.neutralize(frame, "signal", group_cols=["date", "sector"])
    orthogonal = fc.orthogonalize(frame, "signal", exposure_cols=["beta"], by=["date"])
    preprocessed = frame.with_columns(
        fc.rank_normalize(pl.col("signal")).over("date").alias("ranked"),
        fc.winsorize(pl.col("signal"), cutoff=1.0).over("date").alias("winsorized"),
    )

    assert neutral.group_by("sector").agg(pl.col("signal_neutralized").mean().abs().alias("mean"))["mean"].max() < 1e-12
    assert abs(orthogonal.select(pl.corr("signal_orthogonalized", "beta")).item()) < 1e-12
    assert preprocessed["ranked"].min() >= -0.5
    assert preprocessed["ranked"].max() <= 0.5
    assert math.isfinite(preprocessed["winsorized"].mean())
