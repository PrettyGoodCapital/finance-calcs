"""Tests for alpha / signal calculations."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

import finance_calcs as fc


@pytest.fixture
def panel() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n_dates = 60
    n_assets = 30
    fwd = rng.normal(0.0, 0.02, (n_dates, n_assets))
    fwd_z = (fwd - fwd.mean(axis=1, keepdims=True)) / fwd.std(axis=1, keepdims=True)
    noise = rng.normal(0.0, 1.0, (n_dates, n_assets))
    rho = 0.3
    signal = rho * fwd_z + math.sqrt(1.0 - rho * rho) * noise
    dates = np.repeat(np.arange(n_dates), n_assets)
    assets = np.tile(np.arange(n_assets), n_dates)
    return pl.DataFrame(
        {
            "date": dates,
            "asset": assets,
            "signal": signal.flatten(),
            "fwd": fwd.flatten(),
        }
    )


def test_forward_returns():
    df = pl.DataFrame({"p": [100.0, 101.0, 99.0, 102.0]})
    out = df.select(fc.forward_returns(pl.col("p"), 1).alias("f")).to_series()
    assert out[0] == pytest.approx(0.01)
    assert out[1] == pytest.approx(99.0 / 101.0 - 1.0)
    assert out[3] is None


def test_pearson_ic_per_date(panel):
    ic = panel.group_by("date").agg(
        fc.pearson_ic(pl.col("signal"), pl.col("fwd")).alias("ic"),
    )
    mean_ic = float(ic["ic"].mean())
    assert mean_ic > 0.15


def test_spearman_ic_per_date(panel):
    ic = panel.group_by("date").agg(
        fc.spearman_ic(pl.col("signal"), pl.col("fwd")).alias("ic"),
    )
    mean_ic = float(ic["ic"].mean())
    assert mean_ic > 0.15


def test_ic_ir_and_summary_stats(panel):
    ic = panel.group_by("date").agg(fc.spearman_ic(pl.col("signal"), pl.col("fwd")).alias("ic")).sort("date")
    stats = fc.ic_summary_stats(ic["ic"])
    assert stats["n"] == 60
    assert stats["mean"] > 0.0
    assert stats["t_stat"] > 1.0
    ir_expr = float(ic.select(fc.ic_ir(pl.col("ic"))).item())
    assert ir_expr == pytest.approx(stats["ir"], rel=1e-6)


def test_rolling_ic(panel):
    ic = panel.group_by("date").agg(fc.pearson_ic(pl.col("signal"), pl.col("fwd")).alias("ic")).sort("date")
    rolling = ic.with_columns(fc.ic_ir(pl.col("ic"), window=10).alias("r")).drop_nulls()
    assert rolling.height > 0


def test_period_ic_ir_matches_grouped_monthly_result():
    df = pl.DataFrame(
        {
            "date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 2, 1),
                date(2024, 2, 2),
                date(2024, 2, 5),
            ],
            "ic": [0.10, 0.20, -0.05, 0.15, -0.10, 0.05],
        }
    )

    out = df.with_columns(fc.ic_ir(pl.col("ic"), period="month", date=pl.col("date")).alias("period_ir"))
    expected = (
        df.group_by(fc.period_bucket(pl.col("date"), "month").alias("bucket"))
        .agg((pl.col("ic").mean() / pl.col("ic").std()).alias("expected"))
        .sort("bucket")
    )

    assert out["period_ir"].to_list() == pytest.approx([expected["expected"][0]] * 3 + [expected["expected"][1]] * 3)


def test_hit_rate_perfect():
    df = pl.DataFrame({"s": [1.0, -1.0, 1.0, -1.0], "f": [0.1, -0.1, 0.2, -0.2]})
    hr = float(df.select(fc.hit_rate(pl.col("s"), pl.col("f"))).item())
    assert hr == pytest.approx(1.0)


def test_assign_quantile_uniform():
    df = pl.DataFrame({"x": list(range(20))})
    out = df.select(fc.assign_quantile(pl.col("x"), 5).alias("q")).to_series()
    counts = out.value_counts().sort("q")
    assert counts["count"].to_list() == [4, 4, 4, 4, 4]
    assert counts["q"].to_list() == [0, 1, 2, 3, 4]


def test_rank_normalize_bounds():
    df = pl.DataFrame({"x": list(range(11))})
    out = df.select(fc.rank_normalize(pl.col("x")).alias("r")).to_series()
    assert min(out) >= -0.5
    assert max(out) <= 0.5
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-12)


def test_zscore():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = df.select(fc.zscore(pl.col("x")).alias("z")).to_series()
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(out.std()) == pytest.approx(1.0, abs=1e-12)


def test_winsorize_clips():
    df = pl.DataFrame({"x": [-1000.0] + [0.0] * 10 + [1000.0]})
    out = df.select(fc.winsorize(pl.col("x"), 2.0).alias("w")).to_series().to_list()
    assert max(out) < 1000.0
    assert min(out) > -1000.0


def test_long_short_spread(panel):
    panel_q = panel.with_columns(
        fc.assign_quantile(pl.col("signal"), 5).over("date").alias("q"),
    )
    spread = panel_q.group_by("date").agg(
        fc.long_short_spread(pl.col("fwd"), pl.col("q"), upper=4, lower=0).alias("ls"),
    )
    assert float(spread["ls"].mean()) > 0.0


def test_quantile_changed():
    df = pl.DataFrame({"q": [0, 0, 1, 1, 2, 2, 2]})
    out = df.select(fc.quantile_changed(pl.col("q")).alias("c")).to_series().to_list()
    assert out[0] is None
    assert out[1:] == [False, True, False, True, False, False]


def test_namespace_alpha():
    df = pl.DataFrame({"p": [100.0, 101.0, 102.0, 100.0]})
    out = df.select(pl.col("p").finance.forward_returns(1).alias("f"))
    assert out.height == 4
