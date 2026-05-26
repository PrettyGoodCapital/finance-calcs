"""Tests for tail-risk metrics."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

import finance_calcs as fc


def _toy_returns(n: int = 252, mu: float = 0.0005, sigma: float = 0.01, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"ret": rng.normal(mu, sigma, n)})


def test_tail_ratio():
    rng = np.random.default_rng(13)
    df = pl.DataFrame({"r": rng.normal(0, 0.01, 5000)})
    tr = df.select(fc.tail_ratio(pl.col("r")).alias("tr")).item()
    assert tr == pytest.approx(1.0, abs=0.2)


def test_ulcer_index_positive():
    df = _toy_returns(n=500)
    ui = df.select(fc.ulcer_index(pl.col("ret")).alias("ui")).item()
    assert ui >= 0


def test_omega_ratio_around_one_for_zero_mean():
    rng = np.random.default_rng(17)
    df = pl.DataFrame({"r": rng.normal(0, 0.01, 5000)})
    o = df.select(fc.omega_ratio(pl.col("r")).alias("o")).item()
    assert o == pytest.approx(1.0, abs=0.1)


def test_period_tail_ratio_matches_grouped_monthly_result():
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
            "r": [-0.04, -0.01, 0.02, -0.02, 0.01, 0.05],
        }
    )

    out = df.with_columns(fc.tail_ratio(pl.col("r"), period="month", date=pl.col("date")).alias("period_tail"))
    expected = (
        df.group_by(fc.period_bucket(pl.col("date"), "month").alias("bucket"))
        .agg((pl.col("r").quantile(0.95).abs() / pl.col("r").quantile(0.05).abs()).alias("expected"))
        .sort("bucket")
    )

    assert out["period_tail"].to_list() == pytest.approx([expected["expected"][0]] * 3 + [expected["expected"][1]] * 3)


def test_rolling_var_cvar_finite():
    df = _toy_returns(n=500)
    out = df.select(
        fc.value_at_risk(pl.col("ret"), 0.05, window=60).alias("v"),
        fc.conditional_value_at_risk(pl.col("ret"), 0.05, window=60).alias("cv"),
    )
    assert math.isfinite(out["v"][-1])
    assert math.isfinite(out["cv"][-1])
    assert out["cv"][-1] <= out["v"][-1] + 1e-9


def test_gpd_var_cvar_positive_and_ordered():
    rng = np.random.default_rng(19)
    s = pl.Series("r", rng.standard_t(df=4, size=2000) * 0.01)
    var = fc.gpd_var(s, var_p=0.01, threshold_p=0.10)
    cvar = fc.gpd_cvar(s, var_p=0.01, threshold_p=0.10)
    assert var > 0
    assert cvar >= var
