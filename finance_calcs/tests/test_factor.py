"""Tests for factor / capture-ratio metrics."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

import finance_calcs as fc


def _toy_returns(n: int = 252, mu: float = 0.0005, sigma: float = 0.01, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    bench = rng.normal(mu, sigma, n)
    asset = 0.7 * bench + rng.normal(mu, sigma * 0.5, n)
    return pl.DataFrame({"ret": asset, "bench": bench})


def test_beta_close_to_regression():
    df = _toy_returns()
    out = df.select(fc.beta(pl.col("ret"), pl.col("bench")).alias("b")).item()
    r = df["ret"].to_numpy()
    b = df["bench"].to_numpy()
    slope = np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1)
    assert out == pytest.approx(slope, rel=1e-6)


def test_alpha_definition():
    df = _toy_returns()
    out = df.select(fc.alpha(pl.col("ret"), pl.col("bench"), risk_free=0.0).alias("a")).item()
    r = df["ret"].to_numpy()
    b = df["bench"].to_numpy()
    slope = np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1)
    expected = (r.mean() - slope * b.mean()) * 252
    assert out == pytest.approx(expected, rel=1e-6)


def test_alpha_accepts_expr_risk_free():
    df = _toy_returns().with_columns(rf=pl.lit(0.0001))
    scalar = df.select(fc.alpha(pl.col("ret"), pl.col("bench"), risk_free=0.0001 * 252)).item()
    expr = df.select(fc.alpha(pl.col("ret"), pl.col("bench"), risk_free=pl.col("rf"))).item()
    assert scalar == pytest.approx(expr, rel=1e-9)


def test_rolling_beta_tail_value_matches_full_sample():
    df = _toy_returns(n=252)
    rb = df.select(fc.beta(pl.col("ret"), pl.col("bench"), window=252).alias("rb"))
    last = rb["rb"][-1]
    full = df.select(fc.beta(pl.col("ret"), pl.col("bench")).alias("b")).item()
    assert last == pytest.approx(full, rel=5e-3)


def test_up_down_beta_signs():
    df = _toy_returns()
    up = df.select(fc.up_beta(pl.col("ret"), pl.col("bench")).alias("u")).item()
    dn = df.select(fc.down_beta(pl.col("ret"), pl.col("bench")).alias("d")).item()
    assert up > 0
    assert dn > 0


def test_capture_ratios():
    df = _toy_returns()
    uc = df.select(fc.up_capture(pl.col("ret"), pl.col("bench")).alias("u")).item()
    dc = df.select(fc.down_capture(pl.col("ret"), pl.col("bench")).alias("d")).item()
    udc = df.select(fc.up_down_capture(pl.col("ret"), pl.col("bench")).alias("r")).item()
    assert math.isfinite(uc) and math.isfinite(dc)
    assert udc == pytest.approx(uc / dc, rel=1e-9)


def test_batting_average_bounds():
    df = _toy_returns()
    ba = df.select(fc.batting_average(pl.col("ret"), pl.col("bench")).alias("ba")).item()
    assert 0.0 <= ba <= 1.0


def test_rolling_capture_finite():
    df = _toy_returns()
    out = df.select(
        fc.up_capture(pl.col("ret"), pl.col("bench"), window=60).alias("ruc"),
        fc.down_capture(pl.col("ret"), pl.col("bench"), window=60).alias("rdc"),
    )
    assert math.isfinite(out["ruc"][-1])
    assert math.isfinite(out["rdc"][-1])


def test_tracking_error_and_information_ratio():
    df = _toy_returns()
    te = df.select(fc.tracking_error(pl.col("ret"), pl.col("bench")).alias("te")).item()
    ir = df.select(fc.information_ratio(pl.col("ret"), pl.col("bench")).alias("ir")).item()
    diff = (df["ret"] - df["bench"]).to_numpy()
    expected_te = diff.std(ddof=1) * math.sqrt(252)
    expected_ir = diff.mean() / diff.std(ddof=1) * math.sqrt(252)
    assert te == pytest.approx(expected_te, rel=1e-6)
    assert ir == pytest.approx(expected_ir, rel=1e-6)


def test_period_tracking_error_matches_grouped_monthly_result():
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
            "ret": [0.02, 0.01, -0.01, 0.03, -0.02, 0.01],
            "bench": [0.01, 0.00, -0.02, 0.02, -0.01, 0.00],
        }
    )

    out = df.with_columns(
        fc.tracking_error(pl.col("ret"), pl.col("bench"), periods_per_year=1, period="month", date=pl.col("date")).alias("period_te")
    )
    expected = (
        df.with_columns((pl.col("ret") - pl.col("bench")).alias("active"))
        .group_by(fc.period_bucket(pl.col("date"), "month").alias("bucket"))
        .agg(pl.col("active").std().alias("expected"))
        .sort("bucket")
    )

    assert out["period_te"].to_list() == pytest.approx([expected["expected"][0]] * 3 + [expected["expected"][1]] * 3)


def test_namespace_factor():
    df = _toy_returns()
    out = df.select(
        pl.col("ret").finance.beta(pl.col("bench")).alias("b"),
        pl.col("ret").finance.tracking_error(pl.col("bench")).alias("te"),
    )
    assert math.isfinite(out["b"][0])
    assert out["te"][0] > 0
