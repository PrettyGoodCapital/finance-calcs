"""Tests for statistical-validity metrics."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import finance_calcs as fc


def _toy_returns(n: int = 252, mu: float = 0.0005, sigma: float = 0.01, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    bench = rng.normal(mu, sigma, n)
    asset = 0.7 * bench + rng.normal(mu, sigma * 0.5, n)
    return pl.DataFrame({"ret": asset, "bench": bench})


def test_skew_kurt_higher_moments():
    df = _toy_returns(n=2000, seed=11)
    s = df.select(fc.skewness(pl.col("ret")).alias("s")).item()
    k = df.select(fc.kurtosis(pl.col("ret")).alias("k")).item()
    hm = df.select(fc.higher_moments(pl.col("ret")).alias("hm")).item()
    assert abs(s) < 0.5
    assert abs(k) < 1.0
    assert hm["skew"] == pytest.approx(s, rel=1e-9)
    assert hm["kurt"] == pytest.approx(k, rel=1e-9)


def test_stability_of_timeseries():
    n = 500
    rets = pl.DataFrame({"r": [0.001] * n})
    r2 = rets.select(fc.stability_of_timeseries(pl.col("r")).alias("r2")).item()
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_common_sense_ratio_positive():
    df = _toy_returns(n=1000, mu=0.001)
    csr = df.select(fc.common_sense_ratio(pl.col("ret")).alias("csr")).item()
    assert csr > 0


def test_probabilistic_sharpe_high_for_strong_track_record():
    n = 1000
    rng = np.random.default_rng(3)
    s = pl.Series("r", rng.normal(0.001, 0.005, n))
    psr = fc.probabilistic_sharpe(s, benchmark_sr=0.0, periods_per_year=252)
    assert 0.99 < psr <= 1.0


def test_deflated_sharpe_lower_than_psr():
    rng = np.random.default_rng(5)
    s = pl.Series("r", rng.normal(0.0008, 0.01, 800))
    psr = fc.probabilistic_sharpe(s, 0.0, 252)
    dsr = fc.deflated_sharpe(s, n_trials=100, sr_variance=0.5, periods_per_year=252)
    assert dsr <= psr + 1e-6


def test_min_track_record_length_finite():
    rng = np.random.default_rng(7)
    s = pl.Series("r", rng.normal(0.001, 0.01, 2000))
    n = fc.minimum_track_record_length(s, benchmark_sr=0.0, alpha=0.05)
    assert math.isfinite(n) and n > 0


def test_sharpe_ci_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(9)
    s = pl.Series("r", rng.normal(0.001, 0.01, 1000))
    sr, lo, hi = fc.sharpe_ci_bootstrap(s, n_bootstrap=400, seed=1)
    assert lo <= sr <= hi


def test_sharpe_with_ci_brackets_point_estimate():
    rng = np.random.default_rng(11)
    s = pl.Series("r", rng.normal(0.001, 0.01, 1000))
    sr, lo, hi = fc.sharpe_with_ci(s)
    assert lo <= sr <= hi
