"""Tests for portfolio-level exposure / concentration metrics."""

from __future__ import annotations

import polars as pl
import pytest

import finance_calcs as fc


def test_portfolio_exposures():
    weights = pl.DataFrame({"w": [0.3, 0.2, -0.1, -0.4, 0.5]})
    gl = weights.select(fc.gross_leverage(pl.col("w")).alias("gl")).item()
    gx = weights.select(fc.gross_exposure(pl.col("w")).alias("gx")).item()
    nx = weights.select(fc.net_exposure(pl.col("w")).alias("nx")).item()
    lo = weights.select(fc.long_exposure(pl.col("w")).alias("lo")).item()
    sh = weights.select(fc.short_exposure(pl.col("w")).alias("sh")).item()
    assert gl == pytest.approx(1.5)
    assert gx == pytest.approx(1.5)
    assert nx == pytest.approx(0.5)
    assert lo == pytest.approx(1.0)
    assert sh == pytest.approx(-0.5)


def test_concentration_equal_weight():
    n = 10
    weights = pl.DataFrame({"w": [1.0 / n] * n})
    hhi = weights.select(fc.concentration(pl.col("w")).alias("h")).item()
    assert hhi == pytest.approx(1.0 / n, rel=1e-9)


def test_top_n_concentration():
    weights = pl.DataFrame({"w": [0.4, 0.3, 0.15, 0.1, 0.05]})
    top2 = weights.select(fc.top_n_concentration(pl.col("w"), n=2).alias("t")).item()
    assert top2 == pytest.approx(0.7, rel=1e-9)


def test_active_share():
    pw = pl.DataFrame({"w": [0.5, 0.3, 0.2], "b": [0.4, 0.4, 0.2]})
    out = pw.select(fc.active_share(pl.col("w"), pl.col("b")).alias("a")).item()
    assert out == pytest.approx(0.5 * (abs(0.1) + abs(-0.1) + 0.0), rel=1e-9)


def test_namespace_portfolio():
    weights = pl.DataFrame({"w": [0.3, -0.2, 0.5]})
    g = weights.select(pl.col("w").fcalcs.gross_leverage().alias("g")).item()
    assert g == pytest.approx(1.0)
