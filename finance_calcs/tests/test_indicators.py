"""Tests for technical indicators (overlap / momentum / volatility / volume)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import finance_calcs as fc


@pytest.fixture
def ohlcv() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    rets = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    intra = rng.normal(0.0, 0.005, n)
    open_ = close * np.exp(-intra)
    high = np.maximum(close, open_) * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    low = np.minimum(close, open_) * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def monotone_close() -> pl.DataFrame:
    n = 60
    close = np.linspace(100.0, 200.0, n)
    return pl.DataFrame({"close": close})


def _select(df: pl.DataFrame, expr: pl.Expr, name: str = "out") -> pl.Series:
    return df.select(expr.alias(name)).to_series()


def test_sma_constant_series():
    df = pl.DataFrame({"x": [5.0] * 30})
    out = _select(df, fc.sma(pl.col("x"), 10))
    assert out.tail(20).to_list() == [5.0] * 20


def test_ema_runs(ohlcv):
    out = _select(ohlcv, fc.ema(pl.col("close"), 20))
    assert out.len() == ohlcv.height
    assert not math.isnan(out[-1])


def test_wma_recent_weighted_more():
    df = pl.DataFrame({"x": [0.0, 0.0, 0.0, 0.0, 1.0]})
    out = _select(df, fc.wma(pl.col("x"), 5))
    assert out[-1] == pytest.approx(5.0 / 15.0)


def test_bbands_middle_is_sma(ohlcv):
    mid = _select(ohlcv, fc.bbands_middle(pl.col("close"), 20))
    sma = _select(ohlcv, fc.sma(pl.col("close"), 20))
    assert mid.to_list() == sma.to_list()


def test_bbands_upper_above_lower(ohlcv):
    up = _select(ohlcv, fc.bbands_upper(pl.col("close"), 20))
    lo = _select(ohlcv, fc.bbands_lower(pl.col("close"), 20))
    diffs = (up - lo).drop_nulls().to_list()
    assert all(d >= 0 for d in diffs)


def test_donchian_bounds(ohlcv):
    up = _select(ohlcv, fc.donchian_upper(pl.col("high"), 20))
    lo = _select(ohlcv, fc.donchian_lower(pl.col("low"), 20))
    diffs = (up - lo).drop_nulls().to_list()
    assert all(d >= 0 for d in diffs)


def test_rsi_bounds(ohlcv):
    out = [v for v in _select(ohlcv, fc.rsi(pl.col("close"), 14)).drop_nulls().to_list() if not math.isnan(v)]
    assert out
    assert all(0.0 <= v <= 100.0 for v in out)


def test_rsi_uptrend_high(monotone_close):
    out = _select(monotone_close, fc.rsi(pl.col("close"), 14)).drop_nulls().to_list()
    assert out[-1] > 99.0


def test_macd_components(ohlcv):
    line = _select(ohlcv, fc.macd_line(pl.col("close")))
    sig = _select(ohlcv, fc.macd_signal(pl.col("close")))
    hist = _select(ohlcv, fc.macd_hist(pl.col("close")))
    diff = (line - sig - hist).drop_nulls().abs().to_list()
    assert max(diff) < 1e-9


def test_stoch_k_bounds(ohlcv):
    out = (
        _select(
            ohlcv,
            fc.stoch_k(pl.col("high"), pl.col("low"), pl.col("close"), 14),
        )
        .drop_nulls()
        .to_list()
    )
    assert all(0.0 <= v <= 100.0 + 1e-9 for v in out)


def test_willr_bounds(ohlcv):
    out = (
        _select(
            ohlcv,
            fc.willr(pl.col("high"), pl.col("low"), pl.col("close"), 14),
        )
        .drop_nulls()
        .to_list()
    )
    assert all(-100.0 - 1e-9 <= v <= 0.0 + 1e-9 for v in out)


def test_adx_bounds(ohlcv):
    out = [
        v
        for v in _select(
            ohlcv,
            fc.adx(pl.col("high"), pl.col("low"), pl.col("close"), 14),
        )
        .drop_nulls()
        .to_list()
        if not math.isnan(v)
    ]
    assert out
    assert all(0.0 <= v <= 100.0 + 1e-9 for v in out)


def test_cci_runs(ohlcv):
    out = _select(ohlcv, fc.cci(pl.col("high"), pl.col("low"), pl.col("close"), 20))
    assert out.len() == ohlcv.height


def test_roc_zero_for_constant():
    df = pl.DataFrame({"x": [10.0] * 20})
    out = _select(df, fc.roc(pl.col("x"), 5)).drop_nulls().to_list()
    assert all(abs(v) < 1e-12 for v in out)


def test_true_range_nonneg(ohlcv):
    out = (
        _select(
            ohlcv,
            fc.true_range(pl.col("high"), pl.col("low"), pl.col("close")),
        )
        .drop_nulls()
        .to_list()
    )
    assert all(v >= 0.0 for v in out)


def test_atr_positive(ohlcv):
    out = (
        _select(
            ohlcv,
            fc.atr(pl.col("high"), pl.col("low"), pl.col("close"), 14),
        )
        .drop_nulls()
        .to_list()
    )
    assert all(v > 0.0 for v in out)


def test_parkinson_positive(ohlcv):
    out = _select(ohlcv, fc.parkinson_vol(pl.col("high"), pl.col("low"), 20)).drop_nulls().to_list()
    assert all(v > 0.0 for v in out)


def test_garman_klass_runs(ohlcv):
    out = _select(
        ohlcv,
        fc.garman_klass_vol(
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            20,
        ),
    )
    assert out.len() == ohlcv.height


def test_realized_vol_matches_rolling_std(ohlcv):
    rets = ohlcv.select(
        fc.simple_returns(pl.col("close")).alias("r"),
    ).to_series()
    a = _select(pl.DataFrame({"r": rets}), fc.realized_vol(pl.col("r"), 20))
    b = pl.DataFrame({"r": rets}).select(pl.col("r").rolling_std(20).alias("v")).to_series()
    assert a.to_list() == b.to_list()


def test_obv_monotone_up_for_uptrend(monotone_close):
    df = monotone_close.with_columns(pl.lit(1000.0).alias("volume"))
    out = _select(df, fc.obv(pl.col("close"), pl.col("volume"))).to_list()
    diffs = [b - a for a, b in zip(out, out[1:])]
    assert all(d >= 0 for d in diffs)


def test_ad_runs(ohlcv):
    out = _select(
        ohlcv,
        fc.ad(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")),
    )
    assert out.len() == ohlcv.height


def test_adosc_runs(ohlcv):
    out = _select(
        ohlcv,
        fc.adosc(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
        ),
    )
    assert out.len() == ohlcv.height


def test_namespace_single_input(ohlcv):
    out = ohlcv.select(
        pl.col("close").finance.rsi(14).alias("rsi"),
        pl.col("close").finance.sma(20).alias("sma"),
    )
    assert out.height == ohlcv.height
