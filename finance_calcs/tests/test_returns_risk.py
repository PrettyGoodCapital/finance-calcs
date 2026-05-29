"""Tests for finance_calcs core return and basic-risk expressions."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest
from finance_enums import Frequency

import finance_calcs as fc


@pytest.fixture
def constant_returns() -> pl.Series:
    return pl.Series("r", [0.001] * 252)


@pytest.fixture
def two_sided_returns() -> pl.Series:
    return pl.Series("r", [0.01, -0.02, 0.03, -0.01, 0.02, -0.04, 0.05])


@pytest.fixture
def prices() -> pl.Series:
    return pl.Series("p", [100.0, 101.0, 99.0, 102.0, 100.0, 103.0])


def test_simple_returns(prices):
    out = pl.select(fc.simple_returns(pl.lit(prices))).to_series()
    expected = [None, 0.01, -2 / 101, 3 / 99, -2 / 102, 3 / 100]
    assert out[0] is None
    for got, exp in zip(out[1:], expected[1:]):
        assert got == pytest.approx(exp)


def test_log_returns(prices):
    out = pl.select(fc.log_returns(pl.lit(prices))).to_series()
    assert out[0] is None
    assert out[1] == pytest.approx(math.log(101 / 100))
    assert out[5] == pytest.approx(math.log(103 / 100))


def test_cum_returns_final(constant_returns):
    out = pl.select(fc.cum_returns_final(pl.lit(constant_returns))).item()
    assert out == pytest.approx((1.001) ** 252 - 1.0)


def test_cum_returns_series(two_sided_returns):
    out = pl.select(fc.cum_returns(pl.lit(two_sided_returns))).to_series()
    expected = []
    g = 1.0
    for r in two_sided_returns:
        g *= 1 + r
        expected.append(g - 1.0)
    for got, exp in zip(out, expected):
        assert got == pytest.approx(exp)


def test_annualized_return(constant_returns):
    out = pl.select(fc.annualized_return(pl.lit(constant_returns))).item()
    assert out == pytest.approx((1.001) ** 252 - 1.0)


def test_annualized_volatility():
    rets = pl.Series("r", [0.01, -0.01] * 126)
    out = pl.select(fc.annualized_volatility(pl.lit(rets))).item()
    assert out == pytest.approx(rets.std() * math.sqrt(252))


def test_volatility_alias():
    rets = pl.Series("r", [0.01, -0.01] * 126)
    out = pl.select(fc.volatility(pl.lit(rets))).item()
    expected = pl.select(fc.annualized_volatility(pl.lit(rets))).item()
    assert out == pytest.approx(expected)


def test_rolling_returns(two_sided_returns):
    out = pl.select(fc.cum_returns_final(pl.lit(two_sided_returns), window=3)).to_series()
    assert out[0] is None and out[1] is None
    expected_at_2 = (1.01) * (0.98) * (1.03) - 1.0
    assert out[2] == pytest.approx(expected_at_2)


def test_returns_matches_cum_returns_final(two_sided_returns):
    out = pl.select(fc.returns(pl.lit(two_sided_returns))).item()
    expected = pl.select(fc.cum_returns_final(pl.lit(two_sided_returns))).item()
    assert out == pytest.approx(expected)


def test_returns_window(two_sided_returns):
    out = pl.select(fc.returns(pl.lit(two_sided_returns), window=3)).to_series()
    assert out[0] is None and out[1] is None
    expected_at_2 = (1.01) * (0.98) * (1.03) - 1.0
    assert out[2] == pytest.approx(expected_at_2)


def test_returns_rejects_window_and_period(two_sided_returns):
    with pytest.raises(ValueError, match="mutually exclusive"):
        pl.select(fc.returns(pl.lit(two_sided_returns), window=3, period="month"))


def test_period_bucket_accepts_frequency_aliases_and_polars_windows():
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 15)],
        }
    )

    out = df.select(
        fc.period_bucket(pl.col("date"), Frequency.Month).alias("enum_month"),
        fc.period_bucket(pl.col("date"), "monthly").alias("alias_month"),
        fc.period_bucket(pl.col("date"), "2d").alias("two_day"),
    )

    assert out["enum_month"].to_list() == [date(2024, 1, 1), date(2024, 1, 1), date(2024, 2, 1), date(2024, 2, 1)]
    assert out["alias_month"].to_list() == out["enum_month"].to_list()
    assert out["two_day"].null_count() == 0


def test_period_returns_repeat_bucket_compound_return():
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1), date(2024, 2, 2)],
            "r": [0.10, -0.05, 0.02, 0.03],
        }
    )

    out = df.with_columns(
        fc.returns(pl.col("r"), period=Frequency.Month, date=pl.col("date")).alias("period_return"),
        fc.cum_returns(pl.col("r"), period="monthly", date=pl.col("date")).alias("period_path"),
    )

    january = 1.10 * 0.95 - 1.0
    february = 1.02 * 1.03 - 1.0
    assert out["period_return"].to_list() == pytest.approx([january, january, february, february])
    assert out["period_path"].to_list() == pytest.approx([0.10, january, 0.02, february])


def test_period_metrics_require_date(two_sided_returns):
    with pytest.raises(ValueError, match="date is required"):
        pl.select(fc.returns(pl.lit(two_sided_returns), period="month"))


def test_period_metrics_accept_explicit_bucket_expression():
    df = pl.DataFrame(
        {
            "bucket": ["a", "a", "b", "b"],
            "r": [0.10, -0.05, 0.02, 0.03],
        }
    )

    out = df.with_columns(fc.returns(pl.col("r"), period=pl.col("bucket")).alias("period_return"))

    assert out["period_return"].to_list() == pytest.approx([1.10 * 0.95 - 1.0] * 2 + [1.02 * 1.03 - 1.0] * 2)


def test_period_sharpe_matches_grouped_monthly_result():
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
            "r": [0.01, 0.02, -0.01, 0.03, -0.02, 0.01],
        }
    )

    out = df.with_columns(fc.sharpe(pl.col("r"), periods_per_year=1, period="month", date=pl.col("date")).alias("period_sharpe"))
    expected = (
        df.group_by(fc.period_bucket(pl.col("date"), "month").alias("bucket"))
        .agg((pl.col("r").mean() / pl.col("r").std()).alias("expected"))
        .sort("bucket")
    )

    assert out["period_sharpe"].to_list() == pytest.approx([expected["expected"][0]] * 3 + [expected["expected"][1]] * 3)


def test_namespace_period_returns():
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1), date(2024, 2, 2)],
            "r": [0.10, -0.05, 0.02, 0.03],
        }
    )

    out = df.with_columns(pl.col("r").fcalcs.returns(period="month", date=pl.col("date")).alias("period_return"))

    assert out["period_return"].to_list() == pytest.approx([1.10 * 0.95 - 1.0] * 2 + [1.02 * 1.03 - 1.0] * 2)


def test_sharpe_zero_mean_zero():
    rets = pl.Series("r", [0.01, -0.01] * 126)
    sharpe = pl.select(fc.sharpe(pl.lit(rets))).item()
    assert sharpe == pytest.approx(0.0, abs=1e-12)


def test_sharpe_constant_positive(constant_returns):
    val = pl.select(fc.sharpe(pl.lit(constant_returns))).item()
    assert val is None or math.isnan(val) or math.isinf(val) or abs(val) > 1e6


def test_sortino_uses_only_downside(two_sided_returns):
    sortino = pl.select(fc.sortino(pl.lit(two_sided_returns))).item()
    assert sortino is not None and sortino > 0


def test_downside_risk_alias(two_sided_returns):
    out = pl.select(fc.downside_risk(pl.lit(two_sided_returns))).item()
    expected = pl.select(fc.downside_deviation(pl.lit(two_sided_returns))).item()
    assert out == pytest.approx(expected)


def test_max_drawdown_known():
    rets = pl.Series("r", [0.1, -0.1, 0.06060606])
    mdd = pl.select(fc.max_drawdown(pl.lit(rets))).item()
    assert mdd == pytest.approx(0.99 / 1.1 - 1.0)


def test_drawdown_series_nonpositive(two_sided_returns):
    dd = pl.select(fc.drawdown_series(pl.lit(two_sided_returns))).to_series()
    assert (dd <= 1e-12).all()


def test_calmar_finite():
    rets = pl.Series("r", [0.01, -0.005, 0.02, -0.01, 0.015] * 50)
    val = pl.select(fc.calmar(pl.lit(rets))).item()
    assert val is not None and math.isfinite(val)


def test_value_at_risk_quantile(two_sided_returns):
    var5 = pl.select(fc.value_at_risk(pl.lit(two_sided_returns), 0.05)).item()
    expected = two_sided_returns.quantile(0.05)
    assert var5 == pytest.approx(expected)


def test_cvar_tail_mean():
    rets = pl.Series("r", [-0.10, -0.05, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04])
    cvar = pl.select(fc.conditional_value_at_risk(pl.lit(rets), 0.25)).item()
    threshold = rets.quantile(0.25)
    tail = [r for r in rets if r <= threshold]
    assert cvar == pytest.approx(sum(tail) / len(tail))


def test_parametric_var_known_z():
    rets = pl.Series("r", [0.001] * 1000)
    val = pl.select(fc.parametric_var(pl.lit(rets), 0.05)).item()
    assert val == pytest.approx(0.001)


def test_parametric_var_unsupported_cutoff():
    rets = pl.Series("r", [0.001] * 100)
    with pytest.raises(ValueError):
        pl.select(fc.parametric_var(pl.lit(rets), 0.123))


def test_rolling_sharpe_shape():
    rets = pl.Series("r", [0.01, -0.01] * 126)
    s = pl.select(fc.sharpe(pl.lit(rets), window=20)).to_series()
    assert len(s) == len(rets)
    assert s[0] is None
    assert s[-1] is not None and math.isfinite(s[-1])


def test_sharpe_accepts_expr_risk_free():
    df = pl.DataFrame(
        {
            "r": [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.005] * 40,
            "rf": [0.0001] * 280,
        }
    )
    scalar = df.select(fc.sharpe(pl.col("r"), risk_free=0.0001, periods_per_year=1)).item()
    expr = df.select(fc.sharpe(pl.col("r"), risk_free=pl.col("rf"), periods_per_year=1)).item()
    assert math.isfinite(scalar) and math.isfinite(expr)
    assert scalar == pytest.approx(expr, rel=1e-9)


def test_sortino_accepts_expr_required_return():
    df = pl.DataFrame(
        {
            "r": [0.01, -0.02, 0.03, -0.01, 0.02, -0.04, 0.05] * 40,
            "mar": [0.001] * 280,
        }
    )
    scalar = df.select(fc.sortino(pl.col("r"), required_return=0.001, periods_per_year=1)).item()
    expr = df.select(fc.sortino(pl.col("r"), required_return=pl.col("mar"), periods_per_year=1)).item()
    assert scalar == pytest.approx(expr, rel=1e-9)


def test_namespace_on_expr(prices):
    df = pl.DataFrame({"p": prices})
    out = df.select(pl.col("p").fcalcs.log_returns().alias("ret"))["ret"]
    assert out[0] is None
    assert out[1] == pytest.approx(math.log(101 / 100))


def test_namespace_on_series(constant_returns):
    val = constant_returns.fcalcs.sharpe()
    assert val is None or math.isnan(val) or math.isinf(val) or abs(val) > 1e6


def test_namespace_sharpe_handles_exact_zero_variance_series():
    rets = pl.Series("r", [1.0] * 252)
    val = rets.fcalcs.sharpe()
    assert math.isinf(val)


def test_namespace_pipeline(prices):
    df = pl.DataFrame({"p": prices})
    out = df.with_columns(
        pl.col("p").fcalcs.simple_returns().alias("r"),
    ).select(
        pl.col("r").fcalcs.max_drawdown().alias("mdd"),
        pl.col("r").fcalcs.cum_returns_final().alias("total"),
    )
    assert out["total"][0] == pytest.approx(prices[-1] / prices[0] - 1.0)
    assert out["mdd"][0] <= 0.0
