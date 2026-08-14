"""Conformance tests for return and risk metric semantics."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

import finance_calcs as fc


@pytest.fixture
def missing_returns() -> pl.DataFrame:
    return pl.DataFrame({"r": [0.01, None, float("nan"), -0.02]})


def test_null_and_nan_are_equivalent_missing_returns(missing_returns: pl.DataFrame) -> None:
    clean = pl.DataFrame({"r": [0.01, -0.02]})

    actual = missing_returns.select(
        fc.returns(pl.col("r")).alias("compound"),
        fc.sharpe(pl.col("r"), periods_per_year=252).alias("sharpe"),
    ).row(0)
    expected = clean.select(
        fc.returns(pl.col("r")).alias("compound"),
        fc.sharpe(pl.col("r"), periods_per_year=252).alias("sharpe"),
    ).row(0)

    assert actual == pytest.approx(expected)


def test_rolling_compound_uses_native_expression() -> None:
    plan = pl.DataFrame({"r": [0.01, 0.02, -0.01]}).lazy().select(fc.returns(pl.col("r"), window=2).alias("result")).explain()

    assert "rolling_map" not in plan


def test_rolling_compound_handles_total_loss() -> None:
    out = pl.DataFrame({"r": [0.10, -1.0, 0.25]}).select(fc.returns(pl.col("r"), window=2).alias("result"))

    assert out["result"].to_list() == pytest.approx([None, -1.0, -1.0], nan_ok=True)


def test_annualized_return_is_observation_based_for_irregular_dates() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 12, 31)],
            "r": [0.01, 0.02, -0.01],
        }
    )

    actual = frame.select(fc.annualized_return(pl.col("r"), periods_per_year=252, date=pl.col("date"))).item()
    expected = (1.01 * 1.02 * 0.99) ** (252 / 3) - 1.0

    assert actual == pytest.approx(expected)


def test_sharpe_uses_geometric_risk_free_conversion_and_sample_std() -> None:
    values = [0.01, -0.02, 0.03]
    annual_risk_free = 0.05
    per_period_risk_free = (1.0 + annual_risk_free) ** (1.0 / 252) - 1.0
    excess = pl.Series([value - per_period_risk_free for value in values])
    expected = excess.mean() / excess.std(ddof=1) * math.sqrt(252)

    actual = pl.DataFrame({"r": values}).select(fc.sharpe(pl.col("r"), risk_free=annual_risk_free)).item()

    assert actual == pytest.approx(expected)


def test_sortino_uses_full_sample_downside_denominator() -> None:
    values = [0.01, -0.02, 0.03]
    downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in values) / len(values)) * math.sqrt(252)
    expected = (sum(values) / len(values)) * 252 / downside

    actual = pl.DataFrame({"r": values}).select(fc.sortino(pl.col("r"))).item()

    assert actual == pytest.approx(expected)


def test_drawdown_and_ulcer_index_include_initial_equity_baseline() -> None:
    frame = pl.DataFrame({"r": [-0.10, 0.05]})

    out = frame.select(
        fc.drawdown_series(pl.col("r")).alias("drawdown"),
        fc.ulcer_index(pl.col("r")).alias("ulcer"),
    )
    expected_drawdown = [-0.10, -0.055]
    expected_ulcer = math.sqrt(sum(value * value for value in expected_drawdown) / 2)

    assert out["drawdown"].to_list() == pytest.approx(expected_drawdown)
    assert out["ulcer"][0] == pytest.approx(expected_ulcer)


def test_rolling_max_drawdown_rebases_each_window() -> None:
    frame = pl.DataFrame({"r": [1.0, -0.5, 0.0, 0.0]})

    out = frame.select(fc.max_drawdown(pl.col("r"), window=2).alias("max_drawdown"))

    assert out["max_drawdown"].to_list() == pytest.approx([None, -0.5, -0.5, 0.0])


def test_rolling_max_drawdown_uses_native_expression() -> None:
    plan = pl.DataFrame({"r": [0.01, -0.02, 0.03]}).lazy().select(fc.max_drawdown(pl.col("r"), window=2)).explain()

    assert "rolling_map" not in plan
    assert "python_udf" not in plan


def test_rolling_max_drawdown_handles_missing_returns_and_total_loss() -> None:
    frame = pl.DataFrame({"r": [0.10, None, -1.0, 0.25, 0.10]})

    out = frame.select(fc.max_drawdown(pl.col("r"), window=2).alias("max_drawdown"))

    assert out["max_drawdown"].to_list() == pytest.approx([None, 0.0, -1.0, -1.0, 0.0])


def test_rolling_max_drawdown_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        fc.max_drawdown(pl.col("r"), window=0)


def test_expression_metrics_compose_in_lazy_queries(missing_returns: pl.DataFrame) -> None:
    out = (
        missing_returns.lazy()
        .select(
            fc.returns(pl.col("r")).alias("compound"),
            fc.sharpe(pl.col("r")).alias("sharpe"),
            fc.ulcer_index(pl.col("r")).alias("ulcer"),
        )
        .collect()
    )

    assert out.shape == (1, 3)
    assert all(value is not None and math.isfinite(value) for value in out.row(0))
