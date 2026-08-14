"""Report-oriented metric and compatibility tests."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import finance_calcs as fc


def test_report_return_metrics_match_reference_definitions() -> None:
    frame = pl.DataFrame({"r": [0.02, -0.01, 0.0, 0.03, -0.02, None, float("nan")]})

    out = frame.select(
        fc.best_return(pl.col("r")).alias("best"),
        fc.worst_return(pl.col("r")).alias("worst"),
        fc.average_win(pl.col("r")).alias("average_win"),
        fc.average_loss(pl.col("r")).alias("average_loss"),
        fc.gain_to_pain_ratio(pl.col("r")).alias("gain_to_pain"),
        fc.recovery_factor(pl.col("r")).alias("recovery"),
        fc.kelly_criterion(pl.col("r")).alias("kelly"),
        fc.cagr(pl.col("r")).alias("cagr"),
        fc.expected_shortfall(pl.col("r"), cutoff=0.25).alias("expected_shortfall"),
    ).row(0, named=True)

    assert out == pytest.approx(
        {
            "best": 0.03,
            "worst": -0.02,
            "average_win": 0.025,
            "average_loss": -0.015,
            "gain_to_pain": 2.0 / 3.0,
            "recovery": 1.0,
            "kelly": 0.2,
            "cagr": 1.619707506713151,
            "expected_shortfall": -0.015,
        }
    )


def test_best_and_worst_return_support_period_buckets() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1), date(2024, 2, 2)],
            "r": [0.10, -0.05, -0.20, 0.10],
        }
    )

    out = frame.select(
        fc.best_return(pl.col("r"), period="1mo", date=pl.col("date")).alias("best"),
        fc.worst_return(pl.col("r"), period="1mo", date=pl.col("date")).alias("worst"),
    ).row(0, named=True)

    assert out == pytest.approx({"best": 1.10 * 0.95 - 1.0, "worst": 0.80 * 1.10 - 1.0})


def test_compatibility_aliases_preserve_finance_calcs_semantics() -> None:
    frame = pl.DataFrame({"r": [0.01, -0.02, 0.03, -0.01]})

    out = frame.select(
        fc.cagr(pl.col("r")).alias("cagr"),
        fc.annualized_return(pl.col("r")).alias("annualized_return"),
        fc.expected_shortfall(pl.col("r"), cutoff=0.25).alias("expected_shortfall"),
        fc.conditional_value_at_risk(pl.col("r"), cutoff=0.25).alias("cvar"),
        fc.to_drawdown_series(pl.col("r")).alias("to_drawdown"),
        fc.drawdown_series(pl.col("r")).alias("drawdown"),
    )

    assert out["cagr"][0] == pytest.approx(out["annualized_return"][0])
    assert out["expected_shortfall"][0] == pytest.approx(out["cvar"][0])
    assert out["to_drawdown"].to_list() == pytest.approx(out["drawdown"].to_list())


def test_r_squared_is_correlation_squared_against_benchmark() -> None:
    frame = pl.DataFrame({"r": [0.01, 0.02, -0.01, 0.03], "b": [0.02, 0.04, -0.02, 0.06]})

    assert frame.select(fc.r_squared(pl.col("r"), pl.col("b"))).item() == pytest.approx(1.0)


def test_r_squared_supports_rolling_and_missing_values() -> None:
    frame = pl.DataFrame(
        {
            "r": [0.01, float("nan"), 0.02, -0.01, 0.03],
            "b": [0.02, 0.03, 0.04, -0.02, 0.06],
        }
    )

    scalar = frame.select(fc.r_squared(pl.col("r"), pl.col("b"))).item()
    rolling = frame.select(fc.r_squared(pl.col("r"), pl.col("b"), window=3)).to_series()

    assert scalar == pytest.approx(1.0)
    assert rolling[-1] == pytest.approx(1.0)


def test_report_metrics_are_available_through_namespace() -> None:
    frame = pl.DataFrame({"r": [0.01, -0.02, 0.03]})

    out = frame.select(
        pl.col("r").fcalcs.best().alias("best"),
        pl.col("r").fcalcs.avg_loss().alias("average_loss"),
        pl.col("r").fcalcs.kelly_criterion().alias("kelly"),
    ).row(0, named=True)

    assert out == pytest.approx({"best": 0.03, "average_loss": -0.02, "kelly": 1.0 / 3.0})


def test_drawdown_details_identifies_recovered_and_open_periods() -> None:
    returns = pl.Series("r", [0.10, -0.10, -0.10, 0.30, -0.05])
    dates = pl.Series("date", [date(2024, 1, day) for day in range(1, 6)])

    details = fc.drawdown_details(returns, date=dates)

    assert details.to_dicts() == [
        {
            "start": date(2024, 1, 1),
            "valley": date(2024, 1, 3),
            "end": date(2024, 1, 4),
            "duration": 3,
            "max_drawdown": pytest.approx(-0.19),
            "recovered": True,
        },
        {
            "start": date(2024, 1, 4),
            "valley": date(2024, 1, 5),
            "end": None,
            "duration": 1,
            "max_drawdown": pytest.approx(-0.05),
            "recovered": False,
        },
    ]


def test_drawdown_details_validates_lengths_and_empty_input() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        fc.drawdown_details(pl.Series("r", [0.01]), date=pl.Series("date", []))

    details = fc.drawdown_details(pl.Series("r", [], dtype=pl.Float64))
    assert details.is_empty()
    assert details.schema == {
        "start": pl.Int64,
        "valley": pl.Int64,
        "end": pl.Int64,
        "duration": pl.Int64,
        "max_drawdown": pl.Float64,
        "recovered": pl.Boolean,
    }
