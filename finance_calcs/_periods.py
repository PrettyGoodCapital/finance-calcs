"""Period and calendar-bucket helpers for expression metrics."""

from __future__ import annotations

import math
from typing import TypeAlias

import polars as pl
from finance_enums import Frequency, to_frequency

try:
    from finance_dates import period_grid as dates_period_grid
except ImportError:  # pragma: no cover - compatibility with older finance-dates

    def dates_period_grid(date: pl.Expr, period: Frequency | str | pl.Expr) -> pl.Expr:
        if isinstance(period, pl.Expr):
            return period
        if isinstance(period, Frequency):
            return date.dt.truncate(period.polars_truncate)
        value = period.strip()
        if not value:
            raise ValueError("period must not be empty")
        try:
            return date.dt.truncate(to_frequency(value).polars_truncate)
        except ValueError:
            return date.dt.truncate(value)


PeriodLike: TypeAlias = Frequency | str | pl.Expr
FrequencyLike: TypeAlias = Frequency | str | float


def _observations_per_year(frequency: FrequencyLike) -> float:
    """Resolve an annualization frequency to observations per year."""
    if isinstance(frequency, Frequency | str):
        return float(to_frequency(frequency).periods_per_year)
    if isinstance(frequency, bool):
        raise TypeError("frequency must be a Frequency, alias, or positive number")
    value = float(frequency)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("numeric frequency must be finite and positive")
    return value


def _annual_rate_to_observation_rate(rate: float | pl.Expr, observations_per_year: float) -> float | pl.Expr:
    """Convert an annual scalar rate to one observation's rate.

    Expression rates are assumed to already match the observation frequency.
    """
    if isinstance(rate, pl.Expr):
        return rate
    if not math.isfinite(rate) or rate <= -1.0:
        raise ValueError("annual scalar rates must be finite and greater than -1")
    if rate == 0.0:
        return 0.0
    return (1.0 + rate) ** (1.0 / observations_per_year) - 1.0


def period_bucket(date: pl.Expr, period: PeriodLike) -> pl.Expr:
    """Return a period bucket expression for ``date``.

    ``period`` accepts a :class:`finance_enums.Frequency`, any alias
    understood by ``finance_enums.to_frequency()``, any Polars duration
    string accepted by ``dt.truncate()``, or a precomputed bucket
    expression.
    """
    return dates_period_grid(date, period)


def _check_window_period(window: int | None, period: PeriodLike | None) -> None:
    if window is not None and period is not None:
        raise ValueError("window and period are mutually exclusive")


def _bucket_or_none(date: pl.Expr | None, period: PeriodLike | None) -> pl.Expr | None:
    if period is None:
        return None
    if isinstance(period, pl.Expr):
        return period
    if date is None:
        raise ValueError("date is required when period is a Frequency or duration string")
    return period_bucket(date, period)
