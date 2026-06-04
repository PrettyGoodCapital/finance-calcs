"""Period and calendar-bucket helpers for expression metrics."""

from __future__ import annotations

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
