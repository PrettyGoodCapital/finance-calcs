"""Period and calendar-bucket helpers for expression metrics."""

from __future__ import annotations

from typing import TypeAlias

import polars as pl
from finance_enums import Frequency, to_frequency

PeriodLike: TypeAlias = Frequency | str | pl.Expr


def _period_rule(period: Frequency | str) -> str:
    if isinstance(period, Frequency):
        return period.polars_truncate
    if isinstance(period, str):
        value = period.strip()
        if not value:
            raise ValueError("period must not be empty")
        try:
            return to_frequency(value).polars_truncate
        except ValueError:
            return value
    raise TypeError(f"period must be a Frequency, string, or expression; got {type(period).__name__}")


def period_bucket(date: pl.Expr, period: PeriodLike) -> pl.Expr:
    """Return a period bucket expression for ``date``.

    ``period`` accepts a :class:`finance_enums.Frequency`, any alias
    understood by ``finance_enums.to_frequency()``, any Polars duration
    string accepted by ``dt.truncate()``, or a precomputed bucket
    expression.
    """
    if isinstance(period, pl.Expr):
        return period
    return date.dt.truncate(_period_rule(period))


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
