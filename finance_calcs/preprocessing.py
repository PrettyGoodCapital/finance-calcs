"""Signal preprocessing helpers that operate on concrete Polars DataFrames."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

__all__ = ["neutralize", "orthogonalize"]


def neutralize(
    frame: pl.DataFrame,
    signal_col: str,
    *,
    group_cols: Sequence[str] | None = None,
    exposure_cols: Sequence[str] | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    output = output_col or f"{signal_col}_neutralized"
    if exposure_cols:
        return orthogonalize(frame, signal_col, exposure_cols=exposure_cols, by=group_cols, output_col=output)
    signal = pl.col(signal_col)
    residual = signal - signal.mean().over(list(group_cols)) if group_cols else signal - signal.mean()
    return frame.with_columns(residual.alias(output))


def orthogonalize(
    frame: pl.DataFrame,
    signal_col: str,
    *,
    exposure_cols: Sequence[str],
    by: Sequence[str] | None = None,
    output_col: str | None = None,
) -> pl.DataFrame:
    if not exposure_cols:
        raise ValueError("exposure_cols must not be empty")
    output = output_col or f"{signal_col}_orthogonalized"
    groups = frame.partition_by(list(by), maintain_order=True) if by else [frame]
    parts: list[pl.DataFrame] = []
    for group in groups:
        y = group[signal_col].to_numpy().astype(float)
        x = group.select(list(exposure_cols)).to_numpy().astype(float)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        residual = np.full(y.shape, np.nan)
        if valid.sum() > x.shape[1]:
            design = np.column_stack([np.ones(valid.sum()), x[valid]])
            beta = np.linalg.lstsq(design, y[valid], rcond=None)[0]
            residual[valid] = y[valid] - design @ beta
        elif valid.any():
            residual[valid] = y[valid] - y[valid].mean()
        parts.append(group.with_columns(pl.Series(output, residual)))
    return pl.concat(parts, how="vertical")
