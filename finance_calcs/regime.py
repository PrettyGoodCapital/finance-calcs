"""Regime and fractional-differencing helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

__all__ = ["regime_signal", "hurst_exponent", "fractional_difference"]
__finance_namespace__ = ["regime_signal"]


def regime_signal(returns: pl.Expr, *, window: int = 63, threshold: float = 1.0) -> pl.Expr:
    if window < 2:
        raise ValueError("window must be >= 2")
    zscore = returns.rolling_mean(window) / returns.rolling_std(window)
    return pl.when(zscore > threshold).then(1).when(zscore < -threshold).then(-1).otherwise(0)


def _series_array(values: pl.Series | Sequence[float] | np.ndarray) -> np.ndarray:
    if isinstance(values, pl.Series):
        arr = values.drop_nulls().to_numpy()
    else:
        arr = np.asarray(values)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def hurst_exponent(values: pl.Series | Sequence[float] | np.ndarray, *, max_lag: int | None = None) -> float:
    arr = _series_array(values)
    if arr.size < 6:
        return float("nan")
    upper = min(max_lag or arr.size // 2, arr.size - 1)
    lags = np.arange(2, max(3, upper + 1))
    tau = np.array([np.std(arr[lag:] - arr[:-lag], ddof=1) for lag in lags])
    mask = tau > 0.0
    if mask.sum() < 2:
        return 0.0
    slope = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]
    return float(max(0.0, min(1.0, slope)))


def _fracdiff_weights(order: float, threshold: float) -> np.ndarray:
    weights = [1.0]
    k = 1
    while True:
        next_weight = -weights[-1] * (order - k + 1.0) / k
        if abs(next_weight) < threshold:
            break
        weights.append(next_weight)
        k += 1
        if k > 10_000:
            break
    return np.asarray(weights, dtype=float)


def fractional_difference(values: pl.Series | Sequence[float] | np.ndarray, *, order: float, threshold: float = 1e-5) -> pl.Series:
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if isinstance(values, pl.Series):
        name = values.name
        arr = values.to_numpy().astype(float)
    else:
        name = "fractional_difference"
        arr = np.asarray(values, dtype=float).reshape(-1)
    weights = _fracdiff_weights(order, threshold)
    out = np.full(arr.shape[0], np.nan)
    width = weights.size
    reversed_weights = weights[::-1]
    for idx in range(width - 1, arr.shape[0]):
        window = arr[idx - width + 1 : idx + 1]
        if np.isfinite(window).all():
            out[idx] = float(reversed_weights @ window)
    return pl.Series(name, [None if np.isnan(value) else float(value) for value in out])
