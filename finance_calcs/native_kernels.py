"""Native Rust-backed indicator kernels.

These helpers are the phase-8.7 bridge for high-cost computations
before full Polars plugin-kernel migration.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import finance_calcs as _native

__all__ = ["native_adx", "native_parabolic_sar", "native_garch11_variance"]


def _vec(values: Sequence[float]) -> list[float]:
    return np.asarray(values, dtype=float).reshape(-1).tolist()


def _adx_fallback(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = high.size
    tr = np.zeros(n, dtype=float)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if up > down and up > 0.0 else 0.0
        minus_dm[i] = down if down > up and down > 0.0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    alpha = 1.0 / max(period, 1)
    tr_sm = np.zeros(n, dtype=float)
    plus_sm = np.zeros(n, dtype=float)
    minus_sm = np.zeros(n, dtype=float)
    for i in range(1, n):
        tr_sm[i] = alpha * tr[i] + (1.0 - alpha) * tr_sm[i - 1]
        plus_sm[i] = alpha * plus_dm[i] + (1.0 - alpha) * plus_sm[i - 1]
        minus_sm[i] = alpha * minus_dm[i] + (1.0 - alpha) * minus_sm[i - 1]

    p_di = np.divide(100.0 * plus_sm, tr_sm, out=np.zeros(n), where=tr_sm > 0.0)
    m_di = np.divide(100.0 * minus_sm, tr_sm, out=np.zeros(n), where=tr_sm > 0.0)
    denom = p_di + m_di
    dx = np.divide(100.0 * np.abs(p_di - m_di), denom, out=np.zeros(n), where=denom > 0.0)

    out = np.zeros(n, dtype=float)
    for i in range(1, n):
        out[i] = alpha * dx[i] + (1.0 - alpha) * out[i - 1]
    return out


def _parabolic_sar_fallback(high: np.ndarray, low: np.ndarray, af_step: float, af_max: float) -> np.ndarray:
    n = high.size
    out = np.zeros(n, dtype=float)
    long = True
    af = max(af_step, 1e-6)
    ep = high[0]
    out[0] = low[0]
    for i in range(1, n):
        nxt = out[i - 1] + af * (ep - out[i - 1])
        if long:
            nxt = min(nxt, low[i - 1])
            if low[i] < nxt:
                long = False
                nxt = ep
                ep = low[i]
                af = max(af_step, 1e-6)
            elif high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
        else:
            nxt = max(nxt, high[i - 1])
            if high[i] > nxt:
                long = True
                nxt = ep
                ep = high[i]
                af = max(af_step, 1e-6)
            elif low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
        out[i] = nxt
    return out


def _garch11_fallback(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    n = returns.size
    if n == 0:
        return np.array([], dtype=float)
    out = np.zeros(n, dtype=float)
    prev = float(np.mean(returns * returns))
    if not np.isfinite(prev) or prev <= 0.0:
        prev = 1e-8
    for i in range(n):
        value = omega + alpha * returns[i] * returns[i] + beta * prev
        out[i] = max(value, 0.0)
        prev = out[i]
    return out


def native_adx(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> np.ndarray:
    h = np.asarray(high, dtype=float).reshape(-1)
    low_values = np.asarray(low, dtype=float).reshape(-1)
    c = np.asarray(close, dtype=float).reshape(-1)
    if hasattr(_native, "native_adx"):
        return np.asarray(_native.native_adx(h.tolist(), low_values.tolist(), c.tolist(), int(period)), dtype=float)
    return _adx_fallback(h, low_values, c, int(period))


def native_parabolic_sar(high: Sequence[float], low: Sequence[float], af_step: float = 0.02, af_max: float = 0.2) -> np.ndarray:
    h = np.asarray(high, dtype=float).reshape(-1)
    low_values = np.asarray(low, dtype=float).reshape(-1)
    if hasattr(_native, "native_parabolic_sar"):
        return np.asarray(_native.native_parabolic_sar(h.tolist(), low_values.tolist(), float(af_step), float(af_max)), dtype=float)
    return _parabolic_sar_fallback(h, low_values, float(af_step), float(af_max))


def native_garch11_variance(
    returns: Sequence[float],
    *,
    omega: float = 1e-6,
    alpha: float = 0.1,
    beta: float = 0.85,
) -> np.ndarray:
    r = np.asarray(returns, dtype=float).reshape(-1)
    if hasattr(_native, "native_garch11_variance"):
        return np.asarray(_native.native_garch11_variance(r.tolist(), float(omega), float(alpha), float(beta)), dtype=float)
    return _garch11_fallback(r, float(omega), float(alpha), float(beta))
