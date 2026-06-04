from __future__ import annotations

import numpy as np

import finance_calcs as fc


def test_native_adx_shape_and_bounds() -> None:
    high = np.array([10.0, 10.4, 10.6, 10.3, 10.8, 11.0, 10.9, 11.2])
    low = np.array([9.8, 10.1, 10.2, 10.0, 10.4, 10.6, 10.5, 10.9])
    close = np.array([9.9, 10.2, 10.5, 10.1, 10.7, 10.8, 10.6, 11.0])

    out = fc.native_adx(high, low, close, period=3)

    assert out.shape == high.shape
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 100.0))


def test_native_parabolic_sar_shape() -> None:
    high = np.array([10.0, 10.4, 10.6, 10.3, 10.8, 11.0])
    low = np.array([9.8, 10.1, 10.2, 10.0, 10.4, 10.6])

    out = fc.native_parabolic_sar(high, low)

    assert out.shape == high.shape
    assert np.all(np.isfinite(out))


def test_native_garch11_variance_non_negative() -> None:
    returns = np.array([0.01, -0.015, 0.008, -0.004, 0.011])

    out = fc.native_garch11_variance(returns)

    assert out.shape == returns.shape
    assert np.all(out >= 0.0)
