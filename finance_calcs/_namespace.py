"""Polars custom namespace registration.

Registers ``.finance`` on ``pl.Expr`` and ``pl.Series`` so all public
calculations can be invoked fluently::

    df.with_columns(pl.col("close").finance.simple_returns().alias("ret"))
    df.select(pl.col("ret").finance.sharpe_ratio())
"""

from __future__ import annotations

import importlib

import polars as pl

_alpha = importlib.import_module(".alpha", __package__)
_factor = importlib.import_module(".factor", __package__)
_momentum = importlib.import_module(".momentum", __package__)
_overlap = importlib.import_module(".overlap", __package__)
_portfolio = importlib.import_module(".portfolio", __package__)
_post_trade = importlib.import_module(".post_trade", __package__)
_quantile = importlib.import_module(".quantile", __package__)
_returns = importlib.import_module(".returns", __package__)
_risk = importlib.import_module(".risk", __package__)
_stats = importlib.import_module(".stats", __package__)
_tail = importlib.import_module(".tail", __package__)
_volatility = importlib.import_module(".volatility", __package__)
_volume = importlib.import_module(".volume", __package__)


def _bind(target_register, modules):
    funcs: dict[str, callable] = {}
    for module in modules:
        for name in module.__all__:
            funcs[name] = getattr(module, name)

    @target_register("finance")
    class _Finance:
        def __init__(self, expr):
            self._expr = expr

    for name, fn in funcs.items():

        def _make(_fn):
            def method(self, *args, **kwargs):
                return _fn(self._expr, *args, **kwargs)

            method.__name__ = _fn.__name__
            method.__doc__ = _fn.__doc__
            return method

        setattr(_Finance, name, _make(fn))
    return _Finance


_MODULES = [
    _returns,
    _risk,
    _overlap,
    _momentum,
    _volatility,
    _volume,
    _alpha,
    _quantile,
    _factor,
    _stats,
    _tail,
    _portfolio,
    _post_trade,
]

ExprFinance = _bind(pl.api.register_expr_namespace, _MODULES)
SeriesFinance = _bind(pl.api.register_series_namespace, _MODULES)
