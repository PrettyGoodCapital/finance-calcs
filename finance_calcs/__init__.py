"""finance-calcs — financial calculations as polars expressions.

Every public function accepts and returns ``pl.Expr``. Functions are
also exposed on a ``.finance`` custom namespace bound to ``pl.Expr``
and ``pl.Series``::

    import finance_calcs  # noqa: F401  — registers the namespace
    import polars as pl

    df.with_columns(
        pl.col("close").finance.log_returns().alias("ret"),
    ).select(
        pl.col("ret").finance.sharpe().alias("sharpe"),
        pl.col("ret").finance.max_drawdown().alias("mdd"),
    )
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import _namespace  # noqa: F401  — side-effect: register polars namespace
from .alpha import (  # noqa: F401
    forward_returns,
    hit_rate,
    ic_ir,
    ic_summary_stats,
    information_coefficient,
    pearson_ic,
    spearman_ic,
)
from .factor import (  # noqa: F401
    alpha,
    batting_average,
    beta,
    down_alpha,
    down_beta,
    down_capture,
    information_ratio,
    tracking_error,
    up_alpha,
    up_beta,
    up_capture,
    up_down_capture,
)
from .momentum import (  # noqa: F401
    adx,
    cci,
    cmo,
    macd_hist,
    macd_line,
    macd_signal,
    minus_di,
    minus_dm,
    mom,
    plus_di,
    plus_dm,
    roc,
    rocp,
    rocr,
    rocr100,
    rsi,
    stoch_d,
    stoch_k,
    trix,
    willr,
)
from .overlap import (  # noqa: F401
    bbands_lower,
    bbands_middle,
    bbands_upper,
    dema,
    donchian_lower,
    donchian_middle,
    donchian_upper,
    ema,
    midpoint,
    midprice,
    sma,
    tema,
    wma,
)
from .portfolio import (  # noqa: F401
    active_share,
    concentration,
    gross_exposure,
    gross_leverage,
    long_exposure,
    net_exposure,
    short_exposure,
    top_n_concentration,
)
from .post_trade import (  # noqa: F401
    slippage_bps,
    transaction_cost,
    transaction_notional,
    turnover,
)
from .quantile import (  # noqa: F401
    assign_quantile,
    long_short_spread,
    quantile_changed,
    rank_normalize,
    winsorize,
    zscore,
)
from .returns import (  # noqa: F401
    aggregate_returns,
    annualized_return,
    annualized_volatility,
    cum_returns,
    cum_returns_final,
    log_returns,
    period_bucket,
    returns,
    simple_returns,
)
from .risk import (  # noqa: F401
    calmar,
    conditional_value_at_risk,
    downside_deviation,
    downside_risk,
    drawdown_series,
    max_drawdown,
    parametric_var,
    sharpe,
    sortino,
    underwater_series,
    value_at_risk,
    volatility,
)
from .stats import (  # noqa: F401
    common_sense_ratio,
    deflated_sharpe,
    higher_moments,
    kurtosis,
    minimum_track_record_length,
    probabilistic_sharpe,
    sharpe_ci_bootstrap,
    sharpe_with_ci,
    skewness,
    stability_of_timeseries,
)
from .tail import (  # noqa: F401
    gpd_cvar,
    gpd_var,
    omega_ratio,
    tail_ratio,
    ulcer_index,
)
from .volatility import (  # noqa: F401
    atr,
    ewma_vol,
    garman_klass_vol,
    natr,
    parkinson_vol,
    realized_vol,
    rogers_satchell_vol,
    true_range,
    yang_zhang_vol,
)
from .volume import ad, adosc, obv  # noqa: F401

__all__ = [
    # returns
    "period_bucket",
    "simple_returns",
    "log_returns",
    "cum_returns",
    "cum_returns_final",
    "returns",
    "aggregate_returns",
    "annualized_return",
    "annualized_volatility",
    # risk
    "volatility",
    "sharpe",
    "sortino",
    "calmar",
    "downside_risk",
    "downside_deviation",
    "drawdown_series",
    "underwater_series",
    "max_drawdown",
    "value_at_risk",
    "conditional_value_at_risk",
    "parametric_var",
    # overlap
    "sma",
    "ema",
    "wma",
    "dema",
    "tema",
    "midpoint",
    "midprice",
    "bbands_upper",
    "bbands_middle",
    "bbands_lower",
    "donchian_upper",
    "donchian_lower",
    "donchian_middle",
    # momentum
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "mom",
    "roc",
    "rocp",
    "rocr",
    "rocr100",
    "willr",
    "stoch_k",
    "stoch_d",
    "cci",
    "cmo",
    "trix",
    "plus_dm",
    "minus_dm",
    "plus_di",
    "minus_di",
    "adx",
    # volatility
    "true_range",
    "atr",
    "natr",
    "parkinson_vol",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "yang_zhang_vol",
    "ewma_vol",
    "realized_vol",
    # volume
    "obv",
    "ad",
    "adosc",
    # alpha
    "forward_returns",
    "pearson_ic",
    "spearman_ic",
    "information_coefficient",
    "ic_ir",
    "hit_rate",
    "ic_summary_stats",
    # quantile
    "assign_quantile",
    "rank_normalize",
    "zscore",
    "winsorize",
    "long_short_spread",
    "quantile_changed",
    # factor / capture
    "alpha",
    "beta",
    "up_alpha",
    "down_alpha",
    "up_beta",
    "down_beta",
    "up_capture",
    "down_capture",
    "up_down_capture",
    "batting_average",
    "tracking_error",
    "information_ratio",
    # stats / validity
    "skewness",
    "kurtosis",
    "higher_moments",
    "stability_of_timeseries",
    "common_sense_ratio",
    "probabilistic_sharpe",
    "deflated_sharpe",
    "minimum_track_record_length",
    "sharpe_ci_bootstrap",
    "sharpe_with_ci",
    # tail risk
    "tail_ratio",
    "ulcer_index",
    "omega_ratio",
    "gpd_var",
    "gpd_cvar",
    # portfolio
    "gross_leverage",
    "gross_exposure",
    "net_exposure",
    "long_exposure",
    "short_exposure",
    "concentration",
    "top_n_concentration",
    "active_share",
    # post-trade
    "transaction_notional",
    "transaction_cost",
    "slippage_bps",
    "turnover",
]
