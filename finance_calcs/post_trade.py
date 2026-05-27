"""Post-trade transaction cost metrics and trade-quality helpers."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from statistics import median
from typing import Any

import polars as pl

from ._periods import PeriodLike, _bucket_or_none

__all__ = [
    "transaction_notional",
    "transaction_cost",
    "transaction_volume",
    "slippage_bps",
    "implementation_shortfall",
    "vwap_slippage",
    "turnover",
    "cost_per_trade",
    "cost_attribution",
    "extract_round_trips",
    "round_trip_stats",
    "long_short_round_trip_stats",
    "sector_round_trip_stats",
    "win_rate",
    "profit_factor",
    "payoff_ratio",
    "avg_trade_pnl",
    "trade_duration_stats",
    "mae_mfe",
    "consecutive_wins_losses",
    "exit_reason_stats",
    "trade_size_return_correlation",
]

__finance_namespace__ = [
    "transaction_notional",
    "transaction_cost",
    "transaction_volume",
    "slippage_bps",
    "implementation_shortfall",
    "vwap_slippage",
    "turnover",
    "cost_per_trade",
    "win_rate",
    "profit_factor",
    "payoff_ratio",
    "avg_trade_pnl",
    "trade_size_return_correlation",
]


def _expr(value: float | pl.Expr) -> pl.Expr:
    if isinstance(value, pl.Expr):
        return value
    return pl.lit(value)


def transaction_notional(quantity: pl.Expr, price: pl.Expr) -> pl.Expr:
    """Absolute traded notional, ``abs(quantity) * price``."""
    return quantity.abs() * price


def transaction_cost(
    quantity: pl.Expr,
    price: pl.Expr,
    *,
    commission: float | pl.Expr = 0.0,
    fees: float | pl.Expr = 0.0,
    bps: float | pl.Expr = 0.0,
) -> pl.Expr:
    """Per-trade cost from explicit charges plus basis-point slippage.

    ``bps`` is applied to absolute traded notional. ``commission`` and
    ``fees`` may be scalars or expressions aligned to the transaction rows.
    """
    notional = transaction_notional(quantity, price)
    return notional * (_expr(bps) / 10_000.0) + _expr(commission) + _expr(fees)


def transaction_volume(
    quantity: pl.Expr,
    price: pl.Expr,
    *,
    period: PeriodLike | None = None,
    date: pl.Expr | None = None,
) -> pl.Expr:
    """Absolute traded notional, summed over the full sample or period."""
    volume = transaction_notional(quantity, price).sum()
    bucket = _bucket_or_none(date, period)
    if bucket is None:
        return volume
    return volume.over(bucket)


def slippage_bps(
    execution_price: pl.Expr,
    benchmark_price: pl.Expr,
    *,
    side: pl.Expr | None = None,
) -> pl.Expr:
    """Execution slippage in basis points.

    Without ``side``, the result is signed price difference versus the
    benchmark. With ``side``, positive values mean adverse execution cost
    for buy/cover and sell/short transactions.
    """
    raw = (execution_price - benchmark_price) / benchmark_price * 10_000.0
    if side is None:
        return raw

    side_label = side.cast(pl.Utf8).str.to_lowercase()
    return pl.when(side_label.is_in(["buy", "cover"])).then(raw).when(side_label.is_in(["sell", "short"])).then(-raw).otherwise(None)


def implementation_shortfall(
    execution_price: pl.Expr,
    decision_price: pl.Expr,
    *,
    side: pl.Expr | None = None,
) -> pl.Expr:
    """Side-aware execution slippage versus the decision price."""
    return slippage_bps(execution_price, decision_price, side=side)


def vwap_slippage(
    execution_price: pl.Expr,
    vwap: pl.Expr,
    *,
    side: pl.Expr | None = None,
) -> pl.Expr:
    """Side-aware execution slippage versus VWAP."""
    return slippage_bps(execution_price, vwap, side=side)


def turnover(weights: pl.Expr, *, window: int | None = None) -> pl.Expr:
    """Portfolio turnover contribution from position-weight changes.

    Apply over a symbol/security partition, then aggregate by rebalance
    date. The contribution is ``0.5 * abs(weight - prior_weight)``.
    """
    contribution = 0.5 * weights.diff().abs()
    if window is None:
        return contribution
    return contribution.rolling_sum(window)


def cost_per_trade(
    quantity: pl.Expr,
    price: pl.Expr,
    *,
    commission: float | pl.Expr = 0.0,
    fees: float | pl.Expr = 0.0,
    bps: float | pl.Expr = 0.0,
) -> pl.Expr:
    """Alias for per-trade transaction cost."""
    return transaction_cost(quantity, price, commission=commission, fees=fees, bps=bps)


def _optional_col(frame: pl.DataFrame, name: str, default: float = 0.0) -> pl.Expr:
    if name in frame.columns:
        return pl.col(name)
    return pl.lit(default)


def cost_attribution(
    transactions: pl.DataFrame,
    *,
    quantity_col: str = "amount",
    price_col: str = "price",
    commission_col: str = "commission",
    fees_col: str = "fees",
    bps_col: str = "bps",
    spread_bps_col: str = "spread_bps",
    market_impact_bps_col: str = "market_impact_bps",
    slippage_component: str = "slippage",
) -> pl.DataFrame:
    """Summarize transaction costs by component."""
    notional = transaction_notional(pl.col(quantity_col), pl.col(price_col))
    components = transactions.select(
        _optional_col(transactions, commission_col).sum().alias("commission"),
        _optional_col(transactions, fees_col).sum().alias("fees"),
        (notional * (_optional_col(transactions, bps_col) / 10_000.0)).sum().alias(slippage_component),
        (notional * (_optional_col(transactions, spread_bps_col) / 10_000.0)).sum().alias("spread"),
        (notional * (_optional_col(transactions, market_impact_bps_col) / 10_000.0)).sum().alias("market_impact"),
    )
    rows = [{"component": name, "total": float(components[name][0] or 0.0)} for name in components.columns]
    rows = [row for row in rows if row["total"] != 0.0 or row["component"] in {"commission", "fees", slippage_component}]
    total = sum(row["total"] for row in rows)
    for row in rows:
        row["pct_total"] = row["total"] / total if total else float("nan")
    return pl.DataFrame(rows)


def _duration(entry: Any, exit_: Any) -> float:
    delta = exit_ - entry
    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds() / 86_400.0)
    if hasattr(delta, "days"):
        return float(delta.days)
    return float(delta)


def _empty_round_trips() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [],
            "side": [],
            "entry_timestamp": [],
            "exit_timestamp": [],
            "quantity": [],
            "entry_price": [],
            "exit_price": [],
            "pnl": [],
            "return": [],
            "duration": [],
        }
    )


def extract_round_trips(
    transactions: pl.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    quantity_col: str = "amount",
    price_col: str = "price",
) -> pl.DataFrame:
    """Extract FIFO round trips from signed transaction quantities."""
    if transactions.is_empty():
        return _empty_round_trips()

    rows: list[dict[str, Any]] = []
    open_lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for tx in transactions.sort(symbol_col, timestamp_col).iter_rows(named=True):
        symbol = tx[symbol_col]
        signed_quantity = float(tx[quantity_col])
        price = float(tx[price_col])
        timestamp = tx[timestamp_col]
        action_side = "long" if signed_quantity > 0 else "short"
        remaining = abs(signed_quantity)
        opposite = "short" if action_side == "long" else "long"
        lots = open_lots[symbol]

        while remaining > 0 and lots and lots[0]["side"] == opposite:
            lot = lots[0]
            close_quantity = min(remaining, lot["quantity"])
            if lot["side"] == "long":
                pnl = (price - lot["price"]) * close_quantity
            else:
                pnl = (lot["price"] - price) * close_quantity
            denominator = lot["price"] * close_quantity
            rows.append(
                {
                    "symbol": symbol,
                    "side": lot["side"],
                    "entry_timestamp": lot["timestamp"],
                    "exit_timestamp": timestamp,
                    "quantity": close_quantity,
                    "entry_price": lot["price"],
                    "exit_price": price,
                    "pnl": pnl,
                    "return": pnl / denominator if denominator else float("nan"),
                    "duration": _duration(lot["timestamp"], timestamp),
                }
            )
            lot["quantity"] -= close_quantity
            remaining -= close_quantity
            if lot["quantity"] <= 1e-12:
                lots.popleft()

        if remaining > 0:
            lots.append({"side": action_side, "quantity": remaining, "price": price, "timestamp": timestamp})

    if not rows:
        return _empty_round_trips()
    return pl.DataFrame(rows)


def win_rate(pnl: pl.Expr) -> pl.Expr:
    """Fraction of profitable trades."""
    return (pnl > 0).cast(pl.Float64).mean()


def profit_factor(pnl: pl.Expr) -> pl.Expr:
    """Gross profit divided by absolute gross loss."""
    gross_profit = pnl.filter(pnl > 0).sum()
    gross_loss = -pnl.filter(pnl < 0).sum()
    return gross_profit / gross_loss


def payoff_ratio(pnl: pl.Expr) -> pl.Expr:
    """Average winning trade divided by absolute average losing trade."""
    avg_win = pnl.filter(pnl > 0).mean()
    avg_loss = -pnl.filter(pnl < 0).mean()
    return avg_win / avg_loss


def avg_trade_pnl(pnl: pl.Expr) -> pl.Expr:
    """Mean trade PnL."""
    return pnl.mean()


def round_trip_stats(round_trips: pl.DataFrame, *, pnl_col: str = "pnl") -> dict[str, float | int]:
    """Summary statistics for extracted round trips."""
    if round_trips.is_empty():
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "avg_pnl": float("nan"),
            "total_pnl": 0.0,
            "profit_factor": float("nan"),
            "payoff_ratio": float("nan"),
        }
    out = round_trips.select(
        pl.len().alias("n_trades"),
        win_rate(pl.col(pnl_col)).alias("win_rate"),
        avg_trade_pnl(pl.col(pnl_col)).alias("avg_pnl"),
        pl.col(pnl_col).sum().alias("total_pnl"),
        profit_factor(pl.col(pnl_col)).alias("profit_factor"),
        payoff_ratio(pl.col(pnl_col)).alias("payoff_ratio"),
    ).row(0, named=True)
    return {key: int(value) if key == "n_trades" else float(value) for key, value in out.items()}


def long_short_round_trip_stats(round_trips: pl.DataFrame, *, side_col: str = "side", pnl_col: str = "pnl") -> pl.DataFrame:
    """Round-trip statistics split by long and short trades."""
    if round_trips.is_empty():
        return pl.DataFrame({side_col: [], "n_trades": [], "total_pnl": [], "win_rate": [], "avg_pnl": []})
    return round_trips.group_by(side_col).agg(
        pl.len().alias("n_trades"),
        pl.col(pnl_col).sum().alias("total_pnl"),
        win_rate(pl.col(pnl_col)).alias("win_rate"),
        avg_trade_pnl(pl.col(pnl_col)).alias("avg_pnl"),
    )


def sector_round_trip_stats(
    round_trips: pl.DataFrame,
    sector_map: Mapping[str, str],
    *,
    symbol_col: str = "symbol",
    pnl_col: str = "pnl",
) -> pl.DataFrame:
    """Round-trip statistics by sector."""
    if round_trips.is_empty():
        return pl.DataFrame({"sector": [], "n_trades": [], "total_pnl": [], "win_rate": [], "avg_pnl": []})
    sectors = [sector_map.get(symbol, "Unknown") for symbol in round_trips[symbol_col].to_list()]
    frame = round_trips.with_columns(pl.Series("sector", sectors))
    return frame.group_by("sector").agg(
        pl.len().alias("n_trades"),
        pl.col(pnl_col).sum().alias("total_pnl"),
        win_rate(pl.col(pnl_col)).alias("win_rate"),
        avg_trade_pnl(pl.col(pnl_col)).alias("avg_pnl"),
    )


def trade_duration_stats(duration: Iterable[Any]) -> dict[str, float]:
    """Mean, median, and maximum holding duration."""
    values = [float(value) for value in duration if value is not None]
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"mean": sum(values) / len(values), "median": float(median(values)), "max": max(values)}


def mae_mfe(
    trades: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    price_col: str = "price",
) -> pl.DataFrame:
    """Attach maximum adverse and favorable excursion to round trips."""
    out: list[dict[str, Any]] = []
    for trade in trades.iter_rows(named=True):
        symbol = trade[symbol_col]
        entry_time = trade["entry_timestamp"]
        exit_time = trade["exit_timestamp"]
        entry_price = float(trade["entry_price"])
        side = str(trade["side"]).lower()
        window = prices.filter((pl.col(symbol_col) == symbol) & (pl.col(timestamp_col) >= entry_time) & (pl.col(timestamp_col) <= exit_time)).sort(
            timestamp_col
        )
        if window.is_empty() or entry_price == 0.0:
            mae = float("nan")
            mfe = float("nan")
        else:
            price_values = window[price_col].cast(pl.Float64).to_numpy()
            if side == "short":
                excursions = (entry_price - price_values) / entry_price
            else:
                excursions = (price_values - entry_price) / entry_price
            mae = float(min(excursions))
            mfe = float(max(excursions))
        out.append({**trade, "mae": mae, "mfe": mfe})
    return pl.DataFrame(out) if out else trades.with_columns(pl.lit(math.nan).alias("mae"), pl.lit(math.nan).alias("mfe"))


def consecutive_wins_losses(pnl: Iterable[Any]) -> dict[str, int]:
    """Maximum consecutive winning and losing trade counts."""
    max_wins = max_losses = wins = losses = 0
    for value in pnl:
        if value is None:
            continue
        if float(value) > 0:
            wins += 1
            losses = 0
        elif float(value) < 0:
            losses += 1
            wins = 0
        else:
            wins = 0
            losses = 0
        max_wins = max(max_wins, wins)
        max_losses = max(max_losses, losses)
    return {"max_consecutive_wins": max_wins, "max_consecutive_losses": max_losses}


def exit_reason_stats(
    trades: pl.DataFrame,
    *,
    reason_col: str = "exit_reason",
    pnl_col: str = "pnl",
) -> pl.DataFrame:
    """PnL and counts grouped by exit reason."""
    aggregations = [pl.len().alias("count")]
    if pnl_col in trades.columns:
        aggregations.extend([pl.col(pnl_col).sum().alias("total_pnl"), pl.col(pnl_col).mean().alias("avg_pnl")])
    return trades.group_by(reason_col).agg(*aggregations).sort(reason_col)


def trade_size_return_correlation(size: pl.Expr, returns: pl.Expr) -> pl.Expr:
    """Correlation between absolute trade size and trade return."""
    return pl.corr(size.abs(), returns)
