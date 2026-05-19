"""
src/models/analytics.py
------------------------
Strategy Comparison Analytics

Computes interview-grade evaluation metrics across multiple market-making
strategies from their per-step records.

Metrics computed
----------------
total_pnl          : final mark-to-market PnL
realized_pnl       : cash-based PnL (closed positions only)
unrealized_pnl     : mark-to-market on open inventory
sharpe_ratio       : mean(pnl_returns) / std(pnl_returns) × sqrt(252)
max_drawdown       : largest peak-to-trough decline in cumulative PnL
inventory_variance : variance of inventory over time
fill_rate          : fills per quote posted
spread_capture     : cumulative spread earned on round trips
avg_half_spread    : mean quoted half-spread (ask-bid)/2
bid_ask_balance    : bid_fills / ask_fills (1.0 = perfectly balanced)
"""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd


def sharpe_ratio(pnl_series: List[float], annualization: float = 252.0) -> float:
    """
    Compute the Sharpe ratio of a PnL time series.

    Uses step-to-step PnL differences as the return series.
    Annualizes assuming `annualization` steps per year.

    Returns 0.0 if standard deviation is zero (flat PnL).
    """
    if len(pnl_series) < 2:
        return 0.0

    returns = [pnl_series[i] - pnl_series[i - 1] for i in range(1, len(pnl_series))]
    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / max(n - 1, 1)
    std_r = math.sqrt(variance)

    if std_r < 1e-12:
        return 0.0

    return (mean_r / std_r) * math.sqrt(annualization)


def max_drawdown(pnl_series: List[float]) -> float:
    """
    Compute the maximum peak-to-trough drawdown in a PnL series.

    Returns a positive number representing the magnitude of the largest
    drawdown (e.g., 15.3 means the portfolio fell 15.3 units from peak).
    Returns 0.0 if pnl_series is empty or monotonically increasing.
    """
    if len(pnl_series) < 2:
        return 0.0

    peak = pnl_series[0]
    max_dd = 0.0
    for val in pnl_series[1:]:
        if val > peak:
            peak = val
        drawdown = peak - val
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def inventory_variance(inventory_series: List[float]) -> float:
    """Unbiased sample variance of inventory over time."""
    n = len(inventory_series)
    if n < 2:
        return 0.0
    mean = sum(inventory_series) / n
    return sum((x - mean) ** 2 for x in inventory_series) / (n - 1)


def strategy_comparison(strategies: Dict[str, object]) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame for multiple strategies.

    Parameters
    ----------
    strategies : dict[str, BaseMarketMaker]
        Mapping of display name → market maker instance (post-simulation).

    Returns
    -------
    pd.DataFrame
        One row per strategy, one column per metric.
    """
    rows = []
    for name, mm in strategies.items():
        m = mm.mm_metrics
        df = m.to_dataframe()

        pnl_hist = m.pnl_history
        inv_hist = m.inventory_history
        quoted_spreads = df["quoted_spread"].dropna().tolist()

        # Bid/ask fill balance
        bid_fills = m.bid_fills
        ask_fills = m.ask_fills
        bid_ask_bal = bid_fills / ask_fills if ask_fills > 0 else float("nan")

        row = {
            "strategy":            name,
            "total_pnl":           round(m.total_pnl, 4),
            "realized_pnl":        round(m.realized_pnl, 4),
            "unrealized_pnl":      round(m.unrealized_pnl, 4),
            "spread_capture":      round(m.spread_capture, 4),
            "sharpe_ratio":        round(sharpe_ratio(pnl_hist), 4),
            "max_drawdown":        round(max_drawdown(pnl_hist), 4),
            "inventory_variance":  round(inventory_variance(inv_hist), 4),
            "final_inventory":     round(m.inventory, 4),
            "fills_as_maker":      m.fills_as_maker,
            "fill_rate":           round(m.fill_rate, 4),
            "bid_fills":           bid_fills,
            "ask_fills":           ask_fills,
            "bid_ask_balance":     round(bid_ask_bal, 4),
            "avg_half_spread":     round(sum(quoted_spreads) / (2 * len(quoted_spreads)), 4)
                                   if quoted_spreads else float("nan"),
            "quotes_posted":       m.quotes_posted,
            "volume_as_maker":     round(m.volume_as_maker, 4),
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("strategy")
    return df


def print_comparison(df: pd.DataFrame) -> None:
    """Pretty-print the strategy comparison table."""
    print("\n" + "═" * 72)
    print("  STRATEGY COMPARISON")
    print("═" * 72)

    metric_labels = {
        "total_pnl":          "Total PnL",
        "realized_pnl":       "Realized PnL",
        "unrealized_pnl":     "Unrealized PnL",
        "spread_capture":     "Spread Capture",
        "sharpe_ratio":       "Sharpe Ratio",
        "max_drawdown":       "Max Drawdown",
        "inventory_variance": "Inventory Variance",
        "final_inventory":    "Final Inventory",
        "fills_as_maker":     "Fills (Maker)",
        "fill_rate":          "Fill Rate",
        "bid_ask_balance":    "Bid/Ask Balance",
        "avg_half_spread":    "Avg Half-Spread",
        "volume_as_maker":    "Volume (Maker)",
    }

    strategies = list(df.index)
    col_w = 14
    header = f"  {'Metric':<24}" + "".join(f"{s:>{col_w}}" for s in strategies)
    print(header)
    print("  " + "─" * (24 + col_w * len(strategies)))

    for col, label in metric_labels.items():
        if col not in df.columns:
            continue
        row_str = f"  {label:<24}"
        for s in strategies:
            val = df.loc[s, col]
            if isinstance(val, float):
                row_str += f"{val:>{col_w}.4f}"
            else:
                row_str += f"{str(val):>{col_w}}"
        print(row_str)

    print("═" * 72)
