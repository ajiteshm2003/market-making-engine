"""
trade_log.py
------------
Utility layer around the raw trade list.

Provides:
- DataFrame conversion for analysis
- Basic execution statistics
- Spread / midprice time series reconstruction

This is PHASE 1 only. Full analytics (Sharpe, adverse selection, etc.)
will be implemented in later phases.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .trade import Trade


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """
    Convert a list of Trade objects into a tidy pandas DataFrame.

    Columns
    -------
    trade_id, timestamp, price, quantity, aggressor_side,
    maker_order_id, taker_order_id
    """
    if not trades:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "timestamp",
                "price",
                "quantity",
                "aggressor_side",
                "maker_order_id",
                "taker_order_id",
            ]
        )

    df = pd.DataFrame([t.to_dict() for t in trades])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def execution_summary(trades: List[Trade]) -> dict:
    """
    Return a dict of basic execution statistics from a trade list.

    Keys
    ----
    total_trades        : int
    total_volume        : float
    vwap                : float   — volume-weighted average price
    min_price           : float
    max_price           : float
    buy_initiated_pct   : float   — fraction where aggressor = BUY
    sell_initiated_pct  : float
    """
    if not trades:
        return {}

    df = trades_to_dataframe(trades)

    total_vol = df["quantity"].sum()
    vwap = (df["price"] * df["quantity"]).sum() / total_vol if total_vol > 0 else float("nan")
    buy_pct = (df["aggressor_side"] == "buy").mean() * 100
    sell_pct = (df["aggressor_side"] == "sell").mean() * 100

    return {
        "total_trades": len(df),
        "total_volume": total_vol,
        "vwap": round(vwap, 6),
        "min_price": df["price"].min(),
        "max_price": df["price"].max(),
        "buy_initiated_pct": round(buy_pct, 2),
        "sell_initiated_pct": round(sell_pct, 2),
    }
