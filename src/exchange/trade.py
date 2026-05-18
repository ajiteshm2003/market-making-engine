"""
trade.py
--------
Defines the Trade dataclass.

A Trade is an immutable record created every time two orders match.
The matching engine appends these to a trade log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .order import OrderSide


@dataclass(frozen=True)
class Trade:
    """
    Immutable record of a single execution.

    Parameters
    ----------
    trade_id : str
        Unique identifier for this trade.
    timestamp : float
        Unix timestamp of when the match occurred.
    price : float
        The price at which the trade executed.
    quantity : float
        The quantity that changed hands.
    aggressor_side : OrderSide
        The side that crossed the spread (taker).
    maker_order_id : str
        The resting (passive) order that provided liquidity.
    taker_order_id : str
        The aggressive order that consumed liquidity.

    Notes
    -----
    - frozen=True enforces immutability; trade records must never change.
    - aggressor_side BUY  → a buy order lifted the ask.
    - aggressor_side SELL → a sell order hit the bid.
    """

    trade_id: str
    timestamp: float
    price: float
    quantity: float
    aggressor_side: OrderSide
    maker_order_id: str
    taker_order_id: str

    def to_dict(self) -> dict:
        """Serialize to a plain dict (useful for DataFrames)."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "price": self.price,
            "quantity": self.quantity,
            "aggressor_side": self.aggressor_side.value,
            "maker_order_id": self.maker_order_id,
            "taker_order_id": self.taker_order_id,
        }

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.trade_id!r}, price={self.price}, "
            f"qty={self.quantity}, aggressor={self.aggressor_side.value}, "
            f"maker={self.maker_order_id!r}, taker={self.taker_order_id!r})"
        )
