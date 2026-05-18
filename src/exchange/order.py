"""
order.py
--------
Defines the Order dataclass and OrderSide / OrderType enumerations.

Every order that enters the matching engine is represented as one of these.
No logic lives here — this is pure data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


class OrderStatus(Enum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Represents a single order submitted to the exchange.

    Parameters
    ----------
    order_id : str
        Unique identifier for this order.
    side : OrderSide
        BUY or SELL.
    order_type : OrderType
        LIMIT, MARKET, or CANCEL.
    quantity : float
        Total quantity requested.
    price : float, optional
        Limit price. None for market orders.
    timestamp : float, optional
        Unix timestamp (seconds). Auto-filled to now if omitted.

    Notes
    -----
    - `remaining_quantity` tracks how much is left to fill.
    - `status` is updated by the matching engine, not the caller.
    - FIFO priority within a price level is enforced via `timestamp`.
    """

    order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    # Mutable state — set by the engine
    remaining_quantity: float = field(init=False)
    status: OrderStatus = field(default=OrderStatus.OPEN, init=False)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be > 0, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders must specify a price.")
        if self.order_type == OrderType.LIMIT and self.price <= 0:
            raise ValueError(f"Limit price must be > 0, got {self.price}")
        self.remaining_quantity = self.quantity

    @property
    def filled_quantity(self) -> float:
        return self.quantity - self.remaining_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id!r}, side={self.side.value}, "
            f"type={self.order_type.value}, price={self.price}, "
            f"qty={self.quantity}, remaining={self.remaining_quantity:.4f}, "
            f"status={self.status.value})"
        )
