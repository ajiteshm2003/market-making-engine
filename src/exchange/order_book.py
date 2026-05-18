"""
order_book.py
-------------
Implements the limit order book (LOB).

Structure
---------
- bids : dict[price -> deque[Order]]   sorted descending (best bid first)
- asks : dict[price -> deque[Order]]   sorted ascending  (best ask first)

Each price level is a deque of Orders in FIFO insertion order.
This naturally enforces time priority within a price level.

Responsibilities
----------------
- Insert limit orders into the correct side and price level.
- Remove (cancel) orders by order_id.
- Expose best bid / best ask.
- Expose full depth snapshot.
- Does NOT match orders — matching is the engine's job.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from .order import Order, OrderSide, OrderStatus


class OrderBook:
    """
    A price-time priority limit order book.

    Bids are stored in a dict keyed by price (descending).
    Asks are stored in a dict keyed by price (ascending).
    Within each price level, orders are stored in a deque (FIFO).
    """

    def __init__(self) -> None:
        # price -> deque of Orders at that price
        self._bids: Dict[float, Deque[Order]] = {}
        self._asks: Dict[float, Deque[Order]] = {}
        # Fast lookup: order_id -> Order (for cancellations)
        self._order_map: Dict[str, Order] = {}

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add_limit_order(self, order: Order) -> None:
        """
        Insert a resting limit order into the book.

        The order must already have been validated by the caller.
        This does NOT check for immediate crossing — the engine does that.
        """
        if order.order_id in self._order_map:
            raise ValueError(f"Duplicate order_id: {order.order_id!r}")

        levels = self._bids if order.side == OrderSide.BUY else self._asks
        if order.price not in levels:
            levels[order.price] = deque()
        levels[order.price].append(order)
        self._order_map[order.order_id] = order

    # ------------------------------------------------------------------
    # Removal / Cancellation
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> Optional[Order]:
        """
        Remove an order from the book by its id.

        Returns the cancelled Order, or None if not found / already filled.
        Marks the order status as CANCELLED.
        """
        order = self._order_map.get(order_id)
        if order is None:
            return None
        if not order.is_active:
            return None

        levels = self._bids if order.side == OrderSide.BUY else self._asks
        level = levels.get(order.price)
        if level is not None:
            try:
                level.remove(order)
            except ValueError:
                pass  # already consumed by matching
            if not level:
                del levels[order.price]

        order.status = OrderStatus.CANCELLED
        del self._order_map[order_id]
        return order

    def _remove_filled_order(self, order: Order) -> None:
        """
        Called by the matching engine after an order is fully filled.
        Cleans the order out of its price level and the lookup map.
        """
        levels = self._bids if order.side == OrderSide.BUY else self._asks
        level = levels.get(order.price)
        if level is not None:
            try:
                level.remove(order)
            except ValueError:
                pass
            if not level:
                del levels[order.price]
        self._order_map.pop(order.order_id, None)

    # ------------------------------------------------------------------
    # Best quotes
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        """Highest bid price currently in the book."""
        return max(self._bids.keys()) if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest ask price currently in the book."""
        return min(self._asks.keys()) if self._asks else None

    @property
    def spread(self) -> Optional[float]:
        """Current bid-ask spread. None if either side is empty."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def midprice(self) -> Optional[float]:
        """Midpoint of best bid and best ask."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    # ------------------------------------------------------------------
    # Depth helpers (used by the matching engine)
    # ------------------------------------------------------------------

    def best_bid_orders(self) -> Optional[Deque[Order]]:
        """FIFO queue of orders at the best bid price."""
        bb = self.best_bid
        return self._bids[bb] if bb is not None else None

    def best_ask_orders(self) -> Optional[Deque[Order]]:
        """FIFO queue of orders at the best ask price."""
        ba = self.best_ask
        return self._asks[ba] if ba is not None else None

    # ------------------------------------------------------------------
    # Book snapshot (for visualisation / analytics)
    # ------------------------------------------------------------------

    def depth_snapshot(
        self, levels: int = 10
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Return aggregated depth up to `levels` price levels.

        Returns
        -------
        bids : list of (price, total_quantity) sorted descending
        asks : list of (price, total_quantity) sorted ascending
        """
        bid_prices = sorted(self._bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self._asks.keys())[:levels]

        bids_out = [
            (p, sum(o.remaining_quantity for o in self._bids[p]))
            for p in bid_prices
        ]
        asks_out = [
            (p, sum(o.remaining_quantity for o in self._asks[p]))
            for p in ask_prices
        ]
        return bids_out, asks_out

    def order_imbalance(self, levels: int = 5) -> Optional[float]:
        """
        Volume-weighted order imbalance in [-1, +1].

        +1 → all volume is on the bid side (buy pressure)
        -1 → all volume is on the ask side (sell pressure)
        """
        bids, asks = self.depth_snapshot(levels)
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        total = bid_vol + ask_vol
        if total == 0:
            return None
        return (bid_vol - ask_vol) / total

    def get_order(self, order_id: str) -> Optional[Order]:
        """Look up any resting order by id."""
        return self._order_map.get(order_id)

    def __len__(self) -> int:
        return len(self._order_map)

    def __repr__(self) -> str:
        return (
            f"OrderBook(best_bid={self.best_bid}, best_ask={self.best_ask}, "
            f"spread={self.spread}, resting_orders={len(self)})"
        )
