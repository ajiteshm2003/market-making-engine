"""
matching_engine.py
------------------
The core matching engine.

Responsibilities
----------------
- Accept incoming orders (limit, market, cancel).
- Run price-time priority matching.
- Produce Trade records for every execution.
- Maintain the OrderBook state.
- Expose a trade log and order registry.

Matching Rules
--------------
1. Market BUY  → hits lowest ask (ascending ask prices, FIFO within level).
2. Market SELL → hits highest bid (descending bid prices, FIFO within level).
3. Limit BUY   → matches if limit_price >= best_ask; else rests in book.
4. Limit SELL  → matches if limit_price <= best_bid; else rests in book.
5. Within a price level, orders are matched in FIFO (insertion-time) order.
6. Partial fills are supported; a partially filled order stays in the book.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional

from .order import Order, OrderSide, OrderStatus, OrderType
from .order_book import OrderBook
from .trade import Trade


class MatchingEngine:
    """
    Price-time priority continuous matching engine.

    Usage
    -----
    >>> engine = MatchingEngine()
    >>> order = Order("o1", OrderSide.BUY, OrderType.LIMIT, quantity=10, price=100.0)
    >>> trades = engine.submit(order)
    """

    def __init__(self) -> None:
        self.book = OrderBook()
        self._trade_log: List[Trade] = []
        self._order_registry: Dict[str, Order] = {}
        self._trade_counter = itertools.count(1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> List[Trade]:
        """
        Submit an order to the engine.

        Parameters
        ----------
        order : Order
            A fully constructed Order object.

        Returns
        -------
        trades : list[Trade]
            All trades produced by this submission (may be empty).
        """
        if order.order_type == OrderType.CANCEL:
            self._process_cancel(order.order_id)
            return []

        if order.order_id in self._order_registry:
            raise ValueError(f"Duplicate order_id submitted: {order.order_id!r}")

        self._order_registry[order.order_id] = order

        if order.order_type == OrderType.MARKET:
            return self._match_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            return self._match_limit_order(order)
        else:
            raise ValueError(f"Unknown order type: {order.order_type}")

    def cancel(self, order_id: str) -> Optional[Order]:
        """
        Cancel a resting order by its order_id.

        Returns the cancelled Order, or None if not found.
        """
        return self._process_cancel(order_id)

    @property
    def trade_log(self) -> List[Trade]:
        """All trades in chronological order."""
        return list(self._trade_log)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve any order ever submitted by id."""
        return self._order_registry.get(order_id)

    # ------------------------------------------------------------------
    # Internal matching logic
    # ------------------------------------------------------------------

    def _match_market_order(self, taker: Order) -> List[Trade]:
        """
        Match a market order against the opposing side of the book.

        A market order has no price limit — it matches until filled
        or the book is exhausted (in which case the remainder is rejected).
        """
        trades: List[Trade] = []

        if taker.side == OrderSide.BUY:
            trades = self._consume_asks(taker, price_limit=None)
        else:
            trades = self._consume_bids(taker, price_limit=None)

        # Any unfilled remainder is rejected (no resting market orders)
        if taker.remaining_quantity > 0:
            taker.status = OrderStatus.REJECTED

        return trades

    def _match_limit_order(self, taker: Order) -> List[Trade]:
        """
        Match a limit order.

        First tries to execute against the book (aggressive).
        Any unfilled remainder rests in the book (passive).
        """
        trades: List[Trade] = []

        if taker.side == OrderSide.BUY:
            # A buy limit order matches asks up to taker.price
            trades = self._consume_asks(taker, price_limit=taker.price)
        else:
            # A sell limit order matches bids down to taker.price
            trades = self._consume_bids(taker, price_limit=taker.price)

        # If unfilled quantity remains, rest it in the book
        if taker.remaining_quantity > 0 and taker.is_active:
            taker.status = OrderStatus.OPEN
            self.book.add_limit_order(taker)

        return trades

    def _consume_asks(
        self, taker: Order, price_limit: Optional[float]
    ) -> List[Trade]:
        """
        Walk the ask side (ascending price) and fill the taker.

        price_limit=None  → market order, no cap.
        price_limit=X     → limit order, stop if best_ask > X.
        """
        trades: List[Trade] = []

        while taker.remaining_quantity > 0:
            best_ask = self.book.best_ask
            if best_ask is None:
                break
            if price_limit is not None and best_ask > price_limit:
                break

            ask_queue = self.book.best_ask_orders()
            if not ask_queue:
                break

            maker = ask_queue[0]  # FIFO: front of queue
            trade = self._execute(maker, taker, best_ask)
            trades.append(trade)

        return trades

    def _consume_bids(
        self, taker: Order, price_limit: Optional[float]
    ) -> List[Trade]:
        """
        Walk the bid side (descending price) and fill the taker.

        price_limit=None  → market order, no cap.
        price_limit=X     → limit order, stop if best_bid < X.
        """
        trades: List[Trade] = []

        while taker.remaining_quantity > 0:
            best_bid = self.book.best_bid
            if best_bid is None:
                break
            if price_limit is not None and best_bid < price_limit:
                break

            bid_queue = self.book.best_bid_orders()
            if not bid_queue:
                break

            maker = bid_queue[0]  # FIFO: front of queue
            trade = self._execute(maker, taker, best_bid)
            trades.append(trade)

        return trades

    def _execute(self, maker: Order, taker: Order, exec_price: float) -> Trade:
        """
        Execute one maker-taker pair for the maximum available quantity.

        Handles partial fills on both sides.
        Updates order statuses and cleans fully filled makers from the book.
        """
        fill_qty = min(maker.remaining_quantity, taker.remaining_quantity)

        # Deduct from both orders
        maker.remaining_quantity -= fill_qty
        taker.remaining_quantity -= fill_qty

        # Update statuses
        if maker.remaining_quantity == 0:
            maker.status = OrderStatus.FILLED
            self.book._remove_filled_order(maker)
        else:
            maker.status = OrderStatus.PARTIALLY_FILLED

        if taker.remaining_quantity == 0:
            taker.status = OrderStatus.FILLED
        else:
            taker.status = OrderStatus.PARTIALLY_FILLED

        trade = Trade(
            trade_id=f"T{next(self._trade_counter):08d}",
            timestamp=time.time(),
            price=exec_price,
            quantity=fill_qty,
            aggressor_side=taker.side,
            maker_order_id=maker.order_id,
            taker_order_id=taker.order_id,
        )
        self._trade_log.append(trade)
        return trade

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def _process_cancel(self, order_id: str) -> Optional[Order]:
        cancelled = self.book.cancel_order(order_id)
        return cancelled

    # ------------------------------------------------------------------
    # Convenience / diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable engine summary."""
        return (
            f"MatchingEngine | "
            f"book={self.book} | "
            f"total_trades={len(self._trade_log)} | "
            f"total_orders_seen={len(self._order_registry)}"
        )
