"""
src/agents/noise_trader.py
--------------------------
Noise Trader

Purpose
-------
Generates background order flow. Noise traders have no informational edge;
they trade for exogenous reasons (portfolio rebalancing, hedging, etc.).

Behavior
--------
- Each timestep, independently decides whether to act (controlled by `activity_rate`).
- When active, chooses a random side (BUY/SELL) with equal probability.
- Submits either:
    * A LIMIT order, placed near the current midprice with a random offset.
    * A MARKET order, with probability `market_order_prob`.
- Order size is drawn from a log-normal distribution (fat-tailed, realistic).
- Occasionally cancels its own resting orders to simulate order management.

Role in the simulation
----------------------
Noise traders create the raw bid/ask flow that keeps the book populated.
Without them, informed traders and market makers would have no counterparties.
They are the "dumb money" that generates adverse selection for market makers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from .base_agent import BaseAgent
from ..exchange.order import Order, OrderSide

if TYPE_CHECKING:
    from ..simulation.market_state import MarketState


class NoiseTrader(BaseAgent):
    """
    A noise trader that submits randomized limit and market orders.

    Parameters
    ----------
    agent_id : str
        Unique identifier.
    activity_rate : float
        Probability [0, 1] of acting each timestep.
    market_order_prob : float
        Probability [0, 1] of choosing a market order (vs limit).
    order_size_mean : float
        Mean order size (log-normal).
    order_size_std : float
        Std of the log of order size (log-normal shape parameter).
    limit_offset_ticks : float
        Max absolute offset from mid when placing a limit order (in price units).
    max_resting_orders : int
        Maximum number of resting orders to maintain simultaneously.
        Oldest orders are cancelled when this is exceeded.
    initial_cash : float
    random_seed : int, optional
    """

    def __init__(
        self,
        agent_id: str,
        activity_rate: float = 0.4,
        market_order_prob: float = 0.25,
        order_size_mean: float = 3.0,
        order_size_std: float = 0.8,
        limit_offset_ticks: float = 1.0,
        max_resting_orders: int = 5,
        initial_cash: float = 100_000.0,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(agent_id, initial_cash, random_seed)

        if not 0 < activity_rate <= 1:
            raise ValueError(f"activity_rate must be in (0, 1], got {activity_rate}")
        if not 0 <= market_order_prob <= 1:
            raise ValueError(f"market_order_prob must be in [0, 1], got {market_order_prob}")

        self.activity_rate = activity_rate
        self.market_order_prob = market_order_prob
        self.order_size_mean = order_size_mean
        self.order_size_std = order_size_std
        self.limit_offset_ticks = limit_offset_ticks
        self.max_resting_orders = max_resting_orders

        # Queue of resting order ids (FIFO for cancellation management)
        self._resting: List[str] = []

    def act(self, state: "MarketState") -> List[Order]:
        """
        Generate orders for this timestep.

        Returns a list of Order objects (may be empty).
        Cancellation orders are included when the resting queue is full.
        """
        orders: List[Order] = []

        # Decide whether to act this step
        if self._rng.random() > self.activity_rate:
            return orders

        # Prune oldest resting order if queue is full
        if len(self._resting) >= self.max_resting_orders:
            old_id = self._resting.pop(0)
            cancel = self._cancel_order(old_id)
            if cancel is not None:
                orders.append(cancel)
                self.metrics.orders_cancelled += 1

        # Choose side randomly (50/50)
        side = self._rng.choice([OrderSide.BUY, OrderSide.SELL])

        # Sample order size from log-normal
        raw_size = self._rng.lognormvariate(
            self._rng.gauss(self.order_size_mean, 0.1),
            self.order_size_std
        )
        qty = max(0.01, round(raw_size, 4))

        # Market or limit?
        if self._rng.random() < self.market_order_prob:
            # Market order — only submit if there's a book to hit
            if (side == OrderSide.BUY and state.best_ask is not None) or \
               (side == OrderSide.SELL and state.best_bid is not None):
                orders.append(self._market_order(side, qty))
        else:
            # Limit order — placed near mid with random offset
            mid = state.midprice
            if mid is None:
                # No midprice yet — use a reference price if provided
                mid = state.fair_value if state.fair_value is not None else 100.0

            offset = self._rng.uniform(-self.limit_offset_ticks, self.limit_offset_ticks)
            if side == OrderSide.BUY:
                price = max(0.01, mid - abs(offset))
            else:
                price = mid + abs(offset)

            price = round(price, 4)
            order = self._limit_order(side, qty, price)
            self._resting.append(order.order_id)
            orders.append(order)

        return orders

    def _cancel_order(self, order_id: str) -> Optional[Order]:
        """Build a cancel pseudo-order. The simulation engine will process it."""
        from ..exchange.order import OrderType
        # We use a sentinel order to signal cancellation to the simulation.
        # The simulation calls engine.cancel(order_id) directly.
        # Return None here — the simulation handles cancel separately.
        # We expose the order_id for the simulation to cancel.
        self._pending_cancels: List[str] = getattr(self, "_pending_cancels", [])
        self._pending_cancels.append(order_id)
        return None

    def flush_cancels(self) -> List[str]:
        """Return and clear any order ids this agent wants cancelled."""
        cancels = getattr(self, "_pending_cancels", [])
        self._pending_cancels = []
        return cancels
