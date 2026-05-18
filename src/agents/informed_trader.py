"""
src/agents/informed_trader.py
------------------------------
Informed Trader

Purpose
-------
Models agents who possess private information about the true fair value
of the asset. They are the source of ADVERSE SELECTION in the simulation.

When a market maker quotes to an informed trader, the market maker loses
money because the informed trader knows where prices are headed.

Behavior
--------
- The simulation maintains a latent fair value (true_price) that evolves
  via a random walk with occasional jumps.
- Each timestep, the informed trader observes the DEVIATION between the
  fair value and the current best bid/ask midprice.
- If deviation > threshold: trade aggressively toward fair value.
- They submit MARKET orders (maximum urgency) or aggressive LIMIT orders.
- Position limits prevent runaway directional exposure.

Key parameters
--------------
signal_threshold : float
    Minimum |fair_value - midprice| required to act (in price units).
    Acts as a filter for noise in the signal.
aggression : float [0, 1]
    0 = always uses limit orders at mid
    1 = always uses market orders (most aggressive)
trade_size : float
    Base quantity per trade. Scaled by conviction (deviation / threshold).
max_inventory : float
    Soft inventory limit. Reduces trade size as position grows.

Role in the simulation
----------------------
Informed traders are the mechanism through which prices move toward fair
value. Their presence is what makes market making dangerous — a market
maker who doesn't account for informed flow will systematically lose money.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

from .base_agent import BaseAgent
from ..exchange.order import Order, OrderSide

if TYPE_CHECKING:
    from ..simulation.market_state import MarketState


class InformedTrader(BaseAgent):
    """
    An informed trader who acts on a latent fair-value signal.

    Parameters
    ----------
    agent_id : str
    signal_threshold : float
        Minimum deviation from fair value required to trigger a trade.
    aggression : float
        Mix of market (1.0) vs limit (0.0) orders.
    base_trade_size : float
        Base quantity. Actual size scales with conviction.
    max_inventory : float
        Soft position cap — trade size shrinks as this is approached.
    activity_rate : float
        Probability of checking the signal each timestep.
    initial_cash : float
    random_seed : int, optional
    """

    def __init__(
        self,
        agent_id: str,
        signal_threshold: float = 0.10,
        aggression: float = 0.70,
        base_trade_size: float = 5.0,
        max_inventory: float = 50.0,
        activity_rate: float = 0.60,
        initial_cash: float = 100_000.0,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(agent_id, initial_cash, random_seed)

        if signal_threshold <= 0:
            raise ValueError("signal_threshold must be > 0")
        if not 0 <= aggression <= 1:
            raise ValueError("aggression must be in [0, 1]")

        self.signal_threshold = signal_threshold
        self.aggression = aggression
        self.base_trade_size = base_trade_size
        self.max_inventory = max_inventory
        self.activity_rate = activity_rate

        # Internal state
        self._last_signal: Optional[float] = None  # last observed deviation
        self._trades_this_session: int = 0

    def act(self, state: "MarketState") -> List[Order]:
        """
        Check the signal and trade if the deviation is large enough.

        Returns a list with 0 or 1 orders.
        """
        orders: List[Order] = []

        # Stochastic activity: informed traders don't act every tick
        if self._rng.random() > self.activity_rate:
            return orders

        # Need a fair value and a midprice to compute deviation
        fair_value = state.fair_value
        mid = state.midprice

        if fair_value is None or mid is None:
            return orders

        deviation = fair_value - mid
        self._last_signal = deviation

        # Signal too weak — wait
        if abs(deviation) < self.signal_threshold:
            return orders

        # Determine direction to trade
        if deviation > 0:
            # Fair value above mid → price should rise → BUY
            side = OrderSide.BUY
        else:
            # Fair value below mid → price should fall → SELL
            side = OrderSide.SELL

        # Conviction scales with how far deviation exceeds threshold
        conviction = min(abs(deviation) / self.signal_threshold, 3.0)  # cap at 3x

        # Scale size by conviction, then apply inventory penalty
        inv_penalty = self._inventory_penalty()
        qty = max(0.01, self.base_trade_size * conviction * inv_penalty)
        qty = round(qty, 4)

        # Aggressive (market) or slightly patient (limit)?
        if self._rng.random() < self.aggression:
            # Market order — only if there's depth to hit
            if side == OrderSide.BUY and state.best_ask is not None:
                orders.append(self._market_order(side, qty))
            elif side == OrderSide.SELL and state.best_bid is not None:
                orders.append(self._market_order(side, qty))
        else:
            # Aggressive limit — price at or through the best quote
            if side == OrderSide.BUY and state.best_ask is not None:
                # Place limit slightly above best ask to ensure fill
                price = round(state.best_ask + 0.01, 4)
                orders.append(self._limit_order(side, qty, price))
            elif side == OrderSide.SELL and state.best_bid is not None:
                price = round(state.best_bid - 0.01, 4)
                orders.append(self._limit_order(side, qty, price))

        if orders:
            self._trades_this_session += 1

        return orders

    def _inventory_penalty(self) -> float:
        """
        Smooth penalty on trade size as inventory approaches the limit.

        Returns a factor in (0, 1].
        At zero inventory → 1.0 (full size).
        At max_inventory  → ~0.05 (5% of full size).
        """
        abs_inv = abs(self.metrics.inventory)
        if abs_inv >= self.max_inventory:
            return 0.05
        # Cosine decay: full size until 50% of limit, then decays
        fraction = abs_inv / self.max_inventory
        return max(0.05, math.cos(fraction * math.pi / 2))

    @property
    def last_signal(self) -> Optional[float]:
        """Most recent fair-value deviation observed."""
        return self._last_signal
