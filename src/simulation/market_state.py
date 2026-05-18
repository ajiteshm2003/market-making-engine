"""
src/simulation/market_state.py
-------------------------------
MarketState — a read-only snapshot of the exchange at a single timestep.

Design rationale
----------------
Agents must NOT be able to mutate the order book directly.
Instead, they receive a MarketState containing all the information they
need to make decisions, and they return Order objects to the simulation.

This enforces the correct information barrier: agents only know what the
exchange currently shows — not the engine's internal state.

Fields
------
timestep        : int      — current simulation step
fair_value      : float    — latent true price (not directly observable in reality)
midprice        : float    — (best_bid + best_ask) / 2
best_bid        : float    — highest resting bid price
best_ask        : float    — lowest resting ask price
spread          : float    — best_ask - best_bid
bid_depth       : list     — [(price, qty), ...] top N bid levels
ask_depth       : list     — [(price, qty), ...] top N ask levels
order_imbalance : float    — in [-1, +1]; positive = buy pressure
last_trade_price: float    — price of most recent execution
last_trade_qty  : float    — quantity of most recent execution
volume_this_step: float    — total volume traded this timestep
trade_count     : int      — number of trades since session start
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class MarketState:
    """
    Immutable market snapshot passed to each agent at every timestep.
    """

    timestep: int
    fair_value: Optional[float]

    # Best quotes
    best_bid: Optional[float]
    best_ask: Optional[float]

    # Derived
    midprice: Optional[float]
    spread: Optional[float]

    # Depth (list of (price, qty) tuples)
    bid_depth: List[Tuple[float, float]] = field(default_factory=list)
    ask_depth: List[Tuple[float, float]] = field(default_factory=list)

    # Order flow
    order_imbalance: Optional[float] = None

    # Last trade info
    last_trade_price: Optional[float] = None
    last_trade_qty: Optional[float] = None

    # Session accumulators (updated each step)
    volume_this_step: float = 0.0
    trade_count: int = 0

    def __repr__(self) -> str:
        return (
            f"MarketState(t={self.timestep}, fv={self.fair_value:.4f}, "
            f"mid={self.midprice}, spread={self.spread}, "
            f"imb={self.order_imbalance}, trades={self.trade_count})"
        )
