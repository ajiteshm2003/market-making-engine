"""
src/strategies/mm_metrics.py
-----------------------------
MarketMakerMetrics

Dedicated metrics tracking for market makers, extending the base AgentMetrics
with market-making-specific accounting.

The key distinction from a general agent:
- Market makers earn the SPREAD on round-trip fills (maker buy + maker sell).
- They carry INVENTORY RISK: if they buy and prices fall, they lose on the position.
- PnL decomposition:
    realized_pnl     = cash delta from matched round-trips
    unrealized_pnl   = inventory × current_midprice (mark-to-market)
    spread_capture   = total earned from bid-ask spread on round trips
    inventory_cost   = loss from holding directional inventory (adverse selection)

Trajectories
------------
All per-step snapshots are stored so they can be exported to a DataFrame
for plotting inventory paths, PnL curves, and spread capture over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class MMStepRecord:
    """One row of the market maker's per-step history."""
    timestep: int
    inventory: float
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    spread_capture: float           # cumulative gross spread earned
    bid_price: Optional[float]      # quote posted this step
    ask_price: Optional[float]
    quoted_spread: Optional[float]  # ask - bid
    fills_as_maker: int             # cumulative fills as maker
    volume_as_maker: float          # cumulative volume as maker


@dataclass
class MarketMakerMetrics:
    """
    Extended metrics for a market-making agent.

    Attributes
    ----------
    inventory : float
        Net position (+ long, - short).
    cash : float
        Running cash balance. Starts at initial_cash.
    realized_pnl : float
        Cash profit locked in (inventory-adjusted, cost-basis accounting).
    unrealized_pnl : float
        Mark-to-market value of open inventory.
    total_pnl : float
        realized_pnl + unrealized_pnl (net of starting cash).
    spread_capture : float
        Cumulative gross spread income (buy_fill_price - sell_fill_price
        on matched round trips). Positive = maker earned spread.
    fills_as_maker : int
        Number of fills where this MM was the resting (passive) side.
    fills_as_taker : int
        Number of fills where this MM was the aggressor (rare).
    volume_as_maker : float
        Total volume traded as maker.
    volume_as_taker : float
        Total volume traded as taker.
    quotes_posted : int
        Number of quote updates (bid+ask pairs submitted).
    bid_fills : int
        Fills on the bid side (MM bought, taker sold to them).
    ask_fills : int
        Fills on the ask side (MM sold, taker bought from them).
    inventory_history : list[float]
        Per-step inventory snapshots (for plotting).
    pnl_history : list[float]
        Per-step total PnL snapshots.
    step_records : list[MMStepRecord]
        Full per-step records.
    """

    # Running state
    inventory: float = 0.0
    cash: float = 0.0

    # PnL components
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    spread_capture: float = 0.0

    # Fill counts
    fills_as_maker: int = 0
    fills_as_taker: int = 0
    volume_as_maker: float = 0.0
    volume_as_taker: float = 0.0

    # Quote activity
    quotes_posted: int = 0
    bid_fills: int = 0
    ask_fills: int = 0

    # Trajectories (appended each step)
    inventory_history: List[float] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    step_records: List[MMStepRecord] = field(default_factory=list)

    def snapshot(
        self,
        timestep: int,
        bid_price: Optional[float],
        ask_price: Optional[float],
    ) -> None:
        """Append a per-step record to history."""
        self.inventory_history.append(self.inventory)
        self.pnl_history.append(self.total_pnl)
        quoted_spread = (ask_price - bid_price) if (bid_price and ask_price) else None
        self.step_records.append(
            MMStepRecord(
                timestep=timestep,
                inventory=self.inventory,
                cash=self.cash,
                realized_pnl=self.realized_pnl,
                unrealized_pnl=self.unrealized_pnl,
                total_pnl=self.total_pnl,
                spread_capture=self.spread_capture,
                bid_price=bid_price,
                ask_price=ask_price,
                quoted_spread=quoted_spread,
                fills_as_maker=self.fills_as_maker,
                volume_as_maker=self.volume_as_maker,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export step records to a pandas DataFrame indexed by timestep."""
        if not self.step_records:
            return pd.DataFrame()
        return pd.DataFrame(
            [vars(r) for r in self.step_records]
        ).set_index("timestep")

    @property
    def inventory_variance(self) -> float:
        """Variance of inventory over time (measure of inventory risk)."""
        if len(self.inventory_history) < 2:
            return 0.0
        n = len(self.inventory_history)
        mean = sum(self.inventory_history) / n
        return sum((x - mean) ** 2 for x in self.inventory_history) / (n - 1)

    @property
    def fill_rate(self) -> float:
        """Fraction of quotes that resulted in a fill (as maker)."""
        if self.quotes_posted == 0:
            return 0.0
        return self.fills_as_maker / self.quotes_posted

    def summary_dict(self) -> dict:
        return {
            "inventory": round(self.inventory, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "total_pnl": round(self.total_pnl, 4),
            "spread_capture": round(self.spread_capture, 4),
            "fills_as_maker": self.fills_as_maker,
            "fills_as_taker": self.fills_as_taker,
            "volume_as_maker": round(self.volume_as_maker, 4),
            "quotes_posted": self.quotes_posted,
            "bid_fills": self.bid_fills,
            "ask_fills": self.ask_fills,
            "inventory_variance": round(self.inventory_variance, 4),
            "fill_rate": round(self.fill_rate, 4),
        }
