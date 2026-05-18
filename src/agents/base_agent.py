"""
src/agents/base_agent.py
------------------------
Abstract base class for all market participants.

Every agent in the simulation inherits from BaseAgent.
The simulation loop calls agent.act(market_state) once per timestep.

Design contract
---------------
- Agents are stateful (they remember their inventory, PnL, order history).
- Agents receive a MarketState snapshot; they cannot directly mutate the book.
- Agents return a list of Order objects, which the simulation submits to the engine.
- Agents are notified of their fills via notify_fill().

This separation keeps agent logic cleanly decoupled from the matching engine.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from ..exchange.order import Order, OrderSide, OrderType
from ..exchange.trade import Trade

if TYPE_CHECKING:
    from ..simulation.market_state import MarketState


# Shared counter so every agent across the simulation gets unique order ids
_ORDER_COUNTER = itertools.count(1)


def next_order_id(prefix: str = "O") -> str:
    return f"{prefix}_{next(_ORDER_COUNTER):08d}"


@dataclass
class AgentMetrics:
    """Tracks per-agent performance across the simulation."""
    inventory: float = 0.0          # current net position (+ = long, - = short)
    realized_pnl: float = 0.0       # cash locked in from closed trades
    unrealized_pnl: float = 0.0     # mark-to-market on open position
    total_pnl: float = 0.0          # realized + unrealized
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    trades_executed: int = 0
    volume_traded: float = 0.0
    cash: float = 0.0               # running cash balance


class BaseAgent(ABC):
    """
    Abstract base for all simulation agents.

    Parameters
    ----------
    agent_id : str
        Unique identifier (used as prefix in order ids).
    initial_cash : float
        Starting cash balance (informational; used for PnL tracking).
    random_seed : int, optional
        Seed for the agent's private RNG for reproducibility.
    """

    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 100_000.0,
        random_seed: Optional[int] = None,
    ) -> None:
        self.agent_id = agent_id
        self.metrics = AgentMetrics(cash=initial_cash)
        self._initial_cash = initial_cash
        self._active_order_ids: List[str] = []  # orders currently resting in book

        import random
        self._rng = random.Random(random_seed)

    # ------------------------------------------------------------------
    # Interface the simulation calls
    # ------------------------------------------------------------------

    @abstractmethod
    def act(self, state: "MarketState") -> List[Order]:
        """
        Decide what orders to submit this timestep.

        Parameters
        ----------
        state : MarketState
            Read-only snapshot of the current market.

        Returns
        -------
        orders : list[Order]
            Zero or more orders to submit to the engine.
            The simulation will call engine.submit() for each.
        """
        ...

    def notify_fill(self, trade: Trade, as_maker: bool) -> None:
        """
        Called by the simulation each time one of this agent's orders is filled.

        Parameters
        ----------
        trade : Trade
            The executed trade record.
        as_maker : bool
            True if this agent was the resting (passive) side.
        """
        qty = trade.quantity
        price = trade.price

        # Determine direction from agent's perspective
        if trade.aggressor_side == OrderSide.BUY:
            # Taker bought → maker sold
            agent_sold = as_maker
        else:
            # Taker sold → maker bought
            agent_sold = not as_maker

        if agent_sold:
            self.metrics.inventory -= qty
            self.metrics.cash += qty * price
        else:
            self.metrics.inventory += qty
            self.metrics.cash -= qty * price

        self.metrics.trades_executed += 1
        self.metrics.volume_traded += qty

    def update_unrealized_pnl(self, midprice: float) -> None:
        """Mark inventory to current midprice."""
        self.metrics.unrealized_pnl = self.metrics.inventory * midprice
        self.metrics.total_pnl = (
            self.metrics.cash - self._initial_cash + self.metrics.unrealized_pnl
        )

    # ------------------------------------------------------------------
    # Order construction helpers
    # ------------------------------------------------------------------

    def _limit_order(self, side: OrderSide, qty: float, price: float) -> Order:
        oid = next_order_id(self.agent_id)
        self._active_order_ids.append(oid)
        self.metrics.orders_submitted += 1
        return Order(
            order_id=oid,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=round(qty, 8),
            price=round(price, 8),
        )

    def _market_order(self, side: OrderSide, qty: float) -> Order:
        oid = next_order_id(self.agent_id)
        self.metrics.orders_submitted += 1
        return Order(
            order_id=oid,
            side=side,
            order_type=OrderType.MARKET,
            quantity=round(qty, 8),
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.agent_id!r}, "
            f"inv={self.metrics.inventory:.4f}, "
            f"pnl={self.metrics.total_pnl:.4f})"
        )
