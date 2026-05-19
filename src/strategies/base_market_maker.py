"""
src/strategies/base_market_maker.py
------------------------------------
BaseMarketMaker

Abstract base class for all market-making strategies.

Market makers are a special type of agent whose primary activity is to
continuously post limit orders on BOTH sides of the order book (quotes),
earning the bid-ask spread from other participants who cross those quotes.

Core lifecycle (per timestep)
------------------------------
    1. Cancel stale quotes from the previous step.
    2. Compute new bid/ask prices via _compute_quotes() [abstract].
    3. Post new limit orders on both sides.
    4. Receive fill notifications via notify_fill().
    5. Update PnL and snapshot metrics.

Quote management
----------------
A market maker maintains at most ONE resting bid and ONE resting ask at any
time (single-level quoting). Before posting new quotes each step, it cancels
the previous ones. This "cancel-and-requote" pattern is standard practice.

Future extensions (Phase 4+) can implement multi-level quoting or partial
cancellation, but single-level is correct and sufficient here.

PnL accounting
--------------
We use a simple cash-flow model:

    cash += fill_price * fill_qty     when selling (ask fill)
    cash -= fill_price * fill_qty     when buying  (bid fill)
    realized_pnl = cash - initial_cash + (initial_inventory * initial_price)
    unrealized_pnl = inventory * current_midprice
    total_pnl = cash - initial_cash + unrealized_pnl

Spread capture is tracked separately:
    Each time we fill on the ask, we record the ask price.
    Each time we fill on the bid, we record the bid price.
    Round-trip spread = ask_fill_price - bid_fill_price.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..agents.base_agent import next_order_id
from ..exchange.order import Order, OrderSide, OrderStatus, OrderType
from ..exchange.trade import Trade
from .mm_metrics import MarketMakerMetrics

if TYPE_CHECKING:
    from ..simulation.market_state import MarketState


class BaseMarketMaker(ABC):
    """
    Abstract base for all market-making strategies.

    Parameters
    ----------
    agent_id : str
        Unique identifier.
    quote_size : float
        Default quantity for each posted quote (bid and ask).
    initial_cash : float
        Starting cash balance for PnL tracking.
    min_price : float
        Floor on quoted prices (safety guard).
    """

    def __init__(
        self,
        agent_id: str,
        quote_size: float = 5.0,
        initial_cash: float = 100_000.0,
        min_price: float = 0.01,
    ) -> None:
        if quote_size <= 0:
            raise ValueError(f"quote_size must be > 0, got {quote_size}")

        self.agent_id = agent_id
        self.quote_size = quote_size
        self.min_price = min_price
        self._initial_cash = initial_cash

        # Dedicated MM metrics (replaces AgentMetrics from BaseAgent)
        self.mm_metrics = MarketMakerMetrics(cash=initial_cash)

        # Current resting quote ids (None = not currently in book)
        self._bid_order_id: Optional[str] = None
        self._ask_order_id: Optional[str] = None

        # Stale ids to cancel next step
        self._pending_cancels: List[str] = []

        # Last posted quote prices (for snapshot)
        self._last_bid_price: Optional[float] = None
        self._last_ask_price: Optional[float] = None

        # Current timestep (set by act())
        self._current_step: int = 0

        # For spread_capture: track average fill prices per side
        self._bid_fill_prices: List[float] = []  # prices at which we bought (bid fills)
        self._ask_fill_prices: List[float] = []  # prices at which we sold (ask fills)

    # ------------------------------------------------------------------
    # Interface for MarketSimulation (mirrors BaseAgent)
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        """
        Compatibility shim: simulation calls agent.metrics.inventory / total_pnl.
        We forward these from mm_metrics so the simulation summary still works.
        """
        return self.mm_metrics

    def act(self, state: "MarketState") -> List[Order]:
        """
        Called once per timestep by the simulation.

        Sequence:
          1. Schedule cancellation of stale quotes.
          2. Compute new quotes via subclass logic.
          3. Post new bid and ask orders.
          4. Snapshot metrics.

        Returns
        -------
        List[Order] — new orders to submit (does NOT include cancels).
                      Cancels are returned via flush_cancels().
        """
        self._current_step = state.timestep

        # Schedule old quotes for cancellation
        if self._bid_order_id is not None:
            self._pending_cancels.append(self._bid_order_id)
            self._bid_order_id = None
        if self._ask_order_id is not None:
            self._pending_cancels.append(self._ask_order_id)
            self._ask_order_id = None

        # Ask subclass for new quote prices
        bid_price, ask_price = self._compute_quotes(state)

        orders: List[Order] = []

        if bid_price is not None and ask_price is not None:
            # Safety: never post a crossed market
            if bid_price >= ask_price:
                # Widen slightly to avoid crossing
                mid = (bid_price + ask_price) / 2.0
                bid_price = round(mid - 0.01, 6)
                ask_price = round(mid + 0.01, 6)

            bid_price = max(self.min_price, round(bid_price, 6))
            ask_price = max(self.min_price, round(ask_price, 6))

            bid_order = self._make_limit(OrderSide.BUY, self.quote_size, bid_price)
            ask_order = self._make_limit(OrderSide.SELL, self.quote_size, ask_price)

            self._bid_order_id = bid_order.order_id
            self._ask_order_id = ask_order.order_id
            self._last_bid_price = bid_price
            self._last_ask_price = ask_price

            orders = [bid_order, ask_order]
            self.mm_metrics.quotes_posted += 1

        # Snapshot per-step metrics (called after quoting, before fills this step)
        self.mm_metrics.snapshot(
            timestep=state.timestep,
            bid_price=self._last_bid_price,
            ask_price=self._last_ask_price,
        )

        return orders

    def flush_cancels(self) -> List[str]:
        """
        Return and clear the list of order ids to cancel.
        Called by the simulation BEFORE new orders are submitted.
        """
        cancels = list(self._pending_cancels)
        self._pending_cancels.clear()
        return cancels

    def notify_fill(self, trade: Trade, as_maker: bool) -> None:
        """
        Called by the simulation when one of our orders is filled.

        Updates inventory, cash, PnL, and fill counters.
        Also tracks whether the fill was on our bid or ask side.
        """
        qty = trade.quantity
        price = trade.price
        m = self.mm_metrics

        # Determine our side in this fill
        # aggressor_side is the TAKER's side
        if trade.aggressor_side == OrderSide.BUY:
            # Taker bought → we (maker) sold → ASK FILL
            we_sold = as_maker
        else:
            # Taker sold → we (maker) bought → BID FILL
            we_sold = not as_maker

        if we_sold:
            # We sold: cash in, inventory down
            m.cash += price * qty
            m.inventory -= qty
            m.ask_fills += 1
            self._ask_fill_prices.append(price)
        else:
            # We bought: cash out, inventory up
            m.cash -= price * qty
            m.inventory += qty
            m.bid_fills += 1
            self._bid_fill_prices.append(price)

        if as_maker:
            m.fills_as_maker += 1
            m.volume_as_maker += qty
        else:
            m.fills_as_taker += 1
            m.volume_as_taker += qty

        # Update realized PnL (cash-based)
        m.realized_pnl = m.cash - self._initial_cash

        # Update spread capture:
        # For each matched pair (1 bid fill + 1 ask fill), the spread earned is:
        # ask_fill_price - bid_fill_price
        # We use the minimum of the two fill lists to count complete round trips.
        n_complete = min(len(self._bid_fill_prices), len(self._ask_fill_prices))
        if n_complete > 0:
            avg_ask = sum(self._ask_fill_prices[:n_complete]) / n_complete
            avg_bid = sum(self._bid_fill_prices[:n_complete]) / n_complete
            m.spread_capture = (avg_ask - avg_bid) * n_complete * self.quote_size

    def update_unrealized_pnl(self, midprice: float) -> None:
        """Called by simulation each step to mark inventory to market."""
        m = self.mm_metrics
        m.unrealized_pnl = m.inventory * midprice
        m.total_pnl = m.realized_pnl + m.unrealized_pnl

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_quotes(
        self, state: "MarketState"
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute new bid and ask prices for this timestep.

        Parameters
        ----------
        state : MarketState
            Current market snapshot (read-only).

        Returns
        -------
        (bid_price, ask_price) : tuple[float | None, float | None]
            Return (None, None) to sit out this step (post no quotes).
        """
        ...

    # ------------------------------------------------------------------
    # Order construction
    # ------------------------------------------------------------------

    def _make_limit(self, side: OrderSide, qty: float, price: float) -> Order:
        oid = next_order_id(self.agent_id)
        return Order(
            order_id=oid,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=round(qty, 8),
            price=round(price, 8),
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        m = self.mm_metrics
        return (
            f"{self.__class__.__name__}(id={self.agent_id!r}, "
            f"inv={m.inventory:+.4f}, pnl={m.total_pnl:+.4f}, "
            f"bid={self._last_bid_price}, ask={self._last_ask_price})"
        )
