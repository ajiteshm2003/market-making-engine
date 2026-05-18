"""
src/simulation/market_simulation.py
-------------------------------------
MarketSimulation — the core event-driven simulation environment.

Architecture
------------
Each timestep proceeds in this exact order:

    1. FAIR VALUE STEP    — advance the latent fair value process
    2. AGENT ACT          — each agent observes MarketState and returns orders
    3. CANCEL FLUSH       — process any cancellations agents requested
    4. ORDER SUBMISSION   — submit each returned order to the matching engine
    5. FILL NOTIFICATION  — notify agents of any fills from their orders
    6. METRICS RECORD     — snapshot the book state into SimulationMetrics
    7. STATE UPDATE       — build the MarketState for the next timestep

The simulation is intentionally minimal:
- No async, no threads.
- Agents are called sequentially (order randomized each tick to prevent bias).
- No look-ahead: agents only see the state BEFORE their orders are processed.

This matches a synchronous batch auction model. For a continuous model you
would interleave individual agents rather than batching them per tick —
that is a Phase 7 extension.

Usage
-----
    sim = MarketSimulation(
        agents=[NoiseTrader("NT1"), NoiseTrader("NT2"), InformedTrader("IT1")],
        n_steps=500,
        fair_value_config=FairValueConfig(volatility=0.05, jump_prob=0.03),
        random_seed=42,
    )
    result = sim.run()
    df = result.metrics.to_dataframe()
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..exchange.matching_engine import MatchingEngine
from ..exchange.order import OrderSide
from ..exchange.trade import Trade
from ..agents.base_agent import BaseAgent
from ..agents.noise_trader import NoiseTrader
from .fair_value import FairValueConfig, FairValueProcess
from .market_state import MarketState
from .metrics import SimulationMetrics


@dataclass
class SimulationResult:
    """Container for everything produced by a completed simulation."""
    metrics: SimulationMetrics
    engine: MatchingEngine
    fair_value_history: List[float]
    jump_steps: List[int]
    agents: List[BaseAgent]
    n_steps: int

    def summary(self) -> str:
        df = self.metrics.to_dataframe()
        if df.empty:
            return "No data recorded."

        total_trades = df["cumulative_trades"].iloc[-1]
        total_vol = df["cumulative_volume"].iloc[-1]
        mean_spread = df["spread"].mean()
        mean_imb = df["order_imbalance"].mean()

        lines = [
            "=" * 55,
            "  SIMULATION SUMMARY",
            "=" * 55,
            f"  Steps run          : {self.n_steps}",
            f"  Total trades       : {int(total_trades)}",
            f"  Total volume       : {total_vol:.2f}",
            f"  Mean spread        : {mean_spread:.4f}",
            f"  Mean imbalance     : {mean_imb:.4f}" if mean_imb is not None else "",
            f"  Fair value jumps   : {len(self.jump_steps)}",
            f"  Agents             : {len(self.agents)}",
            "-" * 55,
        ]
        for agent in self.agents:
            m = agent.metrics
            lines.append(
                f"  {agent.agent_id:<14} inv={m.inventory:+.2f}  "
                f"pnl={m.total_pnl:+.2f}  trades={m.trades_executed}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)


class MarketSimulation:
    """
    Event-driven market simulation with pluggable agents.

    Parameters
    ----------
    agents : list[BaseAgent]
        All participants. Order of submission is randomised each tick.
    n_steps : int
        Number of simulation timesteps to run.
    fair_value_config : FairValueConfig, optional
        Configuration for the fair value process. Default used if None.
    depth_levels : int
        Number of price levels to snapshot for metrics/MarketState.
    random_seed : int, optional
        Master seed. Each agent and the fair value process derive
        their own seeds from this for full reproducibility.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        n_steps: int = 500,
        fair_value_config: Optional[FairValueConfig] = None,
        depth_levels: int = 5,
        random_seed: Optional[int] = None,
    ) -> None:
        if not agents:
            raise ValueError("At least one agent is required.")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")

        self.agents = agents
        self.n_steps = n_steps
        self.depth_levels = depth_levels
        self._rng = random.Random(random_seed)

        # Build the matching engine fresh
        self.engine = MatchingEngine()

        # Fair value process
        self._fv_process = FairValueProcess(
            config=fair_value_config or FairValueConfig(),
            random_seed=self._rng.randint(0, 2**31),
        )

        # Metrics collector
        self._metrics = SimulationMetrics()

        # Map agent_id → agent for fast fill routing
        self._agent_map: Dict[str, BaseAgent] = {a.agent_id: a for a in agents}

        # Cache: order_id → agent_id (so we can route fills back)
        self._order_owner: Dict[str, str] = {}

        self._ran = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """
        Execute the full simulation and return a SimulationResult.

        Raises RuntimeError if called more than once (use a fresh instance).
        """
        if self._ran:
            raise RuntimeError("Simulation already run. Create a new instance.")
        self._ran = True

        # Build the initial MarketState (book is empty, no fair value step yet)
        current_fv = self._fv_process.value
        state = self._build_state(0, current_fv, trades_this_step=[], volume=0.0)

        for t in range(1, self.n_steps + 1):
            # ── 1. Advance fair value ──────────────────────────────────
            current_fv = self._fv_process.step()

            # ── 2. Agents decide (in random order to prevent bias) ─────
            agent_order = list(self.agents)
            self._rng.shuffle(agent_order)

            all_orders = []
            cancel_ids: List[str] = []

            for agent in agent_order:
                # Update state's fair_value for each agent call
                # (all agents see the same state snapshot per tick)
                proposed = agent.act(state)
                all_orders.extend((agent, o) for o in proposed)

                # Collect cancels from noise traders
                if isinstance(agent, NoiseTrader):
                    cancel_ids.extend(agent.flush_cancels())

            # ── 3. Process cancellations ───────────────────────────────
            for oid in cancel_ids:
                self.engine.cancel(oid)
                # Remove from owner map
                self._order_owner.pop(oid, None)

            # ── 4. Submit orders & collect trades ─────────────────────
            trades_this_step: List[Trade] = []

            for (agent, order) in all_orders:
                # Register order ownership BEFORE submitting
                self._order_owner[order.order_id] = agent.agent_id

                try:
                    new_trades = self.engine.submit(order)
                except ValueError:
                    # Duplicate id or other validation error — skip
                    continue

                trades_this_step.extend(new_trades)

            # ── 5. Route fills back to agents ──────────────────────────
            for trade in trades_this_step:
                self._notify_agents(trade)

            # ── 6. Update unrealised PnL for all agents ────────────────
            mid = self.engine.book.midprice
            if mid is not None:
                for agent in self.agents:
                    agent.update_unrealized_pnl(mid)

            # ── 7. Record metrics ──────────────────────────────────────
            volume_this_step = sum(tr.quantity for tr in trades_this_step)
            bids_snap, asks_snap = self.engine.book.depth_snapshot(self.depth_levels)
            bid_depth_total = sum(q for _, q in bids_snap)
            ask_depth_total = sum(q for _, q in asks_snap)

            self._metrics.record(
                timestep=t,
                fair_value=current_fv,
                midprice=self.engine.book.midprice,
                best_bid=self.engine.book.best_bid,
                best_ask=self.engine.book.best_ask,
                spread=self.engine.book.spread,
                order_imbalance=self.engine.book.order_imbalance(self.depth_levels),
                trades_this_step=len(trades_this_step),
                volume_this_step=volume_this_step,
                book_depth_bids=bid_depth_total,
                book_depth_asks=ask_depth_total,
            )

            # ── 8. Build next state snapshot ───────────────────────────
            last_trade = trades_this_step[-1] if trades_this_step else None
            state = self._build_state(
                t,
                current_fv,
                trades_this_step=trades_this_step,
                volume=volume_this_step,
                last_trade=last_trade,
            )

        return SimulationResult(
            metrics=self._metrics,
            engine=self.engine,
            fair_value_history=self._fv_process.history,
            jump_steps=self._fv_process.jump_steps,
            agents=self.agents,
            n_steps=self.n_steps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _notify_agents(self, trade: Trade) -> None:
        """Route fill notifications to maker and taker agents."""
        maker_agent_id = self._order_owner.get(trade.maker_order_id)
        taker_agent_id = self._order_owner.get(trade.taker_order_id)

        if maker_agent_id and maker_agent_id in self._agent_map:
            self._agent_map[maker_agent_id].notify_fill(trade, as_maker=True)

        if taker_agent_id and taker_agent_id in self._agent_map:
            self._agent_map[taker_agent_id].notify_fill(trade, as_maker=False)

    def _build_state(
        self,
        timestep: int,
        fair_value: float,
        trades_this_step: List[Trade],
        volume: float,
        last_trade: Optional[Trade] = None,
    ) -> MarketState:
        """Build an immutable MarketState snapshot from current engine state."""
        book = self.engine.book
        bids_snap, asks_snap = book.depth_snapshot(self.depth_levels)

        return MarketState(
            timestep=timestep,
            fair_value=fair_value,
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            midprice=book.midprice,
            spread=book.spread,
            bid_depth=bids_snap,
            ask_depth=asks_snap,
            order_imbalance=book.order_imbalance(self.depth_levels),
            last_trade_price=last_trade.price if last_trade else None,
            last_trade_qty=last_trade.quantity if last_trade else None,
            volume_this_step=volume,
            trade_count=len(self.engine.trade_log),
        )
