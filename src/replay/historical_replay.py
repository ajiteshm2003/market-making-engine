"""
src/replay/historical_replay.py
--------------------------------
Historical Replay Engine

Replays a sequence of market data events, reconstructing top-of-book state
and feeding the resulting midprice/spread/volatility series into the
existing regime and volatility estimation infrastructure.

This module bridges the gap between the simulation environment (which uses
a synthetic fair value process) and real market data (which arrives as a
stream of L2 snapshots, incremental updates, and trade events).

Architecture
------------
HistoricalReplayEngine
  └── processes ReplayEventStream
       ├── maintains current TopOfBook state
       ├── records per-timestamp midprice, spread, volume
       ├── feeds midprice into RollingVolatilityEstimator
       ├── classifies regime via RegimeClassifier
       └── emits ReplayResult with full time series

SyntheticDataGenerator
  └── creates realistic-looking synthetic L2 + trade events
       suitable for testing without live API access

These can be replaced by live data feeds in production without
changing the replay engine or downstream analytics.
"""

from __future__ import annotations

import math
import random
import time as time_module
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .market_data import (
    L2Level, L2Snapshot, L2Update, TradeEvent,
    MarketDataEvent, ReplayEventStream, TopOfBook, EventType,
)
from ..models.volatility import RollingVolatilityEstimator, VolatilityConfig
from ..models.regime import RegimeClassifier, RegimeThresholds, VolatilityRegime


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------

@dataclass
class ReplayStep:
    """State captured at each replay step."""
    timestamp:     float
    best_bid:      Optional[float]
    best_ask:      Optional[float]
    midprice:      Optional[float]
    spread:        Optional[float]
    sigma:         float
    regime:        VolatilityRegime
    trade_volume:  float    # total trade volume at this timestamp


@dataclass
class ReplayResult:
    """Full output of a historical replay run."""
    symbol:          str
    n_events:        int
    n_steps:         int
    steps:           List[ReplayStep]
    regime_counts:   Dict[str, int]
    transition_count: int
    mean_spread:     Optional[float]
    mean_sigma:      float
    max_sigma:       float

    def print_summary(self) -> None:
        print(f"  Replay Summary: {self.symbol}")
        print(f"  Events processed : {self.n_events}")
        print(f"  Steps recorded   : {self.n_steps}")
        print(f"  Mean spread      : {self.mean_spread:.5f}" if self.mean_spread else "  Mean spread : N/A")
        print(f"  Mean σ̂           : {self.mean_sigma:.6f}")
        print(f"  Max σ̂            : {self.max_sigma:.6f}")
        print(f"  Regime transitions: {self.transition_count}")
        for regime, count in self.regime_counts.items():
            pct = 100 * count / max(self.n_steps, 1)
            print(f"    {regime:<8} {count:>5} steps  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

class HistoricalReplayEngine:
    """
    Processes a ReplayEventStream, reconstructing market state and
    running volatility/regime analysis on the replayed midprice series.

    Parameters
    ----------
    vol_config   : VolatilityConfig for the rolling σ estimator
    thresholds   : RegimeThresholds for regime classification
    symbol       : instrument name (informational only)
    """

    def __init__(
        self,
        vol_config: Optional[VolatilityConfig] = None,
        thresholds: Optional[RegimeThresholds] = None,
        symbol: str = "SIM",
    ) -> None:
        self.symbol = symbol
        self._vol_est  = RollingVolatilityEstimator(vol_config or VolatilityConfig(window=20))
        self._clf      = RegimeClassifier(thresholds or RegimeThresholds())

        # Current top-of-book state
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None

        # Internal bid/ask tracking (price → quantity)
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}

    def process(self, stream: ReplayEventStream) -> ReplayResult:
        """
        Process all events in the stream and return a ReplayResult.

        Events are grouped by timestamp; all events at the same timestamp
        are processed before recording the step state.
        """
        if not stream.is_ordered():
            raise ValueError("Event stream is not in chronological order.")

        steps: List[ReplayStep] = []

        # Group events by timestamp
        events_by_ts: Dict[float, List[MarketDataEvent]] = {}
        for event in stream:
            events_by_ts.setdefault(event.timestamp, []).append(event)

        for ts in sorted(events_by_ts.keys()):
            trade_vol = 0.0
            for event in events_by_ts[ts]:
                if event.event_type == EventType.SNAPSHOT:
                    self._apply_snapshot(event)
                elif event.event_type == EventType.UPDATE:
                    self._apply_update(event)
                elif event.event_type == EventType.TRADE:
                    trade_vol += event.quantity

            # Compute top of book
            bb = max(self._bids.keys()) if self._bids else None
            ba = min(self._asks.keys()) if self._asks else None
            mid = (bb + ba) / 2.0 if (bb and ba) else None

            # Update estimators
            sigma = self._vol_est.update(mid) if mid else self._vol_est.sigma
            regime = self._clf.update(sigma)

            spread = (ba - bb) if (bb and ba) else None

            steps.append(ReplayStep(
                timestamp=ts,
                best_bid=bb,
                best_ask=ba,
                midprice=mid,
                spread=spread,
                sigma=sigma,
                regime=regime,
                trade_volume=trade_vol,
            ))

        # Summarize
        valid_spreads = [s.spread for s in steps if s.spread is not None]
        mean_spread = sum(valid_spreads) / len(valid_spreads) if valid_spreads else None
        all_sigmas = [s.sigma for s in steps]
        mean_sigma = sum(all_sigmas) / len(all_sigmas) if all_sigmas else 0.0
        max_sigma  = max(all_sigmas) if all_sigmas else 0.0

        regime_counts = {r.value: self._clf.time_in_regime(r) for r in VolatilityRegime}

        return ReplayResult(
            symbol=self.symbol,
            n_events=len(stream),
            n_steps=len(steps),
            steps=steps,
            regime_counts=regime_counts,
            transition_count=self._clf.transition_count,
            mean_spread=mean_spread,
            mean_sigma=round(mean_sigma, 8),
            max_sigma=round(max_sigma, 8),
        )

    def _apply_snapshot(self, snap: L2Snapshot) -> None:
        self._bids = {level.price: level.quantity for level in snap.bids}
        self._asks = {level.price: level.quantity for level in snap.asks}

    def _apply_update(self, update: L2Update) -> None:
        book = self._bids if update.side == "bid" else self._asks
        if update.quantity == 0.0:
            book.pop(update.price, None)
        else:
            book[update.price] = update.quantity


# ---------------------------------------------------------------------------
# Synthetic data generator (no API required)
# ---------------------------------------------------------------------------

class SyntheticMarketDataGenerator:
    """
    Generates a synthetic ReplayEventStream for testing the replay engine
    without any external data source or API connection.

    The generated stream contains:
    - An initial L2Snapshot with realistic bid/ask depth
    - Per-step L2Updates simulating price movement
    - TradeEvents at each step

    The fair value follows a Gaussian random walk with jumps,
    matching the simulation's FairValueProcess behavior.

    Parameters
    ----------
    n_steps       : number of simulation steps to generate
    initial_price : starting midprice
    volatility    : per-step diffusion σ
    jump_prob     : per-step jump probability
    jump_std      : jump size σ
    spread_ticks  : half-spread in price units
    depth_levels  : number of price levels in the book
    seed          : random seed for reproducibility
    """

    def __init__(
        self,
        n_steps:       int   = 300,
        initial_price: float = 100.0,
        volatility:    float = 0.05,
        jump_prob:     float = 0.04,
        jump_std:      float = 0.5,
        spread_ticks:  float = 0.06,
        depth_levels:  int   = 5,
        seed:          int   = 42,
    ) -> None:
        self.n_steps       = n_steps
        self.initial_price = initial_price
        self.volatility    = volatility
        self.jump_prob     = jump_prob
        self.jump_std      = jump_std
        self.spread_ticks  = spread_ticks
        self.depth_levels  = depth_levels
        self._rng = random.Random(seed)

    def generate(self, base_timestamp: float = 0.0) -> ReplayEventStream:
        """
        Generate the full event stream.

        Returns
        -------
        ReplayEventStream in chronological order
        """
        events: List[MarketDataEvent] = []
        fair_value = self.initial_price
        ts = base_timestamp

        # Initial snapshot
        snap = self._make_snapshot(ts, fair_value)
        events.append(snap)
        ts += 1.0

        for step in range(self.n_steps):
            # Advance fair value
            diff = self.volatility * self._rng.gauss(0, 1)
            jump = self._rng.gauss(0, self.jump_std) if self._rng.random() < self.jump_prob else 0.0
            fair_value = max(1.0, fair_value + diff + jump)

            # L2 updates: top bid and ask move with fair value
            best_bid = fair_value - self.spread_ticks
            best_ask = fair_value + self.spread_ticks

            events.append(L2Update(
                timestamp=ts, event_type=EventType.UPDATE, symbol="SIM",
                side="bid", price=round(best_bid, 4),
                quantity=round(self._rng.uniform(1.0, 10.0), 2),
            ))
            events.append(L2Update(
                timestamp=ts, event_type=EventType.UPDATE, symbol="SIM",
                side="ask", price=round(best_ask, 4),
                quantity=round(self._rng.uniform(1.0, 10.0), 2),
            ))

            # Occasional trade
            if self._rng.random() < 0.40:
                side = self._rng.choice(["buy", "sell"])
                price = best_ask if side == "buy" else best_bid
                qty   = round(self._rng.lognormvariate(1.0, 0.5), 2)
                events.append(TradeEvent(
                    timestamp=ts, event_type=EventType.TRADE, symbol="SIM",
                    price=round(price, 4), quantity=qty, aggressor_side=side,
                ))

            ts += 1.0

        return ReplayEventStream(events)

    def _make_snapshot(self, ts: float, mid: float) -> L2Snapshot:
        bids, asks = [], []
        for i in range(self.depth_levels):
            bid_price = mid - self.spread_ticks - i * 0.02
            ask_price = mid + self.spread_ticks + i * 0.02
            qty = round(self._rng.uniform(2.0, 15.0), 2)
            bids.append(L2Level(price=round(bid_price, 4), quantity=qty))
            asks.append(L2Level(price=round(ask_price, 4), quantity=qty))
        return L2Snapshot(
            timestamp=ts, event_type=EventType.SNAPSHOT, symbol="SIM",
            bids=bids, asks=asks,
        )


# ---------------------------------------------------------------------------
# CSV I/O utilities
# ---------------------------------------------------------------------------

def save_replay_to_csv(stream: ReplayEventStream, filepath: str) -> None:
    """
    Save a ReplayEventStream to CSV for persistent storage and reuse.

    Columns: timestamp, event_type, symbol, side, price, quantity, aggressor_side
    """
    import csv
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event_type", "symbol",
                         "side", "price", "quantity", "aggressor_side"])
        for event in stream:
            if event.event_type == EventType.TRADE:
                writer.writerow([
                    event.timestamp, event.event_type.value, event.symbol,
                    "", event.price, event.quantity, event.aggressor_side,
                ])
            elif event.event_type == EventType.UPDATE:
                writer.writerow([
                    event.timestamp, event.event_type.value, event.symbol,
                    event.side, event.price, event.quantity, "",
                ])
            elif event.event_type == EventType.SNAPSHOT:
                for level in event.bids:
                    writer.writerow([
                        event.timestamp, "snapshot_bid", event.symbol,
                        "bid", level.price, level.quantity, "",
                    ])
                for level in event.asks:
                    writer.writerow([
                        event.timestamp, "snapshot_ask", event.symbol,
                        "ask", level.price, level.quantity, "",
                    ])


def load_replay_from_csv(filepath: str) -> ReplayEventStream:
    """
    Load a previously saved event stream from CSV.

    Returns
    -------
    ReplayEventStream
    """
    import csv
    events: List[MarketDataEvent] = []
    snapshot_bids: Dict[float, List[L2Level]] = {}
    snapshot_asks: Dict[float, List[L2Level]] = {}

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            etype = row["event_type"]
            sym = row["symbol"]

            if etype == "trade":
                events.append(TradeEvent(
                    timestamp=ts, event_type=EventType.TRADE, symbol=sym,
                    price=float(row["price"]), quantity=float(row["quantity"]),
                    aggressor_side=row["aggressor_side"],
                ))
            elif etype == "update":
                events.append(L2Update(
                    timestamp=ts, event_type=EventType.UPDATE, symbol=sym,
                    side=row["side"], price=float(row["price"]),
                    quantity=float(row["quantity"]),
                ))
            elif etype in ("snapshot_bid", "snapshot_ask"):
                side_dict = snapshot_bids if etype == "snapshot_bid" else snapshot_asks
                side_dict.setdefault(ts, []).append(
                    L2Level(price=float(row["price"]), quantity=float(row["quantity"]))
                )

    # Reconstruct snapshots
    all_snap_ts = set(snapshot_bids.keys()) | set(snapshot_asks.keys())
    for ts in sorted(all_snap_ts):
        bids = sorted(snapshot_bids.get(ts, []), key=lambda l: -l.price)
        asks = sorted(snapshot_asks.get(ts, []), key=lambda l: l.price)
        events.append(L2Snapshot(
            timestamp=ts, event_type=EventType.SNAPSHOT, symbol="SIM",
            bids=bids, asks=asks,
        ))

    return ReplayEventStream(events)
