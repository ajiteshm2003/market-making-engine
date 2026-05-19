"""
src/replay/market_data.py
--------------------------
Market Data Event Abstractions

Defines the event types used by the historical replay engine.
These structures are designed to be compatible with real exchange data
feeds (Binance, Coinbase) while working entirely on synthetic or CSV data.

Event hierarchy
---------------
MarketDataEvent         (base)
├── L2Snapshot          Full order book snapshot at a point in time
├── L2Update            Incremental bid/ask update
└── TradeEvent          A recorded trade execution

ReplayEventStream       Ordered sequence of events for replay
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple


class EventType(Enum):
    SNAPSHOT = "snapshot"
    UPDATE   = "update"
    TRADE    = "trade"


@dataclass
class MarketDataEvent:
    """Base class for all market data events."""
    timestamp:  float          # Unix timestamp in seconds
    event_type: EventType
    symbol:     str = "SIM"    # instrument identifier


@dataclass
class L2Level:
    """A single price level in the order book."""
    price:    float
    quantity: float


@dataclass
class L2Snapshot(MarketDataEvent):
    """
    Full order book snapshot: top-N bids and asks.

    Contains the complete state of the book at a single point in time.
    Used to initialize the replay engine's internal book state.
    """
    bids:  List[L2Level] = field(default_factory=list)   # sorted descending
    asks:  List[L2Level] = field(default_factory=list)   # sorted ascending

    def __post_init__(self):
        self.event_type = EventType.SNAPSHOT

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def midprice(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


@dataclass
class L2Update(MarketDataEvent):
    """
    Incremental order book update: a single price level change.

    A quantity of 0.0 means the level was removed.
    """
    side:     str    = "bid"   # "bid" or "ask"
    price:    float  = 0.0
    quantity: float  = 0.0

    def __post_init__(self):
        self.event_type = EventType.UPDATE

    @property
    def is_removal(self) -> bool:
        return self.quantity == 0.0


@dataclass
class TradeEvent(MarketDataEvent):
    """
    A recorded market trade.

    Represents a completed execution between a buyer and seller.
    """
    price:          float = 0.0
    quantity:       float = 0.0
    aggressor_side: str   = "buy"   # "buy" = taker bought, "sell" = taker sold

    def __post_init__(self):
        self.event_type = EventType.TRADE


@dataclass
class TopOfBook:
    """
    Current best bid and ask, reconstructed from the event stream.
    Maintained incrementally by the replay engine.
    """
    timestamp: float
    best_bid:  Optional[float]
    best_ask:  Optional[float]

    @property
    def midprice(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


class ReplayEventStream:
    """
    An ordered, iterable sequence of market data events.

    Events are stored in chronological order.
    Used by the HistoricalReplayEngine as its data source.
    """

    def __init__(self, events: Optional[List[MarketDataEvent]] = None) -> None:
        self._events: List[MarketDataEvent] = sorted(
            events or [], key=lambda e: e.timestamp
        )

    def add(self, event: MarketDataEvent) -> None:
        """Insert an event, maintaining chronological order."""
        from bisect import insort_left
        # Find insertion point by timestamp
        ts = event.timestamp
        pos = 0
        for i, e in enumerate(self._events):
            if e.timestamp > ts:
                pos = i
                break
            pos = i + 1
        self._events.insert(pos, event)

    def __iter__(self) -> Iterator[MarketDataEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def n_snapshots(self) -> int:
        return sum(1 for e in self._events if e.event_type == EventType.SNAPSHOT)

    @property
    def n_updates(self) -> int:
        return sum(1 for e in self._events if e.event_type == EventType.UPDATE)

    @property
    def n_trades(self) -> int:
        return sum(1 for e in self._events if e.event_type == EventType.TRADE)

    @property
    def time_range(self) -> Tuple[Optional[float], Optional[float]]:
        if not self._events:
            return None, None
        return self._events[0].timestamp, self._events[-1].timestamp

    def is_ordered(self) -> bool:
        """Verify that all events are in non-decreasing timestamp order."""
        for i in range(1, len(self._events)):
            if self._events[i].timestamp < self._events[i - 1].timestamp:
                return False
        return True
