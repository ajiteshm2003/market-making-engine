"""
tests/test_matching_engine.py
------------------------------
Unit tests for MatchingEngine matching logic.

Covers:
- Limit order resting (no immediate match)
- Limit order immediate match (crossing)
- Market order full fill
- Market order partial fill (book exhausted)
- Partial fills on maker side (residual stays in book)
- FIFO queue priority at same price level
- Cancellation via engine
- Trade log correctness (aggressor side, ids, price)
- Duplicate order_id rejection

Run with:
    pytest tests/test_matching_engine.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.exchange import (
    MatchingEngine,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)


def make_limit(order_id, side, qty, price, ts=None):
    kwargs = dict(order_id=order_id, side=side, order_type=OrderType.LIMIT, quantity=qty, price=price)
    if ts is not None:
        kwargs["timestamp"] = ts
    return Order(**kwargs)


def make_market(order_id, side, qty):
    return Order(order_id=order_id, side=side, order_type=OrderType.MARKET, quantity=qty)


# ---------------------------------------------------------------------------
# Resting (no immediate match)
# ---------------------------------------------------------------------------

class TestResting:

    def test_bid_rests_when_no_asks(self):
        engine = MatchingEngine()
        trades = engine.submit(make_limit("b1", OrderSide.BUY, 10, 100.0))
        assert trades == []
        assert engine.book.best_bid == 100.0
        assert len(engine.book) == 1

    def test_ask_rests_when_no_bids(self):
        engine = MatchingEngine()
        trades = engine.submit(make_limit("a1", OrderSide.SELL, 5, 102.0))
        assert trades == []
        assert engine.book.best_ask == 102.0

    def test_non_crossing_bid_rests(self):
        """Bid at 99 does NOT cross ask at 102 — should rest."""
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 5, 102.0))
        trades = engine.submit(make_limit("b1", OrderSide.BUY, 5, 99.0))
        assert trades == []
        assert len(engine.book) == 2


# ---------------------------------------------------------------------------
# Immediate limit-vs-limit match
# ---------------------------------------------------------------------------

class TestLimitMatch:

    def test_full_match_equal_qty(self):
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 10, 100.0))
        trades = engine.submit(make_limit("b1", OrderSide.BUY, 10, 100.0))

        assert len(trades) == 1
        t = trades[0]
        assert t.price == 100.0
        assert t.quantity == 10
        assert t.aggressor_side == OrderSide.BUY
        assert t.maker_order_id == "a1"
        assert t.taker_order_id == "b1"

        # Both orders fully filled → book empty
        assert len(engine.book) == 0

    def test_taker_larger_than_maker_partial_taker_rest(self):
        """Taker BUY 15, maker SELL 10. Maker fully filled; 5 BUY rests."""
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 10, 100.0))
        trades = engine.submit(make_limit("b1", OrderSide.BUY, 15, 100.0))

        assert len(trades) == 1
        assert trades[0].quantity == 10

        # 5 units of BUY rest in book
        assert engine.book.best_bid == 100.0
        b1 = engine.book.get_order("b1")
        assert b1 is not None
        assert b1.remaining_quantity == pytest.approx(5)
        # Partially-filled taker rests back in book as OPEN (still eligible for matching)
        assert b1.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    def test_maker_larger_than_taker(self):
        """Taker BUY 5, maker SELL 10. 5 of SELL remains."""
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 10, 100.0))
        trades = engine.submit(make_limit("b1", OrderSide.BUY, 5, 100.0))

        assert len(trades) == 1
        assert trades[0].quantity == 5

        a1 = engine.book.get_order("a1")
        assert a1.remaining_quantity == pytest.approx(5)

    def test_buy_limit_sweeps_multiple_ask_levels(self):
        """BUY 20 @ 105 sweeps asks at 100 (qty=8) and 103 (qty=12)."""
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 8, 100.0))
        engine.submit(make_limit("a2", OrderSide.SELL, 12, 103.0))
        trades = engine.submit(make_limit("b1", OrderSide.BUY, 20, 105.0))

        assert len(trades) == 2
        assert trades[0].price == 100.0  # best ask first
        assert trades[0].quantity == 8
        assert trades[1].price == 103.0
        assert trades[1].quantity == 12
        assert len(engine.book) == 0

    def test_buy_limit_stops_at_price(self):
        """BUY 10 @ 101 should NOT lift ask at 102."""
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 10, 102.0))
        trades = engine.submit(make_limit("b1", OrderSide.BUY, 10, 101.0))
        assert trades == []
        assert engine.book.best_bid == 101.0
        assert engine.book.best_ask == 102.0

    def test_sell_limit_matches_bid(self):
        engine = MatchingEngine()
        engine.submit(make_limit("b1", OrderSide.BUY, 10, 100.0))
        trades = engine.submit(make_limit("a1", OrderSide.SELL, 10, 100.0))

        assert len(trades) == 1
        assert trades[0].aggressor_side == OrderSide.SELL
        assert trades[0].maker_order_id == "b1"
        assert trades[0].taker_order_id == "a1"


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------

class TestMarketOrders:

    def test_market_buy_full_fill(self):
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 5, 100.0))
        trades = engine.submit(make_market("m1", OrderSide.BUY, 5))
        assert len(trades) == 1
        assert trades[0].quantity == 5
        assert len(engine.book) == 0

    def test_market_sell_full_fill(self):
        engine = MatchingEngine()
        engine.submit(make_limit("b1", OrderSide.BUY, 5, 99.0))
        trades = engine.submit(make_market("m1", OrderSide.SELL, 5))
        assert len(trades) == 1
        assert trades[0].price == 99.0

    def test_market_buy_exhausts_book_remainder_rejected(self):
        """Market BUY 20 when only 10 available → 10 filled, remainder rejected."""
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 10, 100.0))
        trades = engine.submit(make_market("m1", OrderSide.BUY, 20))

        assert len(trades) == 1
        assert trades[0].quantity == 10

        m1 = engine.get_order("m1")
        assert m1.status == OrderStatus.REJECTED
        assert m1.remaining_quantity == pytest.approx(10)

    def test_market_buy_sweeps_multiple_levels(self):
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 5, 100.0))
        engine.submit(make_limit("a2", OrderSide.SELL, 5, 101.0))
        trades = engine.submit(make_market("m1", OrderSide.BUY, 10))
        assert len(trades) == 2
        assert sum(t.quantity for t in trades) == 10


# ---------------------------------------------------------------------------
# FIFO queue priority
# ---------------------------------------------------------------------------

class TestFIFO:

    def test_fifo_same_price_level(self):
        """
        Three SELL orders at 100.0 submitted in order: a1, a2, a3.
        A BUY 5 should match a1 first, then a2 (not a3).
        """
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 3, 100.0, ts=1.0))
        engine.submit(make_limit("a2", OrderSide.SELL, 3, 100.0, ts=2.0))
        engine.submit(make_limit("a3", OrderSide.SELL, 3, 100.0, ts=3.0))

        trades = engine.submit(make_limit("b1", OrderSide.BUY, 5, 100.0))

        assert len(trades) == 2
        # First trade must be against a1
        assert trades[0].maker_order_id == "a1"
        assert trades[0].quantity == 3
        # Second trade must be against a2 (partial)
        assert trades[1].maker_order_id == "a2"
        assert trades[1].quantity == 2

        # a2 should have 1 remaining, a3 untouched
        a2 = engine.book.get_order("a2")
        assert a2.remaining_quantity == pytest.approx(1)
        a3 = engine.book.get_order("a3")
        assert a3.remaining_quantity == pytest.approx(3)

    def test_fifo_different_price_levels_best_first(self):
        """Best price matched before worse price regardless of insertion order."""
        engine = MatchingEngine()
        engine.submit(make_limit("a_worse", OrderSide.SELL, 5, 102.0, ts=1.0))
        engine.submit(make_limit("a_best", OrderSide.SELL, 5, 100.0, ts=2.0))  # lower ask = better

        trades = engine.submit(make_limit("b1", OrderSide.BUY, 5, 105.0))
        assert trades[0].maker_order_id == "a_best"
        assert trades[0].price == 100.0


# ---------------------------------------------------------------------------
# Cancellations
# ---------------------------------------------------------------------------

class TestCancellations:

    def test_cancel_resting_order(self):
        engine = MatchingEngine()
        engine.submit(make_limit("b1", OrderSide.BUY, 10, 99.0))
        cancelled = engine.cancel("b1")
        assert cancelled is not None
        assert cancelled.status == OrderStatus.CANCELLED
        assert engine.book.best_bid is None

    def test_cancel_nonexistent_returns_none(self):
        engine = MatchingEngine()
        result = engine.cancel("ghost")
        assert result is None

    def test_cannot_match_cancelled_order(self):
        """After cancellation, a crossing order should NOT fill."""
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 10, 100.0))
        engine.cancel("a1")
        trades = engine.submit(make_limit("b1", OrderSide.BUY, 10, 100.0))
        assert trades == []  # a1 gone, b1 rests as new bid


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

class TestTradeLog:

    def test_trade_log_grows_with_matches(self):
        engine = MatchingEngine()
        engine.submit(make_limit("a1", OrderSide.SELL, 5, 100.0))
        engine.submit(make_limit("a2", OrderSide.SELL, 5, 101.0))
        engine.submit(make_market("m1", OrderSide.BUY, 10))
        assert len(engine.trade_log) == 2

    def test_trade_log_no_trades_on_rest(self):
        engine = MatchingEngine()
        engine.submit(make_limit("b1", OrderSide.BUY, 5, 100.0))
        engine.submit(make_limit("a1", OrderSide.SELL, 5, 105.0))
        assert len(engine.trade_log) == 0

    def test_trade_ids_unique(self):
        engine = MatchingEngine()
        for i in range(5):
            engine.submit(make_limit(f"s{i}", OrderSide.SELL, 1, 100.0))
            engine.submit(make_market(f"b{i}", OrderSide.BUY, 1))
        ids = [t.trade_id for t in engine.trade_log]
        assert len(ids) == len(set(ids))

    def test_trade_log_is_copy(self):
        """trade_log property must return a copy so callers can't mutate internal state."""
        engine = MatchingEngine()
        log = engine.trade_log
        log.append(None)  # mutate the copy
        assert len(engine.trade_log) == 0


# ---------------------------------------------------------------------------
# Duplicate order id
# ---------------------------------------------------------------------------

class TestDuplicateOrderId:

    def test_duplicate_order_id_raises(self):
        engine = MatchingEngine()
        engine.submit(make_limit("x1", OrderSide.BUY, 5, 100.0))
        with pytest.raises(ValueError, match="Duplicate order_id"):
            engine.submit(make_limit("x1", OrderSide.BUY, 5, 100.0))
