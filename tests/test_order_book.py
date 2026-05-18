"""
tests/test_order_book.py
------------------------
Unit tests for Order, OrderBook behaviour.

Run with:
    pytest tests/test_order_book.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.exchange import Order, OrderBook, OrderSide, OrderStatus, OrderType


# ---------------------------------------------------------------------------
# Order construction
# ---------------------------------------------------------------------------

class TestOrderConstruction:

    def test_valid_limit_buy(self):
        o = Order("o1", OrderSide.BUY, OrderType.LIMIT, quantity=10, price=100.0)
        assert o.remaining_quantity == 10
        assert o.filled_quantity == 0
        assert o.is_active

    def test_valid_limit_sell(self):
        o = Order("o2", OrderSide.SELL, OrderType.LIMIT, quantity=5, price=105.0)
        assert o.price == 105.0

    def test_market_order_no_price(self):
        o = Order("o3", OrderSide.BUY, OrderType.MARKET, quantity=3)
        assert o.price is None

    def test_limit_order_without_price_raises(self):
        with pytest.raises(ValueError, match="Limit orders must specify"):
            Order("o4", OrderSide.BUY, OrderType.LIMIT, quantity=10)

    def test_zero_quantity_raises(self):
        with pytest.raises(ValueError, match="quantity must be > 0"):
            Order("o5", OrderSide.BUY, OrderType.LIMIT, quantity=0, price=100.0)

    def test_negative_quantity_raises(self):
        with pytest.raises(ValueError):
            Order("o6", OrderSide.SELL, OrderType.LIMIT, quantity=-5, price=99.0)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="price must be > 0"):
            Order("o7", OrderSide.BUY, OrderType.LIMIT, quantity=1, price=-1.0)


# ---------------------------------------------------------------------------
# OrderBook insertion and best quote tracking
# ---------------------------------------------------------------------------

class TestOrderBookInsertion:

    def _make_book_with_bids(self) -> OrderBook:
        book = OrderBook()
        for i, price in enumerate([99.0, 100.0, 101.0], start=1):
            o = Order(f"bid{i}", OrderSide.BUY, OrderType.LIMIT, quantity=5, price=price)
            book.add_limit_order(o)
        return book

    def test_best_bid_is_highest(self):
        book = self._make_book_with_bids()
        assert book.best_bid == 101.0

    def test_best_ask_none_when_empty(self):
        book = OrderBook()
        assert book.best_ask is None

    def test_spread_none_when_one_side_empty(self):
        book = self._make_book_with_bids()
        assert book.spread is None

    def test_spread_computed_correctly(self):
        book = OrderBook()
        b = Order("b1", OrderSide.BUY, OrderType.LIMIT, quantity=1, price=100.0)
        a = Order("a1", OrderSide.SELL, OrderType.LIMIT, quantity=1, price=102.0)
        book.add_limit_order(b)
        book.add_limit_order(a)
        assert book.spread == pytest.approx(2.0)
        assert book.midprice == pytest.approx(101.0)

    def test_duplicate_order_id_raises(self):
        book = OrderBook()
        o = Order("dup", OrderSide.BUY, OrderType.LIMIT, quantity=1, price=100.0)
        book.add_limit_order(o)
        o2 = Order("dup", OrderSide.BUY, OrderType.LIMIT, quantity=1, price=100.0)
        with pytest.raises(ValueError, match="Duplicate order_id"):
            book.add_limit_order(o2)

    def test_len_tracks_resting_orders(self):
        book = OrderBook()
        for i in range(5):
            o = Order(f"o{i}", OrderSide.BUY, OrderType.LIMIT, quantity=1, price=float(i + 100))
            book.add_limit_order(o)
        assert len(book) == 5


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancellation:

    def test_cancel_removes_from_book(self):
        book = OrderBook()
        o = Order("c1", OrderSide.BUY, OrderType.LIMIT, quantity=10, price=99.0)
        book.add_limit_order(o)
        cancelled = book.cancel_order("c1")
        assert cancelled is not None
        assert cancelled.status == OrderStatus.CANCELLED
        assert len(book) == 0

    def test_cancel_unknown_id_returns_none(self):
        book = OrderBook()
        result = book.cancel_order("nonexistent")
        assert result is None

    def test_best_bid_updates_after_cancel(self):
        book = OrderBook()
        o_hi = Order("hi", OrderSide.BUY, OrderType.LIMIT, quantity=1, price=105.0)
        o_lo = Order("lo", OrderSide.BUY, OrderType.LIMIT, quantity=1, price=100.0)
        book.add_limit_order(o_hi)
        book.add_limit_order(o_lo)
        assert book.best_bid == 105.0
        book.cancel_order("hi")
        assert book.best_bid == 100.0


# ---------------------------------------------------------------------------
# Depth snapshot and imbalance
# ---------------------------------------------------------------------------

class TestDepth:

    def test_depth_snapshot_structure(self):
        book = OrderBook()
        for i in range(3):
            b = Order(f"b{i}", OrderSide.BUY, OrderType.LIMIT, quantity=float(i + 1), price=float(100 - i))
            a = Order(f"a{i}", OrderSide.SELL, OrderType.LIMIT, quantity=float(i + 1), price=float(102 + i))
            book.add_limit_order(b)
            book.add_limit_order(a)

        bids, asks = book.depth_snapshot(levels=10)
        # Bids sorted descending
        assert bids[0][0] > bids[-1][0]
        # Asks sorted ascending
        assert asks[0][0] < asks[-1][0]

    def test_order_imbalance_all_bids(self):
        book = OrderBook()
        for i in range(3):
            b = Order(f"b{i}", OrderSide.BUY, OrderType.LIMIT, quantity=10.0, price=float(100 - i))
            book.add_limit_order(b)
        imb = book.order_imbalance()
        assert imb == pytest.approx(1.0)

    def test_order_imbalance_balanced(self):
        book = OrderBook()
        b = Order("b1", OrderSide.BUY, OrderType.LIMIT, quantity=10, price=99.0)
        a = Order("a1", OrderSide.SELL, OrderType.LIMIT, quantity=10, price=101.0)
        book.add_limit_order(b)
        book.add_limit_order(a)
        assert book.order_imbalance() == pytest.approx(0.0)
