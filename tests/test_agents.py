"""
tests/test_agents.py
--------------------
Unit tests for NoiseTrader, InformedTrader, and BaseAgent.

Run with:
    pytest tests/test_agents.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import NoiseTrader, InformedTrader
from src.exchange import OrderSide, OrderType
from src.simulation.market_state import MarketState


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_state(
    timestep=1,
    fair_value=100.0,
    best_bid=99.0,
    best_ask=101.0,
    midprice=100.0,
    spread=2.0,
    order_imbalance=0.0,
):
    return MarketState(
        timestep=timestep,
        fair_value=fair_value,
        best_bid=best_bid,
        best_ask=best_ask,
        midprice=midprice,
        spread=spread,
        order_imbalance=order_imbalance,
    )


def empty_state(timestep=1, fair_value=100.0):
    """State with no book — both sides empty."""
    return MarketState(
        timestep=timestep,
        fair_value=fair_value,
        best_bid=None,
        best_ask=None,
        midprice=None,
        spread=None,
    )


# ─────────────────────────────────────────────────────────────
# NoiseTrader
# ─────────────────────────────────────────────────────────────

class TestNoiseTrader:

    def test_construction_defaults(self):
        nt = NoiseTrader("NT1")
        assert nt.agent_id == "NT1"
        assert nt.activity_rate == pytest.approx(0.4)
        assert nt.market_order_prob == pytest.approx(0.25)

    def test_invalid_activity_rate_raises(self):
        with pytest.raises(ValueError):
            NoiseTrader("NT_bad", activity_rate=1.5)

    def test_invalid_market_order_prob_raises(self):
        with pytest.raises(ValueError):
            NoiseTrader("NT_bad", market_order_prob=-0.1)

    def test_act_returns_list(self):
        nt = NoiseTrader("NT1", random_seed=42)
        state = make_state()
        result = nt.act(state)
        assert isinstance(result, list)

    def test_act_zero_activity_never_trades(self):
        """activity_rate=0 is invalid; activity_rate~=0 should almost never trade."""
        nt = NoiseTrader("NT1", activity_rate=0.001, random_seed=1)
        state = make_state()
        results = [nt.act(state) for _ in range(20)]
        # Most should be empty
        empty_count = sum(1 for r in results if r == [])
        assert empty_count >= 15  # stochastic but high probability

    def test_act_full_activity_always_produces_orders(self):
        nt = NoiseTrader("NT1", activity_rate=1.0, random_seed=99)
        state = make_state()
        for _ in range(20):
            orders = nt.act(state)
            assert len(orders) >= 1

    def test_limit_orders_have_valid_price(self):
        nt = NoiseTrader("NT1", activity_rate=1.0, market_order_prob=0.0, random_seed=7)
        state = make_state()
        for _ in range(30):
            for order in nt.act(state):
                assert order.price is not None
                assert order.price > 0
                assert order.order_type == OrderType.LIMIT

    def test_market_orders_have_no_price(self):
        """With market_order_prob=1, all orders should be market orders."""
        nt = NoiseTrader("NT1", activity_rate=1.0, market_order_prob=1.0, random_seed=3)
        state = make_state()
        orders_seen = []
        for _ in range(30):
            orders_seen.extend(nt.act(state))
        market_orders = [o for o in orders_seen if o.order_type == OrderType.MARKET]
        assert len(market_orders) > 0
        for o in market_orders:
            assert o.price is None

    def test_no_market_order_when_book_empty(self):
        """Market orders require opposing book depth; none submitted on empty book."""
        nt = NoiseTrader("NT1", activity_rate=1.0, market_order_prob=1.0, random_seed=5)
        state = empty_state()
        for _ in range(20):
            orders = nt.act(state)
            for o in orders:
                assert o.order_type != OrderType.MARKET

    def test_order_qty_positive(self):
        nt = NoiseTrader("NT1", activity_rate=1.0, random_seed=11)
        state = make_state()
        for _ in range(30):
            for order in nt.act(state):
                assert order.quantity > 0

    def test_order_ids_unique(self):
        nt = NoiseTrader("NT1", activity_rate=1.0, random_seed=13)
        state = make_state()
        ids = []
        for _ in range(50):
            for o in nt.act(state):
                ids.append(o.order_id)
        assert len(ids) == len(set(ids))

    def test_metrics_orders_submitted_tracked(self):
        nt = NoiseTrader("NT1", activity_rate=1.0, market_order_prob=0.0, random_seed=17)
        state = make_state()
        for _ in range(10):
            nt.act(state)
        assert nt.metrics.orders_submitted >= 1

    def test_max_resting_orders_triggers_cancel(self):
        """When resting queue is full, flush_cancels should return an id."""
        nt = NoiseTrader(
            "NT1",
            activity_rate=1.0,
            market_order_prob=0.0,
            max_resting_orders=2,
            random_seed=19,
        )
        state = make_state()
        all_cancels = []
        for _ in range(20):
            nt.act(state)
            all_cancels.extend(nt.flush_cancels())
        # After filling the queue repeatedly, we should have accumulated cancels
        assert len(all_cancels) > 0

    def test_reproducibility_with_same_seed(self):
        s = make_state()
        nt1 = NoiseTrader("A", activity_rate=1.0, random_seed=42)
        nt2 = NoiseTrader("A", activity_rate=1.0, random_seed=42)
        # Same seed → same first order quantity and side
        o1 = nt1.act(s)
        o2 = nt2.act(s)
        if o1 and o2:
            assert o1[0].side == o2[0].side


# ─────────────────────────────────────────────────────────────
# InformedTrader
# ─────────────────────────────────────────────────────────────

class TestInformedTrader:

    def test_construction_defaults(self):
        it = InformedTrader("IT1")
        assert it.agent_id == "IT1"
        assert it.signal_threshold == pytest.approx(0.10)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            InformedTrader("IT_bad", signal_threshold=0.0)

    def test_invalid_aggression_raises(self):
        with pytest.raises(ValueError):
            InformedTrader("IT_bad", aggression=1.5)

    def test_no_trade_below_threshold(self):
        """Fair value only marginally above mid — should not trade."""
        it = InformedTrader("IT1", signal_threshold=0.5, activity_rate=1.0, random_seed=42)
        # Deviation = 0.05, threshold = 0.5 → no trade
        state = make_state(fair_value=100.05, midprice=100.0)
        for _ in range(20):
            orders = it.act(state)
            assert orders == []

    def test_trades_above_threshold_buy_side(self):
        """Fair value well above mid → informed trader should BUY."""
        it = InformedTrader(
            "IT1",
            signal_threshold=0.10,
            aggression=1.0,   # always market
            activity_rate=1.0,
            random_seed=1,
        )
        # Deviation = +2.0 >> threshold 0.10 → should buy
        state = make_state(fair_value=102.0, midprice=100.0, best_ask=101.0)
        orders = it.act(state)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_trades_above_threshold_sell_side(self):
        """Fair value well below mid → informed trader should SELL."""
        it = InformedTrader(
            "IT1",
            signal_threshold=0.10,
            aggression=1.0,
            activity_rate=1.0,
            random_seed=2,
        )
        state = make_state(fair_value=98.0, midprice=100.0, best_bid=99.0)
        orders = it.act(state)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_no_trade_when_no_midprice(self):
        it = InformedTrader("IT1", activity_rate=1.0, random_seed=3)
        state = empty_state(fair_value=110.0)  # huge deviation, but no mid
        for _ in range(10):
            assert it.act(state) == []

    def test_market_order_when_aggression_1(self):
        it = InformedTrader(
            "IT1", signal_threshold=0.05, aggression=1.0, activity_rate=1.0, random_seed=5
        )
        state = make_state(fair_value=102.0, best_ask=101.0)
        orders = it.act(state)
        assert any(o.order_type == OrderType.MARKET for o in orders)

    def test_limit_order_when_aggression_0(self):
        it = InformedTrader(
            "IT1", signal_threshold=0.05, aggression=0.0, activity_rate=1.0, random_seed=6
        )
        state = make_state(fair_value=102.0, best_ask=101.0)
        orders = it.act(state)
        assert any(o.order_type == OrderType.LIMIT for o in orders)

    def test_inventory_penalty_reduces_size(self):
        """High inventory should reduce the order size."""
        it = InformedTrader(
            "IT1",
            signal_threshold=0.05,
            aggression=1.0,
            base_trade_size=10.0,
            max_inventory=50.0,
            activity_rate=1.0,
            random_seed=7,
        )
        state_buy = make_state(fair_value=102.0, best_ask=101.0)
        # Normal inventory → normal size
        it.metrics.inventory = 0.0
        orders_normal = it.act(state_buy)

        # High inventory → should trade smaller
        it.metrics.inventory = 48.0
        orders_penalised = it.act(state_buy)

        if orders_normal and orders_penalised:
            assert orders_penalised[0].quantity <= orders_normal[0].quantity

    def test_last_signal_updated(self):
        it = InformedTrader("IT1", activity_rate=1.0, random_seed=8)
        assert it.last_signal is None
        state = make_state(fair_value=100.5)
        it.act(state)
        assert it.last_signal is not None


# ─────────────────────────────────────────────────────────────
# BaseAgent fill notification
# ─────────────────────────────────────────────────────────────

class TestFillNotification:

    def _make_trade(self, price, qty, aggressor_side, maker_id="MAKER", taker_id="TAKER"):
        from src.exchange.trade import Trade
        import time
        return Trade(
            trade_id="T001",
            timestamp=time.time(),
            price=price,
            quantity=qty,
            aggressor_side=aggressor_side,
            maker_order_id=maker_id,
            taker_order_id=taker_id,
        )

    def test_buy_fill_increases_inventory(self):
        """Agent fills as taker on a BUY → inventory increases."""
        nt = NoiseTrader("NT1", random_seed=1)
        trade = self._make_trade(100.0, 5.0, OrderSide.BUY, maker_id="MAKER", taker_id="NT1")
        nt.notify_fill(trade, as_maker=False)  # as taker on a buy
        assert nt.metrics.inventory == pytest.approx(5.0)
        assert nt.metrics.cash == pytest.approx(100_000.0 - 500.0)

    def test_sell_fill_decreases_inventory(self):
        """Agent is the TAKER on a SELL (they sold aggressively) → inventory decreases."""
        nt = NoiseTrader("NT1", random_seed=1)
        # aggressor=SELL, as_maker=False → agent is the sell aggressor → agent_sold=True
        trade = self._make_trade(100.0, 3.0, OrderSide.SELL, maker_id="OTHER", taker_id="NT1")
        nt.notify_fill(trade, as_maker=False)  # NT1 is taker on a SELL
        assert nt.metrics.inventory == pytest.approx(-3.0)
        assert nt.metrics.cash == pytest.approx(100_000.0 + 300.0)

    def test_volume_tracked(self):
        nt = NoiseTrader("NT1", random_seed=1)
        trade = self._make_trade(100.0, 7.0, OrderSide.BUY)
        nt.notify_fill(trade, as_maker=False)
        assert nt.metrics.volume_traded == pytest.approx(7.0)
        assert nt.metrics.trades_executed == 1

    def test_unrealised_pnl_update(self):
        nt = NoiseTrader("NT1", random_seed=1)
        nt.metrics.inventory = 10.0
        nt.update_unrealized_pnl(105.0)
        assert nt.metrics.unrealized_pnl == pytest.approx(1050.0)
