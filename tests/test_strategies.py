"""
tests/test_strategies.py
-------------------------
Unit and integration tests for Phase 3 market-making strategies.

Covers:
- MarketMakerMetrics: accounting correctness
- BaseMarketMaker: quote lifecycle, cancel management, PnL routing
- NaiveMarketMaker: symmetric quote generation
- InventoryAwareMarketMaker: skew behavior, spread widening
- Simulation integration: both MMs alongside noise + informed traders

Run with:
    pytest tests/test_strategies.py -v
"""

import sys
import os
import math
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategies import (
    NaiveMarketMaker,
    InventoryAwareMarketMaker,
    MarketMakerMetrics,
)
from src.strategies.base_market_maker import BaseMarketMaker
from src.agents import NoiseTrader, InformedTrader
from src.exchange import Order, OrderSide, OrderType
from src.exchange.trade import Trade
from src.simulation import MarketSimulation, FairValueConfig
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
):
    return MarketState(
        timestep=timestep,
        fair_value=fair_value,
        best_bid=best_bid,
        best_ask=best_ask,
        midprice=midprice,
        spread=spread,
    )


def empty_state(timestep=1, fair_value=100.0):
    return MarketState(
        timestep=timestep,
        fair_value=fair_value,
        best_bid=None,
        best_ask=None,
        midprice=None,
        spread=None,
    )


def make_trade(
    price: float,
    qty: float,
    aggressor_side: OrderSide,
    maker_id: str = "MAKER",
    taker_id: str = "TAKER",
) -> Trade:
    return Trade(
        trade_id=f"T{int(time.time()*1e6)}",
        timestamp=time.time(),
        price=price,
        quantity=qty,
        aggressor_side=aggressor_side,
        maker_order_id=maker_id,
        taker_order_id=taker_id,
    )


def run_sim(n_steps=200, seed=42, mm_list=None, n_noise=3, n_informed=1):
    """Helper to build and run a standard simulation."""
    noise = [NoiseTrader(f"NT{i}", activity_rate=0.6, random_seed=seed + i)
             for i in range(n_noise)]
    informed = [InformedTrader(f"IT{i}", activity_rate=0.5, random_seed=seed + 100 + i)
                for i in range(n_informed)]
    agents = noise + informed + (mm_list or [])
    sim = MarketSimulation(
        agents=agents,
        n_steps=n_steps,
        fair_value_config=FairValueConfig(
            initial_price=100.0, volatility=0.04, jump_prob=0.02
        ),
        random_seed=seed,
    )
    return sim.run()


# ─────────────────────────────────────────────────────────────
# MarketMakerMetrics
# ─────────────────────────────────────────────────────────────

class TestMarketMakerMetrics:

    def test_initial_state(self):
        m = MarketMakerMetrics(cash=50_000.0)
        assert m.inventory == 0.0
        assert m.cash == 50_000.0
        assert m.realized_pnl == 0.0
        assert m.total_pnl == 0.0
        assert m.fills_as_maker == 0

    def test_snapshot_appends_record(self):
        m = MarketMakerMetrics(cash=100_000.0)
        m.snapshot(timestep=1, bid_price=99.0, ask_price=101.0)
        assert len(m.step_records) == 1
        assert m.step_records[0].bid_price == 99.0
        assert m.step_records[0].quoted_spread == pytest.approx(2.0)

    def test_snapshot_none_prices(self):
        m = MarketMakerMetrics()
        m.snapshot(timestep=1, bid_price=None, ask_price=None)
        assert m.step_records[0].quoted_spread is None

    def test_to_dataframe_shape(self):
        m = MarketMakerMetrics()
        for t in range(10):
            m.inventory = float(t)
            m.snapshot(t, 99.0, 101.0)
        df = m.to_dataframe()
        assert df.shape[0] == 10
        assert "inventory" in df.columns

    def test_inventory_variance_zero_constant(self):
        m = MarketMakerMetrics()
        for _ in range(5):
            m.inventory_history.append(10.0)
        assert m.inventory_variance == pytest.approx(0.0)

    def test_inventory_variance_nonzero(self):
        m = MarketMakerMetrics()
        m.inventory_history = [0.0, 10.0, -10.0, 5.0]
        assert m.inventory_variance > 0

    def test_fill_rate_zero_quotes(self):
        m = MarketMakerMetrics()
        assert m.fill_rate == 0.0

    def test_fill_rate_computed(self):
        m = MarketMakerMetrics()
        m.quotes_posted = 10
        m.fills_as_maker = 3
        assert m.fill_rate == pytest.approx(0.3)

    def test_summary_dict_keys(self):
        m = MarketMakerMetrics()
        d = m.summary_dict()
        expected_keys = [
            "inventory", "realized_pnl", "unrealized_pnl", "total_pnl",
            "spread_capture", "fills_as_maker", "fills_as_taker",
            "volume_as_maker", "quotes_posted", "bid_fills", "ask_fills",
            "inventory_variance", "fill_rate",
        ]
        for k in expected_keys:
            assert k in d, f"Missing key: {k}"


# ─────────────────────────────────────────────────────────────
# NaiveMarketMaker — construction
# ─────────────────────────────────────────────────────────────

class TestNaiveMarketMakerConstruction:

    def test_valid_construction(self):
        nmm = NaiveMarketMaker("NMM1", half_spread=0.05)
        assert nmm.half_spread == pytest.approx(0.05)
        assert nmm.quoted_spread == pytest.approx(0.10)

    def test_half_spread_below_min_raises(self):
        with pytest.raises(ValueError, match="min_spread"):
            NaiveMarketMaker("NMM", half_spread=0.0001, min_spread=0.001)

    def test_quote_size_zero_raises(self):
        with pytest.raises(ValueError, match="quote_size"):
            NaiveMarketMaker("NMM", quote_size=0.0)

    def test_negative_quote_size_raises(self):
        with pytest.raises(ValueError):
            NaiveMarketMaker("NMM", quote_size=-1.0)


# ─────────────────────────────────────────────────────────────
# NaiveMarketMaker — quote generation
# ─────────────────────────────────────────────────────────────

class TestNaiveMarketMakerQuotes:

    def test_symmetric_quotes_around_mid(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.10)
        state = make_state(midprice=100.0)
        orders = nmm.act(state)

        assert len(orders) == 2
        bids = [o for o in orders if o.side == OrderSide.BUY]
        asks = [o for o in orders if o.side == OrderSide.SELL]
        assert len(bids) == 1
        assert len(asks) == 1
        assert bids[0].price == pytest.approx(99.90)
        assert asks[0].price == pytest.approx(100.10)

    def test_symmetric_quotes_around_fair_value(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05, use_fair_value=True)
        state = make_state(fair_value=100.0, midprice=99.0)  # FV != mid
        orders = nmm.act(state)
        bids = [o for o in orders if o.side == OrderSide.BUY]
        asks = [o for o in orders if o.side == OrderSide.SELL]
        assert bids[0].price == pytest.approx(99.95)
        assert asks[0].price == pytest.approx(100.05)

    def test_no_quotes_when_no_reference_price(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05, use_fair_value=False)
        state = empty_state(fair_value=None)
        # No midprice and no fair value
        state2 = MarketState(
            timestep=1, fair_value=None,
            best_bid=None, best_ask=None,
            midprice=None, spread=None,
        )
        orders = nmm.act(state2)
        assert orders == []

    def test_falls_back_to_fair_value_when_no_mid(self):
        """Even in midprice mode, falls back to fair_value if mid is unavailable."""
        nmm = NaiveMarketMaker("NMM", half_spread=0.05, use_fair_value=False)
        state = empty_state(fair_value=100.0)
        orders = nmm.act(state)
        assert len(orders) == 2

    def test_bid_always_below_ask(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        for mid in [50.0, 100.0, 200.0, 1000.0]:
            state = make_state(midprice=mid, fair_value=mid)
            nmm2 = NaiveMarketMaker("NMM2", half_spread=0.05)
            orders = nmm2.act(state)
            bids = [o for o in orders if o.side == OrderSide.BUY]
            asks = [o for o in orders if o.side == OrderSide.SELL]
            assert bids[0].price < asks[0].price

    def test_quote_size_is_correct(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05, quote_size=7.5)
        state = make_state(midprice=100.0)
        orders = nmm.act(state)
        for o in orders:
            assert o.quantity == pytest.approx(7.5)

    def test_all_orders_are_limit_type(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        state = make_state(midprice=100.0)
        orders = nmm.act(state)
        for o in orders:
            assert o.order_type == OrderType.LIMIT

    def test_order_ids_unique_across_steps(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        state = make_state(midprice=100.0)
        all_ids = []
        for t in range(1, 10):
            state2 = make_state(timestep=t, midprice=100.0)
            for o in nmm.act(state2):
                all_ids.append(o.order_id)
        assert len(all_ids) == len(set(all_ids))

    def test_quotes_posted_counter_increments(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        state = make_state(midprice=100.0)
        for _ in range(5):
            nmm.act(state)
        assert nmm.mm_metrics.quotes_posted == 5


# ─────────────────────────────────────────────────────────────
# NaiveMarketMaker — cancel management
# ─────────────────────────────────────────────────────────────

class TestNaiveMarketMakerCancels:

    def test_cancel_ids_queued_after_first_act(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        state = make_state(midprice=100.0)
        orders_t1 = nmm.act(state)
        ids_t1 = {o.order_id for o in orders_t1}

        # Before second act, flush should be empty
        assert nmm.flush_cancels() == []

        # Second act: should schedule old quotes for cancellation
        nmm.act(make_state(timestep=2, midprice=100.0))
        cancels = nmm.flush_cancels()
        assert set(cancels) == ids_t1

    def test_flush_cancels_clears_after_call(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        state = make_state(midprice=100.0)
        nmm.act(state)
        nmm.act(make_state(timestep=2, midprice=100.0))
        cancels1 = nmm.flush_cancels()
        cancels2 = nmm.flush_cancels()
        assert len(cancels1) > 0
        assert cancels2 == []


# ─────────────────────────────────────────────────────────────
# NaiveMarketMaker — fill handling and PnL
# ─────────────────────────────────────────────────────────────

class TestNaiveMarketMakerFills:

    def test_ask_fill_reduces_inventory_increases_cash(self):
        """Taker BUY hits our ask → we sold → inventory down, cash up."""
        nmm = NaiveMarketMaker("NMM", half_spread=0.05, initial_cash=100_000.0)
        trade = make_trade(101.0, 5.0, OrderSide.BUY)  # aggressor bought
        nmm.notify_fill(trade, as_maker=True)            # we are the ask (maker)

        assert nmm.mm_metrics.inventory == pytest.approx(-5.0)
        assert nmm.mm_metrics.cash == pytest.approx(100_000.0 + 505.0)
        assert nmm.mm_metrics.ask_fills == 1
        assert nmm.mm_metrics.bid_fills == 0
        assert nmm.mm_metrics.fills_as_maker == 1

    def test_bid_fill_increases_inventory_decreases_cash(self):
        """Taker SELL hits our bid → we bought → inventory up, cash down."""
        nmm = NaiveMarketMaker("NMM", half_spread=0.05, initial_cash=100_000.0)
        trade = make_trade(99.0, 5.0, OrderSide.SELL)   # aggressor sold
        nmm.notify_fill(trade, as_maker=True)             # we are the bid (maker)

        assert nmm.mm_metrics.inventory == pytest.approx(5.0)
        assert nmm.mm_metrics.cash == pytest.approx(100_000.0 - 495.0)
        assert nmm.mm_metrics.bid_fills == 1
        assert nmm.mm_metrics.ask_fills == 0

    def test_round_trip_pnl_positive_spread_capture(self):
        """Buy at 99, sell at 101 → spread_capture > 0."""
        nmm = NaiveMarketMaker("NMM", half_spread=0.05, quote_size=5.0, initial_cash=100_000.0)

        # Bid fill: we bought at 99
        nmm.notify_fill(make_trade(99.0, 5.0, OrderSide.SELL), as_maker=True)
        # Ask fill: we sold at 101
        nmm.notify_fill(make_trade(101.0, 5.0, OrderSide.BUY), as_maker=True)

        assert nmm.mm_metrics.inventory == pytest.approx(0.0)
        assert nmm.mm_metrics.spread_capture > 0

    def test_unrealized_pnl_update(self):
        nmm = NaiveMarketMaker("NMM", initial_cash=100_000.0)
        nmm.mm_metrics.inventory = 10.0
        nmm.update_unrealized_pnl(105.0)
        assert nmm.mm_metrics.unrealized_pnl == pytest.approx(1050.0)

    def test_realized_pnl_updates_after_fill(self):
        nmm = NaiveMarketMaker("NMM", initial_cash=100_000.0)
        trade = make_trade(101.0, 5.0, OrderSide.BUY)
        nmm.notify_fill(trade, as_maker=True)
        # realized_pnl = cash - initial_cash = (100_000 + 505) - 100_000 = 505
        assert nmm.mm_metrics.realized_pnl == pytest.approx(505.0)

    def test_volume_as_maker_tracked(self):
        nmm = NaiveMarketMaker("NMM")
        nmm.notify_fill(make_trade(100.0, 3.0, OrderSide.BUY), as_maker=True)
        nmm.notify_fill(make_trade(100.0, 7.0, OrderSide.SELL), as_maker=True)
        assert nmm.mm_metrics.volume_as_maker == pytest.approx(10.0)


# ─────────────────────────────────────────────────────────────
# InventoryAwareMarketMaker — construction
# ─────────────────────────────────────────────────────────────

class TestInventoryAwareConstruction:

    def test_valid_construction(self):
        iamm = InventoryAwareMarketMaker("IAMM", half_spread=0.05, inventory_skew_factor=0.01)
        assert iamm.half_spread == pytest.approx(0.05)
        assert iamm.inventory_skew_factor == pytest.approx(0.01)

    def test_negative_skew_factor_raises(self):
        with pytest.raises(ValueError, match="inventory_skew_factor"):
            InventoryAwareMarketMaker("IAMM", inventory_skew_factor=-0.01)

    def test_zero_max_inventory_raises(self):
        with pytest.raises(ValueError, match="max_inventory"):
            InventoryAwareMarketMaker("IAMM", max_inventory=0.0)

    def test_negative_spread_widening_raises(self):
        with pytest.raises(ValueError, match="spread_widening"):
            InventoryAwareMarketMaker("IAMM", spread_widening=-0.1)

    def test_half_spread_below_min_raises(self):
        with pytest.raises(ValueError, match="min_spread"):
            InventoryAwareMarketMaker("IAMM", half_spread=0.0001, min_spread=0.01)


# ─────────────────────────────────────────────────────────────
# InventoryAwareMarketMaker — skew behavior
# ─────────────────────────────────────────────────────────────

class TestInventorySkewBehavior:

    def _iamm(self, **kwargs) -> InventoryAwareMarketMaker:
        defaults = dict(
            agent_id="IAMM",
            half_spread=0.10,
            inventory_skew_factor=0.02,
            max_inventory=50.0,
            spread_widening=0.0,  # disable widening for isolation
        )
        defaults.update(kwargs)
        return InventoryAwareMarketMaker(**defaults)

    def test_zero_inventory_quotes_symmetric(self):
        """At zero inventory, quotes should be symmetric around mid."""
        iamm = self._iamm()
        iamm.mm_metrics.inventory = 0.0
        state = make_state(midprice=100.0, fair_value=100.0)
        orders = iamm.act(state)
        bids = [o for o in orders if o.side == OrderSide.BUY]
        asks = [o for o in orders if o.side == OrderSide.SELL]
        mid_quote = (bids[0].price + asks[0].price) / 2.0
        assert mid_quote == pytest.approx(100.0, abs=1e-4)

    def test_long_inventory_shifts_quotes_down(self):
        """Long inventory → reservation price below mid → both quotes lower."""
        iamm_zero = self._iamm()
        iamm_long = self._iamm()

        state = make_state(midprice=100.0, fair_value=100.0)
        iamm_zero.mm_metrics.inventory = 0.0
        iamm_long.mm_metrics.inventory = 20.0

        orders_zero = iamm_zero.act(state)
        orders_long = iamm_long.act(make_state(timestep=2, midprice=100.0, fair_value=100.0))

        bid_zero = next(o.price for o in orders_zero if o.side == OrderSide.BUY)
        bid_long = next(o.price for o in orders_long if o.side == OrderSide.BUY)
        ask_zero = next(o.price for o in orders_zero if o.side == OrderSide.SELL)
        ask_long = next(o.price for o in orders_long if o.side == OrderSide.SELL)

        assert bid_long < bid_zero, "Long inventory should lower bid"
        assert ask_long < ask_zero, "Long inventory should lower ask"

    def test_short_inventory_shifts_quotes_up(self):
        """Short inventory → reservation price above mid → both quotes higher."""
        iamm_zero = self._iamm()
        iamm_short = self._iamm()

        state_zero = make_state(midprice=100.0, fair_value=100.0)
        state_short = make_state(timestep=2, midprice=100.0, fair_value=100.0)

        iamm_zero.mm_metrics.inventory = 0.0
        iamm_short.mm_metrics.inventory = -20.0

        orders_zero = iamm_zero.act(state_zero)
        orders_short = iamm_short.act(state_short)

        bid_zero = next(o.price for o in orders_zero if o.side == OrderSide.BUY)
        bid_short = next(o.price for o in orders_short if o.side == OrderSide.BUY)

        assert bid_short > bid_zero, "Short inventory should raise bid"

    def test_skew_monotone_with_inventory(self):
        """Increasing long inventory → strictly decreasing reservation price."""
        reservations = []
        for inv in [0, 10, 20, 30, 40]:
            iamm = self._iamm()
            iamm.mm_metrics.inventory = float(inv)
            r = iamm.reservation_price(100.0)
            reservations.append(r)

        for i in range(1, len(reservations)):
            assert reservations[i] < reservations[i - 1], \
                f"Reservation price not decreasing: {reservations}"

    def test_inventory_clamped_at_max(self):
        """Inventory beyond max_inventory gives same skew as max_inventory."""
        iamm = self._iamm(max_inventory=50.0)
        iamm.mm_metrics.inventory = 50.0
        r_at_max = iamm.reservation_price(100.0)

        iamm2 = self._iamm(max_inventory=50.0)
        iamm2.mm_metrics.inventory = 200.0  # way over max
        r_over_max = iamm2.reservation_price(100.0)

        assert r_at_max == pytest.approx(r_over_max)

    def test_bid_always_below_ask(self):
        """Must never post a crossed market."""
        for inv in [-100, -50, 0, 50, 100]:
            iamm = InventoryAwareMarketMaker("IAMM", half_spread=0.05,
                                             inventory_skew_factor=0.01, max_inventory=50.0)
            iamm.mm_metrics.inventory = float(inv)
            state = make_state(timestep=1, midprice=100.0, fair_value=100.0)
            orders = iamm.act(state)
            bids = [o for o in orders if o.side == OrderSide.BUY]
            asks = [o for o in orders if o.side == OrderSide.SELL]
            if bids and asks:
                assert bids[0].price < asks[0].price, \
                    f"Crossed market at inventory={inv}"


# ─────────────────────────────────────────────────────────────
# InventoryAwareMarketMaker — spread widening
# ─────────────────────────────────────────────────────────────

class TestSpreadWidening:

    def test_spread_widens_with_inventory(self):
        """Effective spread should be larger at high inventory."""
        iamm_low = InventoryAwareMarketMaker(
            "IAMM", half_spread=0.05, inventory_skew_factor=0.0,
            max_inventory=50.0, spread_widening=1.0
        )
        iamm_high = InventoryAwareMarketMaker(
            "IAMM2", half_spread=0.05, inventory_skew_factor=0.0,
            max_inventory=50.0, spread_widening=1.0
        )
        iamm_low.mm_metrics.inventory = 0.0
        iamm_high.mm_metrics.inventory = 50.0

        assert iamm_high.effective_half_spread() > iamm_low.effective_half_spread()

    def test_no_widening_gives_fixed_spread(self):
        iamm = InventoryAwareMarketMaker(
            "IAMM", half_spread=0.05, spread_widening=0.0
        )
        iamm.mm_metrics.inventory = 1000.0
        assert iamm.effective_half_spread() == pytest.approx(0.05)

    def test_widening_doubles_at_max(self):
        """spread_widening=1.0 should double the half-spread at max_inventory."""
        iamm = InventoryAwareMarketMaker(
            "IAMM", half_spread=0.10, max_inventory=50.0, spread_widening=1.0
        )
        iamm.mm_metrics.inventory = 50.0
        # effective = 0.10 * (1 + 1.0 * 1.0) = 0.20
        assert iamm.effective_half_spread() == pytest.approx(0.20)


# ─────────────────────────────────────────────────────────────
# Simulation integration
# ─────────────────────────────────────────────────────────────

class TestSimulationIntegration:

    def test_naive_mm_runs_without_error(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05, quote_size=5.0)
        result = run_sim(n_steps=100, seed=1, mm_list=[nmm])
        assert result.n_steps == 100

    def test_inventory_aware_mm_runs_without_error(self):
        iamm = InventoryAwareMarketMaker("IAMM", half_spread=0.05, quote_size=5.0)
        result = run_sim(n_steps=100, seed=2, mm_list=[iamm])
        assert result.n_steps == 100

    def test_both_mms_run_together(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        iamm = InventoryAwareMarketMaker("IAMM", half_spread=0.05)
        result = run_sim(n_steps=150, seed=3, mm_list=[nmm, iamm])
        assert result.n_steps == 150

    def test_mm_posts_quotes(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        run_sim(n_steps=100, seed=4, mm_list=[nmm])
        assert nmm.mm_metrics.quotes_posted > 0

    def test_mm_receives_fills(self):
        """Market makers should receive fills from aggressive noise/informed traders."""
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        run_sim(n_steps=300, seed=5, mm_list=[nmm], n_noise=3, n_informed=1)
        assert nmm.mm_metrics.fills_as_maker > 0

    def test_mm_fill_balance_bid_and_ask(self):
        """Over enough steps, MM should fill on both sides."""
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        run_sim(n_steps=400, seed=6, mm_list=[nmm], n_noise=4)
        assert nmm.mm_metrics.bid_fills > 0
        assert nmm.mm_metrics.ask_fills > 0

    def test_inventory_aware_lower_variance_than_naive(self):
        """
        Core property: the inventory-aware MM should have lower inventory
        variance than the naive MM over a long run.
        Both use identical parameters except for the skew factor.
        """
        nmm = NaiveMarketMaker("NMM", half_spread=0.06, quote_size=5.0)
        iamm = InventoryAwareMarketMaker(
            "IAMM",
            half_spread=0.06,
            inventory_skew_factor=0.02,
            max_inventory=40.0,
            spread_widening=0.5,
            quote_size=5.0,
        )
        result = run_sim(n_steps=500, seed=42, mm_list=[nmm, iamm],
                         n_noise=4, n_informed=2)

        nmm_var = nmm.mm_metrics.inventory_variance
        iamm_var = iamm.mm_metrics.inventory_variance

        # IAMM should control inventory better
        assert iamm_var < nmm_var, (
            f"IAMM inventory variance ({iamm_var:.2f}) should be less than "
            f"NMM ({nmm_var:.2f})"
        )

    def test_mm_df_has_correct_length(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        run_sim(n_steps=150, seed=7, mm_list=[nmm])
        df = nmm.mm_metrics.to_dataframe()
        assert len(df) == 150

    def test_mm_pnl_history_length(self):
        iamm = InventoryAwareMarketMaker("IAMM", half_spread=0.05)
        run_sim(n_steps=100, seed=8, mm_list=[iamm])
        assert len(iamm.mm_metrics.pnl_history) == 100

    def test_cumulative_trades_includes_mm_fills(self):
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        result = run_sim(n_steps=200, seed=9, mm_list=[nmm])
        df = result.metrics.to_dataframe()
        total_trades = int(df["cumulative_trades"].iloc[-1])
        # All trades in the simulation include those with NMM as maker
        assert total_trades >= nmm.mm_metrics.fills_as_maker

    def test_stale_quotes_are_cancelled(self):
        """
        Verify the simulation correctly processes MM cancels:
        book should not accumulate unbounded resting MM orders.
        """
        nmm = NaiveMarketMaker("NMM", half_spread=0.05)
        noise = [NoiseTrader("NT1", activity_rate=0.3, random_seed=1)]
        sim = MarketSimulation(
            agents=noise + [nmm],
            n_steps=100,
            fair_value_config=FairValueConfig(initial_price=100.0),
            random_seed=1,
        )
        result = sim.run()
        # At most 2 MM orders should be resting (current bid + ask)
        book = result.engine.book
        mm_resting = [
            oid for oid in book._order_map
            if oid.startswith("NMM")
        ]
        assert len(mm_resting) <= 2, \
            f"Too many stale MM orders in book: {len(mm_resting)}"

    def test_reproducibility(self):
        """Two identical runs should produce identical metrics."""
        def _run(seed):
            nmm = NaiveMarketMaker("NMM", half_spread=0.05, quote_size=5.0)
            run_sim(n_steps=100, seed=seed, mm_list=[nmm])
            return nmm.mm_metrics.fills_as_maker, nmm.mm_metrics.inventory

        r1 = _run(99)
        r2 = _run(99)
        assert r1 == r2

    def test_informed_creates_adverse_selection(self):
        """
        Informed traders create adverse selection: the MM fills more often
        (they trade aggressively) and earns worse average PnL across seeds.

        We run multiple seeds and check that PnL is WORSE with informed traders
        present in the majority of trials — this is the correct statistical test
        for an adversarial effect that is real but has per-seed variance.
        """
        wins_pnl = 0      # count seeds where informed scenario has lower PnL
        wins_fills = 0    # count seeds where informed scenario has more fills

        for seed in range(42, 47):  # 5 seeds
            nmm_i = NaiveMarketMaker("NMM_inf", half_spread=0.05)
            nmm_n = NaiveMarketMaker("NMM_noise", half_spread=0.05)

            noise = [NoiseTrader(f"NT{i}", activity_rate=0.6, random_seed=i) for i in range(3)]
            informed = [InformedTrader("IT1", signal_threshold=0.05,
                                       aggression=0.9, activity_rate=0.8, random_seed=99)]
            sim1 = MarketSimulation(
                agents=noise + informed + [nmm_i],
                n_steps=500,
                fair_value_config=FairValueConfig(volatility=0.05, jump_prob=0.05),
                random_seed=seed,
            )
            sim1.run()

            noise2 = [NoiseTrader(f"NT{i}", activity_rate=0.6, random_seed=i) for i in range(3)]
            sim2 = MarketSimulation(
                agents=noise2 + [nmm_n],
                n_steps=500,
                fair_value_config=FairValueConfig(volatility=0.05, jump_prob=0.05),
                random_seed=seed,
            )
            sim2.run()

            if nmm_i.mm_metrics.total_pnl < nmm_n.mm_metrics.total_pnl:
                wins_pnl += 1
            if nmm_i.mm_metrics.fills_as_maker > nmm_n.mm_metrics.fills_as_maker:
                wins_fills += 1

        # Informed traders should consistently generate more fills (they are aggressive)
        assert wins_fills >= 4, (
            f"Expected informed scenario to have more fills in >=4/5 seeds, got {wins_fills}/5"
        )
        # Informed traders should hurt MM PnL in the majority of trials
        assert wins_pnl >= 3, (
            f"Expected informed scenario to have worse PnL in >=3/5 seeds, got {wins_pnl}/5"
        )
