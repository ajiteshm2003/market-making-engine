"""
tests/test_phase4.py
---------------------
Phase 4 test suite — Avellaneda-Stoikov Market Maker.

Covers:
  - A-S math: reservation price, optimal spread, decomposition
  - Volatility estimator: correctness, bounds, warm-up, EWM
  - Arrival intensity estimator: k dynamics
  - AvellanedaStoikovMarketMaker: quote generation, skew, spread widening
  - Time horizon modes: fixed vs decaying
  - Simulation integration: all three strategies side-by-side
  - Analytics: Sharpe, drawdown, comparison table
  - Reproducibility

Run with:
    pytest tests/test_phase4.py -v
"""

import math
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.avellaneda_stoikov_math import (
    reservation_price,
    optimal_half_spread,
    compute_quotes,
    spread_decomposition,
    implied_k_from_spread,
    sensitivity_analysis,
)
from src.models.volatility import RollingVolatilityEstimator, VolatilityConfig
from src.models.arrival_intensity import ArrivalIntensityEstimator, ArrivalIntensityConfig
from src.models.analytics import sharpe_ratio, max_drawdown, strategy_comparison
from src.strategies import (
    AvellanedaStoikovMarketMaker,
    ASConfig,
    HorizonMode,
    NaiveMarketMaker,
    InventoryAwareMarketMaker,
)
from src.agents import NoiseTrader, InformedTrader
from src.exchange.order import OrderSide
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
    volume_this_step=5.0,
    trade_count=10,
):
    return MarketState(
        timestep=timestep,
        fair_value=fair_value,
        best_bid=best_bid,
        best_ask=best_ask,
        midprice=midprice,
        spread=spread,
        volume_this_step=volume_this_step,
        trade_count=trade_count,
    )


def make_trade(price, qty, aggressor_side, maker_id="MAKER", taker_id="TAKER"):
    return Trade(
        trade_id=f"T{int(time.time()*1e9)}",
        timestamp=time.time(),
        price=price,
        quantity=qty,
        aggressor_side=aggressor_side,
        maker_order_id=maker_id,
        taker_order_id=taker_id,
    )


def make_asmm(gamma=0.1, horizon=1.0, mode=HorizonMode.FIXED, total_steps=500,
              quote_size=5.0, use_fv=False, **kw) -> AvellanedaStoikovMarketMaker:
    config = ASConfig(
        gamma=gamma,
        horizon_mode=mode,
        horizon_steps=horizon,
        total_steps=total_steps,
        use_fair_value=use_fv,
        **kw
    )
    return AvellanedaStoikovMarketMaker("ASMM", config=config, quote_size=quote_size)


def run_sim(n_steps=300, seed=42, mm_list=None, n_noise=3, n_informed=1, vol=0.04, jump=0.025):
    noise    = [NoiseTrader(f"NT{i}", activity_rate=0.6, random_seed=seed + i) for i in range(n_noise)]
    informed = [InformedTrader(f"IT{i}", activity_rate=0.55, random_seed=seed+100+i) for i in range(n_informed)]
    agents   = noise + informed + (mm_list or [])
    sim = MarketSimulation(
        agents=agents,
        n_steps=n_steps,
        fair_value_config=FairValueConfig(
            initial_price=100.0, volatility=vol, jump_prob=jump
        ),
        random_seed=seed,
    )
    return sim.run()


# ─────────────────────────────────────────────────────────────
# A-S Math: reservation_price
# ─────────────────────────────────────────────────────────────

class TestReservationPrice:

    def test_zero_inventory_equals_midprice(self):
        r = reservation_price(100.0, inventory=0.0, gamma=0.1, sigma=0.05, time_remaining=1.0)
        assert r == pytest.approx(100.0)

    def test_long_inventory_below_midprice(self):
        """Long inventory → r < S (want to sell → lower prices)."""
        r = reservation_price(100.0, inventory=10.0, gamma=0.1, sigma=0.05, time_remaining=1.0)
        assert r < 100.0

    def test_short_inventory_above_midprice(self):
        """Short inventory → r > S (want to buy → raise prices)."""
        r = reservation_price(100.0, inventory=-10.0, gamma=0.1, sigma=0.05, time_remaining=1.0)
        assert r > 100.0

    def test_penalty_linear_in_inventory(self):
        """Doubling inventory should double the penalty."""
        r1 = reservation_price(100.0, inventory=5.0,  gamma=0.1, sigma=0.05, time_remaining=1.0)
        r2 = reservation_price(100.0, inventory=10.0, gamma=0.1, sigma=0.05, time_remaining=1.0)
        penalty1 = 100.0 - r1
        penalty2 = 100.0 - r2
        assert penalty2 == pytest.approx(2 * penalty1, rel=1e-6)

    def test_penalty_linear_in_gamma(self):
        """Doubling γ should double the penalty."""
        r1 = reservation_price(100.0, inventory=5.0, gamma=0.1, sigma=0.05, time_remaining=1.0)
        r2 = reservation_price(100.0, inventory=5.0, gamma=0.2, sigma=0.05, time_remaining=1.0)
        assert (100.0 - r2) == pytest.approx(2 * (100.0 - r1), rel=1e-6)

    def test_penalty_quadratic_in_sigma(self):
        """Doubling σ should quadruple the penalty (σ² term)."""
        r1 = reservation_price(100.0, inventory=5.0, gamma=0.1, sigma=0.05, time_remaining=1.0)
        r2 = reservation_price(100.0, inventory=5.0, gamma=0.1, sigma=0.10, time_remaining=1.0)
        assert (100.0 - r2) == pytest.approx(4 * (100.0 - r1), rel=1e-6)

    def test_penalty_linear_in_time_remaining(self):
        """Doubling T-t should double the penalty."""
        r1 = reservation_price(100.0, inventory=5.0, gamma=0.1, sigma=0.05, time_remaining=1.0)
        r2 = reservation_price(100.0, inventory=5.0, gamma=0.1, sigma=0.05, time_remaining=2.0)
        assert (100.0 - r2) == pytest.approx(2 * (100.0 - r1), rel=1e-6)

    def test_zero_time_remaining_equals_midprice(self):
        """At T-t=0, penalty is zero regardless of inventory."""
        r = reservation_price(100.0, inventory=999.0, gamma=10.0, sigma=5.0, time_remaining=0.0)
        assert r == pytest.approx(100.0)

    def test_negative_time_remaining_raises(self):
        with pytest.raises(ValueError):
            reservation_price(100.0, 0.0, 0.1, 0.05, time_remaining=-1.0)

    def test_zero_gamma_raises(self):
        with pytest.raises(ValueError):
            reservation_price(100.0, 0.0, gamma=0.0, sigma=0.05, time_remaining=1.0)

    def test_exact_formula_value(self):
        """Exact numerical check: r = S - q × γ × σ² × (T-t)."""
        S, q, gamma, sigma, Tt = 100.0, 5.0, 0.1, 0.05, 2.0
        expected = S - q * gamma * sigma**2 * Tt
        assert reservation_price(S, q, gamma, sigma, Tt) == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────
# A-S Math: optimal_half_spread
# ─────────────────────────────────────────────────────────────

class TestOptimalHalfSpread:

    def test_always_positive(self):
        d = optimal_half_spread(gamma=0.1, sigma=0.05, time_remaining=1.0, k=1.5)
        assert d > 0

    def test_increases_with_sigma(self):
        d1 = optimal_half_spread(0.1, sigma=0.05, time_remaining=1.0, k=1.5)
        d2 = optimal_half_spread(0.1, sigma=0.15, time_remaining=1.0, k=1.5)
        assert d2 > d1

    def test_increases_with_time_remaining(self):
        d1 = optimal_half_spread(0.1, 0.05, time_remaining=0.5, k=1.5)
        d2 = optimal_half_spread(0.1, 0.05, time_remaining=2.0, k=1.5)
        assert d2 > d1

    def test_decreases_with_k(self):
        """Higher k (more eager takers) → tighter spread needed."""
        d1 = optimal_half_spread(0.1, 0.05, 1.0, k=0.5)  # low k
        d2 = optimal_half_spread(0.1, 0.05, 1.0, k=5.0)  # high k
        assert d1 > d2

    def test_zero_time_remaining_has_liquidity_premium_only(self):
        """At T-t=0, risk_premium = 0, only liquidity premium remains."""
        d = optimal_half_spread(gamma=0.1, sigma=0.05, time_remaining=0.0, k=1.5)
        liquidity_premium = (1.0 / 0.1) * math.log(1.0 + 0.1 / 1.5)
        assert d == pytest.approx(liquidity_premium, rel=1e-6)

    def test_exact_formula_value(self):
        gamma, sigma, Tt, k = 0.1, 0.05, 1.0, 1.5
        expected = (gamma * sigma**2 * Tt) / 2 + (1/gamma) * math.log(1 + gamma/k)
        assert optimal_half_spread(gamma, sigma, Tt, k) == pytest.approx(expected)

    def test_zero_sigma_gives_liquidity_premium_only(self):
        """When σ=0, risk_premium=0, spread = pure liquidity premium."""
        d = optimal_half_spread(gamma=0.1, sigma=0.0, time_remaining=1.0, k=1.5)
        liquidity = (1.0/0.1) * math.log(1 + 0.1/1.5)
        assert d == pytest.approx(liquidity, rel=1e-6)

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            optimal_half_spread(0.1, 0.05, 1.0, k=0.0)

    def test_zero_gamma_raises(self):
        with pytest.raises(ValueError):
            optimal_half_spread(gamma=0.0, sigma=0.05, time_remaining=1.0, k=1.5)

    def test_spread_decomposition_sums_correctly(self):
        gamma, sigma, Tt, k = 0.1, 0.05, 1.0, 1.5
        total, rp, lp = spread_decomposition(gamma, sigma, Tt, k)
        assert total == pytest.approx(rp + lp, rel=1e-9)
        assert total == pytest.approx(optimal_half_spread(gamma, sigma, Tt, k), rel=1e-9)


# ─────────────────────────────────────────────────────────────
# A-S Math: compute_quotes and derived properties
# ─────────────────────────────────────────────────────────────

class TestComputeQuotes:

    def test_bid_below_ask(self):
        bid, ask, r, d = compute_quotes(100.0, 0.0, 0.1, 0.05, 1.0, 1.5)
        assert bid < ask

    def test_bid_ask_symmetric_around_reservation(self):
        bid, ask, r, d = compute_quotes(100.0, 0.0, 0.1, 0.05, 1.0, 1.5)
        assert (bid + ask) / 2 == pytest.approx(r, rel=1e-6)

    def test_half_spread_is_delta(self):
        bid, ask, r, d = compute_quotes(100.0, 0.0, 0.1, 0.05, 1.0, 1.5)
        assert ask - r == pytest.approx(d, rel=1e-6)
        assert r - bid == pytest.approx(d, rel=1e-6)

    def test_long_inventory_shifts_quotes_down(self):
        bid_zero, ask_zero, _, _ = compute_quotes(100.0, 0.0,  0.1, 0.05, 1.0, 1.5)
        bid_long, ask_long, _, _ = compute_quotes(100.0, 10.0, 0.1, 0.05, 1.0, 1.5)
        assert bid_long < bid_zero
        assert ask_long < ask_zero

    def test_short_inventory_shifts_quotes_up(self):
        bid_zero,  ask_zero,  _, _ = compute_quotes(100.0,  0.0, 0.1, 0.05, 1.0, 1.5)
        bid_short, ask_short, _, _ = compute_quotes(100.0, -10.0, 0.1, 0.05, 1.0, 1.5)
        assert bid_short > bid_zero
        assert ask_short > ask_zero

    def test_min_half_spread_enforced(self):
        bid, ask, r, d = compute_quotes(100.0, 0.0, 0.1, 0.0, 0.0, 1.5, min_half_spread=0.10)
        assert d >= 0.10
        assert ask - bid >= 0.20


# ─────────────────────────────────────────────────────────────
# implied_k_from_spread
# ─────────────────────────────────────────────────────────────

class TestImpliedK:

    def test_round_trip_consistency(self):
        """Imply k from A-S spread, then check it reproduces the spread."""
        gamma, sigma, Tt = 0.1, 0.05, 1.0
        k_true = 1.5
        d_true = optimal_half_spread(gamma, sigma, Tt, k_true)
        k_implied = implied_k_from_spread(d_true, gamma, sigma, Tt)
        assert k_implied is not None
        d_check = optimal_half_spread(gamma, sigma, Tt, k_implied)
        assert d_check == pytest.approx(d_true, rel=1e-4)

    def test_very_small_spread_returns_none(self):
        """If spread is too small to explain via A-S, return None."""
        k = implied_k_from_spread(0.0001, gamma=0.1, sigma=0.5, time_remaining=5.0)
        assert k is None


# ─────────────────────────────────────────────────────────────
# RollingVolatilityEstimator
# ─────────────────────────────────────────────────────────────

class TestVolatilityEstimator:

    def test_initial_returns_prior(self):
        est = RollingVolatilityEstimator(VolatilityConfig(initial_vol=0.05))
        assert est.sigma == pytest.approx(0.05)

    def test_single_update_returns_float(self):
        est = RollingVolatilityEstimator()
        sigma = est.update(100.0)
        assert isinstance(sigma, float)

    def test_constant_price_gives_zero_returns(self):
        """Constant price → zero variance → should hit min_vol floor."""
        cfg = VolatilityConfig(window=10, min_vol=1e-6, initial_vol=0.05)
        est = RollingVolatilityEstimator(cfg)
        for _ in range(20):
            est.update(100.0)
        # After warm-up, returns are all zero → sample std = 0 → hits floor
        assert est.sigma == pytest.approx(cfg.min_vol, abs=1e-5)

    def test_high_volatility_prices_give_higher_sigma(self):
        """Noisier prices should give higher σ."""
        import random
        rng = random.Random(1)
        low_vol_prices  = [100.0 + rng.gauss(0, 0.01) for _ in range(50)]
        high_vol_prices = [100.0 + rng.gauss(0, 0.50) for _ in range(50)]

        est_low  = RollingVolatilityEstimator(VolatilityConfig(window=30))
        est_high = RollingVolatilityEstimator(VolatilityConfig(window=30))

        for p in low_vol_prices:
            est_low.update(p)
        for p in high_vol_prices:
            est_high.update(p)

        assert est_high.sigma > est_low.sigma

    def test_min_vol_floor_enforced(self):
        cfg = VolatilityConfig(min_vol=0.01, initial_vol=0.05)
        est = RollingVolatilityEstimator(cfg)
        for _ in range(40):
            est.update(100.0)
        assert est.sigma >= 0.01

    def test_max_vol_cap_enforced(self):
        """Extreme price jumps should be capped at max_vol."""
        cfg = VolatilityConfig(window=5, max_vol=0.10)
        est = RollingVolatilityEstimator(cfg)
        prices = [100.0, 200.0, 50.0, 150.0, 80.0, 300.0]
        for p in prices:
            est.update(p)
        assert est.sigma <= cfg.max_vol

    def test_history_grows_with_updates(self):
        est = RollingVolatilityEstimator()
        for i in range(15):
            est.update(100.0 + i * 0.1)
        assert len(est.history) == 15

    def test_is_warmed_up_flag(self):
        cfg = VolatilityConfig(window=5)
        est = RollingVolatilityEstimator(cfg)
        assert not est.is_warmed_up
        for i in range(6):
            est.update(100.0 + i * 0.01)
        assert est.is_warmed_up

    def test_ewm_mode_runs_without_error(self):
        cfg = VolatilityConfig(window=20, use_ewm=True, ewm_alpha=0.1)
        est = RollingVolatilityEstimator(cfg)
        for i in range(30):
            est.update(100.0 + i * 0.05)
        assert est.sigma > 0

    def test_reset_clears_state(self):
        est = RollingVolatilityEstimator(VolatilityConfig(initial_vol=0.03))
        for i in range(20):
            est.update(100.0 + i)
        est.reset()
        assert est.sigma == pytest.approx(0.03)
        assert len(est.history) == 0
        assert not est.is_warmed_up

    def test_sigma_squared_property(self):
        est = RollingVolatilityEstimator()
        est._current_sigma = 0.05
        assert est.sigma_squared == pytest.approx(0.0025)

    def test_negative_price_skipped(self):
        """Negative / zero price should not crash."""
        est = RollingVolatilityEstimator()
        est.update(100.0)
        sigma = est.update(-1.0)
        assert sigma > 0


# ─────────────────────────────────────────────────────────────
# ArrivalIntensityEstimator
# ─────────────────────────────────────────────────────────────

class TestArrivalIntensity:

    def test_initial_returns_default(self):
        cfg = ArrivalIntensityConfig(k_default=1.5)
        est = ArrivalIntensityEstimator(cfg)
        assert est.k == pytest.approx(1.5)

    def test_zero_fills_returns_default(self):
        est = ArrivalIntensityEstimator()
        for _ in range(20):
            est.update(fills_this_step=0)
        assert est.k == pytest.approx(est.config.k_default)

    def test_high_fills_lowers_k(self):
        """Many fills → takers are eager → lower k → tighter spread."""
        est = ArrivalIntensityEstimator()
        for _ in range(30):
            est.update(fills_this_step=5)  # high fill rate
        k_high_fills = est.k

        est2 = ArrivalIntensityEstimator()
        for _ in range(30):
            est2.update(fills_this_step=1)  # low fill rate
        k_low_fills = est2.k

        assert k_high_fills < k_low_fills

    def test_k_within_bounds(self):
        cfg = ArrivalIntensityConfig(k_min=0.2, k_max=8.0, k_default=1.5)
        est = ArrivalIntensityEstimator(cfg)
        for fills in [0, 10, 100, 0, 0, 0, 1]:
            k = est.update(fills_this_step=fills)
            assert cfg.k_min <= k <= cfg.k_max

    def test_history_grows(self):
        est = ArrivalIntensityEstimator()
        for i in range(10):
            est.update(fills_this_step=i % 3)
        assert len(est.history) == 10

    def test_avg_fills_per_step(self):
        est = ArrivalIntensityEstimator(ArrivalIntensityConfig(window=4))
        for _ in range(4):
            est.update(fills_this_step=2)
        assert est.avg_fills_per_step == pytest.approx(2.0)

    def test_reset(self):
        cfg = ArrivalIntensityConfig(k_default=2.0)
        est = ArrivalIntensityEstimator(cfg)
        for _ in range(10):
            est.update(fills_this_step=5)
        est.reset()
        assert est.k == pytest.approx(2.0)
        assert len(est.history) == 0


# ─────────────────────────────────────────────────────────────
# AvellanedaStoikovMarketMaker — construction
# ─────────────────────────────────────────────────────────────

class TestASMMConstruction:

    def test_valid_construction(self):
        mm = make_asmm(gamma=0.1, horizon=1.0)
        assert mm.gamma == pytest.approx(0.1)

    def test_zero_gamma_raises(self):
        with pytest.raises(ValueError):
            make_asmm(gamma=0.0)

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError):
            make_asmm(gamma=-0.1)

    def test_zero_horizon_raises(self):
        with pytest.raises(ValueError):
            make_asmm(horizon=0.0)

    def test_quote_size_propagated(self):
        mm = make_asmm(quote_size=7.5)
        assert mm.quote_size == pytest.approx(7.5)


# ─────────────────────────────────────────────────────────────
# AvellanedaStoikovMarketMaker — quote generation
# ─────────────────────────────────────────────────────────────

class TestASMMQuotes:

    def test_generates_bid_and_ask(self):
        mm = make_asmm(gamma=0.1)
        state = make_state(midprice=100.0)
        orders = mm.act(state)
        assert len(orders) == 2

    def test_bid_below_ask(self):
        mm = make_asmm(gamma=0.1)
        state = make_state(midprice=100.0)
        orders = mm.act(state)
        bid = next(o.price for o in orders if o.side == OrderSide.BUY)
        ask = next(o.price for o in orders if o.side == OrderSide.SELL)
        assert bid < ask

    def test_no_quotes_without_reference_price(self):
        mm = make_asmm(gamma=0.1, use_fv=False)
        state = MarketState(
            timestep=1, fair_value=None, best_bid=None, best_ask=None,
            midprice=None, spread=None,
        )
        orders = mm.act(state)
        assert orders == []

    def test_falls_back_to_fair_value(self):
        """When midprice is None but fair_value exists, should still quote."""
        mm = make_asmm(gamma=0.1, use_fv=False)
        state = MarketState(
            timestep=1, fair_value=100.0, best_bid=None, best_ask=None,
            midprice=None, spread=None,
        )
        orders = mm.act(state)
        assert len(orders) == 2

    def test_long_inventory_lowers_quotes(self):
        """Long inventory → reservation below mid → both quotes lower."""
        mm_zero = make_asmm(gamma=0.5)
        mm_long = make_asmm(gamma=0.5)

        mm_long.mm_metrics.inventory = 20.0

        orders_zero = mm_zero.act(make_state(midprice=100.0))
        orders_long = mm_long.act(make_state(timestep=2, midprice=100.0))

        bid_zero = next(o.price for o in orders_zero if o.side == OrderSide.BUY)
        bid_long = next(o.price for o in orders_long if o.side == OrderSide.BUY)
        ask_zero = next(o.price for o in orders_zero if o.side == OrderSide.SELL)
        ask_long = next(o.price for o in orders_long if o.side == OrderSide.SELL)

        assert bid_long < bid_zero, "Long inventory should lower bid"
        assert ask_long < ask_zero, "Long inventory should lower ask"

    def test_high_sigma_widens_spread(self):
        """After high-volatility prices, spread should be wider."""
        import random
        rng = random.Random(42)

        mm_quiet  = make_asmm(gamma=0.1, horizon=1.0)
        mm_noisy  = make_asmm(gamma=0.1, horizon=1.0)

        # Feed quiet prices
        for _ in range(35):
            mm_quiet.mm_metrics.inventory = 0.0
            mm_quiet.act(make_state(midprice=100.0 + rng.gauss(0, 0.01)))

        # Feed noisy prices
        rng2 = random.Random(99)
        for _ in range(35):
            mm_noisy.mm_metrics.inventory = 0.0
            mm_noisy.act(make_state(midprice=100.0 + rng2.gauss(0, 0.50)))

        assert mm_noisy.current_sigma > mm_quiet.current_sigma
        assert mm_noisy.current_half_spread > mm_quiet.current_half_spread

    def test_sigma_history_grows(self):
        mm = make_asmm()
        state = make_state()
        for _ in range(5):
            mm.act(state)
        assert len(mm.sigma_history) == 5

    def test_delta_history_grows(self):
        mm = make_asmm()
        state = make_state()
        for _ in range(7):
            mm.act(state)
        assert len(mm.delta_history) == 7

    def test_reservation_history_grows(self):
        mm = make_asmm()
        state = make_state()
        mm.act(state)
        assert len(mm.reservation_history) == 1

    def test_quotes_posted_counter(self):
        mm = make_asmm()
        state = make_state()
        for _ in range(10):
            mm.act(state)
        assert mm.mm_metrics.quotes_posted == 10

    def test_order_ids_unique(self):
        mm = make_asmm()
        all_ids = []
        for t in range(10):
            for o in mm.act(make_state(timestep=t+1)):
                all_ids.append(o.order_id)
        assert len(all_ids) == len(set(all_ids))

    def test_min_spread_enforced(self):
        mm = make_asmm(gamma=0.1, min_half_spread=0.10)
        state = make_state(midprice=100.0)
        for _ in range(5):
            mm.act(state)
        for d in mm.delta_history:
            assert d >= 0.10

    def test_max_spread_enforced(self):
        mm = make_asmm(gamma=100.0, max_half_spread=0.50)  # huge gamma → huge spread
        state = make_state(midprice=100.0)
        for _ in range(5):
            mm.act(state)
        for d in mm.delta_history:
            assert d <= 0.50 + 1e-9


# ─────────────────────────────────────────────────────────────
# Time horizon modes
# ─────────────────────────────────────────────────────────────

class TestTimeHorizon:

    def test_fixed_mode_constant_time_remaining(self):
        mm = make_asmm(gamma=0.1, horizon=2.0, mode=HorizonMode.FIXED, total_steps=100)
        mm.mm_metrics.inventory = 0.0  # zero inv so reservation = mid
        for t in range(1, 10):
            mm.act(make_state(timestep=t, midprice=100.0))
        # All T-t should equal horizon_steps = 2.0
        for tr in mm.time_remaining_history:
            assert tr == pytest.approx(2.0)

    def test_decaying_mode_decreases(self):
        mm = make_asmm(gamma=0.1, horizon=1.0, mode=HorizonMode.DECAYING,
                       total_steps=100)
        mm.mm_metrics.inventory = 0.0
        for t in range(1, 11):
            mm.act(make_state(timestep=t, midprice=100.0))
        # T-t should be strictly decreasing
        for i in range(1, len(mm.time_remaining_history)):
            assert mm.time_remaining_history[i] <= mm.time_remaining_history[i - 1]

    def test_decaying_mode_reaches_zero(self):
        mm = make_asmm(horizon=1.0, mode=HorizonMode.DECAYING, total_steps=10)
        mm.mm_metrics.inventory = 0.0
        for t in range(1, 12):  # past total_steps
            mm.act(make_state(timestep=t, midprice=100.0))
        assert mm.time_remaining_history[-1] == pytest.approx(0.0)

    def test_fixed_spread_widens_with_horizon(self):
        """Fixed mode: larger horizon_steps → wider spread."""
        mm1 = make_asmm(gamma=0.1, horizon=0.5, mode=HorizonMode.FIXED)
        mm2 = make_asmm(gamma=0.1, horizon=3.0, mode=HorizonMode.FIXED)
        mm1.mm_metrics.inventory = mm2.mm_metrics.inventory = 0.0
        state = make_state(midprice=100.0)
        mm1.act(state)
        mm2.act(state)
        assert mm2.delta_history[0] > mm1.delta_history[0]


# ─────────────────────────────────────────────────────────────
# Fill notification and k estimation
# ─────────────────────────────────────────────────────────────

class TestASMMFillHandling:

    def test_ask_fill_decreases_inventory(self):
        mm = make_asmm()
        trade = make_trade(101.0, 5.0, OrderSide.BUY)
        mm.notify_fill(trade, as_maker=True)
        assert mm.mm_metrics.inventory == pytest.approx(-5.0)

    def test_bid_fill_increases_inventory(self):
        mm = make_asmm()
        trade = make_trade(99.0, 5.0, OrderSide.SELL)
        mm.notify_fill(trade, as_maker=True)
        assert mm.mm_metrics.inventory == pytest.approx(5.0)

    def test_maker_fill_counted_for_k(self):
        """Maker fills should increment the internal fill counter."""
        mm = make_asmm()
        trade = make_trade(101.0, 5.0, OrderSide.BUY)
        assert mm._fills_since_last_update == 0
        mm.notify_fill(trade, as_maker=True)
        assert mm._fills_since_last_update == 1

    def test_taker_fill_not_counted_for_k(self):
        """Taker fills should NOT increment the maker fill counter."""
        mm = make_asmm()
        trade = make_trade(101.0, 5.0, OrderSide.BUY)
        mm.notify_fill(trade, as_maker=False)
        assert mm._fills_since_last_update == 0

    def test_fills_reset_after_act(self):
        """Counter should reset to 0 after act() consumes it."""
        mm = make_asmm()
        trade = make_trade(101.0, 5.0, OrderSide.BUY)
        mm.notify_fill(trade, as_maker=True)
        assert mm._fills_since_last_update == 1
        mm.act(make_state(midprice=100.0))
        assert mm._fills_since_last_update == 0


# ─────────────────────────────────────────────────────────────
# Analytics functions
# ─────────────────────────────────────────────────────────────

class TestAnalytics:

    def test_sharpe_flat_returns_zero(self):
        assert sharpe_ratio([100.0] * 50) == pytest.approx(0.0)

    def test_sharpe_positive_for_uptrend(self):
        # Linearly growing PnL with small noise → positive Sharpe
        import random
        rng = random.Random(1)
        pnl = [float(i) + rng.gauss(0, 0.01) for i in range(100)]
        sr = sharpe_ratio(pnl)
        assert sr > 0

    def test_sharpe_negative_for_downtrend(self):
        # Linearly declining PnL with small noise → negative Sharpe
        import random
        rng = random.Random(2)
        pnl = [float(-i) + rng.gauss(0, 0.01) for i in range(100)]
        sr = sharpe_ratio(pnl)
        assert sr < 0

    def test_sharpe_empty_series(self):
        assert sharpe_ratio([]) == pytest.approx(0.0)

    def test_max_drawdown_zero_on_flat(self):
        assert max_drawdown([100.0] * 20) == pytest.approx(0.0)

    def test_max_drawdown_on_decline(self):
        pnl = [100.0, 90.0, 80.0, 70.0]
        dd = max_drawdown(pnl)
        assert dd == pytest.approx(30.0)

    def test_max_drawdown_after_recovery(self):
        pnl = [0.0, 10.0, 5.0, 15.0, 0.0, 20.0]
        dd = max_drawdown(pnl)
        # Peak 15 → trough 0 = drawdown of 15
        assert dd == pytest.approx(15.0)

    def test_max_drawdown_monotone_increase(self):
        pnl = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert max_drawdown(pnl) == pytest.approx(0.0)

    def test_strategy_comparison_runs(self):
        nmm  = NaiveMarketMaker("NMM",  half_spread=0.06)
        iamm = InventoryAwareMarketMaker("IAMM", half_spread=0.06)
        asmm = AvellanedaStoikovMarketMaker("ASMM")
        run_sim(n_steps=100, seed=1, mm_list=[nmm, iamm, asmm])
        df = strategy_comparison({"NMM": nmm, "IAMM": iamm, "ASMM": asmm})
        assert "total_pnl" in df.columns
        assert "sharpe_ratio" in df.columns
        assert "inventory_variance" in df.columns
        assert len(df) == 3


# ─────────────────────────────────────────────────────────────
# Simulation integration
# ─────────────────────────────────────────────────────────────

class TestASMMIntegration:

    def test_asmm_runs_full_simulation(self):
        asmm = make_asmm(gamma=0.1)
        result = run_sim(n_steps=200, seed=42, mm_list=[asmm])
        assert result.n_steps == 200

    def test_asmm_receives_fills(self):
        asmm = make_asmm(gamma=0.1)
        run_sim(n_steps=300, seed=42, mm_list=[asmm], n_noise=3, n_informed=1)
        assert asmm.mm_metrics.fills_as_maker > 0

    def test_asmm_sigma_adapts(self):
        """After warm-up, sigma should vary (not all the same value)."""
        asmm = make_asmm(gamma=0.1)
        run_sim(n_steps=200, seed=42, mm_list=[asmm])
        # After warm-up, not all sigma values should be identical
        sigmas = asmm.sigma_history[50:]  # post-warm-up
        if sigmas:
            assert max(sigmas) - min(sigmas) > 0

    def test_asmm_delta_varies(self):
        """Half-spread should vary across steps (volatility-dependent)."""
        asmm = make_asmm(gamma=0.1)
        run_sim(n_steps=300, seed=42, mm_list=[asmm])
        deltas = asmm.delta_history
        assert max(deltas) > min(deltas)

    def test_asmm_stale_quotes_cancelled(self):
        """Book should not accumulate unbounded ASMM orders."""
        asmm = make_asmm(gamma=0.1)
        noise = [NoiseTrader("NT1", activity_rate=0.3, random_seed=1)]
        sim = MarketSimulation(
            agents=noise + [asmm],
            n_steps=100,
            fair_value_config=FairValueConfig(initial_price=100.0),
            random_seed=1,
        )
        result = sim.run()
        asmm_resting = [
            oid for oid in result.engine.book._order_map
            if oid.startswith("ASMM")
        ]
        assert len(asmm_resting) <= 2

    def test_asmm_df_correct_length(self):
        asmm = make_asmm()
        run_sim(n_steps=150, seed=7, mm_list=[asmm])
        df = asmm.mm_metrics.to_dataframe()
        assert len(df) == 150

    def test_reproducibility(self):
        def _run(seed):
            asmm = make_asmm(gamma=0.1)
            run_sim(n_steps=100, seed=seed, mm_list=[asmm])
            return asmm.mm_metrics.fills_as_maker, round(asmm.mm_metrics.inventory, 6)

        assert _run(77) == _run(77)

    def test_all_three_strategies_run_together(self):
        nmm  = NaiveMarketMaker("NMM",  half_spread=0.06)
        iamm = InventoryAwareMarketMaker("IAMM", half_spread=0.06, inventory_skew_factor=0.012)
        asmm = make_asmm(gamma=0.1)
        result = run_sim(n_steps=200, seed=42, mm_list=[nmm, iamm, asmm])
        assert result.n_steps == 200
        assert nmm.mm_metrics.quotes_posted > 0
        assert iamm.mm_metrics.quotes_posted > 0
        assert asmm.mm_metrics.quotes_posted > 0

    def test_asmm_better_sharpe_than_nmm(self):
        """
        A-S should have a better Sharpe ratio than the Naive MM across seeds.
        We check across multiple seeds and require >=3/5 wins.
        """
        from src.models.analytics import sharpe_ratio as sr
        wins = 0
        for seed in range(42, 47):
            nmm  = NaiveMarketMaker("NMM",  half_spread=0.06)
            asmm = make_asmm(gamma=0.1, horizon=1.0)
            run_sim(n_steps=500, seed=seed, mm_list=[nmm, asmm], n_noise=3, n_informed=2)
            if sr(asmm.mm_metrics.pnl_history) > sr(nmm.mm_metrics.pnl_history):
                wins += 1
        assert wins >= 3, f"ASMM should beat NMM Sharpe in >=3/5 seeds, got {wins}/5"

    def test_decaying_mode_integration(self):
        asmm = make_asmm(gamma=0.1, mode=HorizonMode.DECAYING, total_steps=200, horizon=1.0)
        run_sim(n_steps=200, seed=42, mm_list=[asmm])
        # Time remaining should decrease over the run
        tr = asmm.time_remaining_history
        assert tr[0] > tr[-1]

    def test_previous_tests_still_pass(self):
        """Smoke test: existing strategies still behave correctly."""
        nmm  = NaiveMarketMaker("NMM_chk",  half_spread=0.05)
        iamm = InventoryAwareMarketMaker("IAMM_chk", half_spread=0.05,
                                          inventory_skew_factor=0.012)
        result = run_sim(n_steps=100, seed=1, mm_list=[nmm, iamm])
        assert result.n_steps == 100
        assert nmm.mm_metrics.fills_as_maker >= 0
        assert iamm.mm_metrics.inventory_variance >= 0
