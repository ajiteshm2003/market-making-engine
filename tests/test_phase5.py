"""
tests/test_phase5.py
---------------------
Phase 5 test suite — Regime-Aware Avellaneda-Stoikov Market Maker.

Covers:
  - RegimeClassifier: classification, transitions, history, reset
  - RegimeThresholds: validation
  - RegimeParameters: correctness per regime
  - RegimeAwareAvellanedaStoikovMarketMaker: quotes, skew, spread widening,
    quote size reduction, history tracking, fill routing, integration
  - Four-strategy simulation runs
  - Reproducibility
  - Previous tests unaffected (smoke-checked)

Run with:
    pytest tests/test_phase5.py -v
"""

import math
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.regime import (
    VolatilityRegime,
    RegimeThresholds,
    RegimeClassifier,
    RegimeParameters,
    RegimeTransitionEvent,
)
from src.strategies import (
    NaiveMarketMaker,
    InventoryAwareMarketMaker,
    AvellanedaStoikovMarketMaker,
    ASConfig,
    HorizonMode,
    RegimeAwareAvellanedaStoikovMarketMaker,
    RegimeAwareASConfig,
)
from src.models import VolatilityConfig
from src.agents import NoiseTrader, InformedTrader
from src.exchange.order import OrderSide
from src.exchange.trade import Trade
from src.simulation import MarketSimulation, FairValueConfig
from src.simulation.market_state import MarketState


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = RegimeThresholds(
    low_threshold=0.0008,
    high_threshold=0.0035,
    extreme_threshold=0.0055,
    hysteresis=0.0,           # disable for deterministic tests
)


def make_state(
    timestep=1, fair_value=100.0, midprice=100.0,
    best_bid=99.0, best_ask=101.0, spread=2.0,
    volume_this_step=5.0, trade_count=10,
):
    return MarketState(
        timestep=timestep, fair_value=fair_value,
        best_bid=best_bid, best_ask=best_ask,
        midprice=midprice, spread=spread,
        volume_this_step=volume_this_step, trade_count=trade_count,
    )


def make_trade(price, qty, aggressor_side):
    return Trade(
        trade_id=f"T{int(time.time()*1e9)}",
        timestamp=time.time(),
        price=price, quantity=qty,
        aggressor_side=aggressor_side,
        maker_order_id="MAKER",
        taker_order_id="TAKER",
    )


def make_rasmm(
    gamma=0.1, base_quote_size=5.0, base_max_inv=50.0,
    horizon=1.0, thresholds=None, regime_params=None,
) -> RegimeAwareAvellanedaStoikovMarketMaker:
    base_cfg = ASConfig(gamma=gamma, horizon_steps=horizon)
    ra_cfg = RegimeAwareASConfig(
        base_config=base_cfg,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
        regime_params=regime_params or RegimeParameters(),
        base_quote_size=base_quote_size,
        base_max_inventory=base_max_inv,
    )
    return RegimeAwareAvellanedaStoikovMarketMaker("RASMM", ra_config=ra_cfg)


def run_sim(n_steps=300, seed=42, mm_list=None, n_noise=3, n_informed=1,
            vol=0.05, jump=0.04, jump_std=0.5):
    noise    = [NoiseTrader(f"NT{i}", activity_rate=0.6, random_seed=seed+i) for i in range(n_noise)]
    informed = [InformedTrader(f"IT{i}", activity_rate=0.55, random_seed=seed+100+i) for i in range(n_informed)]
    agents   = noise + informed + (mm_list or [])
    sim = MarketSimulation(
        agents=agents, n_steps=n_steps,
        fair_value_config=FairValueConfig(
            initial_price=100.0, volatility=vol,
            jump_prob=jump, jump_std=jump_std,
        ),
        random_seed=seed,
    )
    return sim.run()


# ─────────────────────────────────────────────────────────────
# RegimeThresholds validation
# ─────────────────────────────────────────────────────────────

class TestRegimeThresholds:

    def test_valid_construction(self):
        t = RegimeThresholds(low_threshold=0.02, high_threshold=0.05, extreme_threshold=0.10)
        assert t.low_threshold == pytest.approx(0.02)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            RegimeThresholds(low_threshold=0.05, high_threshold=0.02, extreme_threshold=0.10)

    def test_low_equals_high_raises(self):
        with pytest.raises(ValueError):
            RegimeThresholds(low_threshold=0.05, high_threshold=0.05, extreme_threshold=0.10)

    def test_negative_hysteresis_raises(self):
        with pytest.raises(ValueError):
            RegimeThresholds(low_threshold=0.02, high_threshold=0.05,
                             extreme_threshold=0.10, hysteresis=-0.001)

    def test_zero_hysteresis_valid(self):
        t = RegimeThresholds(hysteresis=0.0)
        assert t.hysteresis == 0.0


# ─────────────────────────────────────────────────────────────
# RegimeClassifier — basic classification
# ─────────────────────────────────────────────────────────────

class TestRegimeClassifierBasic:
    """Tests use hysteresis=0 for deterministic, threshold-exact behavior."""

    def _clf(self):
        return RegimeClassifier(thresholds=DEFAULT_THRESHOLDS)

    def test_below_low_threshold_gives_low(self):
        clf = self._clf()
        assert clf.update(0.0005) == VolatilityRegime.LOW

    def test_above_low_below_high_gives_medium(self):
        clf = self._clf()
        assert clf.update(0.0020) == VolatilityRegime.MEDIUM

    def test_above_high_below_extreme_gives_high(self):
        clf = self._clf()
        assert clf.update(0.0045) == VolatilityRegime.HIGH

    def test_above_extreme_gives_extreme(self):
        clf = self._clf()
        assert clf.update(0.0080) == VolatilityRegime.EXTREME

    def test_exactly_at_low_threshold_gives_medium(self):
        clf = self._clf()
        assert clf.update(DEFAULT_THRESHOLDS.low_threshold) == VolatilityRegime.MEDIUM

    def test_exactly_at_high_threshold_gives_high(self):
        clf = self._clf()
        assert clf.update(DEFAULT_THRESHOLDS.high_threshold) == VolatilityRegime.HIGH

    def test_exactly_at_extreme_threshold_gives_extreme(self):
        clf = self._clf()
        assert clf.update(DEFAULT_THRESHOLDS.extreme_threshold) == VolatilityRegime.EXTREME

    def test_initial_regime_is_medium(self):
        clf = RegimeClassifier()
        assert clf.current_regime == VolatilityRegime.MEDIUM

    def test_custom_initial_regime(self):
        clf = RegimeClassifier(initial_regime=VolatilityRegime.LOW)
        assert clf.current_regime == VolatilityRegime.LOW


# ─────────────────────────────────────────────────────────────
# RegimeClassifier — transitions and history
# ─────────────────────────────────────────────────────────────

class TestRegimeClassifierTransitions:

    def _clf(self):
        return RegimeClassifier(thresholds=DEFAULT_THRESHOLDS)

    def test_no_transition_on_same_regime(self):
        clf = self._clf()
        for _ in range(5):
            clf.update(0.002)  # always MEDIUM
        assert clf.transition_count == 0

    def test_transition_counted_on_regime_change(self):
        clf = self._clf()
        clf.update(0.0005)   # LOW
        clf.update(0.0045)   # HIGH
        assert clf.transition_count == 2   # start→LOW, LOW→HIGH

    def test_transition_event_fields(self):
        clf = self._clf()
        clf.update(0.0005)   # → LOW
        clf.update(0.0080)   # → EXTREME
        assert len(clf.transitions) == 2
        ev = clf.transitions[1]
        assert ev.from_regime == VolatilityRegime.LOW
        assert ev.to_regime == VolatilityRegime.EXTREME
        assert ev.sigma_at_transition == pytest.approx(0.0080)

    def test_history_length_matches_updates(self):
        clf = self._clf()
        for _ in range(20):
            clf.update(0.002)
        assert len(clf.history) == 20

    def test_regime_counts_accurate(self):
        clf = self._clf()
        for _ in range(3):
            clf.update(0.0005)   # LOW
        for _ in range(5):
            clf.update(0.002)   # MEDIUM
        for _ in range(2):
            clf.update(0.0045)   # HIGH
        assert clf.regime_counts[VolatilityRegime.LOW]    == 3
        assert clf.regime_counts[VolatilityRegime.MEDIUM] == 5
        assert clf.regime_counts[VolatilityRegime.HIGH]   == 2
        assert clf.regime_counts[VolatilityRegime.EXTREME]== 0

    def test_transition_matrix_populated(self):
        clf = self._clf()
        clf.update(0.0005)   # MEDIUM → LOW
        clf.update(0.0080)   # LOW → EXTREME
        key = (VolatilityRegime.LOW, VolatilityRegime.EXTREME)
        assert clf.transition_matrix.get(key, 0) == 1

    def test_regime_fraction_sums_to_one(self):
        clf = self._clf()
        for sigma in [0.0005, 0.002, 0.005, 0.008, 0.002, 0.0005]:
            clf.update(sigma)
        total_frac = sum(clf.regime_fraction(r) for r in VolatilityRegime)
        assert total_frac == pytest.approx(1.0)

    def test_time_in_regime(self):
        clf = self._clf()
        for _ in range(4):
            clf.update(0.0005)  # LOW
        assert clf.time_in_regime(VolatilityRegime.LOW) == 4

    def test_reset_clears_all_state(self):
        clf = self._clf()
        for sigma in [0.01, 0.07, 0.12, 0.03]:
            clf.update(sigma)
        clf.reset()
        assert clf.transition_count == 0
        assert len(clf.history) == 0
        assert clf.current_regime == VolatilityRegime.MEDIUM
        assert sum(clf.regime_counts.values()) == 0


# ─────────────────────────────────────────────────────────────
# RegimeClassifier — hysteresis
# ─────────────────────────────────────────────────────────────

class TestHysteresis:

    def test_hysteresis_prevents_immediate_downgrade(self):
        """With hysteresis, crossing the boundary barely should not switch."""
        t = RegimeThresholds(
            low_threshold=0.0008,
            high_threshold=0.0035,
            extreme_threshold=0.0055,
            hysteresis=0.0008,    # buffer = 0.0008
        )
        clf = RegimeClassifier(thresholds=t, initial_regime=VolatilityRegime.HIGH)
        # σ = 0.003: just below high_threshold=0.0035, but within hysteresis band
        result = clf.update(0.003)
        # Should stay in HIGH due to hysteresis (not drop to MEDIUM yet)
        assert result == VolatilityRegime.HIGH

    def test_hysteresis_allows_switch_when_far_enough(self):
        """Large σ drop should escape hysteresis band."""
        t = RegimeThresholds(
            low_threshold=0.0008,
            high_threshold=0.0035,
            extreme_threshold=0.0055,
            hysteresis=0.0003,
        )
        clf = RegimeClassifier(thresholds=t, initial_regime=VolatilityRegime.HIGH)
        # σ = 0.002: well below high_threshold - hysteresis = 0.0032
        result = clf.update(0.002)
        assert result == VolatilityRegime.MEDIUM

    def test_zero_hysteresis_transitions_immediately(self):
        t = RegimeThresholds(hysteresis=0.0)
        clf = RegimeClassifier(thresholds=t, initial_regime=VolatilityRegime.HIGH)
        result = clf.update(0.003)   # below high threshold
        assert result == VolatilityRegime.MEDIUM


# ─────────────────────────────────────────────────────────────
# RegimeParameters
# ─────────────────────────────────────────────────────────────

class TestRegimeParameters:

    def _params(self):
        return RegimeParameters()

    def test_medium_multipliers_are_unity(self):
        p = self._params()
        gm, sm, qs, im = p.get(VolatilityRegime.MEDIUM)
        assert gm == pytest.approx(1.0)
        assert sm == pytest.approx(1.0)
        assert qs == pytest.approx(1.0)
        assert im == pytest.approx(1.0)

    def test_low_has_smaller_gamma_than_medium(self):
        p = self._params()
        assert p.gamma_mult(VolatilityRegime.LOW) < p.gamma_mult(VolatilityRegime.MEDIUM)

    def test_high_has_larger_gamma_than_medium(self):
        p = self._params()
        assert p.gamma_mult(VolatilityRegime.HIGH) > p.gamma_mult(VolatilityRegime.MEDIUM)

    def test_extreme_has_largest_gamma(self):
        p = self._params()
        assert p.gamma_mult(VolatilityRegime.EXTREME) > p.gamma_mult(VolatilityRegime.HIGH)

    def test_low_has_larger_quote_size_than_medium(self):
        p = self._params()
        assert p.quote_size_mult(VolatilityRegime.LOW) > p.quote_size_mult(VolatilityRegime.MEDIUM)

    def test_high_has_smaller_quote_size_than_medium(self):
        p = self._params()
        assert p.quote_size_mult(VolatilityRegime.HIGH) < p.quote_size_mult(VolatilityRegime.MEDIUM)

    def test_extreme_has_smallest_quote_size(self):
        p = self._params()
        assert p.quote_size_mult(VolatilityRegime.EXTREME) < p.quote_size_mult(VolatilityRegime.HIGH)

    def test_low_has_wider_spread_mult_lt_1(self):
        """LOW regime spread multiplier < 1 (tighter spreads)."""
        p = self._params()
        assert p.spread_mult(VolatilityRegime.LOW) < 1.0

    def test_extreme_has_spread_mult_gt_1(self):
        """EXTREME regime spread multiplier > 1 (wider spreads)."""
        p = self._params()
        assert p.spread_mult(VolatilityRegime.EXTREME) > 1.0

    def test_spread_mult_monotone_with_severity(self):
        """Spread multiplier should be non-decreasing with regime severity."""
        p = self._params()
        mults = [
            p.spread_mult(VolatilityRegime.LOW),
            p.spread_mult(VolatilityRegime.MEDIUM),
            p.spread_mult(VolatilityRegime.HIGH),
            p.spread_mult(VolatilityRegime.EXTREME),
        ]
        for i in range(1, len(mults)):
            assert mults[i] >= mults[i-1], f"Spread mult not monotone: {mults}"

    def test_gamma_mult_monotone_with_severity(self):
        p = self._params()
        mults = [
            p.gamma_mult(VolatilityRegime.LOW),
            p.gamma_mult(VolatilityRegime.MEDIUM),
            p.gamma_mult(VolatilityRegime.HIGH),
            p.gamma_mult(VolatilityRegime.EXTREME),
        ]
        for i in range(1, len(mults)):
            assert mults[i] >= mults[i-1]


# ─────────────────────────────────────────────────────────────
# RegimeAwareASMM — construction
# ─────────────────────────────────────────────────────────────

class TestRAASMMConstruction:

    def test_valid_construction(self):
        mm = make_rasmm(gamma=0.1)
        assert mm.agent_id == "RASMM"
        assert mm.ra_config.base_config.gamma == pytest.approx(0.1)

    def test_base_quote_size_set(self):
        mm = make_rasmm(base_quote_size=8.0)
        assert mm.ra_config.base_quote_size == pytest.approx(8.0)

    def test_invalid_gamma_propagates(self):
        with pytest.raises(ValueError):
            make_rasmm(gamma=0.0)

    def test_classifier_initialised(self):
        mm = make_rasmm()
        assert mm.classifier is not None
        assert mm.classifier.current_regime == VolatilityRegime.MEDIUM


# ─────────────────────────────────────────────────────────────
# RegimeAwareASMM — quote generation
# ─────────────────────────────────────────────────────────────

class TestRAASMMQuotes:

    def test_generates_bid_and_ask(self):
        mm = make_rasmm()
        orders = mm.act(make_state())
        assert len(orders) == 2

    def test_bid_always_below_ask(self):
        mm = make_rasmm()
        state = make_state()
        for _ in range(10):
            orders = mm.act(state)
            bids = [o for o in orders if o.side == OrderSide.BUY]
            asks = [o for o in orders if o.side == OrderSide.SELL]
            if bids and asks:
                assert bids[0].price < asks[0].price

    def test_no_quotes_without_price(self):
        mm = make_rasmm()
        state = MarketState(
            timestep=1, fair_value=None, best_bid=None, best_ask=None,
            midprice=None, spread=None,
        )
        assert mm.act(state) == []

    def test_regime_history_grows(self):
        mm = make_rasmm()
        for t in range(1, 8):
            mm.act(make_state(timestep=t))
        assert len(mm.regime_history) == 7

    def test_gamma_history_grows(self):
        mm = make_rasmm()
        for t in range(1, 6):
            mm.act(make_state(timestep=t))
        assert len(mm.gamma_history) == 5

    def test_quote_size_history_grows(self):
        mm = make_rasmm()
        for t in range(1, 5):
            mm.act(make_state(timestep=t))
        assert len(mm.quote_size_history) == 4

    def test_spread_mult_history_grows(self):
        mm = make_rasmm()
        for t in range(1, 5):
            mm.act(make_state(timestep=t))
        assert len(mm.spread_mult_history) == 4

    def test_order_ids_unique(self):
        mm = make_rasmm()
        all_ids = []
        for t in range(1, 15):
            for o in mm.act(make_state(timestep=t)):
                all_ids.append(o.order_id)
        assert len(all_ids) == len(set(all_ids))

    def test_quote_size_reduces_in_high_regime(self):
        """After high-vol prices, effective quote size should be smaller."""
        import random
        rng = random.Random(1)

        mm_quiet = make_rasmm(base_quote_size=5.0)
        mm_noisy = make_rasmm(base_quote_size=5.0)

        # Feed quiet prices → LOW/MEDIUM regime
        for t in range(1, 40):
            mm_quiet.act(make_state(timestep=t, midprice=100.0 + rng.gauss(0, 0.01)))

        # Feed very noisy prices → HIGH/EXTREME regime
        rng2 = random.Random(99)
        for t in range(1, 40):
            mm_noisy.act(make_state(timestep=t, midprice=100.0 + rng2.gauss(0, 0.50)))

        # Noisy MM should be in a higher regime with smaller quote size
        avg_qs_quiet = sum(mm_quiet.quote_size_history) / len(mm_quiet.quote_size_history)
        avg_qs_noisy = sum(mm_noisy.quote_size_history) / len(mm_noisy.quote_size_history)
        assert avg_qs_noisy < avg_qs_quiet, \
            f"Noisy quote size {avg_qs_noisy:.3f} should be < quiet {avg_qs_quiet:.3f}"

    def test_spread_widens_in_high_regime(self):
        """After high-vol prices, spread should be wider than in quiet market."""
        import random
        mm_quiet = make_rasmm(gamma=0.1)
        mm_noisy = make_rasmm(gamma=0.1)

        rng_q = random.Random(10)
        rng_n = random.Random(20)

        for t in range(1, 50):
            mm_quiet.act(make_state(timestep=t, midprice=100.0 + rng_q.gauss(0, 0.005)))
            mm_noisy.act(make_state(timestep=t, midprice=100.0 + rng_n.gauss(0, 0.80)))

        avg_delta_quiet = sum(mm_quiet.delta_history) / len(mm_quiet.delta_history)
        avg_delta_noisy = sum(mm_noisy.delta_history) / len(mm_noisy.delta_history)
        assert avg_delta_noisy > avg_delta_quiet

    def test_gamma_increases_with_regime_severity(self):
        """Effective gamma should be higher when regime is HIGH/EXTREME."""
        # Directly test: at same sigma, higher regime → higher effective gamma
        mm = make_rasmm(gamma=0.1)
        rp = mm.ra_config.regime_params
        base_gamma = 0.1

        gamma_low = base_gamma * rp.gamma_mult(VolatilityRegime.LOW)
        gamma_med = base_gamma * rp.gamma_mult(VolatilityRegime.MEDIUM)
        gamma_high = base_gamma * rp.gamma_mult(VolatilityRegime.HIGH)
        gamma_ext = base_gamma * rp.gamma_mult(VolatilityRegime.EXTREME)

        assert gamma_low < gamma_med < gamma_high < gamma_ext


# ─────────────────────────────────────────────────────────────
# RegimeAwareASMM — fill handling
# ─────────────────────────────────────────────────────────────

class TestRAASMMFills:

    def test_ask_fill_reduces_inventory(self):
        mm = make_rasmm()
        trade = make_trade(101.0, 5.0, OrderSide.BUY)
        mm.notify_fill(trade, as_maker=True)
        assert mm.mm_metrics.inventory == pytest.approx(-5.0)

    def test_bid_fill_increases_inventory(self):
        mm = make_rasmm()
        trade = make_trade(99.0, 5.0, OrderSide.SELL)
        mm.notify_fill(trade, as_maker=True)
        assert mm.mm_metrics.inventory == pytest.approx(5.0)

    def test_regime_fill_counter(self):
        """Fills should be routed to correct regime bucket."""
        mm = make_rasmm()
        # Advance to a known regime by calling act() first
        mm.act(make_state(midprice=100.0))  # establishes current regime
        regime_before = mm.current_regime

        trade = make_trade(101.0, 3.0, OrderSide.BUY)
        mm.notify_fill(trade, as_maker=True)

        assert mm._regime_fills[regime_before] >= 1


# ─────────────────────────────────────────────────────────────
# Regime metrics
# ─────────────────────────────────────────────────────────────

class TestRegimeMetrics:

    def test_metrics_summary_has_all_regimes(self):
        mm = make_rasmm()
        result = run_sim(n_steps=200, seed=42, mm_list=[mm])
        summary = mm.regime_metrics_summary()
        for r in VolatilityRegime:
            assert r.value in summary

    def test_metrics_steps_sum_to_total(self):
        mm = make_rasmm()
        result = run_sim(n_steps=200, seed=42, mm_list=[mm])
        summary = mm.regime_metrics_summary()
        total_steps = sum(v["steps"] for v in summary.values())
        assert total_steps == 200

    def test_metrics_pct_time_sums_to_100(self):
        mm = make_rasmm()
        result = run_sim(n_steps=200, seed=42, mm_list=[mm])
        summary = mm.regime_metrics_summary()
        total_pct = sum(v["pct_time"] for v in summary.values())
        assert total_pct == pytest.approx(100.0, abs=0.2)

    def test_avg_spread_higher_in_extreme_than_low(self):
        """After a volatile run, average spread in EXTREME should exceed LOW."""
        mm = make_rasmm()
        result = run_sim(n_steps=500, seed=42, mm_list=[mm],
                         vol=0.06, jump=0.06, jump_std=0.6)
        summary = mm.regime_metrics_summary()
        low_spr = summary.get("low", {}).get("avg_half_spread", 0)
        ext_spr = summary.get("extreme", {}).get("avg_half_spread", 0)
        # Only check if both regimes were visited
        if low_spr > 0 and ext_spr > 0:
            assert ext_spr > low_spr, \
                f"EXTREME spread {ext_spr:.5f} should exceed LOW {low_spr:.5f}"


# ─────────────────────────────────────────────────────────────
# Simulation integration
# ─────────────────────────────────────────────────────────────

class TestRAASMMIntegration:

    def test_rasmm_runs_full_simulation(self):
        mm = make_rasmm()
        result = run_sim(n_steps=200, seed=1, mm_list=[mm])
        assert result.n_steps == 200

    def test_rasmm_posts_quotes(self):
        mm = make_rasmm()
        run_sim(n_steps=200, seed=2, mm_list=[mm])
        assert mm.mm_metrics.quotes_posted > 0

    def test_rasmm_receives_fills(self):
        mm = make_rasmm()
        run_sim(n_steps=400, seed=3, mm_list=[mm], n_noise=4, n_informed=2)
        assert mm.mm_metrics.fills_as_maker > 0

    def test_rasmm_stale_quotes_cancelled(self):
        """Book should not accumulate unbounded R-ASMM orders."""
        mm = make_rasmm()
        noise = [NoiseTrader("NT1", activity_rate=0.3, random_seed=1)]
        sim = MarketSimulation(
            agents=noise + [mm], n_steps=100,
            fair_value_config=FairValueConfig(initial_price=100.0),
            random_seed=1,
        )
        result = sim.run()
        resting = [oid for oid in result.engine.book._order_map if oid.startswith("RASMM")]
        assert len(resting) <= 2

    def test_rasmm_df_correct_length(self):
        mm = make_rasmm()
        run_sim(n_steps=150, seed=4, mm_list=[mm])
        df = mm.mm_metrics.to_dataframe()
        assert len(df) == 150

    def test_regime_history_length_matches_steps(self):
        mm = make_rasmm()
        run_sim(n_steps=200, seed=5, mm_list=[mm])
        assert len(mm.regime_history) == 200

    def test_gamma_history_length_matches_steps(self):
        mm = make_rasmm()
        run_sim(n_steps=150, seed=6, mm_list=[mm])
        assert len(mm.gamma_history) == 150

    def test_transitions_occur_in_volatile_sim(self):
        """High volatility + jumps should trigger regime transitions."""
        mm = make_rasmm()
        run_sim(n_steps=500, seed=42, mm_list=[mm],
                vol=0.06, jump=0.06, jump_std=0.6)
        assert mm.classifier.transition_count > 0

    def test_all_four_strategies_run_together(self):
        nmm   = NaiveMarketMaker("NMM",  half_spread=0.06)
        iamm  = InventoryAwareMarketMaker("IAMM", half_spread=0.06, inventory_skew_factor=0.012)
        asmm  = AvellanedaStoikovMarketMaker("ASMM", config=ASConfig(gamma=0.1))
        rasmm = make_rasmm(gamma=0.1)
        result = run_sim(n_steps=200, seed=42, mm_list=[nmm, iamm, asmm, rasmm])
        assert result.n_steps == 200
        for mm in [nmm, iamm, asmm, rasmm]:
            assert mm.mm_metrics.quotes_posted > 0

    def test_reproducibility(self):
        def _run(seed):
            mm = make_rasmm(gamma=0.1)
            run_sim(n_steps=100, seed=seed, mm_list=[mm])
            return (mm.mm_metrics.fills_as_maker,
                    mm.classifier.transition_count,
                    round(mm.mm_metrics.total_pnl, 4))
        assert _run(77) == _run(77)

    def test_rasmm_lower_drawdown_than_nmm_in_volatile_market(self):
        """
        In a highly volatile market, R-ASMM should have lower max drawdown
        than the Naive MM in the majority of seeds (≥3/5).
        """
        from src.models.analytics import max_drawdown
        wins = 0
        for seed in range(42, 47):
            nmm_  = NaiveMarketMaker("NMM",  half_spread=0.06)
            rasmm = make_rasmm(gamma=0.1)
            run_sim(n_steps=500, seed=seed, mm_list=[nmm_, rasmm],
                    n_noise=3, n_informed=2, vol=0.06, jump=0.06, jump_std=0.5)
            dd_nmm   = max_drawdown(nmm_.mm_metrics.pnl_history)
            dd_rasmm = max_drawdown(rasmm.mm_metrics.pnl_history)
            if dd_rasmm < dd_nmm:
                wins += 1
        assert wins >= 3, \
            f"RASMM should have lower drawdown in >=3/5 volatile seeds, got {wins}/5"

    def test_sigma_history_populated(self):
        mm = make_rasmm()
        run_sim(n_steps=100, seed=1, mm_list=[mm])
        assert len(mm.sigma_history) == 100

    def test_classifier_accessible(self):
        mm = make_rasmm()
        run_sim(n_steps=100, seed=1, mm_list=[mm])
        assert mm.classifier is not None
        assert isinstance(mm.current_regime, VolatilityRegime)

    def test_previous_phase_smoke(self):
        """Smoke test: previous strategies still behave correctly."""
        nmm  = NaiveMarketMaker("NMM_ck", half_spread=0.05)
        iamm = InventoryAwareMarketMaker("IAMM_ck", half_spread=0.05,
                                          inventory_skew_factor=0.012)
        asmm = AvellanedaStoikovMarketMaker("ASMM_ck", config=ASConfig(gamma=0.1))
        result = run_sim(n_steps=100, seed=1, mm_list=[nmm, iamm, asmm])
        assert result.n_steps == 100
        for mm in [nmm, iamm, asmm]:
            assert mm.mm_metrics.quotes_posted > 0
