"""
tests/test_simulation.py
-------------------------
Unit tests for FairValueProcess, MarketState, and MarketSimulation.

Run with:
    pytest tests/test_simulation.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import InformedTrader, NoiseTrader
from src.simulation import (
    FairValueConfig,
    FairValueProcess,
    MarketSimulation,
    MarketState,
)


# ─────────────────────────────────────────────────────────────
# FairValueProcess
# ─────────────────────────────────────────────────────────────

class TestFairValueProcess:

    def test_initial_value(self):
        fv = FairValueProcess(FairValueConfig(initial_price=50.0))
        assert fv.value == pytest.approx(50.0)

    def test_step_returns_float(self):
        fv = FairValueProcess()
        v = fv.step()
        assert isinstance(v, float)

    def test_history_grows(self):
        fv = FairValueProcess(random_seed=1)
        assert len(fv.history) == 1
        for _ in range(10):
            fv.step()
        assert len(fv.history) == 11

    def test_price_never_negative(self):
        cfg = FairValueConfig(volatility=5.0, jump_prob=0.5, jump_std=5.0, min_price=1.0)
        fv = FairValueProcess(cfg, random_seed=42)
        for _ in range(1000):
            v = fv.step()
            assert v >= 1.0

    def test_reproducible_with_seed(self):
        fv1 = FairValueProcess(random_seed=7)
        fv2 = FairValueProcess(random_seed=7)
        seq1 = [fv1.step() for _ in range(20)]
        seq2 = [fv2.step() for _ in range(20)]
        assert seq1 == seq2

    def test_different_seeds_differ(self):
        fv1 = FairValueProcess(random_seed=1)
        fv2 = FairValueProcess(random_seed=2)
        seq1 = [fv1.step() for _ in range(10)]
        seq2 = [fv2.step() for _ in range(10)]
        assert seq1 != seq2

    def test_jumps_recorded(self):
        cfg = FairValueConfig(jump_prob=1.0)  # jump every step
        fv = FairValueProcess(cfg, random_seed=1)
        for _ in range(10):
            fv.step()
        assert len(fv.jump_steps) == 10

    def test_no_jumps_when_prob_zero(self):
        cfg = FairValueConfig(jump_prob=0.0)
        fv = FairValueProcess(cfg, random_seed=1)
        for _ in range(100):
            fv.step()
        assert fv.jump_steps == []

    def test_reset(self):
        fv = FairValueProcess(FairValueConfig(initial_price=200.0), random_seed=1)
        for _ in range(10):
            fv.step()
        fv.reset()
        assert fv.value == pytest.approx(200.0)
        assert len(fv.history) == 1
        assert fv.jump_steps == []


# ─────────────────────────────────────────────────────────────
# MarketState
# ─────────────────────────────────────────────────────────────

class TestMarketState:

    def test_immutable(self):
        state = MarketState(
            timestep=1, fair_value=100.0,
            best_bid=99.0, best_ask=101.0,
            midprice=100.0, spread=2.0,
        )
        with pytest.raises(Exception):
            state.midprice = 99.0  # frozen dataclass

    def test_empty_state(self):
        state = MarketState(
            timestep=0, fair_value=100.0,
            best_bid=None, best_ask=None,
            midprice=None, spread=None,
        )
        assert state.best_bid is None
        assert state.spread is None


# ─────────────────────────────────────────────────────────────
# MarketSimulation — construction
# ─────────────────────────────────────────────────────────────

class TestMarketSimulationConstruction:

    def test_no_agents_raises(self):
        with pytest.raises(ValueError, match="At least one agent"):
            MarketSimulation(agents=[], n_steps=10)

    def test_zero_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            MarketSimulation(agents=[NoiseTrader("NT1")], n_steps=0)

    def test_double_run_raises(self):
        sim = MarketSimulation(
            agents=[NoiseTrader("NT1", random_seed=1)],
            n_steps=5,
            random_seed=1,
        )
        sim.run()
        with pytest.raises(RuntimeError, match="already run"):
            sim.run()


# ─────────────────────────────────────────────────────────────
# MarketSimulation — correctness
# ─────────────────────────────────────────────────────────────

class TestMarketSimulationRun:

    def _run(self, n_steps=50, seed=42, n_noise=3, n_informed=1):
        agents = [
            NoiseTrader(f"NT{i}", activity_rate=0.8, random_seed=seed + i)
            for i in range(n_noise)
        ]
        agents += [
            InformedTrader(f"IT{i}", activity_rate=0.7, random_seed=seed + 100 + i)
            for i in range(n_informed)
        ]
        sim = MarketSimulation(
            agents=agents,
            n_steps=n_steps,
            fair_value_config=FairValueConfig(
                initial_price=100.0, volatility=0.05, jump_prob=0.03
            ),
            random_seed=seed,
        )
        return sim.run()

    def test_result_has_correct_step_count(self):
        result = self._run(n_steps=100)
        df = result.metrics.to_dataframe()
        assert len(df) == 100

    def test_fair_value_history_length(self):
        result = self._run(n_steps=50)
        # history starts at t=0, then 50 steps → 51 entries
        assert len(result.fair_value_history) == 51

    def test_metrics_cumulative_volume_non_negative(self):
        result = self._run(n_steps=50)
        df = result.metrics.to_dataframe()
        assert (df["cumulative_volume"] >= 0).all()

    def test_metrics_cumulative_trades_monotone(self):
        result = self._run(n_steps=100)
        df = result.metrics.to_dataframe()
        assert df["cumulative_trades"].is_monotonic_increasing

    def test_cumulative_volume_monotone(self):
        result = self._run(n_steps=100)
        df = result.metrics.to_dataframe()
        assert df["cumulative_volume"].is_monotonic_increasing

    def test_spread_non_negative_when_present(self):
        result = self._run(n_steps=100)
        df = result.metrics.to_dataframe()
        spreads = df["spread"].dropna()
        assert (spreads >= 0).all()

    def test_midprice_between_bid_ask(self):
        result = self._run(n_steps=100)
        df = result.metrics.to_dataframe()
        both = df.dropna(subset=["midprice", "best_bid", "best_ask"])
        assert ((both["midprice"] >= both["best_bid"]) & (both["midprice"] <= both["best_ask"])).all()

    def test_fair_value_always_positive(self):
        result = self._run(n_steps=200)
        assert all(v > 0 for v in result.fair_value_history)

    def test_order_imbalance_in_valid_range(self):
        result = self._run(n_steps=100)
        df = result.metrics.to_dataframe()
        imb = df["order_imbalance"].dropna()
        assert (imb >= -1.0).all() and (imb <= 1.0).all()

    def test_trade_log_matches_metrics(self):
        result = self._run(n_steps=100)
        df = result.metrics.to_dataframe()
        assert df["cumulative_trades"].iloc[-1] == len(result.engine.trade_log)

    def test_agent_inventory_changes(self):
        """At least one agent should have non-zero inventory after running."""
        result = self._run(n_steps=200)
        inventories = [a.metrics.inventory for a in result.agents]
        assert any(abs(inv) > 0 for inv in inventories)

    def test_reproducible_with_seed(self):
        r1 = self._run(n_steps=50, seed=99)
        r2 = self._run(n_steps=50, seed=99)
        df1 = r1.metrics.to_dataframe()
        df2 = r2.metrics.to_dataframe()
        assert list(df1["cumulative_trades"]) == list(df2["cumulative_trades"])

    def test_different_seeds_differ(self):
        r1 = self._run(n_steps=50, seed=1)
        r2 = self._run(n_steps=50, seed=2)
        df1 = r1.metrics.to_dataframe()
        df2 = r2.metrics.to_dataframe()
        # Should differ somewhere across 50 steps
        assert list(df1["cumulative_trades"]) != list(df2["cumulative_trades"]) or \
               list(df1["fair_value"]) != list(df2["fair_value"])

    def test_summary_is_string(self):
        result = self._run(n_steps=50)
        s = result.summary()
        assert isinstance(s, str)
        assert "SIMULATION SUMMARY" in s

    def test_noise_only_simulation(self):
        """Simulation with only noise traders should still produce trades."""
        agents = [NoiseTrader(f"NT{i}", activity_rate=1.0, random_seed=i) for i in range(4)]
        sim = MarketSimulation(agents=agents, n_steps=100, random_seed=0)
        result = sim.run()
        df = result.metrics.to_dataframe()
        assert df["cumulative_trades"].iloc[-1] > 0

    def test_informed_only_simulation(self):
        """Informed traders with no noise traders — may produce few/no trades (empty book)."""
        agents = [InformedTrader("IT1", activity_rate=1.0, random_seed=1)]
        sim = MarketSimulation(agents=agents, n_steps=50, random_seed=0)
        result = sim.run()
        # Should complete without error; trades may be zero (no opposing side)
        assert result.n_steps == 50

    def test_many_agents_stable(self):
        """10 agents running 300 steps should not crash."""
        agents = [NoiseTrader(f"NT{i}", random_seed=i) for i in range(7)]
        agents += [InformedTrader(f"IT{i}", random_seed=i + 100) for i in range(3)]
        sim = MarketSimulation(agents=agents, n_steps=300, random_seed=42)
        result = sim.run()
        assert len(result.metrics) == 300
