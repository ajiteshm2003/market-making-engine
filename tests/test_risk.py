"""
tests/test_risk.py
-------------------
Tests for VaR, Expected Shortfall, rolling VaR, and portfolio exposure.
"""

import math
import random
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.risk.var import (
    historical_var, parametric_var, expected_shortfall,
    rolling_var, pnl_distribution_stats, _normal_ppf,
)
from src.risk.portfolio import PortfolioExposureTracker


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def normal_pnl(n=200, mu=0.0, sigma=1.0, seed=42):
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


# ─────────────────────────────────────────────────────────────
# normal_ppf
# ─────────────────────────────────────────────────────────────

class TestNormalPPF:
    def test_95th_percentile(self):
        z = _normal_ppf(0.95)
        assert z == pytest.approx(1.645, abs=0.01)

    def test_99th_percentile(self):
        z = _normal_ppf(0.99)
        assert z == pytest.approx(2.326, abs=0.01)

    def test_50th_percentile(self):
        z = _normal_ppf(0.50)
        assert z == pytest.approx(0.0, abs=0.01)

    def test_symmetry(self):
        assert _normal_ppf(0.95) == pytest.approx(-_normal_ppf(0.05), abs=1e-3)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _normal_ppf(0.0)
        with pytest.raises(ValueError):
            _normal_ppf(1.0)


# ─────────────────────────────────────────────────────────────
# historical_var
# ─────────────────────────────────────────────────────────────

class TestHistoricalVaR:

    def test_returns_var_result(self):
        pnl = normal_pnl()
        r = historical_var(pnl, 0.95)
        assert r.method == "historical"
        assert r.confidence == pytest.approx(0.95)
        assert r.n_obs == 200

    def test_var_positive(self):
        """VaR should be a positive loss magnitude."""
        pnl = normal_pnl()
        r = historical_var(pnl, 0.95)
        assert r.var >= 0

    def test_var_99_exceeds_var_95(self):
        """Higher confidence → larger VaR."""
        pnl = normal_pnl(n=500)
        r95 = historical_var(pnl, 0.95)
        r99 = historical_var(pnl, 0.99)
        assert r99.var >= r95.var

    def test_zero_returns_zero_var(self):
        """Flat P&L has zero VaR."""
        pnl = [0.0] * 50
        r = historical_var(pnl, 0.95)
        assert r.var == pytest.approx(0.0, abs=1e-6)

    def test_all_positive_pnl_zero_var(self):
        """If all P&L is positive, loss quantile is zero (or negative → VaR=0)."""
        pnl = [1.0] * 50
        r = historical_var(pnl, 0.95)
        assert r.var <= 0  # positive skew means no loss at 95th percentile

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            historical_var([1.0, 2.0], 0.95)

    def test_invalid_confidence_raises(self):
        pnl = normal_pnl()
        with pytest.raises(ValueError):
            historical_var(pnl, 1.5)
        with pytest.raises(ValueError):
            historical_var(pnl, -0.1)

    def test_nan_filtered(self):
        """NaN values should be filtered out."""
        pnl = normal_pnl(n=100) + [float("nan")] * 10
        r = historical_var(pnl, 0.95)
        assert r.n_obs == 100

    def test_large_n_converges_to_normal_quantile(self):
        """With large N, historical VaR at 95% ≈ 1.645σ (for N(0,1))."""
        pnl = normal_pnl(n=10_000, mu=0.0, sigma=1.0, seed=1)
        r = historical_var(pnl, 0.95)
        # For N(0,1): 95% VaR ≈ 1.645
        assert r.var == pytest.approx(1.645, abs=0.15)


# ─────────────────────────────────────────────────────────────
# parametric_var
# ─────────────────────────────────────────────────────────────

class TestParametricVaR:

    def test_returns_var_result(self):
        pnl = normal_pnl()
        r = parametric_var(pnl, 0.95)
        assert r.method == "parametric_gaussian"

    def test_known_distribution(self):
        """For N(0,1) with large sample, parametric VaR ≈ 1.645."""
        pnl = normal_pnl(n=10_000, mu=0.0, sigma=1.0, seed=2)
        r = parametric_var(pnl, 0.95)
        assert r.var == pytest.approx(1.645, abs=0.05)

    def test_positive_mean_reduces_var(self):
        """Positive expected return reduces VaR."""
        pnl_pos = normal_pnl(n=1000, mu=1.0, sigma=1.0, seed=3)
        pnl_zero = normal_pnl(n=1000, mu=0.0, sigma=1.0, seed=3)
        r_pos  = parametric_var(pnl_pos, 0.95)
        r_zero = parametric_var(pnl_zero, 0.95)
        assert r_pos.var < r_zero.var

    def test_higher_sigma_increases_var(self):
        pnl_low  = normal_pnl(n=500, mu=0.0, sigma=0.5, seed=4)
        pnl_high = normal_pnl(n=500, mu=0.0, sigma=2.0, seed=4)
        r_low  = parametric_var(pnl_low,  0.95)
        r_high = parametric_var(pnl_high, 0.95)
        assert r_high.var > r_low.var


# ─────────────────────────────────────────────────────────────
# expected_shortfall
# ─────────────────────────────────────────────────────────────

class TestExpectedShortfall:

    def test_es_exceeds_var(self):
        """ES must always be >= VaR at same confidence."""
        pnl = normal_pnl(n=500)
        result = expected_shortfall(pnl, 0.95)
        assert result.es >= result.var

    def test_tail_obs_count(self):
        """About 5% of 500 observations should be in the 95% tail."""
        pnl = normal_pnl(n=500)
        result = expected_shortfall(pnl, 0.95)
        assert 0 <= result.tail_obs <= 50  # 10% tolerance

    def test_es_99_exceeds_es_95(self):
        pnl = normal_pnl(n=500)
        es95 = expected_shortfall(pnl, 0.95)
        es99 = expected_shortfall(pnl, 0.99)
        assert es99.es >= es95.es

    def test_known_distribution_es(self):
        """For N(0,1) at 95%: ES ≈ E[X | X < -1.645] ≈ 2.063."""
        pnl = normal_pnl(n=20_000, mu=0.0, sigma=1.0, seed=5)
        result = expected_shortfall(pnl, 0.95)
        assert result.es == pytest.approx(2.063, abs=0.15)

    def test_flat_pnl_es_zero(self):
        pnl = [0.0] * 50
        result = expected_shortfall(pnl, 0.95)
        assert result.es == pytest.approx(0.0, abs=0.01)


# ─────────────────────────────────────────────────────────────
# rolling_var
# ─────────────────────────────────────────────────────────────

class TestRollingVaR:

    def test_output_length_matches_input(self):
        pnl = normal_pnl(n=200)
        rv = rolling_var(pnl, window=50)
        assert len(rv) == 200

    def test_first_window_minus_1_are_none(self):
        pnl = normal_pnl(n=100)
        rv = rolling_var(pnl, window=30)
        for i in range(29):
            assert rv[i] is None

    def test_after_window_are_float(self):
        pnl = normal_pnl(n=100)
        rv = rolling_var(pnl, window=30)
        for i in range(29, 100):
            assert rv[i] is not None
            assert isinstance(rv[i], float)

    def test_rolling_var_non_negative(self):
        pnl = normal_pnl(n=100)
        rv = rolling_var(pnl, window=20)
        for v in rv:
            if v is not None:
                assert v >= 0

    def test_parametric_method(self):
        pnl = normal_pnl(n=150)
        rv = rolling_var(pnl, window=50, method="parametric_gaussian")
        non_none = [v for v in rv if v is not None]
        assert len(non_none) > 0


# ─────────────────────────────────────────────────────────────
# pnl_distribution_stats
# ─────────────────────────────────────────────────────────────

class TestPnLDistributionStats:

    def test_n_correct(self):
        pnl = normal_pnl(n=150)
        stats = pnl_distribution_stats(pnl)
        assert stats.n == 150

    def test_mean_reasonable(self):
        pnl = normal_pnl(n=500, mu=0.5, sigma=1.0, seed=7)
        stats = pnl_distribution_stats(pnl)
        assert stats.mean == pytest.approx(0.5, abs=0.15)

    def test_std_reasonable(self):
        pnl = normal_pnl(n=500, mu=0.0, sigma=2.0, seed=8)
        stats = pnl_distribution_stats(pnl)
        assert stats.std == pytest.approx(2.0, abs=0.3)

    def test_percentiles_ordered(self):
        pnl = normal_pnl(n=200)
        stats = pnl_distribution_stats(pnl)
        assert stats.min <= stats.p5 <= stats.p25 <= stats.median
        assert stats.median <= stats.p75 <= stats.p95 <= stats.p99 <= stats.max

    def test_positive_days_range(self):
        pnl = normal_pnl(n=200)
        stats = pnl_distribution_stats(pnl)
        assert 0 <= stats.positive_days <= 1

    def test_sharpe_sign(self):
        """Positive mean → positive Sharpe."""
        pnl = normal_pnl(n=500, mu=0.5, sigma=0.3, seed=9)
        stats = pnl_distribution_stats(pnl)
        assert stats.sharpe > 0


# ─────────────────────────────────────────────────────────────
# PortfolioExposureTracker
# ─────────────────────────────────────────────────────────────

class TestPortfolioExposureTracker:

    def _tracker_with_data(self, n=20, seed=1) -> PortfolioExposureTracker:
        rng = random.Random(seed)
        tracker = PortfolioExposureTracker(initial_cash=100_000.0)
        for t in range(n):
            inv = rng.uniform(-10, 10)
            price = 100.0 + rng.gauss(0, 1)
            cash = 100_000.0 - inv * price
            tracker.update(t, inv, price, cash)
        return tracker

    def test_snapshots_count(self):
        tracker = self._tracker_with_data(n=15)
        assert len(tracker.snapshots) == 15

    def test_notional_is_abs_inventory_times_price(self):
        tracker = PortfolioExposureTracker()
        snap = tracker.update(1, inventory=5.0, mark_price=100.0, cash=99_500.0)
        assert snap.notional_exposure == pytest.approx(500.0)

    def test_negative_inventory_notional_positive(self):
        tracker = PortfolioExposureTracker()
        snap = tracker.update(1, inventory=-3.0, mark_price=100.0, cash=100_300.0)
        assert snap.notional_exposure == pytest.approx(300.0)  # |−3| × 100

    def test_zero_inventory_zero_notional(self):
        tracker = PortfolioExposureTracker()
        snap = tracker.update(1, inventory=0.0, mark_price=100.0, cash=100_000.0)
        assert snap.notional_exposure == pytest.approx(0.0)

    def test_leverage_positive(self):
        tracker = PortfolioExposureTracker()
        tracker.update(1, 10.0, 100.0, 99_000.0)
        snap = tracker.snapshots[-1]
        assert snap.leverage >= 0

    def test_concentration_bounded(self):
        tracker = self._tracker_with_data(n=20)
        for s in tracker.snapshots:
            assert 0 <= s.inv_concentration <= 1.0 + 1e-9

    def test_summarize_equity_drawdown_non_negative(self):
        tracker = self._tracker_with_data(n=30)
        summary = tracker.summarize()
        assert summary.equity_drawdown >= 0

    def test_summarize_n_steps(self):
        tracker = self._tracker_with_data(n=25)
        summary = tracker.summarize()
        assert summary.n_steps == 25

    def test_reset_clears_snapshots(self):
        tracker = self._tracker_with_data(n=10)
        tracker.reset()
        assert len(tracker.snapshots) == 0

    def test_empty_tracker_raises(self):
        tracker = PortfolioExposureTracker()
        with pytest.raises(RuntimeError):
            tracker.summarize()

    def test_update_from_mm(self):
        """Test compatibility with MarketMakerMetrics objects."""
        from src.strategies import NaiveMarketMaker
        from src.agents import NoiseTrader
        from src.simulation import MarketSimulation, FairValueConfig

        nmm = NaiveMarketMaker("NMM_t", half_spread=0.05)
        noise = [NoiseTrader("NT0", activity_rate=0.5, random_seed=1)]
        sim = MarketSimulation(agents=noise + [nmm], n_steps=50,
                               fair_value_config=FairValueConfig(), random_seed=1)
        sim.run()

        tracker = PortfolioExposureTracker(initial_cash=100_000.0)
        mkt_df = sim._metrics.to_dataframe()
        for t, row in mkt_df.iterrows():
            mid = row["midprice"]
            if mid is not None and not (isinstance(mid, float) and math.isnan(mid)):
                tracker.update_from_mm(t, nmm.mm_metrics, mid)

        assert len(tracker.snapshots) > 0
