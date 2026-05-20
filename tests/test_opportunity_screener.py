"""
tests/test_opportunity_screener.py
------------------------------------
Tests for the opportunity screener extension. Synthetic data only.

Run with:
    pytest tests/test_opportunity_screener.py -v
"""

import os
import sys
import math
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.screener.market_data import TickerData
from src.screener.factors import FactorEngine
from src.screener.universe import (
    INSTITUTIONAL, EMERGING, SPECULATIVE, ALL_UNIVERSES,
    CapBucket, VolTolerance, get_universe, all_tickers,
)
from src.screener.opportunity import (
    OpportunityFactors, OpportunityResult, OpportunityPipeline,
    OpportunityScorer, compute_opportunity_factors,
    save_opportunity_csv, save_opportunity_markdown,
    print_opportunity_table, OPPORTUNITY_WEIGHTS,
    _cap_bonus, _vol_is_constructive, _generate_insight,
)
from src.models.regime import VolatilityRegime


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

N = 252

def make_td(
    ticker="TEST",
    drift=0.001,
    sigma=0.012,
    adv=50_000_000.0,
    seed=42,
    n=N,
) -> TickerData:
    np.random.seed(seed)
    rets = np.random.randn(n) * sigma + drift
    price = 100.0 * np.exp(np.cumsum(rets))
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": price * 0.999, "High": price * 1.005,
        "Low": price * 0.995, "Close": price,
        "Volume": volume, "Adj_Close": price,
    }, index=dates)
    return TickerData(
        ticker=ticker, df=df,
        start_date=str(dates[0].date()), end_date=str(dates[-1].date()),
        n_days=n, avg_dollar_volume=adv,
    )


def make_factors(drift=0.001, sigma=0.012, adv=50_000_000, seed=42) -> FactorEngine:
    return FactorEngine()


def compute_base_and_opp(ticker="T", drift=0.001, sigma=0.012, adv=50e6, seed=42,
                          cap=CapBucket.MID):
    td = make_td(ticker, drift=drift, sigma=sigma, adv=adv, seed=seed)
    base = FactorEngine().compute(td)
    opp = compute_opportunity_factors(base, td, cap_bucket=cap)
    return base, opp, td


# ─────────────────────────────────────────────────────────────
# Universe definitions
# ─────────────────────────────────────────────────────────────

class TestUniverseDefinitions:

    def test_all_universes_exist(self):
        assert set(ALL_UNIVERSES.keys()) == {"institutional", "emerging", "speculative"}

    def test_each_universe_has_tickers(self):
        for name, u in ALL_UNIVERSES.items():
            assert len(u.tickers) > 0, f"{name} has no tickers"

    def test_get_universe_known(self):
        u = get_universe("emerging")
        assert u.name == "emerging"

    def test_get_universe_unknown_raises(self):
        with pytest.raises(KeyError):
            get_universe("nonexistent")

    def test_all_tickers_no_duplicates(self):
        tickers = all_tickers()
        assert len(tickers) == len(set(tickers))

    def test_institutional_has_low_vol_tolerance(self):
        assert INSTITUTIONAL.vol_tolerance == VolTolerance.LOW

    def test_emerging_has_medium_vol_tolerance(self):
        assert EMERGING.vol_tolerance == VolTolerance.MEDIUM

    def test_speculative_has_high_vol_tolerance(self):
        assert SPECULATIVE.vol_tolerance == VolTolerance.HIGH

    def test_ticker_cap_assignments(self):
        assert INSTITUTIONAL.ticker_caps.get("MSFT") == CapBucket.MEGA
        assert EMERGING.ticker_caps.get("CRWD") == CapBucket.LARGE
        assert SPECULATIVE.ticker_caps.get("IONQ") == CapBucket.SMALL

    def test_spy_is_etf(self):
        assert INSTITUTIONAL.ticker_caps.get("SPY") == CapBucket.ETF


# ─────────────────────────────────────────────────────────────
# Cap bonus
# ─────────────────────────────────────────────────────────────

class TestCapBonus:

    def test_small_has_highest_bonus(self):
        assert _cap_bonus(CapBucket.SMALL) > _cap_bonus(CapBucket.MID)
        assert _cap_bonus(CapBucket.MID)   > _cap_bonus(CapBucket.LARGE)
        assert _cap_bonus(CapBucket.LARGE) > _cap_bonus(CapBucket.MEGA)

    def test_mega_has_zero_bonus(self):
        assert _cap_bonus(CapBucket.MEGA) == pytest.approx(0.0)

    def test_micro_has_negative_bonus(self):
        assert _cap_bonus(CapBucket.MICRO) < 0

    def test_etf_has_zero_bonus(self):
        assert _cap_bonus(CapBucket.ETF) == pytest.approx(0.0)

    def test_all_buckets_return_float(self):
        for bucket in CapBucket:
            bonus = _cap_bonus(bucket)
            assert isinstance(bonus, float)


# ─────────────────────────────────────────────────────────────
# Vol quality
# ─────────────────────────────────────────────────────────────

class TestVolQuality:

    def _base_with_regime(self, regime: VolatilityRegime, drawdown=0.10,
                           vol_pct=50.0, ret_20d=0.05):
        from src.screener.factors import TickerFactors
        f = TickerFactors(ticker="T")
        f.regime = regime
        f.max_drawdown = drawdown
        f.vol_pct = vol_pct
        f.ret_20d = ret_20d
        f.error = None
        f.vol_20d = 0.20
        return f

    def test_extreme_regime_not_constructive(self):
        base = self._base_with_regime(VolatilityRegime.EXTREME)
        assert not _vol_is_constructive(base)

    def test_high_drawdown_not_constructive(self):
        base = self._base_with_regime(VolatilityRegime.HIGH, drawdown=0.60)
        assert not _vol_is_constructive(base)

    def test_medium_regime_constructive(self):
        base = self._base_with_regime(VolatilityRegime.MEDIUM)
        assert _vol_is_constructive(base)

    def test_low_regime_constructive(self):
        base = self._base_with_regime(VolatilityRegime.LOW)
        assert _vol_is_constructive(base)

    def test_high_vol_pct_with_negative_return_not_constructive(self):
        base = self._base_with_regime(VolatilityRegime.HIGH, vol_pct=95.0, ret_20d=-0.15)
        assert not _vol_is_constructive(base)


# ─────────────────────────────────────────────────────────────
# Opportunity factor computation
# ─────────────────────────────────────────────────────────────

class TestOpportunityFactors:

    def test_momentum_slope_computed(self):
        base, opp, _ = compute_base_and_opp(drift=0.002, seed=1)
        if base.ret_20d is not None and base.ret_60d is not None:
            assert opp.momentum_slope is not None

    def test_rel_volume_positive(self):
        base, opp, _ = compute_base_and_opp(seed=2)
        if opp.rel_volume is not None:
            assert opp.rel_volume > 0

    def test_pct_from_52w_high_negative_or_zero(self):
        """52-week high proximity is always ≤ 0 (can't be above the high)."""
        base, opp, _ = compute_base_and_opp(seed=3)
        if opp.pct_from_52w_high is not None:
            assert opp.pct_from_52w_high <= 0 + 1e-9

    def test_cap_bucket_propagated(self):
        base, opp, _ = compute_base_and_opp(cap=CapBucket.SMALL)
        assert opp.cap_bucket == CapBucket.SMALL

    def test_cap_bonus_matches_bucket(self):
        _, opp, _ = compute_base_and_opp(cap=CapBucket.SMALL)
        assert opp.cap_bonus == _cap_bonus(CapBucket.SMALL)

    def test_high_risk_flag_for_extreme_regime(self):
        # Use very high sigma to trigger EXTREME regime
        td = make_td(drift=-0.001, sigma=0.04, seed=5)
        base = FactorEngine().compute(td)
        opp = compute_opportunity_factors(base, td, CapBucket.SMALL)
        if base.regime == VolatilityRegime.EXTREME:
            assert opp.high_risk_flag

    def test_vol_constructive_accessible(self):
        base, opp, _ = compute_base_and_opp(seed=6)
        assert isinstance(opp.vol_is_constructive, bool)


# ─────────────────────────────────────────────────────────────
# Opportunity scorer
# ─────────────────────────────────────────────────────────────

class TestOpportunityScorer:

    def _score(self, universe=EMERGING, drift=0.001, sigma=0.012, seed=1, cap=CapBucket.MID):
        td = make_td(drift=drift, sigma=sigma, seed=seed)
        base = FactorEngine().compute(td)
        opp = compute_opportunity_factors(base, td, cap)
        scorer = OpportunityScorer(universe)
        total, components = scorer.score(base, opp)
        return total, components

    def test_score_in_range(self):
        total, _ = self._score()
        assert 0 <= total <= 100

    def test_all_components_in_range(self):
        _, components = self._score()
        for k, v in components.items():
            assert 0 <= v <= 100, f"{k}={v} out of range"

    def test_component_keys_correct(self):
        _, components = self._score()
        assert set(components.keys()) == set(OPPORTUNITY_WEIGHTS.keys())

    def test_weights_sum_to_one(self):
        assert abs(sum(OPPORTUNITY_WEIGHTS.values()) - 1.0) < 1e-9

    def test_high_drift_scores_higher(self):
        high, _ = self._score(drift=0.003, sigma=0.008, seed=1)
        low, _  = self._score(drift=-0.002, sigma=0.008, seed=1)
        assert high > low

    def test_small_cap_bonus_applied(self):
        small, _ = self._score(cap=CapBucket.SMALL)
        mega, _  = self._score(cap=CapBucket.MEGA)
        # Small cap should get a bonus in opportunity_bonus component
        _, comp_small = self._score(cap=CapBucket.SMALL)
        _, comp_mega  = self._score(cap=CapBucket.MEGA)
        assert comp_small["opportunity_bonus"] >= comp_mega["opportunity_bonus"]

    def test_institutional_penalises_high_vol_more(self):
        """Institutional universe should give lower regime score for HIGH vol."""
        td = make_td(sigma=0.03, seed=7)   # ~47% annualised → HIGH
        base = FactorEngine().compute(td)
        opp = compute_opportunity_factors(base, td, CapBucket.LARGE)
        if base.regime == VolatilityRegime.HIGH:
            _, comp_inst  = OpportunityScorer(INSTITUTIONAL).score(base, opp)
            _, comp_emerg = OpportunityScorer(EMERGING).score(base, opp)
            assert comp_inst["regime_quality"] <= comp_emerg["regime_quality"]


# ─────────────────────────────────────────────────────────────
# Opportunity pipeline
# ─────────────────────────────────────────────────────────────

class TestOpportunityPipeline:

    def _make_fetch_data(self, n=4, seed_offset=0):
        return {
            f"T{i}": make_td(f"T{i}", drift=0.001*(i-2), seed=seed_offset+i)
            for i in range(n)
        }

    def test_pipeline_returns_results(self):
        fetch = self._make_fetch_data()
        pipeline = OpportunityPipeline(EMERGING)
        results = pipeline.run(fetch)
        assert len(results) > 0

    def test_results_sorted_descending(self):
        fetch = self._make_fetch_data(n=5)
        results = OpportunityPipeline(EMERGING).run(fetch)
        scores = [r.opportunity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_in_range(self):
        fetch = self._make_fetch_data(n=4)
        results = OpportunityPipeline(EMERGING).run(fetch)
        for r in results:
            assert 0 <= r.opportunity_score <= 100

    def test_universe_name_in_result(self):
        fetch = self._make_fetch_data(n=3)
        results = OpportunityPipeline(EMERGING).run(fetch)
        for r in results:
            assert r.universe == "emerging"

    def test_insight_string_not_empty(self):
        fetch = self._make_fetch_data(n=3)
        results = OpportunityPipeline(EMERGING).run(fetch)
        for r in results:
            assert isinstance(r.insight, str)
            assert len(r.insight) > 0

    def test_cap_bucket_propagated(self):
        universe = EMERGING
        td = make_td("CRWD")
        fetch = {"CRWD": td}
        results = OpportunityPipeline(universe).run(fetch)
        assert results[0].cap_bucket == universe.ticker_caps.get("CRWD", CapBucket.UNKNOWN)

    def test_avoid_for_extreme_regime(self):
        td = make_td(sigma=0.05, drift=-0.001, seed=99)   # very high vol
        fetch = {"EXTREME": td}
        results = OpportunityPipeline(SPECULATIVE).run(fetch)
        # If regime is EXTREME, result should be marked avoid
        for r in results:
            if r.regime == VolatilityRegime.EXTREME:
                assert r.is_avoid

    def test_illiquid_gets_low_liquidity_score(self):
        td = make_td("ILLIQ", adv=100_000)   # $100K ADV — very illiquid
        fetch = {"ILLIQ": td}
        results = OpportunityPipeline(EMERGING).run(fetch)
        if results:
            assert results[0].liquidity_score < 30

    def test_summary_row_has_required_keys(self):
        fetch = self._make_fetch_data(n=2)
        results = OpportunityPipeline(EMERGING).run(fetch)
        if results:
            row = results[0].summary_row()
            required = {"Ticker", "Universe", "OppScore", "Regime", "Insight",
                        "Ret20d%", "MaxDD%", "MomSlope"}
            assert required.issubset(set(row.keys()))

    def test_reproducible(self):
        fetch = self._make_fetch_data(n=4, seed_offset=10)
        r1 = OpportunityPipeline(EMERGING).run(fetch)
        r2 = OpportunityPipeline(EMERGING).run(fetch)
        assert [r.opportunity_score for r in r1] == [r.opportunity_score for r in r2]


# ─────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────

class TestOpportunityReports:

    def _make_results(self, n=4) -> dict:
        fetch = {f"T{i}": make_td(f"T{i}", drift=0.001*(i-2), seed=i) for i in range(n)}
        results = OpportunityPipeline(EMERGING).run(fetch)
        return {"emerging": results}

    def test_save_csv(self):
        results_by_uni = self._make_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opp.csv")
            result_path = save_opportunity_csv(results_by_uni, path)
            assert os.path.exists(result_path)
            df = pd.read_csv(result_path)
            assert "Ticker" in df.columns
            assert "OppScore" in df.columns

    def test_save_markdown(self):
        results_by_uni = self._make_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opp.md")
            result_path = save_opportunity_markdown(results_by_uni, path)
            assert os.path.exists(result_path)
            content = open(result_path).read()
            assert "Opportunity Screener" in content
            assert "NOT FINANCIAL ADVICE" in content

    def test_print_opportunity_table_runs(self, capsys):
        fetch = {f"T{i}": make_td(f"T{i}", seed=i) for i in range(3)}
        results = OpportunityPipeline(EMERGING).run(fetch)
        print_opportunity_table(results, "Emerging Leaders")
        captured = capsys.readouterr()
        assert "EMERGING LEADERS" in captured.out

    def test_csv_contains_all_tickers(self):
        fetch = {f"T{i}": make_td(f"T{i}", seed=i) for i in range(4)}
        results = OpportunityPipeline(EMERGING).run(fetch)
        results_by_uni = {"emerging": results}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "opp.csv")
            save_opportunity_csv(results_by_uni, path)
            df = pd.read_csv(path)
            for r in results:
                assert r.ticker in df["Ticker"].values


# ─────────────────────────────────────────────────────────────
# Insight generation
# ─────────────────────────────────────────────────────────────

class TestInsightGeneration:

    def _make_base_and_opp(self, drift=0.001, sigma=0.012, seed=1):
        td = make_td(drift=drift, sigma=sigma, seed=seed)
        base = FactorEngine().compute(td)
        opp = compute_opportunity_factors(base, td, CapBucket.MID)
        return base, opp

    def test_insight_is_string(self):
        base, opp = self._make_base_and_opp()
        insight = _generate_insight(base, opp, EMERGING)
        assert isinstance(insight, str)

    def test_insight_non_empty(self):
        base, opp = self._make_base_and_opp()
        insight = _generate_insight(base, opp, EMERGING)
        assert len(insight) > 0

    def test_insight_mentions_regime(self):
        base, opp = self._make_base_and_opp(sigma=0.005)  # low vol → LOW regime
        insight = _generate_insight(base, opp, EMERGING)
        # Should mention regime
        assert any(word in insight.lower() for word in ["vol", "regime", "stable"])

    def test_extreme_regime_mentioned(self):
        # Create a ticker in EXTREME regime
        td = make_td(sigma=0.045, drift=-0.001, seed=99)
        base = FactorEngine().compute(td)
        opp = compute_opportunity_factors(base, td, CapBucket.SMALL)
        if base.regime == VolatilityRegime.EXTREME:
            insight = _generate_insight(base, opp, SPECULATIVE)
            assert "extreme" in insight.lower() or "dangerous" in insight.lower()


# ─────────────────────────────────────────────────────────────
# Stable ranking behavior
# ─────────────────────────────────────────────────────────────

class TestStableRanking:

    def test_clearly_better_ticker_ranks_higher(self):
        """Strong uptrend should rank higher than downtrend."""
        fetch = {
            "GOOD": make_td("GOOD", drift=0.003, sigma=0.008, seed=1),
            "BAD":  make_td("BAD",  drift=-0.003, sigma=0.025, seed=2),
        }
        results = OpportunityPipeline(EMERGING).run(fetch)
        scores = {r.ticker: r.opportunity_score for r in results}
        assert scores["GOOD"] > scores["BAD"]

    def test_mid_cap_ranks_above_mega_same_factors(self):
        """Mid-cap gets an opportunity bonus that mega-cap does not."""
        td = make_td("TEST", drift=0.002, sigma=0.012, seed=5)
        base = FactorEngine().compute(td)
        opp_mid  = compute_opportunity_factors(base, td, CapBucket.MID)
        opp_mega = compute_opportunity_factors(base, td, CapBucket.MEGA)
        scorer = OpportunityScorer(EMERGING)
        score_mid,  _ = scorer.score(base, opp_mid)
        score_mega, _ = scorer.score(base, opp_mega)
        assert score_mid > score_mega

    def test_illiquid_penalised(self):
        liquid   = make_td("LIQ",   adv=500_000_000, seed=3)
        illiquid = make_td("ILLIQ", adv=500_000, seed=3)
        res_liq  = OpportunityPipeline(EMERGING).run({"LIQ":   liquid})
        res_ill  = OpportunityPipeline(EMERGING).run({"ILLIQ": illiquid})
        assert res_liq[0].liquidity_score > res_ill[0].liquidity_score
