"""
tests/test_screener.py
-----------------------
Tests for the stock screener. Uses only synthetic data — no yfinance calls.

Run with:
    pytest tests/test_screener.py -v
"""

import math
import os
import sys
import tempfile
import random

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.screener.market_data import TickerData, FetchResult, DEFAULT_UNIVERSE
from src.screener.factors import (
    FactorEngine, TickerFactors,
    _log_returns, _total_log_return, _realized_vol, _max_drawdown,
    EQUITY_THRESHOLDS,
)
from src.screener.scoring import Scorer, ScoredTicker, WEIGHTS
from src.screener.report import (
    save_csv, save_markdown, print_terminal_table, DISCLAIMER
)
from src.models.regime import VolatilityRegime


# ─────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────

def make_price_series(
    n: int = 252,
    drift: float = 0.0003,
    sigma: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """Create a synthetic daily price series."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, sigma, n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


def make_ticker_data(
    ticker: str = "TEST",
    n: int = 252,
    drift: float = 0.0003,
    sigma: float = 0.01,
    adv: float = 50_000_000.0,
    seed: int = 42,
) -> TickerData:
    """Create a synthetic TickerData."""
    price = make_price_series(n=n, drift=drift, sigma=sigma, seed=seed)
    volume = pd.Series(
        np.random.default_rng(seed).integers(1_000_000, 5_000_000, n).astype(float),
        index=price.index,
    )
    df = pd.DataFrame({
        "Open":      price * 0.999,
        "High":      price * 1.005,
        "Low":       price * 0.995,
        "Close":     price,
        "Volume":    volume,
        "Adj_Close": price,
    })
    return TickerData(
        ticker=ticker,
        df=df,
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
        n_days=n,
        avg_dollar_volume=adv,
    )


def make_invalid_ticker_data(ticker: str = "BAD", n: int = 10) -> TickerData:
    """Create a TickerData with too few days."""
    price = make_price_series(n=n)
    df = pd.DataFrame({
        "Open": price, "High": price, "Low": price,
        "Close": price, "Volume": pd.Series([1e6]*n, index=price.index),
        "Adj_Close": price,
    })
    return TickerData(ticker=ticker, df=df, start_date="2023-01-01",
                      end_date="2023-01-14", n_days=n, avg_dollar_volume=1e6)


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

class TestHelpers:

    def test_log_returns_length(self):
        price = make_price_series(n=100)
        lr = _log_returns(price)
        assert len(lr) == 99  # n-1

    def test_log_returns_values(self):
        prices = pd.Series([100.0, 110.0, 99.0])
        lr = _log_returns(prices)
        assert lr.iloc[0] == pytest.approx(math.log(110/100), rel=1e-6)
        assert lr.iloc[1] == pytest.approx(math.log(99/110), rel=1e-6)

    def test_total_log_return_correct(self):
        prices = pd.Series([100.0] * 60 + [110.0])
        r = _total_log_return(prices, 20)
        assert r == pytest.approx(math.log(110/100), rel=1e-4)

    def test_total_log_return_insufficient_data(self):
        prices = pd.Series([100.0, 101.0])
        assert _total_log_return(prices, 20) is None

    def test_realized_vol_zero_for_flat(self):
        lr = pd.Series([0.0] * 20)
        assert _realized_vol(lr, 20) == pytest.approx(0.0)

    def test_realized_vol_positive_for_noisy(self):
        np.random.seed(1)
        lr = pd.Series(np.random.randn(50) * 0.01)
        assert _realized_vol(lr, 20) > 0

    def test_max_drawdown_flat_series(self):
        price = pd.Series([100.0] * 50)
        assert _max_drawdown(price) == pytest.approx(0.0)

    def test_max_drawdown_known_case(self):
        # Peak 100, trough 80 → 20% drawdown
        price = pd.Series([80.0, 90.0, 100.0, 90.0, 80.0, 85.0])
        dd = _max_drawdown(price)
        assert dd == pytest.approx(0.20, abs=0.01)

    def test_max_drawdown_monotone_increase(self):
        price = pd.Series([90.0, 95.0, 100.0, 105.0])
        assert _max_drawdown(price) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────
# FactorEngine
# ─────────────────────────────────────────────────────────────

class TestFactorEngine:

    def _engine(self):
        return FactorEngine(thresholds=EQUITY_THRESHOLDS)

    def test_returns_ticker_factors(self):
        td = make_ticker_data()
        f = self._engine().compute(td)
        assert isinstance(f, TickerFactors)
        assert f.ticker == "TEST"

    def test_is_valid_with_good_data(self):
        td = make_ticker_data(n=252)
        f = self._engine().compute(td)
        assert f.is_valid

    def test_is_invalid_with_insufficient_data(self):
        td = make_invalid_ticker_data(n=10)
        f = self._engine().compute(td)
        assert not f.is_valid

    def test_returns_computed(self):
        td = make_ticker_data(n=252, drift=0.0005)   # positive drift
        f = self._engine().compute(td)
        assert f.ret_20d is not None
        assert f.ret_60d is not None

    def test_positive_drift_gives_positive_returns(self):
        td = make_ticker_data(n=252, drift=0.002, sigma=0.001, seed=1)
        f = self._engine().compute(td)
        assert f.ret_20d > 0
        assert f.ret_60d > 0

    def test_negative_drift_gives_negative_returns(self):
        td = make_ticker_data(n=252, drift=-0.002, sigma=0.001, seed=2)
        f = self._engine().compute(td)
        assert f.ret_60d < 0

    def test_vol_20d_annualised_plausible(self):
        # sigma=0.01 daily → ~16% annualised
        td = make_ticker_data(n=252, sigma=0.01, seed=3)
        f = self._engine().compute(td)
        assert f.vol_20d is not None
        assert 0.05 < f.vol_20d < 0.50   # plausible range

    def test_high_sigma_gives_high_vol(self):
        td_low  = make_ticker_data(n=252, sigma=0.005, seed=4)
        td_high = make_ticker_data(n=252, sigma=0.03, seed=4)
        f_low  = self._engine().compute(td_low)
        f_high = self._engine().compute(td_high)
        assert f_high.vol_20d > f_low.vol_20d

    def test_regime_classification_low(self):
        # Very low daily vol → LOW annualised regime
        td = make_ticker_data(n=252, sigma=0.004, seed=5)
        f = self._engine().compute(td)
        # vol_20d ~ 6% annualised → should be LOW
        assert f.regime in (VolatilityRegime.LOW, VolatilityRegime.MEDIUM)

    def test_regime_classification_extreme(self):
        # Very high daily vol → EXTREME annualised regime
        td = make_ticker_data(n=252, sigma=0.04, seed=6)
        f = self._engine().compute(td)
        # vol_20d ~ 63% annualised → should be HIGH or EXTREME
        assert f.regime in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME)

    def test_var_95_non_negative(self):
        td = make_ticker_data(n=252, seed=7)
        f = self._engine().compute(td)
        if f.var_95 is not None:
            assert f.var_95 >= 0

    def test_es_exceeds_var(self):
        td = make_ticker_data(n=252, seed=8)
        f = self._engine().compute(td)
        if f.var_95 is not None and f.es_95 is not None:
            assert f.es_95 >= f.var_95 - 1e-9

    def test_max_drawdown_non_negative(self):
        td = make_ticker_data(n=252, seed=9)
        f = self._engine().compute(td)
        assert f.max_drawdown >= 0

    def test_downside_vol_non_negative(self):
        td = make_ticker_data(n=252, seed=10)
        f = self._engine().compute(td)
        if f.downside_vol is not None:
            assert f.downside_vol >= 0

    def test_vol_pct_in_range(self):
        td = make_ticker_data(n=252, seed=11)
        f = self._engine().compute(td)
        if f.vol_pct is not None:
            assert 0.0 <= f.vol_pct <= 100.0

    def test_price_vs_ma20_correct_sign(self):
        """If price is monotonically increasing, it should be above MA20."""
        td = make_ticker_data(n=252, drift=0.002, sigma=0.0001, seed=12)
        f = self._engine().compute(td)
        if f.price_vs_ma20 is not None:
            assert f.price_vs_ma20 > 0   # price > MA20 for strong uptrend

    def test_compute_batch(self):
        tds = {
            "A": make_ticker_data("A", seed=1),
            "B": make_ticker_data("B", seed=2),
        }
        engine = self._engine()
        result = engine.compute_batch(tds)
        assert set(result.keys()) == {"A", "B"}
        for f in result.values():
            assert isinstance(f, TickerFactors)


# ─────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────

class TestScorer:

    def _score(self, **kwargs) -> ScoredTicker:
        td = make_ticker_data(**kwargs)
        engine = FactorEngine()
        factors = engine.compute(td)
        scorer = Scorer()
        return scorer.score(factors)

    def test_score_in_range(self):
        s = self._score(seed=1)
        assert 0 <= s.total_score <= 100

    def test_high_drift_scores_higher_than_low(self):
        high = self._score(drift=0.002, sigma=0.005, seed=3)
        low  = self._score(drift=-0.002, sigma=0.005, seed=3)
        assert high.total_score > low.total_score

    def test_low_vol_scores_higher_regime(self):
        low_vol  = self._score(sigma=0.004, drift=0.001, seed=5)
        high_vol = self._score(sigma=0.040, drift=0.001, seed=5)
        assert low_vol.regime_score > high_vol.regime_score

    def test_extreme_regime_triggers_avoid(self):
        s = self._score(sigma=0.040, seed=6)   # ~63% annualised
        if s.regime == VolatilityRegime.EXTREME:
            assert s.is_avoid

    def test_invalid_factors_score_zero(self):
        td = make_invalid_ticker_data(n=5)
        factors = FactorEngine().compute(td)
        scorer = Scorer()
        s = scorer.score(factors)
        assert s.total_score == pytest.approx(0.0)
        assert s.is_avoid

    def test_score_all_returns_sorted(self):
        tds = {f"T{i}": make_ticker_data(f"T{i}", seed=i) for i in range(5)}
        factors = FactorEngine().compute_batch(tds)
        scorer = Scorer()
        scored = scorer.score_all(factors)
        scores = [s.total_score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_reason_string_not_empty(self):
        s = self._score(seed=7)
        assert isinstance(s.reason, str)
        assert len(s.reason) > 0

    def test_regime_label_string(self):
        s = self._score(seed=8)
        assert s.regime_label in ("LOW", "MEDIUM", "HIGH", "EXTREME", "UNKNOWN")

    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_liquidity_score_high_for_large_adv(self):
        td = make_ticker_data(adv=500_000_000)   # $500M ADV
        f = FactorEngine().compute(td)
        s = Scorer().score(f)
        assert s.liquidity_score > 60

    def test_summary_row_has_required_keys(self):
        s = self._score(seed=9)
        row = s.summary_row()
        required = {"Ticker", "Score", "RegimeLabel", "Ret20d%", "MaxDD%", "VaR95%", "Reason"}
        assert required.issubset(set(row.keys()))


# ─────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────

class TestReportGeneration:

    def _scored_list(self, n: int = 5) -> list:
        tds = {f"T{i}": make_ticker_data(f"T{i}", seed=i, drift=0.001*(i-2)) for i in range(n)}
        factors = FactorEngine().compute_batch(tds)
        return Scorer().score_all(factors)

    def test_save_csv(self):
        scored = self._scored_list()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            result = save_csv(scored, path)
            assert os.path.exists(result)
            df = pd.read_csv(result)
            assert len(df) == len(scored)
            assert "Ticker" in df.columns
            assert "Score" in df.columns

    def test_save_markdown(self):
        scored = self._scored_list()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.md")
            result = save_markdown(scored, path, test_portfolio_size=1000.0)
            assert os.path.exists(result)
            content = open(result).read()
            assert "Stock Screener Report" in content
            assert "DISCLAIMER" in content or "NOT FINANCIAL ADVICE" in content

    def test_markdown_contains_tickers(self):
        scored = self._scored_list(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.md")
            save_markdown(scored, path)
            content = open(path).read()
            for s in scored:
                assert s.ticker in content

    def test_csv_score_matches(self):
        scored = self._scored_list(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            save_csv(scored, path)
            df = pd.read_csv(path)
            for s in scored:
                row = df[df["Ticker"] == s.ticker].iloc[0]
                assert float(row["Score"]) == pytest.approx(s.total_score, abs=0.1)

    def test_print_terminal_table_runs(self, capsys):
        scored = self._scored_list(3)
        print_terminal_table(scored, top_n=3)
        captured = capsys.readouterr()
        assert "STOCK SCREENER REPORT" in captured.out

    def test_disclaimer_in_terminal_output(self, capsys):
        scored = self._scored_list(2)
        print_terminal_table(scored)
        captured = capsys.readouterr()
        assert "NOT FINANCIAL ADVICE" in captured.out


# ─────────────────────────────────────────────────────────────
# Default universe
# ─────────────────────────────────────────────────────────────

class TestDefaultUniverse:

    def test_default_universe_not_empty(self):
        assert len(DEFAULT_UNIVERSE) > 0

    def test_default_universe_contains_spy(self):
        assert "SPY" in DEFAULT_UNIVERSE

    def test_default_universe_no_duplicates(self):
        assert len(DEFAULT_UNIVERSE) == len(set(DEFAULT_UNIVERSE))


# ─────────────────────────────────────────────────────────────
# Regime classification boundaries
# ─────────────────────────────────────────────────────────────

class TestRegimeBoundaries:

    def test_low_sigma_low_regime(self):
        engine = FactorEngine(thresholds=EQUITY_THRESHOLDS)
        # 5% annualised → LOW
        assert engine._classify_regime(0.05) == VolatilityRegime.LOW

    def test_medium_regime(self):
        engine = FactorEngine(thresholds=EQUITY_THRESHOLDS)
        # 18% annualised → MEDIUM
        assert engine._classify_regime(0.18) == VolatilityRegime.MEDIUM

    def test_high_regime(self):
        engine = FactorEngine(thresholds=EQUITY_THRESHOLDS)
        # 35% annualised → HIGH
        assert engine._classify_regime(0.35) == VolatilityRegime.HIGH

    def test_extreme_regime(self):
        engine = FactorEngine(thresholds=EQUITY_THRESHOLDS)
        # 60% annualised → EXTREME
        assert engine._classify_regime(0.60) == VolatilityRegime.EXTREME

    def test_boundary_at_low_threshold(self):
        engine = FactorEngine(thresholds=EQUITY_THRESHOLDS)
        t = EQUITY_THRESHOLDS.low_threshold
        assert engine._classify_regime(t - 0.001) == VolatilityRegime.LOW
        assert engine._classify_regime(t + 0.001) == VolatilityRegime.MEDIUM

    def test_boundary_at_extreme_threshold(self):
        engine = FactorEngine(thresholds=EQUITY_THRESHOLDS)
        t = EQUITY_THRESHOLDS.extreme_threshold
        assert engine._classify_regime(t + 0.001) == VolatilityRegime.EXTREME


# ─────────────────────────────────────────────────────────────
# Illiquid rejection
# ─────────────────────────────────────────────────────────────

class TestIlliquidRejection:

    def test_illiquid_ticker_gets_low_liquidity_score(self):
        td = make_ticker_data("ILLIQ", adv=500_000)   # only $500K ADV
        factors = FactorEngine().compute(td)
        scored = Scorer().score(factors)
        assert scored.liquidity_score < 30

    def test_liquid_ticker_gets_high_liquidity_score(self):
        td = make_ticker_data("LIQUID", adv=1_000_000_000)  # $1B ADV
        factors = FactorEngine().compute(td)
        scored = Scorer().score(factors)
        assert scored.liquidity_score > 60


# ─────────────────────────────────────────────────────────────
# Stable ranking behavior
# ─────────────────────────────────────────────────────────────

class TestStableRanking:

    def test_reproducible_scores(self):
        """Same synthetic data → same score every time."""
        td = make_ticker_data("REP", seed=42)
        f = FactorEngine().compute(td)
        s1 = Scorer().score(f)
        s2 = Scorer().score(f)
        assert s1.total_score == pytest.approx(s2.total_score)

    def test_ranking_order_stable(self):
        """Adding noise doesn't flip the order of clearly separated tickers."""
        td_good = make_ticker_data("GOOD", drift=0.002, sigma=0.005, seed=1)
        td_bad  = make_ticker_data("BAD",  drift=-0.002, sigma=0.03, seed=2)
        f_good = FactorEngine().compute(td_good)
        f_bad  = FactorEngine().compute(td_bad)
        s_good = Scorer().score(f_good)
        s_bad  = Scorer().score(f_bad)
        assert s_good.total_score > s_bad.total_score
