"""
src/screener/scoring.py
------------------------
Composite Scoring Model

Converts raw factor values into a transparent 0–100 composite score.
Each component is scored independently, then weighted.

Score components (weights)
---------------------------
  Trend score     : 30%  — are prices moving up consistently?
  Regime score    : 25%  — is volatility low and stable?
  Risk score      : 25%  — is drawdown and tail risk controlled?
  Liquidity score : 10%  — is the name liquid and tradeable?
  Quality score   : 10%  — is the trend clean (not a chaotic spike)?

Design philosophy
-----------------
The scorer is deliberately transparent: every scoring function is a
monotone transformation of a single factor.  There are no interaction
terms or non-linear combinations that would obscure the reasoning.

This matches how quantitative screening desks actually build factor models:
each factor is independently validated before being combined.  Hidden
interactions are a common source of overfitting in screening tools.

Regime penalty
--------------
EXTREME volatility regime is treated as a hard penalty.  Any ticker in
the EXTREME regime receives a 30-point deduction on the regime component
score, regardless of trend or momentum quality.  This reflects the
market-making intuition from the engine: EXTREME is when you step back
from the book.  The same logic applies to equity screening.

Disclaimer integration
-----------------------
Every scored ticker includes a 'reason' string explaining the ranking.
This is not investment advice. Scores are based on historical patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .factors import TickerFactors
from ..models.regime import VolatilityRegime


# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "trend":     0.30,
    "regime":    0.25,
    "risk":      0.25,
    "liquidity": 0.10,
    "quality":   0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Scored ticker output
# ---------------------------------------------------------------------------

@dataclass
class ScoredTicker:
    """Full scoring output for a single ticker."""
    ticker:           str
    total_score:      float          # 0–100
    trend_score:      float
    regime_score:     float
    risk_score:       float
    liquidity_score:  float
    quality_score:    float
    regime:           Optional[VolatilityRegime]
    vol_20d:          Optional[float]
    vol_pct:          Optional[float]
    ret_20d:          Optional[float]
    ret_60d:          Optional[float]
    max_drawdown:     Optional[float]
    var_95:           Optional[float]
    es_95:            Optional[float]
    avg_dollar_vol:   Optional[float]
    reason:           str = ""
    is_avoid:         bool = False   # True if regime=EXTREME or score < threshold

    @property
    def regime_label(self) -> str:
        return self.regime.value.upper() if self.regime else "UNKNOWN"

    def summary_row(self) -> dict:
        """Return a dict suitable for a DataFrame row."""
        return {
            "Ticker":        self.ticker,
            "Score":         round(self.total_score, 1),
            "Trend":         round(self.trend_score, 1),
            "Regime":        round(self.regime_score, 1),
            "Risk":          round(self.risk_score, 1),
            "Liquidity":     round(self.liquidity_score, 1),
            "Quality":       round(self.quality_score, 1),
            "RegimeLabel":   self.regime_label,
            "Vol20d%":       f"{self.vol_20d:.1%}" if self.vol_20d else "N/A",
            "VolPct":        f"{self.vol_pct:.0f}" if self.vol_pct else "N/A",
            "Ret20d%":       f"{self.ret_20d:+.1%}" if self.ret_20d else "N/A",
            "Ret60d%":       f"{self.ret_60d:+.1%}" if self.ret_60d else "N/A",
            "MaxDD%":        f"{self.max_drawdown:.1%}" if self.max_drawdown else "N/A",
            "VaR95%":        f"{self.var_95:.2%}" if self.var_95 else "N/A",
            "ES95%":         f"{self.es_95:.2%}" if self.es_95 else "N/A",
            "AvgDolVol$M":   f"{self.avg_dollar_vol/1e6:.0f}" if self.avg_dollar_vol else "N/A",
            "Avoid":         "YES" if self.is_avoid else "",
            "Reason":        self.reason,
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Scorer:
    """
    Converts a TickerFactors object into a ScoredTicker.

    All individual scoring functions map a factor value to [0, 100].
    The composite score is a weighted average.
    """

    AVOID_SCORE_THRESHOLD = 35.0   # scores below this → avoid list
    EXTREME_REGIME_PENALTY = 30.0  # subtracted from regime component

    def score(self, factors: TickerFactors) -> ScoredTicker:
        """Score a single ticker."""
        if not factors.is_valid:
            return ScoredTicker(
                ticker=factors.ticker,
                total_score=0.0,
                trend_score=0.0, regime_score=0.0,
                risk_score=0.0, liquidity_score=0.0, quality_score=0.0,
                regime=None, vol_20d=None, vol_pct=None,
                ret_20d=None, ret_60d=None, max_drawdown=None,
                var_95=None, es_95=None, avg_dollar_vol=None,
                reason=f"Invalid: {factors.error}",
                is_avoid=True,
            )

        # ── Component scores ──────────────────────────────────────────────
        trend_score     = self._trend_score(factors)
        regime_score    = self._regime_score(factors)
        risk_score      = self._risk_score(factors)
        liquidity_score = self._liquidity_score(factors)
        quality_score   = self._quality_score(factors)

        total = (
            WEIGHTS["trend"]     * trend_score
          + WEIGHTS["regime"]    * regime_score
          + WEIGHTS["risk"]      * risk_score
          + WEIGHTS["liquidity"] * liquidity_score
          + WEIGHTS["quality"]   * quality_score
        )
        total = max(0.0, min(100.0, total))

        is_avoid = (
            total < self.AVOID_SCORE_THRESHOLD
            or factors.regime == VolatilityRegime.EXTREME
        )

        reason = self._build_reason(
            factors, total, trend_score, regime_score, risk_score,
            liquidity_score, quality_score,
        )

        return ScoredTicker(
            ticker=factors.ticker,
            total_score=round(total, 2),
            trend_score=round(trend_score, 2),
            regime_score=round(regime_score, 2),
            risk_score=round(risk_score, 2),
            liquidity_score=round(liquidity_score, 2),
            quality_score=round(quality_score, 2),
            regime=factors.regime,
            vol_20d=factors.vol_20d,
            vol_pct=factors.vol_pct,
            ret_20d=factors.ret_20d,
            ret_60d=factors.ret_60d,
            max_drawdown=factors.max_drawdown,
            var_95=factors.var_95,
            es_95=factors.es_95,
            avg_dollar_vol=factors.avg_dollar_vol,
            reason=reason,
            is_avoid=is_avoid,
        )

    def score_all(self, factors_dict: Dict[str, TickerFactors]) -> List[ScoredTicker]:
        """Score all tickers and return list sorted by total_score descending."""
        scored = [self.score(f) for f in factors_dict.values()]
        return sorted(scored, key=lambda s: s.total_score, reverse=True)

    # ------------------------------------------------------------------
    # Component scoring functions
    # ------------------------------------------------------------------

    def _trend_score(self, f: TickerFactors) -> float:
        """
        Score based on 20d and 60d returns and position vs moving averages.
        Rewards consistent uptrend; penalizes negative or mixed signals.
        """
        score = 50.0   # start at neutral

        # 20d return contribution (±20 points)
        if f.ret_20d is not None:
            # Sigmoid-like: full credit at +10%, full penalty at -10%
            score += _clamp(f.ret_20d / 0.10 * 20, -20, 20)

        # 60d return contribution (±15 points)
        if f.ret_60d is not None:
            score += _clamp(f.ret_60d / 0.15 * 15, -15, 15)

        # Position vs MA20 (±10 points)
        if f.price_vs_ma20 is not None:
            score += _clamp(f.price_vs_ma20 / 0.05 * 10, -10, 10)

        # Position vs MA50 (±5 points)
        if f.price_vs_ma50 is not None:
            score += _clamp(f.price_vs_ma50 / 0.05 * 5, -5, 5)

        return _clamp(score, 0, 100)

    def _regime_score(self, f: TickerFactors) -> float:
        """
        Score based on volatility regime and percentile.
        LOW regime = good. EXTREME = very bad.
        """
        if f.regime is None:
            return 30.0

        base = {
            VolatilityRegime.LOW:     90.0,
            VolatilityRegime.MEDIUM:  65.0,
            VolatilityRegime.HIGH:    35.0,
            VolatilityRegime.EXTREME: 10.0,
        }[f.regime]

        # Adjust for vol percentile: high percentile = elevated within regime
        if f.vol_pct is not None:
            # Penalise up to 15 points for being at the high end of vol distribution
            adjustment = -((f.vol_pct / 100.0) * 15)
            base = max(0, base + adjustment)

        return _clamp(base, 0, 100)

    def _risk_score(self, f: TickerFactors) -> float:
        """
        Score based on drawdown, VaR, and ES.
        Lower drawdown and tail risk = higher score.
        """
        score = 80.0

        # Penalise drawdown (up to -40 points for DD > 40%)
        if f.max_drawdown is not None:
            dd_penalty = min(40, f.max_drawdown / 0.40 * 40)
            score -= dd_penalty

        # Penalise high VaR (up to -20 points for VaR > 4%)
        if f.var_95 is not None:
            var_penalty = min(20, f.var_95 / 0.04 * 20)
            score -= var_penalty

        # Penalise high ES relative to VaR (up to -10 points)
        if f.es_95 is not None and f.var_95 is not None and f.var_95 > 0:
            es_ratio = f.es_95 / f.var_95
            tail_penalty = min(10, (es_ratio - 1.0) * 20)
            score -= max(0, tail_penalty)

        return _clamp(score, 0, 100)

    def _liquidity_score(self, f: TickerFactors) -> float:
        """
        Score based on average dollar volume and volume stability.
        Very liquid names score near 100.
        """
        score = 0.0

        # ADV component (0–70 points)
        if f.avg_dollar_vol is not None:
            adv_m = f.avg_dollar_vol / 1e6   # in millions
            if adv_m >= 500:
                score += 70
            elif adv_m >= 100:
                score += 50 + (adv_m - 100) / 400 * 20
            elif adv_m >= 10:
                score += 20 + (adv_m - 10) / 90 * 30
            else:
                score += max(0, adv_m / 10 * 20)

        # Volume stability component (0–30 points)
        if f.vol_stability is not None:
            score += f.vol_stability * 30

        return _clamp(score, 0, 100)

    def _quality_score(self, f: TickerFactors) -> float:
        """
        Reward positive trend with controlled volatility.
        Penalise high-volatility names that happen to have good returns
        (these are more likely to be chaotic spikes than genuine trends).
        """
        if f.ret_20d is None or f.vol_20d is None:
            return 50.0

        # If trend is negative: quality is poor regardless
        if f.ret_20d <= 0:
            return max(0, 50 + f.ret_20d / 0.05 * 50)

        # Sharpe-like ratio: return / volatility (rough, over 20d)
        # Annualise return (×252/20) then divide by annualised vol
        ann_ret_20d = f.ret_20d * (252 / 20)
        sharpe_proxy = ann_ret_20d / max(f.vol_20d, 0.01)

        # Score from 0–100: full credit at sharpe_proxy = 2.0
        score = min(100, sharpe_proxy / 2.0 * 100)

        # Downside vol penalty: if downside vol is high relative to overall vol
        if f.downside_vol is not None and f.vol_20d is not None and f.vol_20d > 0:
            downside_ratio = f.downside_vol / f.vol_20d
            # High downside_ratio (skewed to the downside) → penalty
            if downside_ratio > 0.8:
                score *= 0.85

        return _clamp(score, 0, 100)

    # ------------------------------------------------------------------
    # Reason string
    # ------------------------------------------------------------------

    def _build_reason(
        self,
        f: TickerFactors,
        total: float,
        trend: float,
        regime: float,
        risk: float,
        liq: float,
        quality: float,
    ) -> str:
        parts = []

        if f.regime == VolatilityRegime.EXTREME:
            parts.append("EXTREME volatility regime — avoid")
        elif f.regime == VolatilityRegime.HIGH:
            parts.append("HIGH volatility regime")
        elif f.regime == VolatilityRegime.LOW:
            parts.append("LOW volatility regime (favorable)")

        if f.ret_20d is not None:
            parts.append(f"20d return {f.ret_20d:+.1%}")

        if f.ret_60d is not None:
            parts.append(f"60d return {f.ret_60d:+.1%}")

        if f.max_drawdown is not None and f.max_drawdown > 0.25:
            parts.append(f"drawdown {f.max_drawdown:.0%} (elevated)")

        if f.var_95 is not None and f.var_95 > 0.03:
            parts.append(f"VaR95 {f.var_95:.2%} (high tail risk)")

        if liq < 40:
            parts.append("low liquidity")

        if total >= 65:
            parts.append("→ in scope for monitoring")
        elif total < self.AVOID_SCORE_THRESHOLD:
            parts.append("→ avoid")

        return "; ".join(parts) if parts else "no notable signals"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
