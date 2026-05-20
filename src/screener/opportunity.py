"""
src/screener/opportunity.py
-----------------------------
Opportunity Factor Engine and Scorer

Extends the base screener with factors designed to surface smaller/mid-cap
names with improving momentum structure — without abandoning the project's
core risk discipline.

Philosophy
----------
The base screener (scoring.py) rewards stability. That naturally favours
mega-caps and ETFs, which have low volatility, deep liquidity, and modest
drawdowns. This is correct — for that universe.

The opportunity screener asks a different question:
    "Within the acceptable risk envelope, which names have the strongest
     improving trend structure and the best risk-adjusted acceleration?"

Key differences from the base scorer
--------------------------------------
1. Trend acceleration replaces raw trend magnitude.
   A name that has moved from -5% to +8% (20d) is more interesting than
   one that has been at +12% for three months. The direction of change matters.

2. Market-cap opportunity bonus.
   $2B–$30B names get a scoring boost. This is not a size bet — it is
   recognition that price discovery is less efficient in smaller names,
   and genuine trend structures in those names are less crowded.

3. Volume expansion proxy.
   Current volume vs rolling average signals institutional accumulation.
   A name that has been quietly trending higher on declining volume is
   less interesting than one where volume is expanding into the move.

4. Universe-relative scoring.
   Each universe is ranked within itself. Emerging Leaders are compared
   against Emerging Leaders, not against SPY.

5. Volatility quality (not just volatility level).
   Moderate-to-high vol is acceptable in the Emerging universe IF:
   - trend is intact (no broken structure)
   - drawdown is survivable (not a parabolic blowoff)
   - vol_pct is not extreme (current vol is not in the historical tail)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .factors import TickerFactors, FactorEngine, _log_returns
from .scoring import ScoredTicker, Scorer, _clamp
from .universe import UniverseSpec, CapBucket, VolTolerance
from ..models.regime import VolatilityRegime


# ---------------------------------------------------------------------------
# Additional opportunity factors
# ---------------------------------------------------------------------------

@dataclass
class OpportunityFactors:
    """
    Extended factors computed on top of base TickerFactors.
    """
    ticker: str

    # Trend acceleration: how fast is momentum improving?
    momentum_slope:     Optional[float] = None   # ret_20d - ret_60d (improving if positive)
    accel_score:        Optional[float] = None   # normalized [−1, +1]

    # Relative volume (current 5d avg volume / 60d avg volume)
    rel_volume:         Optional[float] = None   # >1 = expanding, <1 = contracting

    # Breakout proximity: how close to 52-week high?
    pct_from_52w_high:  Optional[float] = None   # 0 = at high, −0.2 = 20% below

    # Market cap bucket
    cap_bucket:         CapBucket = CapBucket.UNKNOWN

    # Opportunity bonus (0–30 points, added to total)
    cap_bonus:          float = 0.0

    # Vol quality assessment
    vol_is_constructive: bool = True    # False if chaotic spike or parabolic

    # Final opportunity score (0–100)
    opportunity_score:  float = 0.0

    # Human-readable insight
    insight:            str = ""

    # Whether to flag as highest-risk in its universe
    high_risk_flag:     bool = False


# ---------------------------------------------------------------------------
# Opportunity factor computation
# ---------------------------------------------------------------------------

def compute_opportunity_factors(
    base: TickerFactors,
    ticker_data,
    cap_bucket: CapBucket = CapBucket.UNKNOWN,
) -> OpportunityFactors:
    """
    Compute opportunity factors from base TickerFactors + raw price data.

    Parameters
    ----------
    base       : TickerFactors already computed by FactorEngine
    ticker_data: TickerData from MarketDataFetcher
    cap_bucket : CapBucket assigned by the universe definition

    Returns
    -------
    OpportunityFactors
    """
    of = OpportunityFactors(ticker=base.ticker, cap_bucket=cap_bucket)

    try:
        price = ticker_data.adj_close
        volume = ticker_data.volume

        # ── Momentum slope (trend acceleration) ────────────────────────────
        if base.ret_20d is not None and base.ret_60d is not None:
            # ret_20d is over 20 days, ret_60d over 60.
            # To compare on same timeframe: annualise both
            ann_20d = base.ret_20d * (252 / 20)
            ann_60d = base.ret_60d * (252 / 60)
            of.momentum_slope = ann_20d - ann_60d
            # Normalise: slope of ±1.0 (100% annualised diff) → ±1 accel score
            of.accel_score = _clamp(of.momentum_slope, -1.0, 1.0)

        # ── Relative volume ─────────────────────────────────────────────────
        if len(volume) >= 65:
            vol_5d  = volume.iloc[-5:].mean()
            vol_60d = volume.iloc[-65:-5].mean()
            if vol_60d > 0:
                of.rel_volume = float(vol_5d / vol_60d)

        # ── 52-week proximity ───────────────────────────────────────────────
        lookback = min(len(price), 252)
        high_52w = price.iloc[-lookback:].max()
        if high_52w > 0:
            of.pct_from_52w_high = float(price.iloc[-1] / high_52w - 1.0)

        # ── Cap opportunity bonus ────────────────────────────────────────────
        of.cap_bonus = _cap_bonus(cap_bucket)

        # ── Volatility quality ───────────────────────────────────────────────
        of.vol_is_constructive = _vol_is_constructive(base)

        # ── High-risk flag ───────────────────────────────────────────────────
        of.high_risk_flag = (
            base.regime == VolatilityRegime.EXTREME
            or (base.max_drawdown is not None and base.max_drawdown > 0.60)
        )

    except Exception as e:
        of.insight = f"Factor error: {str(e)[:100]}"

    return of


def _cap_bonus(bucket: CapBucket) -> float:
    """
    Opportunity bonus by market cap bucket.
    SMALL and MID caps in improving trends are harder to find →
    they get a score bonus when their other factors are strong.
    """
    return {
        CapBucket.MEGA:    0.0,   # already fully discovered
        CapBucket.LARGE:   5.0,   # slight discount for size
        CapBucket.MID:    15.0,   # sweet spot for opportunity
        CapBucket.SMALL:  20.0,   # highest reward for finding quality here
        CapBucket.MICRO:  -5.0,   # too small → liquidity penalty
        CapBucket.ETF:     0.0,   # no cap bonus for ETFs
        CapBucket.UNKNOWN: 5.0,   # moderate bonus without classification
    }.get(bucket, 0.0)


def _vol_is_constructive(f: TickerFactors) -> bool:
    """
    True if volatility is in a constructive range — elevated but not chaotic.
    Parabolic blowoffs (high vol + high drawdown simultaneously) are NOT constructive.
    """
    if f.regime == VolatilityRegime.EXTREME:
        return False
    if f.max_drawdown is not None and f.max_drawdown > 0.50:
        return False
    # High vol percentile combined with negative recent returns = distribution, not accumulation
    if (f.vol_pct is not None and f.vol_pct > 90
            and f.ret_20d is not None and f.ret_20d < -0.10):
        return False
    return True


# ---------------------------------------------------------------------------
# Opportunity scorer
# ---------------------------------------------------------------------------

OPPORTUNITY_WEIGHTS = {
    "trend_quality":       0.25,
    "trend_acceleration":  0.20,
    "regime_quality":      0.20,
    "risk_control":        0.15,
    "liquidity":           0.10,
    "opportunity_bonus":   0.10,
}

assert abs(sum(OPPORTUNITY_WEIGHTS.values()) - 1.0) < 1e-9


class OpportunityScorer:
    """
    Scores tickers within a specific universe for opportunity potential.

    The scorer is universe-aware: it applies different vol tolerance
    rules based on the universe's VolTolerance setting.

    Parameters
    ----------
    universe : UniverseSpec
        The universe being scored (used for vol tolerance adjustments).
    """

    def __init__(self, universe: UniverseSpec) -> None:
        self.universe = universe

    def score(
        self,
        base: TickerFactors,
        opp: OpportunityFactors,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the opportunity score.

        Returns
        -------
        (total_score, component_scores_dict)
        """
        components = {}

        # ── Trend quality (0–100) ────────────────────────────────────────────
        tq = 50.0
        if base.ret_20d is not None:
            tq += _clamp(base.ret_20d / 0.10 * 20, -20, 25)
        if base.ret_60d is not None:
            tq += _clamp(base.ret_60d / 0.15 * 15, -15, 20)
        if base.price_vs_ma20 is not None:
            tq += _clamp(base.price_vs_ma20 / 0.05 * 10, -10, 10)
        components["trend_quality"] = _clamp(tq, 0, 100)

        # ── Trend acceleration (0–100) ────────────────────────────────────────
        ta = 50.0
        if opp.accel_score is not None:
            ta += opp.accel_score * 40   # full reward for improving momentum
        if opp.pct_from_52w_high is not None:
            # Reward being near highs, but not at extreme blowoff
            proximity = 1.0 + opp.pct_from_52w_high   # 1.0 = at high
            if proximity > 0.80:
                ta += 10  # near 52w high = constructive
            elif proximity < 0.60:
                ta -= 15  # far from high = structural damage
        if opp.rel_volume is not None:
            # Volume expansion into move = institutional participation
            if opp.rel_volume > 1.5:
                ta += 10
            elif opp.rel_volume > 1.0:
                ta += 5
            elif opp.rel_volume < 0.6:
                ta -= 10
        components["trend_acceleration"] = _clamp(ta, 0, 100)

        # ── Regime quality (0–100) ─────────────────────────────────────────────
        rq = {
            VolatilityRegime.LOW:     85.0,
            VolatilityRegime.MEDIUM:  70.0,
            VolatilityRegime.HIGH:    40.0,
            VolatilityRegime.EXTREME: 10.0,
        }.get(base.regime, 50.0)

        # Universe vol tolerance adjustments
        tol = self.universe.vol_tolerance
        if base.regime == VolatilityRegime.HIGH:
            if tol == VolTolerance.HIGH:
                rq = 55.0    # less penalty for high-beta universe
            elif tol == VolTolerance.MEDIUM:
                rq = 45.0
        if base.regime == VolatilityRegime.MEDIUM:
            if tol == VolTolerance.LOW:
                rq = 60.0    # more penalty for stability universe

        # Vol quality bonus/penalty
        if not opp.vol_is_constructive:
            rq = max(0, rq - 20)
        elif base.regime in (VolatilityRegime.HIGH, VolatilityRegime.MEDIUM) and opp.vol_is_constructive:
            rq = min(100, rq + 10)  # constructive vol bonus

        if base.vol_pct is not None:
            rq -= (base.vol_pct / 100) * 15

        components["regime_quality"] = _clamp(rq, 0, 100)

        # ── Risk control (0–100) ──────────────────────────────────────────────
        rc = 80.0
        if base.max_drawdown is not None:
            dd_threshold = {
                VolTolerance.LOW:    0.20,
                VolTolerance.MEDIUM: 0.35,
                VolTolerance.HIGH:   0.55,
            }[tol]
            dd_penalty = min(50, (base.max_drawdown / dd_threshold) * 40)
            rc -= dd_penalty
        if base.var_95 is not None:
            var_threshold = {
                VolTolerance.LOW:    0.025,
                VolTolerance.MEDIUM: 0.04,
                VolTolerance.HIGH:   0.06,
            }[tol]
            var_penalty = min(20, (base.var_95 / var_threshold) * 20)
            rc -= var_penalty
        if base.es_95 is not None and base.var_95 is not None and base.var_95 > 0:
            es_ratio = base.es_95 / base.var_95
            if es_ratio > 1.5:
                rc -= min(10, (es_ratio - 1.5) * 15)
        components["risk_control"] = _clamp(rc, 0, 100)

        # ── Liquidity (0–100) ─────────────────────────────────────────────────
        lq = 0.0
        if base.avg_dollar_vol is not None:
            adv = base.avg_dollar_vol / 1e6
            if adv >= 200:
                lq = 75
            elif adv >= 50:
                lq = 55 + (adv - 50) / 150 * 20
            elif adv >= 10:
                lq = 30 + (adv - 10) / 40 * 25
            elif adv >= 2:
                lq = 10 + (adv - 2) / 8 * 20
            else:
                lq = max(0, adv / 2 * 10)
        if base.vol_stability is not None:
            lq += base.vol_stability * 25
        components["liquidity"] = _clamp(lq, 0, 100)

        # ── Opportunity bonus (0–100, from cap bucket) ────────────────────────
        ob = _clamp(50 + opp.cap_bonus * 2, 0, 100)
        components["opportunity_bonus"] = ob

        # ── Weighted total ────────────────────────────────────────────────────
        total = sum(OPPORTUNITY_WEIGHTS[k] * v for k, v in components.items())
        total = _clamp(total, 0, 100)

        return round(total, 2), {k: round(v, 2) for k, v in components.items()}


# ---------------------------------------------------------------------------
# Opportunity result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OpportunityResult:
    """Full opportunity analysis result for a single ticker."""
    ticker:              str
    universe:            str
    opportunity_score:   float
    base_score:          float
    trend_quality:       float
    trend_acceleration:  float
    regime_quality:      float
    risk_control:        float
    liquidity_score:     float
    opportunity_bonus:   float

    # Regime and risk
    regime:              Optional[VolatilityRegime]
    vol_20d:             Optional[float]
    drawdown:            Optional[float]
    var_95:              Optional[float]
    es_95:               Optional[float]
    ret_20d:             Optional[float]
    ret_60d:             Optional[float]

    # Opportunity-specific
    momentum_slope:      Optional[float]
    rel_volume:          Optional[float]
    pct_from_52w_high:   Optional[float]
    cap_bucket:          CapBucket
    vol_is_constructive: bool
    high_risk:           bool

    # Narrative
    insight:             str
    is_avoid:            bool

    @property
    def regime_label(self) -> str:
        return self.regime.value.upper() if self.regime else "UNKNOWN"

    @property
    def cap_label(self) -> str:
        return self.cap_bucket.value.upper()

    def summary_row(self) -> dict:
        return {
            "Ticker":       self.ticker,
            "Universe":     self.universe,
            "OppScore":     round(self.opportunity_score, 1),
            "BaseScore":    round(self.base_score, 1),
            "TrendQ":       round(self.trend_quality, 1),
            "TrendAccel":   round(self.trend_acceleration, 1),
            "RegimeQ":      round(self.regime_quality, 1),
            "RiskCtrl":     round(self.risk_control, 1),
            "Liquidity":    round(self.liquidity_score, 1),
            "OppBonus":     round(self.opportunity_bonus, 1),
            "Regime":       self.regime_label,
            "CapBucket":    self.cap_label,
            "Vol20d%":      f"{self.vol_20d:.1%}" if self.vol_20d else "N/A",
            "Ret20d%":      f"{self.ret_20d:+.1%}" if self.ret_20d else "N/A",
            "Ret60d%":      f"{self.ret_60d:+.1%}" if self.ret_60d else "N/A",
            "MaxDD%":       f"{self.drawdown:.1%}" if self.drawdown else "N/A",
            "VaR95%":       f"{self.var_95:.2%}" if self.var_95 else "N/A",
            "MomSlope":     f"{self.momentum_slope:+.2f}" if self.momentum_slope else "N/A",
            "RelVol":       f"{self.rel_volume:.2f}x" if self.rel_volume else "N/A",
            "PctFrom52wH":  f"{self.pct_from_52w_high:+.1%}" if self.pct_from_52w_high else "N/A",
            "VolConstructive": "YES" if self.vol_is_constructive else "NO",
            "HighRisk":     "YES" if self.high_risk else "",
            "Avoid":        "YES" if self.is_avoid else "",
            "Insight":      self.insight,
        }


# ---------------------------------------------------------------------------
# Full opportunity pipeline
# ---------------------------------------------------------------------------

class OpportunityPipeline:
    """
    Runs the full opportunity screening pipeline for a single universe.

    Parameters
    ----------
    universe_spec : UniverseSpec
    """

    AVOID_SCORE     = 30.0
    HIGH_RISK_SCORE = 45.0

    def __init__(self, universe_spec: UniverseSpec) -> None:
        self.universe = universe_spec
        self._factor_engine = FactorEngine()
        self._base_scorer   = Scorer()
        self._opp_scorer    = OpportunityScorer(universe_spec)

    def run(self, fetch_data: dict) -> List[OpportunityResult]:
        """
        Run full pipeline on pre-fetched data.

        Parameters
        ----------
        fetch_data : dict[ticker, TickerData]  from MarketDataFetcher

        Returns
        -------
        List[OpportunityResult] sorted by opportunity_score descending
        """
        results = []

        for ticker, td in fetch_data.items():
            # Base factors
            base_factors = self._factor_engine.compute(td)
            if not base_factors.is_valid:
                continue

            # Base score
            base_scored = self._base_scorer.score(base_factors)

            # Opportunity factors
            cap = self.universe.ticker_caps.get(ticker, CapBucket.UNKNOWN)
            opp_factors = compute_opportunity_factors(base_factors, td, cap_bucket=cap)

            # Opportunity score
            opp_score, components = self._opp_scorer.score(base_factors, opp_factors)

            # Insight generation
            insight = _generate_insight(base_factors, opp_factors, self.universe)

            is_avoid = (
                opp_score < self.AVOID_SCORE
                or base_factors.regime == VolatilityRegime.EXTREME
                or not base_factors.is_valid
            )

            results.append(OpportunityResult(
                ticker=ticker,
                universe=self.universe.name,
                opportunity_score=opp_score,
                base_score=base_scored.total_score,
                trend_quality=components["trend_quality"],
                trend_acceleration=components["trend_acceleration"],
                regime_quality=components["regime_quality"],
                risk_control=components["risk_control"],
                liquidity_score=components["liquidity"],
                opportunity_bonus=components["opportunity_bonus"],
                regime=base_factors.regime,
                vol_20d=base_factors.vol_20d,
                drawdown=base_factors.max_drawdown,
                var_95=base_factors.var_95,
                es_95=base_factors.es_95,
                ret_20d=base_factors.ret_20d,
                ret_60d=base_factors.ret_60d,
                momentum_slope=opp_factors.momentum_slope,
                rel_volume=opp_factors.rel_volume,
                pct_from_52w_high=opp_factors.pct_from_52w_high,
                cap_bucket=cap,
                vol_is_constructive=opp_factors.vol_is_constructive,
                high_risk=opp_factors.high_risk_flag,
                insight=insight,
                is_avoid=is_avoid,
            ))

        return sorted(results, key=lambda r: r.opportunity_score, reverse=True)


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------

def _generate_insight(
    f: TickerFactors,
    of: OpportunityFactors,
    universe: UniverseSpec,
) -> str:
    """
    Generate a plain-English explanation of why a ticker ranked where it did.
    Goal: specific, not generic. "Why this name, in this regime, right now."
    """
    parts = []

    # Regime context
    if f.regime == VolatilityRegime.LOW:
        parts.append("stable low-vol regime")
    elif f.regime == VolatilityRegime.MEDIUM:
        parts.append("medium-vol regime (normal)")
    elif f.regime == VolatilityRegime.HIGH:
        if of.vol_is_constructive:
            parts.append("elevated but constructive volatility")
        else:
            parts.append("elevated and disruptive volatility")
    elif f.regime == VolatilityRegime.EXTREME:
        parts.append("EXTREME volatility — structurally dangerous")

    # Trend description
    if f.ret_20d is not None and f.ret_60d is not None:
        if f.ret_20d > 0.05 and f.ret_60d > 0.10:
            parts.append("strong sustained uptrend")
        elif f.ret_20d > 0.05 and f.ret_60d <= 0:
            parts.append("recent momentum recovering from longer-term weakness")
        elif f.ret_20d > 0 and f.ret_60d > 0:
            parts.append("modest positive trend")
        elif f.ret_20d <= 0 and f.ret_60d > 0:
            parts.append("near-term pullback within longer uptrend")
        elif f.ret_20d <= 0 and f.ret_60d <= 0:
            parts.append("downtrend on both timeframes")

    # Trend acceleration
    if of.accel_score is not None:
        if of.accel_score > 0.3:
            parts.append("accelerating momentum (20d outpacing 60d)")
        elif of.accel_score < -0.3:
            parts.append("decelerating momentum (20d lagging 60d)")

    # Volume signal
    if of.rel_volume is not None:
        if of.rel_volume > 1.5:
            parts.append(f"volume expanding {of.rel_volume:.1f}x above average (accumulation signal)")
        elif of.rel_volume < 0.6:
            parts.append("volume contracting (low conviction)")

    # 52-week proximity
    if of.pct_from_52w_high is not None:
        if of.pct_from_52w_high > -0.05:
            parts.append("near 52-week high")
        elif of.pct_from_52w_high < -0.40:
            parts.append(f"{of.pct_from_52w_high:.0%} below 52-week high")

    # Drawdown note
    if f.max_drawdown is not None:
        if f.max_drawdown > 0.50:
            parts.append(f"severe drawdown {f.max_drawdown:.0%} — structural damage")
        elif f.max_drawdown > 0.30:
            parts.append(f"significant drawdown {f.max_drawdown:.0%}")
        elif f.max_drawdown < 0.15:
            parts.append("drawdown well-controlled")

    # Cap bucket context
    if of.cap_bucket == CapBucket.SMALL:
        parts.append("small-cap (higher discovery potential, higher risk)")
    elif of.cap_bucket == CapBucket.MID:
        parts.append("mid-cap opportunity range")
    elif of.cap_bucket == CapBucket.MEGA:
        parts.append("mega-cap (mature, limited upside surprise)")

    return "; ".join(parts) if parts else "insufficient data for narrative"


# ---------------------------------------------------------------------------
# Multi-universe report utilities
# ---------------------------------------------------------------------------

def save_opportunity_csv(
    results_by_universe: Dict[str, List[OpportunityResult]],
    filepath: str,
) -> str:
    """Save all universe results to a single CSV."""
    import os
    import pandas as pd
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    all_rows = []
    for universe_name, results in results_by_universe.items():
        for r in results:
            all_rows.append(r.summary_row())
    df = pd.DataFrame(all_rows)
    df.to_csv(filepath, index=False)
    return filepath


def save_opportunity_markdown(
    results_by_universe: Dict[str, List[OpportunityResult]],
    filepath: str,
) -> str:
    """Save full opportunity report as Markdown."""
    import os
    from datetime import datetime
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    date = datetime.today().strftime("%Y-%m-%d")
    lines = [
        "# Opportunity Screener Report",
        "",
        f"**Generated:** {date}",
        "",
        "> ⚠ NOT FINANCIAL ADVICE. This report identifies names worth further investigation "
        "based on historical patterns. It does not predict future performance. "
        "All trading involves risk of loss.",
        "",
        "---",
        "",
    ]

    for universe_name, results in results_by_universe.items():
        from .universe import ALL_UNIVERSES
        spec = ALL_UNIVERSES.get(universe_name)
        label = spec.label if spec else universe_name

        eligible = [r for r in results if not r.is_avoid]
        avoid    = [r for r in results if r.is_avoid]

        lines += [
            f"## {label}",
            "",
            f"*{spec.description if spec else ''}*",
            "",
            f"**Eligible:** {len(eligible)}  |  **Avoid:** {len(avoid)}",
            "",
        ]

        # Top candidates
        lines += ["### Top Candidates", "", _opp_md_table(eligible[:8]), ""]

        # Insights for top 5
        if eligible:
            lines += ["### Insights", ""]
            for r in eligible[:5]:
                lines.append(f"**{r.ticker}** (score={r.opportunity_score:.0f}, {r.regime_label}):")
                lines.append(f"> {r.insight}")
                lines.append("")

        # Avoid
        if avoid:
            lines += ["### Avoid", ""]
            for r in avoid:
                lines.append(f"- **{r.ticker}** ({r.regime_label}, score={r.opportunity_score:.0f}): {r.insight}")
            lines.append("")

        lines += ["---", ""]

    # Cross-universe analysis
    all_results = [r for results in results_by_universe.values() for r in results]
    high_risk = [r for r in all_results if r.high_risk and not r.is_avoid]
    if high_risk:
        lines += [
            "## Most Dangerous Names (High Risk, Not Avoided)",
            "",
            "These names scored above the avoid threshold but carry elevated structural risk:",
            "",
        ]
        for r in sorted(high_risk, key=lambda x: x.opportunity_score, reverse=True)[:5]:
            lines.append(f"- **{r.ticker}** ({r.universe}, score={r.opportunity_score:.0f}): {r.insight}")
        lines += ["", "---", ""]

    lines += [
        "## Score Interpretation",
        "",
        "| Score | Meaning |",
        "|---|---|",
        "| 65–100 | Strong candidate — watch closely |",
        "| 50–65  | Developing structure — monitor |",
        "| 30–50  | Weak — pass |",
        "| 0–30   | Avoid |",
        "",
        "## Factor Weights",
        "",
        "| Factor | Weight |",
        "|---|---|",
        *[f"| {k.replace('_', ' ').title()} | {v:.0%} |"
          for k, v in OPPORTUNITY_WEIGHTS.items()],
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def _opp_md_table(results: List[OpportunityResult]) -> str:
    if not results:
        return "_No eligible tickers._"
    header = "| # | Ticker | OppScore | Regime | Cap | Vol20d | Ret20d | Ret60d | MomSlope | RelVol |"
    sep    = "|---|---|---|---|---|---|---|---|---|---|"
    rows   = [header, sep]
    for i, r in enumerate(results, 1):
        vol = f"{r.vol_20d:.0%}" if r.vol_20d else "—"
        r20 = f"{r.ret_20d:+.1%}" if r.ret_20d else "—"
        r60 = f"{r.ret_60d:+.1%}" if r.ret_60d else "—"
        ms  = f"{r.momentum_slope:+.2f}" if r.momentum_slope else "—"
        rv  = f"{r.rel_volume:.1f}x" if r.rel_volume else "—"
        rows.append(
            f"| {i} | **{r.ticker}** | {r.opportunity_score:.1f} | {r.regime_label} "
            f"| {r.cap_label} | {vol} | {r20} | {r60} | {ms} | {rv} |"
        )
    return "\n".join(rows)


def print_opportunity_table(
    results: List[OpportunityResult],
    universe_label: str,
    top_n: int = 10,
) -> None:
    """Print a compact terminal table for one universe."""
    print(f"\n  {'─'*90}")
    print(f"  {universe_label.upper()}")
    print(f"  {'─'*90}")
    print(f"  {'#':<3} {'Ticker':<7} {'Score':>6} {'Regime':<8} {'Cap':<6} "
          f"{'Vol20d':>7} {'Ret20d':>7} {'MomSlope':>9} {'RelVol':>7} {'Insight'}")
    print(f"  {'─'*90}")

    for i, r in enumerate(results[:top_n], 1):
        flag = " ⚠" if r.high_risk else "  "
        vol  = f"{r.vol_20d:.0%}" if r.vol_20d else "N/A"
        r20  = f"{r.ret_20d:+.1%}" if r.ret_20d else "N/A"
        ms   = f"{r.momentum_slope:+.2f}" if r.momentum_slope else "N/A"
        rv   = f"{r.rel_volume:.1f}x" if r.rel_volume else "N/A"
        ins  = r.insight[:35] + "..." if len(r.insight) > 35 else r.insight

        print(f"  {i:<3} {r.ticker:<7}{flag} {r.opportunity_score:>5.1f}  "
              f"{r.regime_label:<8} {r.cap_label:<6} {vol:>7} {r20:>7} "
              f"{ms:>9} {rv:>7}  {ins}")

    avoid = [r for r in results if r.is_avoid]
    if avoid:
        print(f"\n  AVOID: {', '.join(r.ticker for r in avoid)}")
    print(f"  {'─'*90}")
