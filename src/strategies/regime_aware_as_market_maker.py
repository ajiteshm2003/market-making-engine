"""
src/strategies/regime_aware_as_market_maker.py
------------------------------------------------
Regime-Aware Avellaneda-Stoikov Market Maker (R-ASMM)

This is the flagship strategy of the engine.  It extends the static
AvellanedaStoikovMarketMaker by making every model parameter a function
of the current volatility regime.

Architecture
------------
R-ASMM wraps a RegimeClassifier around the existing ASMM logic.
Each timestep:

    1. Update σ estimate (via parent's vol estimator)
    2. Classify regime: LOW / MEDIUM / HIGH / EXTREME
    3. Apply regime multipliers to: γ, spread, quote_size, max_inventory
    4. Compute A-S reservation price with effective_gamma
    5. Compute A-S optimal spread with effective_gamma
    6. Scale spread by spread_multiplier
    7. Scale quote size by quote_size_multiplier
    8. Post bid = r - δ_eff, ask = r + δ_eff

Why this is better than static A-S
-----------------------------------
Static ASMM uses a single γ for the entire trading day.  This is correct
if volatility is stationary — but real markets are not.

The problem manifests in two scenarios:

1. A jump happens (HIGH/EXTREME regime):
   Static ASMM continues with its baseline γ.  The inventory penalty is
   proportional to γσ²(T-t) — so σ alone drives the widening, but γ stays
   small.  The spread widens somewhat but not enough: informed traders who
   know the new fair value still find it profitable to hit the quotes.
   R-ASMM simultaneously raises γ AND lets σ rise, creating a multiplicative
   defense.  The combined effect is much stronger protection.

2. Market is quiet (LOW regime):
   Static ASMM with calibrated γ for volatile periods is too conservative
   in quiet markets — too wide, too cautious, loses fill rate to competitors.
   R-ASMM reduces γ in LOW regime: tighter spreads attract more volume,
   improving fill rate and spread income during low-risk periods.

The result: better TAIL PROTECTION (less drawdown in high/extreme) and
better FILL INCOME (more competitive in low/medium), leading to improved
risk-adjusted Sharpe ratio.

Per-regime metrics
------------------
The strategy records a breakdown of PnL, fills, and inventory by regime,
enabling the post-hoc analysis required for the interview research paper.

This is the "regime-aware" feature that distinguishes the project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .avellaneda_stoikov_market_maker import (
    AvellanedaStoikovMarketMaker,
    ASConfig,
    HorizonMode,
)
from ..models.regime import (
    RegimeClassifier,
    RegimeParameters,
    RegimeThresholds,
    VolatilityRegime,
)
from ..models import (
    reservation_price,
    optimal_half_spread,
    compute_quotes as as_compute_quotes,
)

if TYPE_CHECKING:
    from ..simulation.market_state import MarketState


@dataclass
class RegimeAwareASConfig:
    """
    Full configuration for the Regime-Aware ASMM.

    Combines a base ASConfig (static A-S parameters) with regime
    detection thresholds and per-regime multipliers.

    Parameters
    ----------
    base_config : ASConfig
        Base A-S parameters (gamma, horizon, sigma/k estimator configs).
        These serve as the MEDIUM-regime baseline.

    thresholds : RegimeThresholds
        σ boundary values for regime classification.

    regime_params : RegimeParameters
        Multipliers applied to base parameters per regime.

    base_quote_size : float
        Quote size at MEDIUM regime.  Scaled by regime multiplier.

    base_max_inventory : float
        Inventory limit at MEDIUM regime.  Scaled by regime multiplier.
        Used in inventory penalty clamping.
    """
    base_config:        ASConfig          = field(default_factory=ASConfig)
    thresholds:         RegimeThresholds  = field(default_factory=RegimeThresholds)
    regime_params:      RegimeParameters  = field(default_factory=RegimeParameters)
    base_quote_size:    float             = 5.0
    base_max_inventory: float             = 50.0


class RegimeAwareAvellanedaStoikovMarketMaker(AvellanedaStoikovMarketMaker):
    """
    Regime-Aware Avellaneda-Stoikov Market Maker.

    Extends AvellanedaStoikovMarketMaker by adjusting all A-S parameters
    dynamically based on the current volatility regime.

    Parameters
    ----------
    agent_id : str
    ra_config : RegimeAwareASConfig
        Full regime-aware configuration.  If None, sensible defaults used.
    initial_cash : float

    Key additional attributes (beyond ASMM)
    ----------------------------------------
    regime_history      : list[VolatilityRegime]   per-step regime
    gamma_history       : list[float]              effective gamma per step
    quote_size_history  : list[float]              effective quote size per step
    spread_mult_history : list[float]              spread multiplier per step
    max_inv_history     : list[float]              effective max inventory per step
    regime_metrics      : dict[str, dict]          per-regime breakdowns
    classifier          : RegimeClassifier         access to transition data
    """

    def __init__(
        self,
        agent_id: str,
        ra_config: Optional[RegimeAwareASConfig] = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        self.ra_config = ra_config or RegimeAwareASConfig()

        # Initialise parent with the base ASConfig and base_quote_size
        super().__init__(
            agent_id=agent_id,
            config=self.ra_config.base_config,
            quote_size=self.ra_config.base_quote_size,
            initial_cash=initial_cash,
        )

        # Regime detection
        self._classifier = RegimeClassifier(
            thresholds=self.ra_config.thresholds,
            initial_regime=VolatilityRegime.MEDIUM,
        )

        # Per-step diagnostic histories
        self.regime_history:      List[VolatilityRegime] = []
        self.gamma_history:       List[float] = []
        self.quote_size_history:  List[float] = []
        self.spread_mult_history: List[float] = []
        self.max_inv_history:     List[float] = []

        # Per-regime metric accumulators
        self._regime_pnl_delta: Dict[VolatilityRegime, List[float]] = {
            r: [] for r in VolatilityRegime
        }
        self._regime_fills: Dict[VolatilityRegime, int] = {
            r: 0 for r in VolatilityRegime
        }
        self._regime_inventory_snap: Dict[VolatilityRegime, List[float]] = {
            r: [] for r in VolatilityRegime
        }
        self._regime_half_spread: Dict[VolatilityRegime, List[float]] = {
            r: [] for r in VolatilityRegime
        }
        self._regime_spread_capture: Dict[VolatilityRegime, float] = {
            r: 0.0 for r in VolatilityRegime
        }

        # Track PnL at last step for delta computation
        self._last_total_pnl: float = 0.0

        # Track current regime for fill routing
        self._current_regime_for_fill: VolatilityRegime = VolatilityRegime.MEDIUM

    # ------------------------------------------------------------------
    # Core override: regime-aware quote computation
    # ------------------------------------------------------------------

    def _compute_quotes(
        self, state: "MarketState"
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Override parent's _compute_quotes to inject regime-aware parameters.

        Steps:
          1. Get reference price
          2. Update σ via parent's estimator
          3. Classify regime from σ
          4. Look up regime multipliers
          5. Compute effective γ, spread multiplier, quote size, max inventory
          6. Compute A-S quotes with effective_gamma
          7. Scale spread by spread_multiplier
          8. Set self.quote_size so BaseMarketMaker.act() uses correct size
          9. Record diagnostics
        """
        # ── 1. Reference price ──────────────────────────────────────────
        if self.config.use_fair_value:
            ref = state.fair_value
        else:
            ref = state.midprice

        if ref is None:
            ref = state.fair_value
        if ref is None:
            return None, None

        # ── 2. Update σ ─────────────────────────────────────────────────
        sigma = self._vol_estimator.update(ref)

        # ── 3. Classify regime ───────────────────────────────────────────
        regime = self._classifier.update(sigma)
        self._current_regime_for_fill = regime

        # ── 4. Regime multipliers ────────────────────────────────────────
        rp = self.ra_config.regime_params
        gamma_mult  = rp.gamma_mult(regime)
        spr_mult    = rp.spread_mult(regime)
        qs_mult     = rp.quote_size_mult(regime)
        inv_mult    = rp.max_inv_mult(regime)

        effective_gamma   = self.ra_config.base_config.gamma * gamma_mult
        effective_quote_sz = self.ra_config.base_quote_size * qs_mult
        effective_max_inv  = self.ra_config.base_max_inventory * inv_mult

        # ── 5. Update arrival intensity k ───────────────────────────────
        fills_now = self._fills_since_last_update
        self._fills_since_last_update = 0
        k = self._k_estimator.update(
            fills_this_step=fills_now,
            volume_this_step=state.volume_this_step,
        )

        # ── 6. Time remaining ────────────────────────────────────────────
        t_remaining = self._time_remaining(state.timestep)

        # ── 7. A-S reservation price with EFFECTIVE gamma ────────────────
        inventory = self.mm_metrics.inventory

        # Clamp inventory for penalty computation
        clamped_inv = max(-effective_max_inv, min(effective_max_inv, inventory))

        r = reservation_price(
            midprice=ref,
            inventory=clamped_inv,
            gamma=effective_gamma,
            sigma=sigma,
            time_remaining=t_remaining,
        )

        # ── 8. A-S optimal spread with EFFECTIVE gamma, then scale ────────
        raw_delta = optimal_half_spread(
            gamma=effective_gamma,
            sigma=sigma,
            time_remaining=t_remaining,
            k=k,
        )
        delta = raw_delta * spr_mult
        delta = max(self.config.min_half_spread, delta)
        delta = min(self.config.max_half_spread, delta)

        bid = r - delta
        ask = r + delta

        # ── 9. Apply effective quote size ────────────────────────────────
        # Override self.quote_size so BaseMarketMaker.act() submits correct qty
        self.quote_size = round(max(0.01, effective_quote_sz), 4)

        # ── 10. Record diagnostics ────────────────────────────────────────
        self.regime_history.append(regime)
        self.gamma_history.append(effective_gamma)
        self.quote_size_history.append(self.quote_size)
        self.spread_mult_history.append(spr_mult)
        self.max_inv_history.append(effective_max_inv)

        # Append to parent's diagnostic histories
        self.sigma_history.append(sigma)
        self.k_history.append(k)
        self.reservation_history.append(r)
        self.delta_history.append(delta)
        self.time_remaining_history.append(t_remaining)

        # Per-regime inventory snapshot and spread tracking
        self._regime_inventory_snap[regime].append(inventory)
        self._regime_half_spread[regime].append(delta)

        # PnL delta for per-regime accounting
        current_pnl = self.mm_metrics.total_pnl
        pnl_delta = current_pnl - self._last_total_pnl
        self._regime_pnl_delta[regime].append(pnl_delta)
        self._last_total_pnl = current_pnl

        return bid, ask

    # ------------------------------------------------------------------
    # Override notify_fill to track per-regime fills
    # ------------------------------------------------------------------

    def notify_fill(self, trade, as_maker: bool) -> None:
        """Extend fill handling with per-regime fill counting."""
        super().notify_fill(trade, as_maker)
        if as_maker:
            regime = self._current_regime_for_fill
            self._regime_fills[regime] = self._regime_fills.get(regime, 0) + 1

    # ------------------------------------------------------------------
    # Regime metrics
    # ------------------------------------------------------------------

    @property
    def classifier(self) -> RegimeClassifier:
        """Access the underlying regime classifier."""
        return self._classifier

    def regime_metrics_summary(self) -> Dict[str, dict]:
        """
        Return a per-regime breakdown of key metrics.

        Returns dict keyed by regime name, each containing:
            steps          : steps spent in this regime
            pct_time       : fraction of total time
            total_fills    : number of maker fills in this regime
            avg_inventory  : mean |inventory| during this regime
            avg_half_spread: mean δ* during this regime
            pnl_sum        : sum of step PnL deltas in this regime
        """
        result = {}
        for regime in VolatilityRegime:
            inv_snaps = self._regime_inventory_snap[regime]
            spr_snaps = self._regime_half_spread[regime]
            pnl_deltas = self._regime_pnl_delta[regime]
            steps = self._classifier.time_in_regime(regime)
            total_steps = sum(self._classifier.regime_counts.values())

            result[regime.value] = {
                "steps":           steps,
                "pct_time":        round(100 * steps / total_steps, 1) if total_steps else 0.0,
                "total_fills":     self._regime_fills.get(regime, 0),
                "avg_inventory":   round(sum(abs(x) for x in inv_snaps) / len(inv_snaps), 4)
                                   if inv_snaps else 0.0,
                "avg_half_spread": round(sum(spr_snaps) / len(spr_snaps), 6)
                                   if spr_snaps else 0.0,
                "pnl_sum":         round(sum(pnl_deltas), 4),
            }
        return result

    # ------------------------------------------------------------------
    # Diagnostic properties
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> VolatilityRegime:
        return self._classifier.current_regime

    @property
    def transition_count(self) -> int:
        return self._classifier.transition_count

    @property
    def regime_transitions(self):
        return self._classifier.transitions
