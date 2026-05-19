"""
src/strategies/avellaneda_stoikov_market_maker.py
---------------------------------------------------
Avellaneda-Stoikov Market Maker — Strategy C

This is the mathematically principled strategy derived from stochastic
optimal control theory.  It subsumes both Strategy A (NaiveMarketMaker)
and Strategy B (InventoryAwareMarketMaker) as special cases.

How it generalizes the prior strategies
----------------------------------------
NaiveMarketMaker (Strategy A):
    - Fixed spread: δ = half_spread (constant)
    - No inventory adjustment: r = S
    - A-S reduces to this when γ → 0 (zero risk aversion)

InventoryAwareMarketMaker (Strategy B):
    - Linear inventory skew: r = S - γ_eff × q
    - Fixed (widening) spread
    - A-S reduces to this when T-t = 1 (constant horizon) and
      the spread formula is simplified to a linear approximation

A-S (Strategy C):
    - Spread is DYNAMIC: scales with σ²(T-t) — reacts to volatility
    - Inventory skew is DYNAMIC: also scales with σ²(T-t) — time-varying
    - Both components are jointly optimized under the same γ parameter
    - k is estimated from actual market order flow

The critical advantage
----------------------
In volatile periods:
    - σ rises → risk_premium rises → spread widens automatically
    - σ rises → inventory penalty grows → MM skews quotes harder
    Both effects compound, making the MM safer when it's most at risk.

In quiet periods:
    - σ falls → spread narrows → more competitive quotes → more fills
    - Inventory penalty shrinks → quotes stay near mid → less information
    Both effects compound, making the MM more aggressive when it's safe.

This volatility-adaptive behavior is impossible with fixed-spread strategies
and is the reason A-S outperforms IAMM on a risk-adjusted basis.

Time horizon
------------
(T-t) represents remaining time in the trading session.  In our simulation,
we implement two modes:

1. FIXED_HORIZON: T-t = horizon_steps (constant)
   - Same spread all day, regardless of when in the session we are
   - Mathematically equivalent to IAMM with proper γ

2. DECAYING_HORIZON: T-t = (total_steps - current_step) / total_steps
   - Spread NARROWS as session progresses
   - MM becomes more aggressive near close (dumps inventory)
   - Most realistic; matches actual A-S behavior

Architecture
------------
The strategy wraps:
    RollingVolatilityEstimator  — σ estimate (updated each step)
    ArrivalIntensityEstimator   — k estimate (updated from fills)
    avellaneda_stoikov_math     — pure formula functions

Internal state exposed:
    sigma_history               — for plotting volatility
    k_history                   — for plotting fill intensity
    reservation_history         — for diagnostics
    delta_history               — for spread analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from .base_market_maker import BaseMarketMaker
from ..models.volatility import RollingVolatilityEstimator, VolatilityConfig
from ..models.arrival_intensity import ArrivalIntensityEstimator, ArrivalIntensityConfig
from ..models import (
    reservation_price,
    optimal_half_spread,
    compute_quotes as as_compute_quotes,
)

if TYPE_CHECKING:
    from ..simulation.market_state import MarketState


class HorizonMode:
    """Constants for time horizon calculation mode."""
    FIXED   = "fixed"    # T-t constant across all steps
    DECAYING = "decaying" # T-t shrinks from 1.0 to 0.0 over total_steps


@dataclass
class ASConfig:
    """
    Full configuration for the Avellaneda-Stoikov market maker.

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient (γ).  Controls the trade-off between
        inventory risk and spread income.
        - Small γ (≈0.01): spread-focused, tolerates large inventory swings
        - Large γ (≈1.0):  inventory-focused, quotes very defensively
        Typical calibrated range: 0.05 – 0.5

    sigma_config : VolatilityConfig
        Configuration for the rolling volatility estimator.

    k_config : ArrivalIntensityConfig
        Configuration for the arrival intensity estimator.

    horizon_mode : str
        'fixed'   → T-t = horizon_steps throughout
        'decaying'→ T-t = (total - current) / total

    horizon_steps : float
        T-t when mode is FIXED.  Typical values: 0.5 – 5.0.
        Larger values → wider spreads (MM sees more residual risk).

    total_steps : int
        Total simulation steps, used only in DECAYING mode.

    min_half_spread : float
        Absolute floor on δ* (prevents zero-spread quoting).

    max_half_spread : float
        Absolute cap on δ* (prevents infinite spread in edge cases).

    use_fair_value : bool
        If True, use state.fair_value as reference S.
        If False, use observable state.midprice (realistic mode).
    """
    gamma: float = 0.1
    sigma_config: VolatilityConfig = field(default_factory=VolatilityConfig)
    k_config: ArrivalIntensityConfig = field(default_factory=ArrivalIntensityConfig)
    horizon_mode: str = HorizonMode.FIXED
    horizon_steps: float = 1.0
    total_steps: int = 500
    min_half_spread: float = 0.001
    max_half_spread: float = 2.0
    use_fair_value: bool = False


class AvellanedaStoikovMarketMaker(BaseMarketMaker):
    """
    Optimal market maker derived from the Avellaneda-Stoikov (2008) model.

    Quotes dynamically adapting spread and reservation price jointly
    as functions of: inventory, volatility, time horizon, and fill intensity.

    Parameters
    ----------
    agent_id : str
    config : ASConfig
        Full model configuration.  Default values are reasonable starting points.
    quote_size : float
        Quantity per quote.
    initial_cash : float
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[ASConfig] = None,
        quote_size: float = 5.0,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            quote_size=quote_size,
            initial_cash=initial_cash,
        )
        self.config = config or ASConfig()

        if self.config.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.config.gamma}")
        if self.config.horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be > 0, got {self.config.horizon_steps}")

        # Estimators
        self._vol_estimator = RollingVolatilityEstimator(self.config.sigma_config)
        self._k_estimator   = ArrivalIntensityEstimator(self.config.k_config)

        # Diagnostic histories (appended every step)
        self.sigma_history:       List[float] = []
        self.k_history:           List[float] = []
        self.reservation_history: List[float] = []
        self.delta_history:       List[float] = []
        self.time_remaining_history: List[float] = []

        # Track fills since last k update
        self._fills_since_last_update: int = 0

    # ------------------------------------------------------------------
    # Core strategy logic
    # ------------------------------------------------------------------

    def _compute_quotes(
        self, state: "MarketState"
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Implement the A-S optimal quoting strategy.

        Steps:
          1. Update volatility estimate from latest midprice.
          2. Update arrival intensity estimate from recent fills.
          3. Compute time remaining (T-t).
          4. Compute reservation price r = S - q×γ×σ²×(T-t).
          5. Compute optimal half-spread δ* = risk_premium + liquidity_premium.
          6. Return bid = r - δ*, ask = r + δ*.
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

        # ── 2. Update volatility ────────────────────────────────────────
        sigma = self._vol_estimator.update(ref)

        # ── 3. Update arrival intensity ─────────────────────────────────
        # Count fills received since last act() call
        fills_now = self._fills_since_last_update
        self._fills_since_last_update = 0   # reset counter
        k = self._k_estimator.update(
            fills_this_step=fills_now,
            volume_this_step=state.volume_this_step,
        )

        # ── 4. Time remaining ───────────────────────────────────────────
        t_remaining = self._time_remaining(state.timestep)

        # ── 5. A-S formulas ─────────────────────────────────────────────
        inventory = self.mm_metrics.inventory

        bid, ask, r, delta = as_compute_quotes(
            midprice=ref,
            inventory=inventory,
            gamma=self.config.gamma,
            sigma=sigma,
            time_remaining=t_remaining,
            k=k,
            min_half_spread=self.config.min_half_spread,
        )

        # Apply max spread cap
        if delta > self.config.max_half_spread:
            delta = self.config.max_half_spread
            bid = r - delta
            ask = r + delta

        # ── 6. Record diagnostics ────────────────────────────────────────
        self.sigma_history.append(sigma)
        self.k_history.append(k)
        self.reservation_history.append(r)
        self.delta_history.append(delta)
        self.time_remaining_history.append(t_remaining)

        return bid, ask

    # ------------------------------------------------------------------
    # Override notify_fill to count fills for k estimator
    # ------------------------------------------------------------------

    def notify_fill(self, trade, as_maker: bool) -> None:
        """
        Extend base fill handling to track fills for k estimation.
        """
        super().notify_fill(trade, as_maker)
        if as_maker:
            self._fills_since_last_update += 1

    # ------------------------------------------------------------------
    # Time horizon
    # ------------------------------------------------------------------

    def _time_remaining(self, current_step: int) -> float:
        """
        Compute (T - t) based on the configured horizon mode.

        FIXED:   always returns horizon_steps (constant)
        DECAYING: linearly decays from horizon_steps to min_half_spread
                  over total_steps
        """
        if self.config.horizon_mode == HorizonMode.FIXED:
            return self.config.horizon_steps

        elif self.config.horizon_mode == HorizonMode.DECAYING:
            if self.config.total_steps <= 0:
                return self.config.horizon_steps
            # Decay from horizon_steps at t=0 to 0 at t=total_steps
            fraction_remaining = max(
                0.0,
                1.0 - current_step / self.config.total_steps
            )
            return self.config.horizon_steps * fraction_remaining

        return self.config.horizon_steps

    # ------------------------------------------------------------------
    # Diagnostic properties
    # ------------------------------------------------------------------

    @property
    def current_sigma(self) -> float:
        """Current volatility estimate."""
        return self._vol_estimator.sigma

    @property
    def current_k(self) -> float:
        """Current arrival intensity estimate."""
        return self._k_estimator.k

    @property
    def current_reservation_price(self) -> Optional[float]:
        """Most recently computed reservation price."""
        return self.reservation_history[-1] if self.reservation_history else None

    @property
    def current_half_spread(self) -> Optional[float]:
        """Most recently computed optimal half-spread."""
        return self.delta_history[-1] if self.delta_history else None

    @property
    def gamma(self) -> float:
        """Risk aversion parameter."""
        return self.config.gamma
