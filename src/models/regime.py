"""
src/models/regime.py
---------------------
Volatility Regime Classifier

This module implements a four-state volatility regime classifier that
converts a continuous σ estimate into one of four discrete regimes:
LOW, MEDIUM, HIGH, EXTREME.

Economic motivation
-------------------
Financial markets operate in qualitatively different states.  A market
maker behaving optimally in a quiet, low-volatility environment should
behave very differently in a crisis or after a large news shock.

The A-S model's γ parameter is intended to capture risk aversion, but
a static γ cannot represent the changing nature of market risk:

- In LOW volatility: competition is intense, spreads are tight.
  A market maker should be aggressive — small γ, tight spread, large size.

- In MEDIUM volatility: baseline A-S behavior applies.

- In HIGH volatility: adverse selection increases sharply.
  Fair value is moving faster than quotes can track.
  Informed traders have a strong edge.  Widen spreads, reduce size.

- In EXTREME volatility: the market is dislocated.
  Quote at all only with maximum protection: very wide spreads,
  minimal size, strict inventory limits.

Implementation
--------------
We use simple threshold-based classification on a smoothed σ estimate.
More sophisticated approaches (HMM, Markov-switching) are Phase 6+.

The classifier maintains:
- current regime
- full regime history (one entry per step)
- transition count matrix (from_regime → to_regime → count)
- time spent in each regime

This data enables post-hoc analysis of how the market maker behaved
across different market conditions.

Hysteresis (optional)
---------------------
Simple threshold classifiers can chatter rapidly between regimes
when σ hovers near a boundary.  The `hysteresis` parameter adds
a dead-band: once you enter a regime, you stay there until σ moves
more than `hysteresis × threshold` away from the boundary.
Set hysteresis=0 to disable (pure threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class VolatilityRegime(Enum):
    """Four-state volatility regime classification."""
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    EXTREME = "extreme"

    def __lt__(self, other: "VolatilityRegime") -> bool:
        order = [self.LOW, self.MEDIUM, self.HIGH, self.EXTREME]
        return order.index(self) < order.index(other)

    def __le__(self, other: "VolatilityRegime") -> bool:
        return self == other or self < other


@dataclass
class RegimeThresholds:
    """
    Volatility thresholds for regime boundaries.

    Interpretation (σ is per-step standard deviation of log-returns):
        σ < low_threshold               → LOW
        low_threshold ≤ σ < high_threshold → MEDIUM
        high_threshold ≤ σ < extreme_threshold → HIGH
        σ ≥ extreme_threshold           → EXTREME

    Default values calibrated to a simulation where σ_true ≈ 0.04/step
    and jumps drive σ_estimated into 0.06–0.12+ range.
    """
    low_threshold:     float = 0.0008  # below this → LOW    (calibrated to simulation σ scale)
    high_threshold:    float = 0.0035  # above this → HIGH
    extreme_threshold: float = 0.0055  # above this → EXTREME
    hysteresis:        float = 0.0002  # dead-band to prevent rapid switching

    def __post_init__(self) -> None:
        if not (0 < self.low_threshold < self.high_threshold < self.extreme_threshold):
            raise ValueError(
                f"Thresholds must satisfy 0 < low < high < extreme, got "
                f"{self.low_threshold}, {self.high_threshold}, {self.extreme_threshold}"
            )
        if self.hysteresis < 0:
            raise ValueError(f"hysteresis must be >= 0, got {self.hysteresis}")


@dataclass
class RegimeTransitionEvent:
    """Records a single regime transition."""
    timestep:  int
    from_regime: VolatilityRegime
    to_regime:   VolatilityRegime
    sigma_at_transition: float


class RegimeClassifier:
    """
    Classifies the current volatility regime from a σ estimate.

    Uses configurable thresholds with optional hysteresis to avoid
    rapid regime switching near boundaries.

    Parameters
    ----------
    thresholds : RegimeThresholds
        Boundary values for regime classification.
    initial_regime : VolatilityRegime
        Starting regime before any data arrives.

    Usage
    -----
    classifier = RegimeClassifier()
    regime = classifier.update(sigma=0.07)   # → HIGH
    regime = classifier.update(sigma=0.03)   # → MEDIUM (or LOW with hysteresis)
    """

    def __init__(
        self,
        thresholds: Optional[RegimeThresholds] = None,
        initial_regime: VolatilityRegime = VolatilityRegime.MEDIUM,
    ) -> None:
        self.thresholds = thresholds or RegimeThresholds()
        self._current_regime = initial_regime
        self._history: List[VolatilityRegime] = []
        self._transitions: List[RegimeTransitionEvent] = []
        self._regime_counts: Dict[VolatilityRegime, int] = {
            r: 0 for r in VolatilityRegime
        }
        self._transition_matrix: Dict[
            Tuple[VolatilityRegime, VolatilityRegime], int
        ] = {}
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, sigma: float) -> VolatilityRegime:
        """
        Classify the current regime from a volatility estimate.

        Parameters
        ----------
        sigma : float
            Current per-step volatility estimate (σ, not σ²).

        Returns
        -------
        VolatilityRegime : current regime after update
        """
        self._step += 1
        new_regime = self._classify(sigma)

        if new_regime != self._current_regime:
            # Record transition
            event = RegimeTransitionEvent(
                timestep=self._step,
                from_regime=self._current_regime,
                to_regime=new_regime,
                sigma_at_transition=sigma,
            )
            self._transitions.append(event)

            key = (self._current_regime, new_regime)
            self._transition_matrix[key] = self._transition_matrix.get(key, 0) + 1
            self._current_regime = new_regime

        self._history.append(self._current_regime)
        self._regime_counts[self._current_regime] += 1
        return self._current_regime

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(self, sigma: float) -> VolatilityRegime:
        """
        Apply threshold + hysteresis classification.

        Hysteresis: to escape a regime, σ must cross the boundary by
        more than the hysteresis margin.  This prevents chattering.
        """
        t = self.thresholds
        h = t.hysteresis
        cur = self._current_regime

        # Pure threshold classification (no hysteresis)
        if sigma >= t.extreme_threshold:
            pure = VolatilityRegime.EXTREME
        elif sigma >= t.high_threshold:
            pure = VolatilityRegime.HIGH
        elif sigma >= t.low_threshold:
            pure = VolatilityRegime.MEDIUM
        else:
            pure = VolatilityRegime.LOW

        # Apply hysteresis: only switch if the new regime is different
        # and σ has moved sufficiently past the boundary
        if pure == cur:
            return cur

        # Allow downgrade (lower regime) only if σ is well below the boundary
        if pure < cur:
            boundary = self._lower_boundary(cur)
            if boundary is not None and sigma > boundary - h:
                return cur   # stay in current regime (haven't moved far enough)

        # Allow upgrade (higher regime) only if σ is well above the boundary
        if pure > cur:
            boundary = self._upper_boundary(cur)
            if boundary is not None and sigma < boundary + h:
                return cur   # stay in current regime

        return pure

    def _lower_boundary(self, regime: VolatilityRegime) -> Optional[float]:
        """The σ threshold below which we exit this regime downward."""
        t = self.thresholds
        return {
            VolatilityRegime.MEDIUM:  t.low_threshold,
            VolatilityRegime.HIGH:    t.high_threshold,
            VolatilityRegime.EXTREME: t.extreme_threshold,
        }.get(regime)

    def _upper_boundary(self, regime: VolatilityRegime) -> Optional[float]:
        """The σ threshold above which we exit this regime upward."""
        t = self.thresholds
        return {
            VolatilityRegime.LOW:    t.low_threshold,
            VolatilityRegime.MEDIUM: t.high_threshold,
            VolatilityRegime.HIGH:   t.extreme_threshold,
        }.get(regime)

    # ------------------------------------------------------------------
    # Properties and diagnostics
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> VolatilityRegime:
        """Current volatility regime."""
        return self._current_regime

    @property
    def history(self) -> List[VolatilityRegime]:
        """Full per-step regime history."""
        return list(self._history)

    @property
    def transitions(self) -> List[RegimeTransitionEvent]:
        """List of all regime transition events."""
        return list(self._transitions)

    @property
    def transition_count(self) -> int:
        """Total number of regime transitions."""
        return len(self._transitions)

    @property
    def regime_counts(self) -> Dict[VolatilityRegime, int]:
        """Number of steps spent in each regime."""
        return dict(self._regime_counts)

    @property
    def transition_matrix(self) -> Dict[Tuple[VolatilityRegime, VolatilityRegime], int]:
        """Count of transitions between each pair of regimes."""
        return dict(self._transition_matrix)

    def time_in_regime(self, regime: VolatilityRegime) -> int:
        """Steps spent in a given regime."""
        return self._regime_counts.get(regime, 0)

    def regime_fraction(self, regime: VolatilityRegime) -> float:
        """Fraction of time spent in a given regime [0,1]."""
        total = sum(self._regime_counts.values())
        if total == 0:
            return 0.0
        return self._regime_counts.get(regime, 0) / total

    def reset(self) -> None:
        """Reset classifier to initial state."""
        self._current_regime = VolatilityRegime.MEDIUM
        self._history.clear()
        self._transitions.clear()
        self._regime_counts = {r: 0 for r in VolatilityRegime}
        self._transition_matrix.clear()
        self._step = 0

    def print_summary(self) -> None:
        """Print a human-readable summary of regime history."""
        total = sum(self._regime_counts.values())
        print(f"  Regime Summary ({total} steps, {self.transition_count} transitions)")
        for r in VolatilityRegime:
            cnt = self._regime_counts[r]
            pct = 100 * cnt / total if total else 0
            print(f"    {r.value:<8}  {cnt:>5} steps  ({pct:5.1f}%)")
        if self._transitions:
            print(f"  Last transition: step {self._transitions[-1].timestep}  "
                  f"{self._transitions[-1].from_regime.value} → "
                  f"{self._transitions[-1].to_regime.value}")

    def __repr__(self) -> str:
        return (
            f"RegimeClassifier(current={self._current_regime.value}, "
            f"transitions={self.transition_count})"
        )


# ---------------------------------------------------------------------------
# Regime-aware parameter set
# ---------------------------------------------------------------------------

@dataclass
class RegimeParameters:
    """
    Multipliers applied to A-S base parameters in each regime.

    All values are multiplicative factors applied to the base ASConfig values:
        effective_gamma     = base_gamma     × gamma_mult
        effective_half_spr  = computed_delta × spread_mult
        effective_quote_sz  = base_quote_sz  × quote_size_mult
        effective_max_inv   = base_max_inv   × max_inv_mult

    The min_spread_mult applies to the min_half_spread config value.

    Design rationale for defaults
    ------------------------------
    LOW:
        - Tighter spreads attract more flow in quiet markets
        - Larger size earns more spread income per fill
        - Lower γ → lighter inventory penalty → quotes closer to fair value
        - Competitive edge vs other MMs who also sit out in quiet markets

    HIGH / EXTREME:
        - γ increases → heavier inventory penalty → more aggressive skew
        - Spread multiplier widens quotes → higher protection per fill
        - Smaller size → limits exposure per quote
        - These protect against adverse selection during jumps
    """
    # Per-regime multipliers: (gamma, spread, quote_size, max_inventory)
    low_mult:     Tuple[float, float, float, float] = (0.6, 0.7, 1.4, 1.5)
    medium_mult:  Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    high_mult:    Tuple[float, float, float, float] = (2.0, 1.8, 0.6, 0.6)
    extreme_mult: Tuple[float, float, float, float] = (4.0, 3.0, 0.25, 0.3)

    def get(self, regime: VolatilityRegime) -> Tuple[float, float, float, float]:
        """Return (gamma_mult, spread_mult, quote_size_mult, max_inv_mult)."""
        return {
            VolatilityRegime.LOW:     self.low_mult,
            VolatilityRegime.MEDIUM:  self.medium_mult,
            VolatilityRegime.HIGH:    self.high_mult,
            VolatilityRegime.EXTREME: self.extreme_mult,
        }[regime]

    def gamma_mult(self, regime: VolatilityRegime) -> float:
        return self.get(regime)[0]

    def spread_mult(self, regime: VolatilityRegime) -> float:
        return self.get(regime)[1]

    def quote_size_mult(self, regime: VolatilityRegime) -> float:
        return self.get(regime)[2]

    def max_inv_mult(self, regime: VolatilityRegime) -> float:
        return self.get(regime)[3]
