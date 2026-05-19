"""
src/models/arrival_intensity.py
---------------------------------
Order Arrival Intensity Estimator

In the Avellaneda-Stoikov model, k is the order arrival intensity parameter.
It governs how quickly incoming orders deplete the market maker's quotes.

Interpretation of k
--------------------
The A-S model assumes market orders arrive as a Poisson process, and that
the probability of a market order filling a quote at distance δ from the
midprice follows an exponential law:

    P(fill at depth δ) = A × exp(-k × δ)

Where:
    A  = baseline arrival rate (orders per unit time)
    k  = depth sensitivity (how fast fill probability decays with distance)

A higher k means:
    - Traders are LESS willing to pay away from mid
    - The optimal spread should be TIGHTER (MM must stay competitive)

A lower k means:
    - Traders WILL fill even at wide spreads
    - The optimal spread should be WIDER (MM can charge more)

In the optimal spread formula:
    δ* = (γσ²(T-t))/2 + (1/γ) × ln(1 + γ/k)

The term (1/γ) × ln(1 + γ/k) is the LIQUIDITY PREMIUM component.
As k → ∞, this term → 0 (very eager market takers, spread is just risk).
As k → 0, this term → ∞ (reluctant takers, MM demands large spread).

Estimation approach
-------------------
We estimate k from observable data using two signals:

1. Fill frequency: how often do our quotes get filled per unit time?
   Higher frequency → lower k → tighter spread.
   
2. Trade volume rate: total market order flow in recent window.
   More aggressive flow → lower k.

We use a rolling window estimator:
    k_estimate = base_k × (1 / normalized_fill_rate)
    
Bounded to [k_min, k_max] to prevent degenerate spreads.

If no fills are observed in the window, we fall back to k_default
(a conservative prior that produces a moderately wide spread).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


@dataclass
class ArrivalIntensityConfig:
    """Configuration for the arrival intensity estimator."""
    window: int = 50            # number of timesteps in rolling window
    k_default: float = 1.5     # prior used before enough fills observed
    k_min: float = 0.1         # minimum allowed k (prevents infinite spread)
    k_max: float = 10.0        # maximum allowed k (prevents zero spread)
    fill_scale: float = 2.0    # scaling factor: fills_per_step → k estimate


class ArrivalIntensityEstimator:
    """
    Estimates the order arrival intensity parameter k from fill observations.

    k is directly observable in our simulation (we track fills), which is
    the advantage of simulated market making over real-world deployment.

    In production, k would be estimated from Level 2 order flow data and
    calibrated against historical fill records.

    Parameters
    ----------
    config : ArrivalIntensityConfig

    Usage
    -----
    estimator = ArrivalIntensityEstimator()
    estimator.update(fills_this_step=2, volume_this_step=10.0)
    k = estimator.k    # current intensity estimate
    """

    def __init__(self, config: Optional[ArrivalIntensityConfig] = None) -> None:
        self.config = config or ArrivalIntensityConfig()
        self._fill_counts: Deque[int] = deque(maxlen=self.config.window)
        self._volume_obs: Deque[float] = deque(maxlen=self.config.window)
        self._k_history: List[float] = []
        self._current_k: float = self.config.k_default

    def update(self, fills_this_step: int, volume_this_step: float = 0.0) -> float:
        """
        Update the intensity estimate with new observations.

        Parameters
        ----------
        fills_this_step : int
            Number of market-order fills received this step (as maker).
        volume_this_step : float
            Total market order volume this step.

        Returns
        -------
        float : updated k estimate
        """
        self._fill_counts.append(fills_this_step)
        self._volume_obs.append(volume_this_step)

        n = len(self._fill_counts)
        if n == 0:
            self._k_history.append(self._current_k)
            return self._current_k

        avg_fills_per_step = sum(self._fill_counts) / n

        if avg_fills_per_step < 1e-6:
            # No fills observed — use conservative default
            k = self.config.k_default
        else:
            # k scales inversely with fill frequency:
            # More fills → lower k (takers are eager → tighter spread needed)
            # Fewer fills → higher k (takers are selective → can widen spread)
            k = self.config.fill_scale / avg_fills_per_step

        k = max(self.config.k_min, min(self.config.k_max, k))
        self._current_k = k
        self._k_history.append(k)
        return k

    @property
    def k(self) -> float:
        """Current arrival intensity estimate."""
        return self._current_k

    @property
    def history(self) -> List[float]:
        """Full history of k estimates."""
        return list(self._k_history)

    @property
    def avg_fills_per_step(self) -> float:
        """Recent average fills per timestep."""
        if not self._fill_counts:
            return 0.0
        return sum(self._fill_counts) / len(self._fill_counts)

    def reset(self) -> None:
        """Reset to initial state."""
        self._fill_counts.clear()
        self._volume_obs.clear()
        self._k_history.clear()
        self._current_k = self.config.k_default

    def __repr__(self) -> str:
        return (
            f"ArrivalIntensityEstimator(k={self._current_k:.4f}, "
            f"avg_fills/step={self.avg_fills_per_step:.3f}, "
            f"window={self.config.window})"
        )
