"""
src/models/volatility.py
-------------------------
Rolling Realized Volatility Estimator

The Avellaneda-Stoikov model requires a real-time estimate of σ, the
per-step price volatility.  In practice this is not known — it must be
inferred from recent price history.

Approach: Rogers-Satchell style rolling window
----------------------------------------------
We use a simple rolling window of log midprice returns:

    r_t = ln(mid_t / mid_{t-1})

    σ²_realized = (1 / (n-1)) × Σ (r_i - r̄)²   over the last n steps

This is the standard unbiased sample variance estimator applied to
log-returns.  It converges to the true diffusion coefficient as n → ∞.

Design choices
--------------
1. Minimum variance floor: σ is clamped to `min_vol` to avoid numerical
   issues (division by near-zero in the spread formula).
2. Maximum variance cap: σ is clamped to `max_vol` to prevent extreme
   spreads during data artifacts.
3. Warm-up period: before the window is full, the estimator returns the
   initial_vol prior — a reasonable Bayesian-like initialization.
4. Exponential weighting (optional): recent returns weighted more heavily
   via an EWM-style decay.  This makes the estimator react faster to
   volatility regime changes — important for Phase 4 and later.

Economic intuition
------------------
- Low σ → A-S formula gives a TIGHTER spread.  The MM is willing to
  provide more liquidity when the asset is quiet.
- High σ → A-S formula gives a WIDER spread.  The MM demands higher
  compensation per unit of inventory risk.
This is the key channel through which regime-awareness emerges even
without an explicit regime classifier.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


@dataclass
class VolatilityConfig:
    """Configuration for the rolling volatility estimator."""
    window: int = 30          # number of midprice observations in rolling window
    min_vol: float = 1e-4     # absolute floor on σ (prevents divide-by-zero)
    max_vol: float = 5.0      # absolute cap on σ
    initial_vol: float = 0.05 # prior used during warm-up before window is full
    use_ewm: bool = False      # if True, use exponential weighting
    ewm_alpha: float = 0.10   # EWM decay per step (higher = faster adaptation)


class RollingVolatilityEstimator:
    """
    Maintains a rolling window of midprice observations and computes
    realized volatility as the standard deviation of log-returns.

    Parameters
    ----------
    config : VolatilityConfig

    Usage
    -----
    estimator = RollingVolatilityEstimator()
    for midprice in price_series:
        estimator.update(midprice)
        sigma = estimator.sigma   # current estimate
    """

    def __init__(self, config: Optional[VolatilityConfig] = None) -> None:
        self.config = config or VolatilityConfig()
        # Store raw midprices (one extra needed to compute returns)
        self._prices: Deque[float] = deque(maxlen=self.config.window + 1)
        self._vol_history: List[float] = []
        self._current_sigma: float = self.config.initial_vol

    def update(self, midprice: float) -> float:
        """
        Ingest a new midprice observation and return updated σ estimate.

        Parameters
        ----------
        midprice : float
            Current midprice from the order book.

        Returns
        -------
        float : updated σ estimate (per-step standard deviation of log-return)
        """
        if midprice <= 0:
            self._vol_history.append(self._current_sigma)
            return self._current_sigma

        self._prices.append(midprice)

        # Need at least 2 prices to compute a return, and window+1 for full estimate
        if len(self._prices) < 2:
            self._vol_history.append(self._current_sigma)
            return self._current_sigma

        # Compute log-returns from the stored window
        prices = list(self._prices)
        returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > 0 and prices[i] > 0
        ]

        if not returns:
            self._vol_history.append(self._current_sigma)
            return self._current_sigma

        # Warm-up: window not yet full — return initial prior blended with sample
        if len(returns) < self.config.window:
            fill_ratio = len(returns) / self.config.window
            if self.config.use_ewm:
                sigma = self._ewm_vol(returns)
            else:
                sigma = self._sample_std(returns)
            # Blend toward initial prior when data is sparse
            sigma = fill_ratio * sigma + (1 - fill_ratio) * self.config.initial_vol
        else:
            if self.config.use_ewm:
                sigma = self._ewm_vol(returns)
            else:
                sigma = self._sample_std(returns)

        # Apply bounds
        sigma = max(self.config.min_vol, min(self.config.max_vol, sigma))
        self._current_sigma = sigma
        self._vol_history.append(sigma)
        return sigma

    @property
    def sigma(self) -> float:
        """Current volatility estimate."""
        return self._current_sigma

    @property
    def sigma_squared(self) -> float:
        """Current variance estimate (σ²)."""
        return self._current_sigma ** 2

    @property
    def history(self) -> List[float]:
        """Full history of per-step σ estimates."""
        return list(self._vol_history)

    @property
    def is_warmed_up(self) -> bool:
        """True once the rolling window is fully populated."""
        return len(self._prices) >= self.config.window + 1

    def reset(self) -> None:
        """Reset estimator to initial state."""
        self._prices.clear()
        self._vol_history.clear()
        self._current_sigma = self.config.initial_vol

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_std(self, returns: List[float]) -> float:
        """Unbiased sample standard deviation of log-returns."""
        n = len(returns)
        if n < 2:
            return self.config.initial_vol
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        return math.sqrt(variance)

    def _ewm_vol(self, returns: List[float]) -> float:
        """
        Exponentially weighted standard deviation.
        Mimics pandas ewm(alpha=α).std() without the library dependency.
        """
        alpha = self.config.ewm_alpha
        if not returns:
            return self.config.initial_vol

        # EWM mean
        ewm_mean = returns[0]
        for r in returns[1:]:
            ewm_mean = alpha * r + (1 - alpha) * ewm_mean

        # EWM variance
        ewm_var = 0.0
        for r in returns:
            ewm_var = alpha * (r - ewm_mean) ** 2 + (1 - alpha) * ewm_var

        return math.sqrt(max(ewm_var, 0.0))

    def __repr__(self) -> str:
        return (
            f"RollingVolatilityEstimator(σ={self._current_sigma:.6f}, "
            f"window={self.config.window}, warmed_up={self.is_warmed_up})"
        )
