"""
src/models/__init__.py
Public API for the models package.
"""

from .avellaneda_stoikov_math import (
    compute_quotes,
    optimal_half_spread,
    reservation_price,
    sensitivity_analysis,
    spread_decomposition,
    implied_k_from_spread,
)
from .arrival_intensity import ArrivalIntensityConfig, ArrivalIntensityEstimator
from .volatility import RollingVolatilityEstimator, VolatilityConfig
from .analytics import (
    max_drawdown,
    print_comparison,
    sharpe_ratio,
    strategy_comparison,
)
from .regime import (
    VolatilityRegime,
    RegimeThresholds,
    RegimeClassifier,
    RegimeParameters,
    RegimeTransitionEvent,
)

__all__ = [
    "compute_quotes",
    "optimal_half_spread",
    "reservation_price",
    "sensitivity_analysis",
    "spread_decomposition",
    "implied_k_from_spread",
    "ArrivalIntensityConfig",
    "ArrivalIntensityEstimator",
    "RollingVolatilityEstimator",
    "VolatilityConfig",
    "max_drawdown",
    "print_comparison",
    "sharpe_ratio",
    "strategy_comparison",
    "VolatilityRegime",
    "RegimeThresholds",
    "RegimeClassifier",
    "RegimeParameters",
    "RegimeTransitionEvent",
]
