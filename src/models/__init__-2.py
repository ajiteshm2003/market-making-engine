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
)
from .arrival_intensity import ArrivalIntensityConfig, ArrivalIntensityEstimator
from .volatility import RollingVolatilityEstimator, VolatilityConfig

__all__ = [
    "ArrivalIntensityConfig",
    "ArrivalIntensityEstimator",
    "RollingVolatilityEstimator",
    "VolatilityConfig",
    "compute_quotes",
    "optimal_half_spread",
    "reservation_price",
    "sensitivity_analysis",
    "spread_decomposition",
]

from .analytics import (
    max_drawdown,
    print_comparison,
    sharpe_ratio,
    strategy_comparison,
)

__all__ += [
    "max_drawdown",
    "print_comparison",
    "sharpe_ratio",
    "strategy_comparison",
]
