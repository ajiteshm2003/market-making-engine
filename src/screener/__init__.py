"""src/screener/__init__.py"""
from .market_data import MarketDataFetcher, TickerData, FetchResult, DEFAULT_UNIVERSE
from .factors import FactorEngine, TickerFactors, EQUITY_THRESHOLDS
from .scoring import Scorer, ScoredTicker, WEIGHTS
from .report import save_csv, save_markdown, print_terminal_table

__all__ = [
    "MarketDataFetcher", "TickerData", "FetchResult", "DEFAULT_UNIVERSE",
    "FactorEngine", "TickerFactors", "EQUITY_THRESHOLDS",
    "Scorer", "ScoredTicker", "WEIGHTS",
    "save_csv", "save_markdown", "print_terminal_table",
]

from .universe import UniverseSpec, CapBucket, VolTolerance, ALL_UNIVERSES, get_universe, all_tickers
from .universe import INSTITUTIONAL, EMERGING, SPECULATIVE
from .opportunity import (
    OpportunityFactors, OpportunityResult, OpportunityPipeline,
    OpportunityScorer, compute_opportunity_factors,
    save_opportunity_csv, save_opportunity_markdown,
    print_opportunity_table, OPPORTUNITY_WEIGHTS,
)
