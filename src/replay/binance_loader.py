"""
src/replay/binance_loader.py
-----------------------------
Binance Market Data Loader

Provides a consistent interface for loading Binance L2 order book data,
either from local CSV files or (optionally) from the Binance REST/WebSocket API.

This module is designed to work entirely offline using cached CSV files.
Live API integration is stubbed and clearly marked for future implementation.

Binance public data is available at:
  https://data.binance.vision/
  - Spot daily/monthly snapshots
  - Trade data: SYMBOL-trades-DATE.zip
  - Book ticker: SYMBOL-bookTicker-DATE.zip
  - Depth snapshots: SYMBOL-depth-DATE.zip

CSV format for depth snapshots (Binance):
  timestamp, first_update_id, last_update_id,
  side, price, quantity (one row per level change)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .market_data import (
    L2Level, L2Snapshot, L2Update, TradeEvent,
    MarketDataEvent, ReplayEventStream, EventType,
)
from .historical_replay import SyntheticMarketDataGenerator


@dataclass
class BinanceLoaderConfig:
    """Configuration for the Binance data loader."""
    symbol:       str   = "BTCUSDT"
    data_dir:     str   = "data/binance"
    use_synthetic: bool = True     # set False to require real files


class BinanceMarketDataLoader:
    """
    Loads Binance market data into a ReplayEventStream.

    In offline mode (use_synthetic=True or no real files found):
        Generates synthetic data matching Binance format conventions.

    In live mode (not yet implemented):
        Would connect to Binance WebSocket depth stream.

    Parameters
    ----------
    config : BinanceLoaderConfig
    seed   : random seed for synthetic data generation
    """

    def __init__(
        self,
        config: Optional[BinanceLoaderConfig] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or BinanceLoaderConfig()
        self._seed  = seed

    def load_from_csv(self, filepath: str) -> ReplayEventStream:
        """
        Load a Binance-format depth snapshot CSV file.

        Expected columns:
          timestamp, side, price, quantity

        Returns
        -------
        ReplayEventStream
        """
        from .historical_replay import load_replay_from_csv
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Binance data file not found: {filepath}\n"
                f"Download from https://data.binance.vision/ or use synthetic mode."
            )
        return load_replay_from_csv(filepath)

    def load_synthetic(
        self,
        n_steps: int = 300,
        initial_price: float = 30000.0,  # BTC-like
        volatility: float = 0.05,
        jump_prob: float = 0.04,
        jump_std: float = 150.0,
    ) -> ReplayEventStream:
        """
        Generate a synthetic Binance-like event stream for testing.
        Prices and spread ticks are scaled to a BTC/USDT-like instrument.
        """
        gen = SyntheticMarketDataGenerator(
            n_steps=n_steps,
            initial_price=initial_price,
            volatility=volatility,
            jump_prob=jump_prob,
            jump_std=jump_std,
            spread_ticks=initial_price * 0.001,  # 10 bps spread
            depth_levels=5,
            seed=self._seed,
        )
        return gen.generate()

    def load(self, filepath: Optional[str] = None, **kwargs) -> ReplayEventStream:
        """
        Primary entry point: load data from file or fall back to synthetic.
        """
        if filepath and os.path.exists(filepath):
            return self.load_from_csv(filepath)
        return self.load_synthetic(**kwargs)

    # ------------------------------------------------------------------
    # Future live integration stubs
    # ------------------------------------------------------------------

    def connect_websocket(self, symbol: str, depth_levels: int = 10) -> None:
        """
        [STUB] Connect to Binance WebSocket depth stream.

        Would establish a WSS connection to:
          wss://stream.binance.com:9443/ws/{symbol.lower()}@depth{depth_levels}

        Not implemented. Requires: websockets, API key (for signed endpoints).
        """
        raise NotImplementedError(
            "Live WebSocket integration not yet implemented. "
            "Use load_synthetic() or load_from_csv() for offline testing."
        )

    def fetch_snapshot(self, symbol: str, limit: int = 1000) -> None:
        """
        [STUB] Fetch current order book snapshot via REST API.

        Would call: GET https://api.binance.com/api/v3/depth
        Not implemented. No API key required for public endpoints.
        """
        raise NotImplementedError(
            "Live REST integration not yet implemented."
        )
