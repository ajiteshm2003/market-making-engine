"""
src/replay/coinbase_loader.py
------------------------------
Coinbase Market Data Loader

Provides a consistent interface for loading Coinbase Advanced Trade (formerly
Coinbase Pro) L2 order book data, either from local CSV files or (optionally)
from the Coinbase WebSocket API.

Coinbase public data:
  - Historical fills: https://api.exchange.coinbase.com/products/{id}/trades
  - Level 2 data: WebSocket channel "level2" on wss://advanced-trade-ws.coinbase.com

CSV format (Coinbase historical fills):
  trade_id, product_id, price, size, time, side, bid, ask, volume
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .market_data import ReplayEventStream
from .historical_replay import SyntheticMarketDataGenerator


@dataclass
class CoinbaseLoaderConfig:
    """Configuration for the Coinbase data loader."""
    product_id:    str  = "BTC-USD"
    data_dir:      str  = "data/coinbase"
    use_synthetic: bool = True


class CoinbaseMarketDataLoader:
    """
    Loads Coinbase market data into a ReplayEventStream.

    In offline mode (default): generates synthetic data matching
    Coinbase format conventions.

    In live mode (stubbed): would connect to Coinbase WebSocket.
    """

    def __init__(
        self,
        config: Optional[CoinbaseLoaderConfig] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or CoinbaseLoaderConfig()
        self._seed  = seed

    def load_from_csv(self, filepath: str) -> ReplayEventStream:
        """
        Load a Coinbase-format historical trades CSV.

        Expected columns (subset of Coinbase fills export):
          time, price, size, side
        """
        from .historical_replay import load_replay_from_csv
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Coinbase data file not found: {filepath}\n"
                f"Download from the Coinbase Exchange API or use synthetic mode."
            )
        return load_replay_from_csv(filepath)

    def load_synthetic(
        self,
        n_steps: int = 300,
        initial_price: float = 30000.0,
        volatility: float = 0.04,
        jump_prob: float = 0.03,
        jump_std: float = 120.0,
    ) -> ReplayEventStream:
        """Generate a synthetic Coinbase-like event stream."""
        gen = SyntheticMarketDataGenerator(
            n_steps=n_steps,
            initial_price=initial_price,
            volatility=volatility,
            jump_prob=jump_prob,
            jump_std=jump_std,
            spread_ticks=initial_price * 0.0008,   # ~8 bps
            depth_levels=5,
            seed=self._seed,
        )
        return gen.generate()

    def load(self, filepath: Optional[str] = None, **kwargs) -> ReplayEventStream:
        """Primary entry point: file or synthetic fallback."""
        if filepath and os.path.exists(filepath):
            return self.load_from_csv(filepath)
        return self.load_synthetic(**kwargs)

    # ------------------------------------------------------------------
    # Future live integration stubs
    # ------------------------------------------------------------------

    def connect_websocket(self, product_ids: list, channels: list = None) -> None:
        """
        [STUB] Connect to Coinbase Advanced Trade WebSocket.

        Would establish a WSS connection to:
          wss://advanced-trade-ws.coinbase.com

        Subscribe with:
          {"type": "subscribe", "product_ids": product_ids,
           "channel": "level2", ...}

        Not implemented. Requires: websockets, Coinbase API key for auth.
        """
        raise NotImplementedError(
            "Live WebSocket integration not yet implemented. "
            "Use load_synthetic() or load_from_csv() for offline testing."
        )

    def fetch_trades(self, product_id: str, limit: int = 1000) -> None:
        """
        [STUB] Fetch recent trades via REST API.

        Would call: GET https://api.exchange.coinbase.com/products/{product_id}/trades
        No API key required for public endpoints.
        """
        raise NotImplementedError("Live REST integration not yet implemented.")
