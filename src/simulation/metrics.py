"""
src/simulation/metrics.py
--------------------------
SimulationMetrics

Collects a time series of market statistics at every simulation timestep.
After the simulation completes, these are exported to a pandas DataFrame
for analysis and visualisation.

Tracked per timestep
--------------------
- timestep
- fair_value         — latent true price
- midprice           — (best_bid + best_ask) / 2
- best_bid
- best_ask
- spread             — best_ask - best_bid
- order_imbalance    — in [-1, +1]
- trades_this_step   — number of executions this tick
- volume_this_step   — total quantity traded this tick
- cumulative_trades  — running total of all trades
- cumulative_volume  — running total of all volume
- book_depth_bids    — total qty on bid side (top 5 levels)
- book_depth_asks    — total qty on ask side (top 5 levels)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class StepRecord:
    timestep: int
    fair_value: Optional[float]
    midprice: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    order_imbalance: Optional[float]
    trades_this_step: int
    volume_this_step: float
    cumulative_trades: int
    cumulative_volume: float
    book_depth_bids: float
    book_depth_asks: float


class SimulationMetrics:
    """
    Append-only ledger of per-step market statistics.
    """

    def __init__(self) -> None:
        self._records: List[StepRecord] = []
        self._cumulative_trades: int = 0
        self._cumulative_volume: float = 0.0

    def record(
        self,
        timestep: int,
        fair_value: Optional[float],
        midprice: Optional[float],
        best_bid: Optional[float],
        best_ask: Optional[float],
        spread: Optional[float],
        order_imbalance: Optional[float],
        trades_this_step: int,
        volume_this_step: float,
        book_depth_bids: float,
        book_depth_asks: float,
    ) -> None:
        self._cumulative_trades += trades_this_step
        self._cumulative_volume += volume_this_step

        self._records.append(
            StepRecord(
                timestep=timestep,
                fair_value=fair_value,
                midprice=midprice,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                order_imbalance=order_imbalance,
                trades_this_step=trades_this_step,
                volume_this_step=volume_this_step,
                cumulative_trades=self._cumulative_trades,
                cumulative_volume=self._cumulative_volume,
                book_depth_bids=book_depth_bids,
                book_depth_asks=book_depth_asks,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all step records to a tidy pandas DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(
            [vars(r) for r in self._records]
        ).set_index("timestep")

    def __len__(self) -> int:
        return len(self._records)
