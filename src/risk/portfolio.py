"""
src/risk/portfolio.py
----------------------
Portfolio Exposure Tracker

Computes standard portfolio risk metrics from market maker position data.
Compatible with MarketMakerMetrics inventory trajectories.

Metrics
-------
notional_exposure : |inventory| × mark_price
net_exposure      : inventory × mark_price (signed)
gross_exposure    : sum of |position| × price across all holdings
leverage          : gross_exposure / equity
inventory_concentration : |inventory| / max_inventory_observed

These metrics are used by institutional risk teams to monitor dealer
book risk in real-time.  The inventory_concentration metric in particular
is used to detect when a market maker has accumulated an outsized position
relative to its normal operating range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class ExposureSnapshot:
    """Point-in-time portfolio exposure."""
    timestep:              int
    inventory:             float
    mark_price:            float
    cash:                  float
    notional_exposure:     float   # |inventory| × mark_price
    net_exposure:          float   # inventory × mark_price (signed)
    equity:                float   # cash + net_exposure
    leverage:              float   # |net_exposure| / max(equity, 1)
    inv_concentration:     float   # |inventory| / rolling_max_abs_inv

    def __str__(self) -> str:
        return (
            f"t={self.timestep:>4}  inv={self.inventory:+.3f}  "
            f"notional={self.notional_exposure:.2f}  "
            f"equity={self.equity:.2f}  leverage={self.leverage:.3f}x  "
            f"conc={self.inv_concentration:.3f}"
        )


@dataclass
class PortfolioRiskSummary:
    """Aggregate risk summary over the full simulation horizon."""
    n_steps:               int
    max_notional:          float
    mean_notional:         float
    max_leverage:          float
    mean_leverage:         float
    max_inventory:         float
    min_inventory:         float
    max_concentration:     float
    time_overleveraged:    float   # fraction of steps where leverage > 1.0
    equity_drawdown:       float   # max peak-to-trough decline in equity
    final_equity:          float

    def __str__(self) -> str:
        lines = [
            "Portfolio Risk Summary",
            f"  Max Notional Exposure : {self.max_notional:.2f}",
            f"  Mean Notional         : {self.mean_notional:.2f}",
            f"  Max Leverage          : {self.max_leverage:.3f}x",
            f"  Mean Leverage         : {self.mean_leverage:.3f}x",
            f"  Inventory Range       : [{self.min_inventory:.3f}, {self.max_inventory:.3f}]",
            f"  Max Concentration     : {self.max_concentration:.3f}",
            f"  Time Overleveraged    : {self.time_overleveraged:.1%}",
            f"  Equity Drawdown       : {self.equity_drawdown:.2f}",
            f"  Final Equity          : {self.final_equity:.2f}",
        ]
        return "\n".join(lines)


class PortfolioExposureTracker:
    """
    Tracks portfolio exposure metrics over a simulation run.

    Usage
    -----
    tracker = PortfolioExposureTracker(initial_cash=100_000)
    tracker.update(timestep=1, inventory=5.0, mark_price=100.5, cash=99_497.5)
    summary = tracker.summarize()
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        leverage_threshold: float = 1.0,
    ) -> None:
        self._initial_cash = initial_cash
        self._leverage_threshold = leverage_threshold
        self._snapshots: List[ExposureSnapshot] = []
        self._max_abs_inv: float = 1e-9  # running max |inventory|

    def update(
        self,
        timestep: int,
        inventory: float,
        mark_price: float,
        cash: float,
    ) -> ExposureSnapshot:
        """
        Record a new exposure snapshot.

        Parameters
        ----------
        timestep   : simulation step number
        inventory  : current net position (signed)
        mark_price : current midprice or fair value
        cash       : current cash balance
        """
        abs_inv = abs(inventory)
        self._max_abs_inv = max(self._max_abs_inv, abs_inv, 1e-9)

        notional   = abs_inv * abs(mark_price)
        net_exp    = inventory * mark_price
        equity     = cash + net_exp
        leverage   = notional / max(abs(equity), 1.0)
        conc       = abs_inv / self._max_abs_inv

        snap = ExposureSnapshot(
            timestep=timestep,
            inventory=inventory,
            mark_price=mark_price,
            cash=cash,
            notional_exposure=notional,
            net_exposure=net_exp,
            equity=equity,
            leverage=leverage,
            inv_concentration=conc,
        )
        self._snapshots.append(snap)
        return snap

    def update_from_mm(self, timestep: int, mm_metrics, mark_price: float) -> ExposureSnapshot:
        """
        Convenience wrapper: update from a MarketMakerMetrics object.
        """
        return self.update(
            timestep=timestep,
            inventory=mm_metrics.inventory,
            mark_price=mark_price,
            cash=mm_metrics.cash,
        )

    @property
    def snapshots(self) -> List[ExposureSnapshot]:
        return list(self._snapshots)

    def equity_series(self) -> List[float]:
        return [s.equity for s in self._snapshots]

    def notional_series(self) -> List[float]:
        return [s.notional_exposure for s in self._snapshots]

    def leverage_series(self) -> List[float]:
        return [s.leverage for s in self._snapshots]

    def summarize(self) -> PortfolioRiskSummary:
        """Compute aggregate risk metrics over all recorded snapshots."""
        if not self._snapshots:
            raise RuntimeError("No snapshots recorded yet.")

        notionals  = self.notional_series()
        leverages  = self.leverage_series()
        equities   = self.equity_series()
        inventories = [s.inventory for s in self._snapshots]
        concs      = [s.inv_concentration for s in self._snapshots]

        # Equity drawdown
        peak = equities[0]
        max_dd = 0.0
        for e in equities[1:]:
            peak = max(peak, e)
            max_dd = max(max_dd, peak - e)

        n = len(self._snapshots)
        overleveraged = sum(1 for lev in leverages if lev > self._leverage_threshold) / n

        return PortfolioRiskSummary(
            n_steps=n,
            max_notional=max(notionals),
            mean_notional=sum(notionals) / n,
            max_leverage=max(leverages),
            mean_leverage=sum(leverages) / n,
            max_inventory=max(inventories),
            min_inventory=min(inventories),
            max_concentration=max(concs),
            time_overleveraged=overleveraged,
            equity_drawdown=max_dd,
            final_equity=equities[-1],
        )

    def reset(self) -> None:
        self._snapshots.clear()
        self._max_abs_inv = 1e-9
