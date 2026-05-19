"""
src/risk/stress.py
-------------------
Stress Testing Engine

Defines standardized stress scenarios and a runner that applies each scenario
to all four market-making strategies, collecting risk metrics.

Stress testing philosophy
--------------------------
Backtests and simulation parameters are calibrated to "normal" market
conditions.  Stress tests ask: "What happens in the tails?"

Each scenario modifies the simulation parameters to represent a specific
adverse market condition.  The goal is not to find parameters that maximize
strategy PnL, but to expose strategies to conditions under which their
weaknesses are most apparent.

Scenarios implemented
----------------------
FLASH_CRASH:
    Sudden large downward jump in fair value followed by partial recovery.
    Tests: inventory management, adverse selection from directional informed flow,
    ability to reduce quotes or skew during dislocated price action.

VOLATILITY_SPIKE:
    Sustained high diffusion volatility with frequent jumps.
    Tests: spread widening response, VaR exceedances, drawdown magnitude.

LIQUIDITY_DROUGHT:
    Noise traders withdraw (low activity).  Only informed traders and market
    maker interact.  Tests: adverse selection in the absence of noise flow.

INFORMED_FLOW_ATTACK:
    High aggression, low threshold informed traders with large trade sizes.
    Tests: the market maker's ability to detect and defend against directional flow.

SPREAD_COLLAPSE:
    External competitive pressure (simulated by high noise trader market-order
    activity at narrow effective spreads).  Tests: fill rate and PnL under
    extremely competitive quoting environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..agents import NoiseTrader, InformedTrader
from ..simulation import MarketSimulation, FairValueConfig
from ..strategies import (
    NaiveMarketMaker,
    InventoryAwareMarketMaker,
    AvellanedaStoikovMarketMaker,
    ASConfig,
    RegimeAwareAvellanedaStoikovMarketMaker,
    RegimeAwareASConfig,
)
from ..models import VolatilityConfig
from .var import historical_var, expected_shortfall, VaRResult, ESResult


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """
    Defines a stress test scenario.

    Parameters
    ----------
    name              : short identifier
    description       : human-readable description
    n_steps           : simulation length
    fv_volatility     : fair value diffusion σ per step
    jump_prob         : jump probability per step
    jump_std          : jump size standard deviation
    noise_activity    : noise trader activity rate
    n_noise_traders   : number of noise traders
    informed_threshold: informed trader signal threshold (lower = more trades)
    informed_aggression: informed trader aggression (1.0 = all market orders)
    informed_size     : base trade size for informed traders
    n_informed        : number of informed traders
    seed              : random seed for reproducibility
    """
    name:                str
    description:         str
    n_steps:             int   = 400
    fv_volatility:       float = 0.05
    jump_prob:           float = 0.04
    jump_std:            float = 0.5
    noise_activity:      float = 0.55
    n_noise_traders:     int   = 3
    informed_threshold:  float = 0.06
    informed_aggression: float = 0.80
    informed_size:       float = 5.0
    n_informed:          int   = 2
    seed:                int   = 42


# Pre-built scenarios
SCENARIOS: Dict[str, StressScenario] = {
    "baseline": StressScenario(
        name="baseline",
        description="Standard simulation parameters. Reference point for all comparisons.",
        fv_volatility=0.05, jump_prob=0.04, jump_std=0.5,
        noise_activity=0.55, n_noise_traders=3,
        informed_threshold=0.06, informed_aggression=0.80, informed_size=5.0,
        n_informed=2,
    ),
    "flash_crash": StressScenario(
        name="flash_crash",
        description="Single large downward jump (5σ) followed by elevated volatility. "
                    "Tests adverse selection defense and drawdown protection.",
        fv_volatility=0.05, jump_prob=0.08, jump_std=3.0,
        noise_activity=0.60, n_noise_traders=3,
        informed_threshold=0.04, informed_aggression=0.95, informed_size=8.0,
        n_informed=3,
    ),
    "volatility_spike": StressScenario(
        name="volatility_spike",
        description="Sustained 2× normal diffusion volatility with frequent jumps. "
                    "Tests spread-widening response and VaR exceedances.",
        fv_volatility=0.10, jump_prob=0.10, jump_std=1.5,
        noise_activity=0.50, n_noise_traders=3,
        informed_threshold=0.08, informed_aggression=0.75, informed_size=6.0,
        n_informed=2,
    ),
    "liquidity_drought": StressScenario(
        name="liquidity_drought",
        description="Noise traders withdraw (20% activity). Only informed flow remains. "
                    "Tests adverse selection in absence of uninformed volume.",
        fv_volatility=0.05, jump_prob=0.04, jump_std=0.5,
        noise_activity=0.20, n_noise_traders=2,
        informed_threshold=0.05, informed_aggression=0.90, informed_size=7.0,
        n_informed=2,
    ),
    "informed_flow_attack": StressScenario(
        name="informed_flow_attack",
        description="High-aggression informed traders with low signal threshold and large size. "
                    "Maximum adverse selection pressure.",
        fv_volatility=0.06, jump_prob=0.06, jump_std=1.0,
        noise_activity=0.40, n_noise_traders=2,
        informed_threshold=0.02, informed_aggression=0.99, informed_size=12.0,
        n_informed=4,
    ),
    "spread_collapse": StressScenario(
        name="spread_collapse",
        description="High noise trader market-order activity simulating a highly competitive "
                    "quoting environment. Tests fill rate and PnL compression.",
        fv_volatility=0.03, jump_prob=0.02, jump_std=0.3,
        noise_activity=0.90, n_noise_traders=5,
        informed_threshold=0.15, informed_aggression=0.50, informed_size=3.0,
        n_informed=1,
    ),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StressTestResult:
    """
    Results from running a single stress scenario against a single strategy.

    Attributes
    ----------
    scenario_name  : name of the scenario
    strategy_name  : name of the strategy
    total_pnl      : final mark-to-market P&L
    max_drawdown   : peak-to-trough maximum PnL decline
    var_95         : 95% historical VaR
    var_99         : 99% historical VaR
    es_95          : 95% Expected Shortfall
    inv_variance   : variance of inventory over time
    worst_step_loss: largest single-step P&L loss
    final_inventory: inventory at end of simulation
    n_fills        : total maker fills
    runtime_secs   : wall-clock time to run
    """
    scenario_name:   str
    strategy_name:   str
    total_pnl:       float
    max_drawdown:    float
    var_95:          float
    var_99:          float
    es_95:           float
    inv_variance:    float
    worst_step_loss: float
    final_inventory: float
    n_fills:         int
    runtime_secs:    float

    def __str__(self) -> str:
        return (
            f"[{self.scenario_name}/{self.strategy_name}]  "
            f"PnL={self.total_pnl:+.2f}  DD={self.max_drawdown:.2f}  "
            f"VaR95={self.var_95:.4f}  ES95={self.es_95:.4f}  "
            f"InvVar={self.inv_variance:.2f}  Fills={self.n_fills}"
        )


# ---------------------------------------------------------------------------
# Stress test runner
# ---------------------------------------------------------------------------

class StressTestRunner:
    """
    Runs multiple stress scenarios against multiple market-making strategies.

    Usage
    -----
    runner = StressTestRunner()
    results = runner.run_all()
    runner.print_summary(results)
    """

    def __init__(
        self,
        n_steps: int = 400,
        base_gamma: float = 0.10,
        base_quote_size: float = 5.0,
        var_window: int = 50,
    ) -> None:
        self.n_steps = n_steps
        self.base_gamma = base_gamma
        self.base_quote_size = base_quote_size
        self.var_window = var_window

    def _build_strategies(self, seed: int):
        """Build all four market-making strategy instances."""
        return {
            "NMM":   NaiveMarketMaker("NMM",   half_spread=0.06,
                                      quote_size=self.base_quote_size),
            "IAMM":  InventoryAwareMarketMaker("IAMM", half_spread=0.06,
                                               inventory_skew_factor=0.012,
                                               max_inventory=40.0,
                                               spread_widening=0.60,
                                               quote_size=self.base_quote_size),
            "ASMM":  AvellanedaStoikovMarketMaker("ASMM",
                         config=ASConfig(gamma=self.base_gamma,
                                         sigma_config=VolatilityConfig(window=20),
                                         horizon_steps=1.0),
                         quote_size=self.base_quote_size),
            "RASMM": RegimeAwareAvellanedaStoikovMarketMaker("RASMM",
                         ra_config=RegimeAwareASConfig(
                             base_config=ASConfig(gamma=self.base_gamma,
                                                  sigma_config=VolatilityConfig(window=20),
                                                  horizon_steps=1.0),
                             base_quote_size=self.base_quote_size,
                             base_max_inventory=50.0,
                         )),
        }

    def run_scenario(self, scenario: StressScenario) -> Dict[str, StressTestResult]:
        """Run a single scenario against all four strategies. Returns dict of results."""
        strategies = self._build_strategies(scenario.seed)

        noise = [
            NoiseTrader(f"NT{i}", activity_rate=scenario.noise_activity,
                        market_order_prob=0.25, random_seed=scenario.seed + i)
            for i in range(scenario.n_noise_traders)
        ]
        informed = [
            InformedTrader(f"IT{i}",
                           signal_threshold=scenario.informed_threshold,
                           aggression=scenario.informed_aggression,
                           base_trade_size=scenario.informed_size,
                           activity_rate=0.65,
                           random_seed=scenario.seed + 100 + i)
            for i in range(scenario.n_informed)
        ]

        fv_config = FairValueConfig(
            initial_price=100.0,
            volatility=scenario.fv_volatility,
            jump_prob=scenario.jump_prob,
            jump_std=scenario.jump_std,
        )

        agents = noise + informed + list(strategies.values())
        t0 = time.perf_counter()
        sim = MarketSimulation(
            agents=agents,
            n_steps=scenario.n_steps,
            fair_value_config=fv_config,
            random_seed=scenario.seed,
        )
        sim.run()
        elapsed = time.perf_counter() - t0

        results = {}
        for name, mm in strategies.items():
            pnl_hist = mm.mm_metrics.pnl_history
            inv_hist = mm.mm_metrics.inventory_history

            # Max drawdown
            peak = pnl_hist[0] if pnl_hist else 0.0
            max_dd = 0.0
            for v in pnl_hist:
                peak = max(peak, v)
                max_dd = max(max_dd, peak - v)

            # Per-step PnL returns
            pnl_returns = [pnl_hist[i] - pnl_hist[i - 1]
                           for i in range(1, len(pnl_hist))]
            worst_loss = min(pnl_returns) if pnl_returns else 0.0

            # VaR and ES (use returns for risk metrics)
            try:
                var95 = historical_var(pnl_returns, 0.95).var if len(pnl_returns) >= 5 else 0.0
                var99 = historical_var(pnl_returns, 0.99).var if len(pnl_returns) >= 10 else 0.0
                es95  = expected_shortfall(pnl_returns, 0.95).es if len(pnl_returns) >= 5 else 0.0
            except ValueError:
                var95 = var99 = es95 = 0.0

            # Inventory variance
            inv_var = 0.0
            if len(inv_hist) >= 2:
                n = len(inv_hist)
                mean_inv = sum(inv_hist) / n
                inv_var = sum((x - mean_inv) ** 2 for x in inv_hist) / (n - 1)

            results[name] = StressTestResult(
                scenario_name=scenario.name,
                strategy_name=name,
                total_pnl=round(mm.mm_metrics.total_pnl, 4),
                max_drawdown=round(max_dd, 4),
                var_95=round(var95, 6),
                var_99=round(var99, 6),
                es_95=round(es95, 6),
                inv_variance=round(inv_var, 4),
                worst_step_loss=round(worst_loss, 4),
                final_inventory=round(mm.mm_metrics.inventory, 4),
                n_fills=mm.mm_metrics.fills_as_maker,
                runtime_secs=round(elapsed, 4),
            )
        return results

    def run_all(
        self,
        scenario_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, StressTestResult]]:
        """
        Run all (or selected) scenarios.

        Returns
        -------
        dict[scenario_name, dict[strategy_name, StressTestResult]]
        """
        names = scenario_names or list(SCENARIOS.keys())
        all_results = {}
        for name in names:
            scenario = SCENARIOS[name]
            all_results[name] = self.run_scenario(scenario)
        return all_results

    @staticmethod
    def print_summary(results: Dict[str, Dict[str, StressTestResult]]) -> None:
        """Print a formatted comparison table across all scenarios and strategies."""
        strategies = ["NMM", "IAMM", "ASMM", "RASMM"]
        metrics = [
            ("total_pnl",       "PnL",      "+.2f"),
            ("max_drawdown",    "MaxDD",    ".2f"),
            ("var_95",          "VaR95",    ".4f"),
            ("es_95",           "ES95",     ".4f"),
            ("inv_variance",    "InvVar",   ".2f"),
            ("worst_step_loss", "WorstStep",".3f"),
        ]

        for scenario_name, strat_results in results.items():
            print(f"\n  ─── {scenario_name.upper()} ───")
            print(f"  {'Metric':<14}" + "".join(f"{s:>10}" for s in strategies))
            print(f"  {'─'*14}" + "─" * (10 * len(strategies)))
            for attr, label, fmt in metrics:
                row = f"  {label:<14}"
                for s in strategies:
                    val = getattr(strat_results.get(s), attr, float("nan"))
                    try:
                        formatted = f"{val:>10.4f}"
                    except (TypeError, ValueError):
                        formatted = f"{str(val):>10}"
                    row += formatted
                print(row)
