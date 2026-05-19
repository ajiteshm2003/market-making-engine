"""
examples/demo_risk_engine.py
-----------------------------
Demonstrates the VaR / Expected Shortfall / Portfolio Exposure engine.

Run:
    python examples/demo_risk_engine.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
from src.risk import (
    historical_var, parametric_var, expected_shortfall,
    rolling_var, pnl_distribution_stats,
    PortfolioExposureTracker,
)
from src.agents import NoiseTrader, InformedTrader
from src.strategies import AvellanedaStoikovMarketMaker, ASConfig
from src.simulation import MarketSimulation, FairValueConfig
from src.models import VolatilityConfig

def divider(t=""): print(f"\n{'═'*60}\n  {t}\n{'═'*60}" if t else "\n"+"─"*60)

SEED = 42

# ── 1. Generate P&L series from a live simulation ────────────────────────────
divider("1. RUNNING SIMULATION")

asmm = AvellanedaStoikovMarketMaker("ASMM",
    config=ASConfig(gamma=0.1, sigma_config=VolatilityConfig(window=25)),
    quote_size=5.0)
noise = [NoiseTrader(f"NT{i}", activity_rate=0.6, random_seed=SEED+i) for i in range(3)]
informed = [InformedTrader("IT1", signal_threshold=0.06, aggression=0.8,
                            activity_rate=0.65, random_seed=SEED+100)]

sim = MarketSimulation(
    agents=noise + informed + [asmm],
    n_steps=600,
    fair_value_config=FairValueConfig(volatility=0.05, jump_prob=0.05, jump_std=1.5),
    random_seed=SEED,
)
result = sim.run()
pnl = asmm.mm_metrics.pnl_history
pnl_returns = [pnl[i] - pnl[i-1] for i in range(1, len(pnl))]
print(f"  Simulation complete: {len(pnl)} steps, {asmm.mm_metrics.fills_as_maker} fills")

# ── 2. VaR ────────────────────────────────────────────────────────────────────
divider("2. VALUE-AT-RISK")

for conf in [0.90, 0.95, 0.99]:
    h = historical_var(pnl_returns, conf)
    p = parametric_var(pnl_returns, conf)
    print(f"  {conf:.0%}  Historical: {h.var:>8.4f}  Parametric: {p.var:>8.4f}")

# ── 3. Expected Shortfall ─────────────────────────────────────────────────────
divider("3. EXPECTED SHORTFALL (CVaR)")

for conf in [0.90, 0.95, 0.99]:
    es = expected_shortfall(pnl_returns, conf)
    print(f"  {conf:.0%}  VaR={es.var:>8.4f}  ES={es.es:>8.4f}  "
          f"tail_obs={es.tail_obs:>4}/{es.n_obs}")

# ── 4. Rolling VaR ────────────────────────────────────────────────────────────
divider("4. ROLLING 95% VAR (window=100)")

rv = rolling_var(pnl_returns, window=100, confidence=0.95)
non_none = [v for v in rv if v is not None]
if non_none:
    print(f"  Min rolling VaR : {min(non_none):.4f}")
    print(f"  Mean rolling VaR: {sum(non_none)/len(non_none):.4f}")
    print(f"  Max rolling VaR : {max(non_none):.4f}")

# ── 5. Distribution stats ─────────────────────────────────────────────────────
divider("5. P&L DISTRIBUTION STATISTICS")
stats = pnl_distribution_stats(pnl_returns)
print(stats)

# ── 6. Portfolio exposure ─────────────────────────────────────────────────────
divider("6. PORTFOLIO EXPOSURE TRACKER")

tracker = PortfolioExposureTracker(initial_cash=100_000.0)
mkt_df = result.metrics.to_dataframe()
m = asmm.mm_metrics

for t in mkt_df.index:
    mid = mkt_df.loc[t, "midprice"]
    if mid is not None and mid == mid:  # not NaN
        # Approximate cash from PnL history
        step_pnl = pnl[t-1] if t <= len(pnl) else 0.0
        cash = 100_000.0 + step_pnl - m.inventory * mid
        tracker.update(int(t), m.inventory, mid, cash)

summary = tracker.summarize()
print(summary)

divider("DONE")
print("  Risk engine demo complete.")
