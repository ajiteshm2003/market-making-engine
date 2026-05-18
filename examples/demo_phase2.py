"""
examples/demo_phase2.py
-----------------------
Phase 2 Demo — Simulated Market Participants

Run locally:
    python examples/demo_phase2.py

Run in Google Colab:
    !git clone https://github.com/YOUR/market_making_engine
    %cd market_making_engine
    !pip install -r requirements.txt
    %run examples/demo_phase2.py

What this demonstrates
----------------------
1. Fair value process (random walk + jumps)
2. 3 NoiseTraders generating background flow
3. 2 InformedTraders trading on the fair value signal
4. Full simulation run (500 steps)
5. Time series analysis of:
   - Fair value vs midprice
   - Bid-ask spread over time
   - Order imbalance over time
   - Cumulative volume
   - Execution price histogram
   - Per-agent PnL and inventory
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # safe for Colab; switch to 'TkAgg' for local interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.agents import InformedTrader, NoiseTrader
from src.simulation import FairValueConfig, MarketSimulation
from src.exchange.trade_log import trades_to_dataframe, execution_summary

# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_STEPS = 500

def divider(title=""):
    print("\n" + "═" * 60)
    if title:
        print(f"  {title}")
        print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build agents
# ─────────────────────────────────────────────────────────────────────────────

divider("1. BUILDING AGENTS")

noise_traders = [
    NoiseTrader(
        agent_id=f"NT{i+1}",
        activity_rate=0.50,
        market_order_prob=0.20,
        order_size_mean=3.0,
        order_size_std=0.6,
        limit_offset_ticks=0.50,
        max_resting_orders=6,
        random_seed=RANDOM_SEED + i,
    )
    for i in range(3)
]

informed_traders = [
    InformedTrader(
        agent_id="IT1",
        signal_threshold=0.08,
        aggression=0.80,
        base_trade_size=4.0,
        max_inventory=40.0,
        activity_rate=0.65,
        random_seed=RANDOM_SEED + 100,
    ),
    InformedTrader(
        agent_id="IT2",
        signal_threshold=0.15,    # higher bar → trades less but larger
        aggression=0.60,
        base_trade_size=8.0,
        max_inventory=60.0,
        activity_rate=0.50,
        random_seed=RANDOM_SEED + 200,
    ),
]

all_agents = noise_traders + informed_traders

for a in all_agents:
    print(f"  {a}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Configure fair value process
# ─────────────────────────────────────────────────────────────────────────────

divider("2. FAIR VALUE PROCESS")

fv_config = FairValueConfig(
    initial_price=100.0,
    drift=0.0,
    volatility=0.04,      # ~4 ticks per step standard deviation
    jump_prob=0.025,      # jump roughly every 40 steps
    jump_std=0.40,        # jump size std
    min_price=1.0,
)
print(f"  Initial price : {fv_config.initial_price}")
print(f"  Volatility    : {fv_config.volatility} per step")
print(f"  Jump prob     : {fv_config.jump_prob:.1%} per step")
print(f"  Jump std      : {fv_config.jump_std}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run simulation
# ─────────────────────────────────────────────────────────────────────────────

divider(f"3. RUNNING SIMULATION ({N_STEPS} steps, {len(all_agents)} agents)")

sim = MarketSimulation(
    agents=all_agents,
    n_steps=N_STEPS,
    fair_value_config=fv_config,
    depth_levels=5,
    random_seed=RANDOM_SEED,
)

result = sim.run()
print(result.summary())

# ─────────────────────────────────────────────────────────────────────────────
# 4. Extract data
# ─────────────────────────────────────────────────────────────────────────────

divider("4. DATA EXTRACTION")

metrics_df = result.metrics.to_dataframe()
trades_df = trades_to_dataframe(result.engine.trade_log)

print(f"\n  Metrics DataFrame shape : {metrics_df.shape}")
print(f"  Trades DataFrame shape  : {trades_df.shape}")
print(f"\n  Jump steps (first 10)   : {result.jump_steps[:10]}")

print("\n  Metrics sample (last 5 rows):")
print(metrics_df[["fair_value", "midprice", "spread", "order_imbalance",
                   "trades_this_step", "cumulative_trades"]].tail(5).to_string())

if not trades_df.empty:
    print("\n  Execution summary:")
    summary = execution_summary(result.engine.trade_log)
    for k, v in summary.items():
        print(f"    {k:<26} {v}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-agent metrics
# ─────────────────────────────────────────────────────────────────────────────

divider("5. AGENT METRICS")

agent_rows = []
for agent in result.agents:
    m = agent.metrics
    agent_rows.append({
        "agent_id": agent.agent_id,
        "type": agent.__class__.__name__,
        "inventory": round(m.inventory, 4),
        "cash_delta": round(m.cash - 100_000.0, 4),
        "unrealized_pnl": round(m.unrealized_pnl, 4),
        "total_pnl": round(m.total_pnl, 4),
        "trades": m.trades_executed,
        "volume": round(m.volume_traded, 4),
        "orders_submitted": m.orders_submitted,
    })

agent_df = pd.DataFrame(agent_rows).set_index("agent_id")
print(agent_df.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────────────────────

divider("6. GENERATING PLOTS")

steps = metrics_df.index
fv = metrics_df["fair_value"]
mid = metrics_df["midprice"]
spread = metrics_df["spread"]
imb = metrics_df["order_imbalance"]
cum_vol = metrics_df["cumulative_volume"]
trades_per_step = metrics_df["trades_this_step"]

fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    f"Phase 2 — Market Simulation  ({N_STEPS} steps, "
    f"{len(noise_traders)} noise traders, {len(informed_traders)} informed traders)",
    fontsize=13,
    fontweight="bold",
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: Fair value vs midprice ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])  # full width
ax1.plot(steps, fv, color="#e74c3c", linewidth=1.5, label="Fair Value (latent)", alpha=0.9)
ax1.plot(steps, mid, color="#2980b9", linewidth=1.0, label="Midprice (observed)", alpha=0.85)

# Mark jumps
for jt in result.jump_steps:
    if jt <= N_STEPS:
        ax1.axvline(jt, color="#e74c3c", alpha=0.15, linewidth=0.7)

ax1.set_title("Fair Value vs Midprice (red verticals = fair value jumps)")
ax1.set_ylabel("Price")
ax1.set_xlabel("Timestep")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.25)

# ── Panel 2: Spread over time ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(steps, spread, color="#8e44ad", linewidth=0.9, alpha=0.85)
ax2.fill_between(steps, spread.fillna(0), alpha=0.2, color="#8e44ad")
ax2.axhline(spread.mean(), color="#8e44ad", linestyle="--", linewidth=1, label=f"Mean={spread.mean():.4f}")
ax2.set_title("Bid-Ask Spread")
ax2.set_ylabel("Spread (price units)")
ax2.set_xlabel("Timestep")
ax2.legend()
ax2.grid(alpha=0.25)

# ── Panel 3: Order imbalance ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(steps, imb, color="#27ae60", linewidth=0.8, alpha=0.85)
ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax3.fill_between(steps, imb.fillna(0), 0, where=(imb.fillna(0) > 0),
                 alpha=0.25, color="#27ae60", label="Buy pressure")
ax3.fill_between(steps, imb.fillna(0), 0, where=(imb.fillna(0) < 0),
                 alpha=0.25, color="#e74c3c", label="Sell pressure")
ax3.set_title("Order Imbalance (±1)")
ax3.set_ylabel("Imbalance")
ax3.set_xlabel("Timestep")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.25)

# ── Panel 4: Cumulative volume ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(steps, cum_vol, color="#d35400", linewidth=1.2)
ax4.set_title("Cumulative Volume")
ax4.set_ylabel("Total Quantity Traded")
ax4.set_xlabel("Timestep")
ax4.grid(alpha=0.25)

# ── Panel 5: Trade execution price histogram ───────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
if not trades_df.empty:
    ax5.hist(trades_df["price"], bins=40, color="#2980b9", alpha=0.75, edgecolor="white")
    ax5.axvline(trades_df["price"].mean(), color="#e74c3c", linestyle="--",
                linewidth=1.5, label=f"Mean={trades_df['price'].mean():.3f}")
    ax5.set_title("Trade Price Distribution")
    ax5.set_xlabel("Execution Price")
    ax5.set_ylabel("Count")
    ax5.legend()
    ax5.grid(alpha=0.25, axis="y")
else:
    ax5.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax5.transAxes)
    ax5.set_title("Trade Price Distribution")

plt.savefig("phase2_demo_output.png", dpi=150, bbox_inches="tight")
print("  Plot saved → phase2_demo_output.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Microstructure observations (educational)
# ─────────────────────────────────────────────────────────────────────────────

divider("7. MICROSTRUCTURE OBSERVATIONS")

mean_spread = spread.dropna().mean()
mean_imb = imb.dropna().mean()
price_range = (fv.max() - fv.min())
tracking_error = (fv - mid).dropna().abs().mean()

jump_count = len(result.jump_steps)
total_trades = int(metrics_df["cumulative_trades"].iloc[-1])
total_vol = metrics_df["cumulative_volume"].iloc[-1]

print(f"""
  Fair value range    : {fv.min():.3f} → {fv.max():.3f}  (Δ={price_range:.3f})
  Mean spread         : {mean_spread:.4f}
  Mean |tracking err| : {tracking_error:.4f}  (how well mid tracks fair value)
  Mean imbalance      : {mean_imb:.4f}  (+ = buy-heavy book)
  Fair value jumps    : {jump_count}
  Total executions    : {total_trades}
  Total volume        : {total_vol:.2f}

  WHAT TO NOTICE:
  ──────────────
  • The midprice (blue) converges toward the fair value (red) over time.
    This is price discovery — informed traders push quotes toward true value.

  • After jump events (vertical red lines), midprice takes several steps
    to catch up. This lag is the informed trader's trading edge.

  • Spread widens when the book is thin or when volatility spikes.
    A wider spread = higher cost to trade = higher market maker protection.

  • Order imbalance flips sign around jumps — informed traders one-sidedly
    consume liquidity in the direction of the signal.

  Phase 3 will add a Market Maker who must quote into this flow and
  manage inventory risk against the informed traders above.
""")

divider("DONE — Phase 2 complete")
