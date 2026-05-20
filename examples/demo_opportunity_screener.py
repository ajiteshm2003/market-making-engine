"""
examples/demo_opportunity_screener.py
---------------------------------------
Opportunity Screener Demo — Multi-Universe Ranking

Screens three universes (Institutional, Emerging, Speculative) and generates:
- Terminal ranked tables per universe
- CSV and Markdown reports
- Scatter plots (score vs vol, drawdown vs return, regime distribution)

Run:
    python examples/demo_opportunity_screener.py

Output:
    reports/opportunity_screen.csv
    reports/opportunity_screen.md
    reports/figures/score_vs_vol.png
    reports/figures/drawdown_vs_return.png
    reports/figures/regime_distribution.png
    reports/figures/top_scores.png

IMPORTANT: NOT FINANCIAL ADVICE. For educational/experimental use only.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

from src.screener import (
    MarketDataFetcher,
    save_opportunity_csv, save_opportunity_markdown, print_opportunity_table,
)
from src.screener.universe import INSTITUTIONAL, EMERGING, SPECULATIVE, ALL_UNIVERSES
from src.screener.opportunity import OpportunityPipeline, OpportunityResult
from src.models.regime import VolatilityRegime

REPORT_DIR  = os.path.join(os.path.dirname(__file__), "..", "reports")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "..", ".data_cache")

REGIME_COLORS = {
    "LOW":     "#2ecc71",
    "MEDIUM":  "#3498db",
    "HIGH":    "#e67e22",
    "EXTREME": "#e74c3c",
    "UNKNOWN": "#95a5a6",
}

UNIVERSE_COLORS = {
    "institutional": "#3498db",
    "emerging":      "#27ae60",
    "speculative":   "#e67e22",
}


def divider(t=""):
    print(f"\n{'═'*68}\n  {t}\n{'═'*68}" if t else "\n" + "─" * 68)


os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Fetch data for all universes
# ─────────────────────────────────────────────────────────────────────────────

divider("1. FETCHING MARKET DATA")

fetcher = MarketDataFetcher(
    lookback_days=365,
    cache_dir=CACHE_DIR,
    min_trading_days=60,
    min_adv_usd=2_000_000,   # lower floor for emerging/speculative
    request_delay=0.35,
)

all_results: dict = {}
fetch_by_universe: dict = {}

for universe_spec in [INSTITUTIONAL, EMERGING, SPECULATIVE]:
    print(f"\n  Fetching {universe_spec.label} ({len(universe_spec.tickers)} tickers)...")
    result = fetcher.fetch(universe_spec.tickers)
    result.print_summary()
    fetch_by_universe[universe_spec.name] = result

# ─────────────────────────────────────────────────────────────────────────────
# 2. Run opportunity pipeline per universe
# ─────────────────────────────────────────────────────────────────────────────

divider("2. COMPUTING OPPORTUNITY SCORES")

results_by_universe: dict = {}

for spec in [INSTITUTIONAL, EMERGING, SPECULATIVE]:
    fetch_data = fetch_by_universe[spec.name].data
    if not fetch_data:
        print(f"  {spec.label}: no data — skipping")
        continue

    pipeline = OpportunityPipeline(spec)
    results  = pipeline.run(fetch_data)
    results_by_universe[spec.name] = results

    eligible = [r for r in results if not r.is_avoid]
    avoid    = [r for r in results if r.is_avoid]
    print(f"\n  {spec.label}: {len(results)} scored, "
          f"{len(eligible)} eligible, {len(avoid)} avoid")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Print ranked tables
# ─────────────────────────────────────────────────────────────────────────────

divider("3. RANKED TABLES")

for spec in [INSTITUTIONAL, EMERGING, SPECULATIVE]:
    if spec.name not in results_by_universe:
        continue
    print_opportunity_table(
        results_by_universe[spec.name],
        universe_label=spec.label,
        top_n=8,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-universe highlights
# ─────────────────────────────────────────────────────────────────────────────

divider("4. CROSS-UNIVERSE INSIGHTS")

all_scored = [r for results in results_by_universe.values() for r in results]

if all_scored:
    # Best emerging leaders
    emerging_results = results_by_universe.get("emerging", [])
    if emerging_results:
        print("\n  TOP EMERGING LEADERS:")
        for r in [r for r in emerging_results if not r.is_avoid][:5]:
            print(f"  {r.ticker:<8} score={r.opportunity_score:.0f}  "
                  f"regime={r.regime_label:<8}  {r.insight[:60]}")

    # Highest risk-adjusted momentum (positive momentum slope, constructive vol)
    best_accel = sorted(
        [r for r in all_scored if r.momentum_slope and r.momentum_slope > 0
         and r.vol_is_constructive and not r.is_avoid],
        key=lambda r: r.momentum_slope, reverse=True,
    )
    if best_accel:
        print("\n  HIGHEST TREND ACCELERATION:")
        for r in best_accel[:5]:
            print(f"  {r.ticker:<8} momentum_slope={r.momentum_slope:+.2f}  "
                  f"regime={r.regime_label}  ret20={r.ret_20d:+.1%}" if r.ret_20d else "")

    # Dangerous names
    dangerous = [r for r in all_scored if r.high_risk]
    if dangerous:
        print("\n  MOST DANGEROUS NAMES (high risk):")
        for r in sorted(dangerous, key=lambda r: r.opportunity_score, reverse=True)[:5]:
            print(f"  {r.ticker:<8} score={r.opportunity_score:.0f}  "
                  f"regime={r.regime_label}  drawdown={r.drawdown:.0%}" if r.drawdown else "")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save reports
# ─────────────────────────────────────────────────────────────────────────────

divider("5. SAVING REPORTS")

if results_by_universe:
    csv_path = os.path.join(REPORT_DIR, "opportunity_screen.csv")
    md_path  = os.path.join(REPORT_DIR, "opportunity_screen.md")

    save_opportunity_csv(results_by_universe, csv_path)
    print(f"  CSV → {csv_path}")

    save_opportunity_markdown(results_by_universe, md_path)
    print(f"  Markdown → {md_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

divider("6. GENERATING PLOTS")

if not all_scored:
    print("  No data to plot.")
else:
    # ── Plot 1: Score vs Volatility scatter ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for spec in [INSTITUTIONAL, EMERGING, SPECULATIVE]:
        subset = results_by_universe.get(spec.name, [])
        xs = [r.vol_20d * 100 for r in subset if r.vol_20d]
        ys = [r.opportunity_score for r in subset if r.vol_20d]
        labels = [r.ticker for r in subset if r.vol_20d]
        color = UNIVERSE_COLORS[spec.name]
        ax.scatter(xs, ys, c=color, alpha=0.75, s=60, label=spec.label, zorder=3)
        for x, y, lbl in zip(xs[:8], ys[:8], labels[:8]):
            ax.annotate(lbl, (x, y), fontsize=6, alpha=0.8,
                        xytext=(2, 2), textcoords="offset points")

    ax.axhline(65, color="green",  linestyle="--", alpha=0.5, linewidth=0.8, label="Eligible threshold")
    ax.axhline(30, color="red",    linestyle="--", alpha=0.5, linewidth=0.8, label="Avoid threshold")
    ax.axvline(25, color="orange", linestyle=":",  alpha=0.5, linewidth=0.8, label="HIGH vol boundary")
    ax.axvline(45, color="red",    linestyle=":",  alpha=0.5, linewidth=0.8, label="EXTREME vol boundary")
    ax.set_xlabel("20d Realized Volatility (%, annualised)")
    ax.set_ylabel("Opportunity Score")
    ax.set_title("Opportunity Score vs Realized Volatility by Universe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, "score_vs_vol.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  {p1}")

    # ── Plot 2: Drawdown vs Return scatter ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for spec in [INSTITUTIONAL, EMERGING, SPECULATIVE]:
        subset = results_by_universe.get(spec.name, [])
        xs = [r.ret_60d * 100 for r in subset if r.ret_60d and r.drawdown]
        ys = [r.drawdown * 100 for r in subset if r.ret_60d and r.drawdown]
        tks = [r.ticker for r in subset if r.ret_60d and r.drawdown]
        scores = [r.opportunity_score for r in subset if r.ret_60d and r.drawdown]
        color = UNIVERSE_COLORS[spec.name]
        sc = ax.scatter(xs, ys, c=scores, cmap="RdYlGn", vmin=0, vmax=100,
                        alpha=0.8, s=60, label=spec.label, zorder=3)
        for x, y, lbl in zip(xs[:6], ys[:6], tks[:6]):
            ax.annotate(lbl, (x, y), fontsize=6, alpha=0.8,
                        xytext=(2, 2), textcoords="offset points")

    plt.colorbar(sc, ax=ax, label="Opportunity Score")
    ax.set_xlabel("60d Log Return (%)")
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Drawdown vs 60d Return (color = Opportunity Score)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, "drawdown_vs_return.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"  {p2}")

    # ── Plot 3: Regime distribution by universe ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    regime_order = ["LOW", "MEDIUM", "HIGH", "EXTREME"]

    for ax, spec in zip(axes, [INSTITUTIONAL, EMERGING, SPECULATIVE]):
        subset = results_by_universe.get(spec.name, [])
        counts = {r: 0 for r in regime_order}
        for res in subset:
            counts[res.regime_label] = counts.get(res.regime_label, 0) + 1
        colors = [REGIME_COLORS[r] for r in regime_order]
        bars = ax.bar(regime_order, [counts[r] for r in regime_order], color=colors, alpha=0.85)
        ax.set_title(spec.label, fontsize=9)
        ax.set_ylabel("# Tickers")
        ax.set_xlabel("Regime")
        for bar, r in zip(bars, regime_order):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(int(h)),
                        ha="center", fontsize=8)

    plt.suptitle("Regime Distribution by Universe", fontsize=12)
    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, "regime_distribution.png")
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"  {p3}")

    # ── Plot 4: Top 10 scores bar chart ───────────────────────────────────
    top_all = sorted(all_scored, key=lambda r: r.opportunity_score, reverse=True)[:12]
    if top_all:
        fig, ax = plt.subplots(figsize=(12, 6))
        tickers = [r.ticker for r in top_all]
        scores  = [r.opportunity_score for r in top_all]
        colors  = [UNIVERSE_COLORS.get(r.universe, "#95a5a6") for r in top_all]
        bars = ax.barh(tickers[::-1], scores[::-1], color=colors[::-1], alpha=0.85)
        ax.axvline(65, color="green", linestyle="--", alpha=0.6, label="Eligible threshold")
        ax.axvline(30, color="red",   linestyle="--", alpha=0.6, label="Avoid threshold")
        ax.set_xlabel("Opportunity Score")
        ax.set_title("Top 12 Opportunity Scores (All Universes)")
        ax.legend(fontsize=8)
        patches = [mpatches.Patch(color=c, label=l)
                   for l, c in UNIVERSE_COLORS.items()]
        ax.legend(handles=patches + [
            mpatches.Patch(color="green", label="Eligible (≥65)"),
            mpatches.Patch(color="red",   label="Avoid (<30)"),
        ], fontsize=7, loc="lower right")
        ax.grid(alpha=0.2, axis="x")
        plt.tight_layout()
        p4 = os.path.join(FIGURES_DIR, "top_scores.png")
        plt.savefig(p4, dpi=150)
        plt.close()
        print(f"  {p4}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary
# ─────────────────────────────────────────────────────────────────────────────

divider("7. METHODOLOGY SUMMARY")

print(f"""
  FACTOR WEIGHTS (Opportunity Score):
    Trend Quality       : 25%  — sustained positive returns vs MAs
    Trend Acceleration  : 20%  — 20d momentum improving vs 60d; vol expansion
    Regime Quality      : 20%  — vol regime + constructiveness
    Risk Control        : 15%  — drawdown + VaR + ES tail ratio
    Liquidity           : 10%  — average dollar volume + vol stability
    Opportunity Bonus   : 10%  — market cap discovery potential

  UNIVERSE DESIGN:
    Institutional  → favour stability; heavy vol penalty
    Emerging       → accept moderate vol; reward acceleration
    Speculative    → allow high vol; require coherent structure

  ⚠  NOT FINANCIAL ADVICE.
     Scores reflect historical factor patterns, not future returns.
     Always verify findings with fundamental and market context.
""")

divider("DONE")
print(f"  Reports → {REPORT_DIR}")
print(f"  Figures → {FIGURES_DIR}")
