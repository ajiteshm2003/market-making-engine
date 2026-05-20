"""
examples/demo_stock_screener.py
---------------------------------
Regime-Aware Stock Screener Demo

Downloads 1 year of price data for the default universe, computes
volatility regime, trend, and risk factors, ranks tickers, and saves
a CSV + Markdown report.

Run:
    python examples/demo_stock_screener.py

Output files:
    reports/stock_screen_report.csv
    reports/stock_screen_report.md

Requires:
    pip install yfinance

IMPORTANT: This is not financial advice. Scores are based on historical
patterns and do not predict future performance. All trading involves
risk of loss.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

from src.screener import (
    MarketDataFetcher, FactorEngine, Scorer,
    save_csv, save_markdown, print_terminal_table,
    DEFAULT_UNIVERSE,
)

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", ".data_cache")


def divider(t=""):
    print(f"\n{'═'*65}\n  {t}\n{'═'*65}" if t else "\n" + "─"*65)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fetch data
# ─────────────────────────────────────────────────────────────────────────────

divider("1. FETCHING MARKET DATA")

print(f"  Universe: {len(DEFAULT_UNIVERSE)} tickers")
print(f"  {DEFAULT_UNIVERSE}")
print(f"  Lookback: 365 calendar days")
print(f"  Cache: {CACHE_DIR}")
print()

fetcher = MarketDataFetcher(
    lookback_days=365,
    cache_dir=CACHE_DIR,
    min_trading_days=60,
    min_adv_usd=10_000_000,
    request_delay=0.3,
)

t0 = time.time()
fetch_result = fetcher.fetch(DEFAULT_UNIVERSE)
elapsed = time.time() - t0

print(f"  Completed in {elapsed:.1f}s")
fetch_result.print_summary()

if not fetch_result.successful:
    print("  No data successfully fetched. Check internet connection.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute factors
# ─────────────────────────────────────────────────────────────────────────────

divider("2. COMPUTING FACTORS")

engine  = FactorEngine()
factors = engine.compute_batch(fetch_result.data)

print(f"  Computed factors for {len(factors)} tickers")
print()
for ticker, f in sorted(factors.items()):
    print(f"  {f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Score and rank
# ─────────────────────────────────────────────────────────────────────────────

divider("3. SCORING AND RANKING")

scorer = Scorer()
scored = scorer.score_all(factors)

eligible = [s for s in scored if not s.is_avoid and s.total_score >= 65]
avoid    = [s for s in scored if s.is_avoid]

print(f"  Total scored   : {len(scored)}")
print(f"  Eligible (≥65) : {len(eligible)}")
print(f"  Avoid list     : {len(avoid)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Terminal output
# ─────────────────────────────────────────────────────────────────────────────

divider("4. TOP 10 RANKED TICKERS")
print_terminal_table(scored, top_n=10, show_avoid=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save reports
# ─────────────────────────────────────────────────────────────────────────────

divider("5. SAVING REPORTS")

os.makedirs(REPORT_DIR, exist_ok=True)
csv_path = os.path.join(REPORT_DIR, "stock_screen_report.csv")
md_path  = os.path.join(REPORT_DIR, "stock_screen_report.md")

save_csv(scored, csv_path)
print(f"  CSV saved  → {csv_path}")

save_markdown(scored, md_path, test_portfolio_size=1_000.0)
print(f"  Markdown   → {md_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Regime summary
# ─────────────────────────────────────────────────────────────────────────────

divider("6. REGIME DISTRIBUTION")

from collections import Counter
regime_dist = Counter(s.regime_label for s in scored)
for label, count in sorted(regime_dist.items()):
    pct = 100 * count / len(scored)
    bar = "█" * int(pct / 5)
    print(f"  {label:<8} {count:>3} ({pct:4.0f}%) {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Risk notes
# ─────────────────────────────────────────────────────────────────────────────

divider("7. RISK NOTES")

if avoid:
    print(f"  AVOID ({len(avoid)} tickers):")
    for s in avoid:
        print(f"    {s.ticker}: {s.reason}")

print(f"""
  SCORING METHODOLOGY:
    Trend score     (30%): 20d+60d returns, price vs MA20/MA50
    Regime score    (25%): volatility regime label + percentile
    Risk score      (25%): drawdown + VaR95 + ES95 tail ratio
    Liquidity score (10%): average dollar volume + stability
    Quality score   (10%): return / volatility ratio (Sharpe-proxy)

  REGIME THRESHOLDS (annualised realized vol):
    LOW      < 12%   |   MEDIUM  12–25%   |   HIGH  25–45%   |   EXTREME > 45%

  ⚠ NOT FINANCIAL ADVICE. For educational/experimental use only.
    Historical screening does not guarantee future returns.
    All trading involves risk of loss.
""")

divider("DONE")
print(f"  Reports saved to: {REPORT_DIR}")
print(f"  Run `cat {md_path}` to read the full report.")
