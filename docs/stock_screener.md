# Regime-Aware Stock Screener

A disciplined regime/risk/momentum screening tool that applies the same volatility regime detection and risk measurement philosophy used in the market making engine to publicly available equity and ETF data.

---

## What This Is

The screener is a **quantitative ranking tool**, not a prediction engine. It answers the question: *"Given what the market looked like over the past year, which instruments are currently in healthier regimes for monitoring and small-scale experimental validation?"*

It does not predict future prices. It does not claim alpha. It surfaces instruments that are:
- trending consistently (not chaotically)
- in LOW or MEDIUM volatility regimes
- showing controlled drawdown and tail risk
- liquid enough to enter and exit cleanly

This is the same philosophy applied in the market making engine: the Regime-Aware ASMM steps back from the book during EXTREME volatility. A disciplined discretionary investor should apply the same filter.

---

## How It Connects to the Market Making Engine

The screener directly reuses three modules from the engine:

| Engine Module | Screener Usage |
|---|---|
| `src/models/regime.py` | `RegimeClassifier` + `VolatilityRegime` for equity volatility classification |
| `src/risk/var.py` | `historical_var()` + `expected_shortfall()` on daily log-returns |
| `src/models/volatility.py` | `RollingVolatilityEstimator` for σ estimation |

The screening philosophy mirrors the market making insight: the same σ that drives the A-S optimal spread also determines which instruments are safe to approach. During HIGH/EXTREME volatility, adverse selection increases for both market makers and discretionary traders.

---

## Data

**Source**: Yahoo Finance via `yfinance` library.
**Type**: Adjusted close, OHLCV daily.
**Lookback**: 1 year (365 calendar days ≈ 252 trading days).
**Minimum requirements**: ≥60 trading days, ≥$10M average dollar volume.

Adjusted close prices account for stock splits and dividends, ensuring clean return calculations.

---

## Default Universe

```
SPY, QQQ, IWM, DIA          (broad market ETFs)
NVDA, MSFT, AAPL, AMZN,
META, GOOGL, AVGO, AMD, TSLA (large-cap tech)
XLE, XLF, XLK               (sector ETFs)
USO, GLD, TLT               (commodity / bond)
```

To screen a custom universe, pass a list to `MarketDataFetcher.fetch()`.

---

## Factor Definitions

### Trend Factors

| Factor | Formula | Interpretation |
|---|---|---|
| `ret_20d` | log(P_t / P_{t-20}) | 20-day total log return |
| `ret_60d` | log(P_t / P_{t-60}) | 60-day total log return |
| `price_vs_ma20` | P_t / MA20 − 1 | Position above/below 20-day SMA |
| `price_vs_ma50` | P_t / MA50 − 1 | Position above/below 50-day SMA |

### Volatility and Regime

| Factor | Formula | Interpretation |
|---|---|---|
| `vol_20d` | std(log-returns, 20d) × √252 | Annualised 20-day realized vol |
| `vol_60d` | std(log-returns, 60d) × √252 | Annualised 60-day realized vol |
| `vol_pct` | percentile of vol_20d in history | Where current vol sits (0=low, 100=high) |
| `regime` | threshold classifier on vol_20d | LOW / MEDIUM / HIGH / EXTREME |

Regime thresholds (annualised):

| Regime | Threshold | Typical equity examples |
|---|---|---|
| LOW | σ < 12% | Calm ETFs, defensive names |
| MEDIUM | 12–25% | Normal large-cap equities |
| HIGH | 25–45% | Tech growth, sector ETFs in stress |
| EXTREME | σ > 45% | Meme stocks, dislocated markets |

### Risk Factors

| Factor | Definition |
|---|---|
| `max_drawdown` | Max peak-to-trough decline over full lookback |
| `var_95` | 95% Historical VaR on daily log-returns (positive = loss) |
| `es_95` | 95% Expected Shortfall (average of worst 5% of days) |
| `downside_vol` | Annualised std of negative daily returns only |

VaR and ES are computed using the same functions as the risk engine (`src/risk/var.py`), on the daily log-return distribution.

### Liquidity Factors

| Factor | Definition |
|---|---|
| `avg_dollar_vol` | Mean daily close × volume over lookback |
| `vol_stability` | 1 − coefficient_of_variation(daily volume) |

---

## Scoring Model

Each component is independently mapped to [0, 100], then combined:

| Component | Weight | Rewards |
|---|---|---|
| Trend | 30% | Positive 20d and 60d returns; price above moving averages |
| Regime | 25% | LOW or MEDIUM regime; low volatility percentile |
| Risk | 25% | Low drawdown; low VaR; low ES/VaR ratio |
| Liquidity | 10% | High average dollar volume; stable volume |
| Quality | 10% | High risk-adjusted return (Sharpe-like proxy) |

**Key design choices:**
- EXTREME regime never scores above 10 on the regime component, regardless of trend
- High drawdown (>40%) removes up to 40 points from the risk component
- The quality score penalises high downside volatility even when returns look good

**Score interpretation:**

| Score | Status |
|---|---|
| 70–100 | Strong candidate — favorable regime, trend, and risk |
| 55–70 | Watch — mixed signals; monitor but do not initiate |
| 35–55 | Weak — avoid new entries |
| 0–35 | Avoid — extreme volatility or poor risk/reward |

---

## Report Output

### CSV (`reports/stock_screen_report.csv`)

Machine-readable table with one row per ticker. All factors and component scores included. Sortable and importable into any spreadsheet or analysis tool.

### Markdown (`reports/stock_screen_report.md`)

Human-readable report including:
- Top candidates table
- Full rankings
- Avoid list with reasons
- Regime distribution
- Suggested tiny test allocation framework
- Risk notes and score interpretation guide

---

## Allocation Framework

The report includes an illustrative allocation for a small test portfolio. **This is for paper trading or tiny real-money validation of the screening methodology only.**

Rules applied:
- Only tickers with total score ≥ 65 and regime ≠ EXTREME
- Maximum 10% of test portfolio per ticker
- Score-weighted allocation within eligible set
- Minimum 30% held in cash at all times
- No leverage, no options, no margin

> ⚠ **NOT FINANCIAL ADVICE.** Position sizes shown are illustrative only. All trading involves risk of loss. Past performance does not predict future returns.

---

## Limitations

**Historical basis only.** All factors are computed from past data. The screen tells you what a ticker looked like — not what it will do.

**Regime detection lags.** The 20-day rolling window means the regime label responds to the past 20 trading days. A sudden shock that begins today will not appear until sufficient data accumulates.

**Survivorship bias.** The universe is a fixed list. Tickers that were delisted or halted are not included. This introduces survivorship bias in any long-term backtest of this methodology.

**No alpha claim.** The score is a risk-and-regime filter, not a return predictor. A score of 80 does not mean the ticker will go up. It means the ticker is currently in a regime and trend state that the methodology considers favorable for monitoring.

**Correlation ignored.** The screener scores each ticker independently. It does not account for correlation between names. A portfolio of the top 10 scorers may be highly concentrated in the same factor exposure (e.g., all large-cap tech in a bull market).

**No transaction costs.** The scoring model does not account for bid-ask spreads, commissions, or market impact.

---

## How To Run

```bash
# Install dependency
pip install yfinance

# Run screener on default universe
python examples/demo_stock_screener.py

# Output files
cat reports/stock_screen_report.md    # readable report
open reports/stock_screen_report.csv  # data in spreadsheet
```

### Custom Universe

```python
from src.screener import MarketDataFetcher, FactorEngine, Scorer
from src.screener import save_csv, save_markdown, print_terminal_table

my_universe = ["AAPL", "MSFT", "JPM", "BRK-B", "V"]

fetcher = MarketDataFetcher(lookback_days=365, cache_dir=".cache")
result  = fetcher.fetch(my_universe)

factors = FactorEngine().compute_batch(result.data)
scored  = Scorer().score_all(factors)

print_terminal_table(scored)
save_markdown(scored, "reports/custom_report.md")
```

---

## Testing

The test suite (`tests/test_screener.py`) uses only synthetic data. No yfinance calls are made during testing, ensuring reproducibility and offline compatibility.

```bash
pytest tests/test_screener.py -v    # 56 tests
pytest tests/ -q                    # 510 total tests passing
```
