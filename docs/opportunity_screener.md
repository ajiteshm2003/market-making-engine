# Opportunity Screener

An extension of the base regime-aware screener that surfaces smaller/mid-cap names with improving momentum structure — without abandoning the project's core risk discipline.

---

## Why the Base Screener Favours Mega-Caps

The base screener rewards regime stability and drawdown control. These qualities are most reliably found in mega-cap stocks and broad ETFs: SPY has never had a 50% drawdown (over any recent rolling year), MSFT has deep liquidity and low annualised volatility relative to individual stocks.

This is correct behaviour for the "institutional stability" universe. But it means emerging names with genuine trend momentum are systematically underweighted — even when their structure is compelling.

The opportunity screener addresses this with three changes:

1. **Three separate universes** — each scored within itself, preventing SPY from crowding out CRWD.
2. **Market-cap opportunity bonus** — $2B–$30B names receive a scoring boost that reflects the higher discovery potential in less-crowded price discovery.
3. **Trend acceleration replaces trend magnitude** — a name recovering from −5% to +10% is more interesting than one that has been at +15% for six months. Direction of change matters.

---

## The Three Universes

### 1. Institutional Stability

Large-cap and ETF universe. SPY, QQQ, Mag-7, quality financials and healthcare.

**Philosophy:** The risk anchor. Rewards regime stability and drawdown control. Penalises elevated volatility more aggressively than the other universes. HIGH volatility regime gets a score of 40 (vs 55 in Emerging).

**Use:** Reference for regime context and capital preservation.

### 2. Emerging Leaders

Mid-to-small-cap names with genuine revenue, institutional ownership, and improving momentum structure. CRWD, NET, PLTR, ONON, CAVA, DUOL, SOFI, etc.

**Philosophy:** Find names where the trend structure is genuine and improving. Moderate volatility is acceptable if:
- Vol is constructive (not a parabolic blowoff)
- Drawdown is within the tolerance for the universe (<35%)
- Volume is expanding into the move (institutional accumulation)

**Use:** Primary opportunity hunting ground. The names most likely to be "interesting" that the base screener misses.

### 3. Speculative High Beta

Small-cap, high-growth, high-volatility names. IONQ, ASTS, RGTI, COIN, etc.

**Philosophy:** Watchlist awareness only. Not for position sizing. The screener applies strict filters: EXTREME regime and >60% drawdown trigger automatic avoidance. Names that survive the filter have coherent (if elevated) trend structure.

**Use:** Monitor only. Identify which speculative names have the least chaotic structure, not which ones are "most explosive."

---

## New Factors

### Trend Acceleration

```
momentum_slope = ann(ret_20d) - ann(ret_60d)
```

Annualises both returns to the same per-year basis before subtracting. Positive slope means the 20-day trajectory is steeper than the 60-day — the trend is accelerating.

A strong trend acceleration with constructive volume is the hallmark of genuine institutional accumulation, not a short squeeze.

### Volume Expansion

```
rel_volume = avg_volume_5d / avg_volume_60d
```

Values > 1.0 indicate expanding volume. Values > 1.5 are scored as an accumulation signal. Volume contraction (<0.6) is a negative signal even when price is rising.

### 52-Week High Proximity

```
pct_from_52w_high = current_price / 52w_high - 1
```

Ranges from 0 (at the high) to −1 (at zero). Names within 5% of their 52-week high score a structural bonus. Names more than 40% below their high receive a structural penalty.

This is not a momentum signal — it measures structural integrity. A name recovering to near its high has resolved its technical damage.

### Market Cap Opportunity Bonus

| Cap Bucket | Bonus Points | Rationale |
|---|---|---|
| SMALL ($2B–$10B) | +20 | Highest discovery potential |
| MID ($10B–$100B) | +15 | Solid opportunity range |
| LARGE ($100B–$500B) | +5 | Mild bonus |
| MEGA (>$500B) | 0 | Fully discovered |
| MICRO (<$2B) | −5 | Liquidity penalty |
| ETF | 0 | No cap applicable |

### Volatility Quality

A name with HIGH volatility is assessed as "constructive" if:
- Regime is not EXTREME
- Drawdown does not exceed the universe tolerance
- Volume is not contracting while volatility is rising

Constructive volatility (elevated but trend-following) receives a bonus. Non-constructive volatility (chaotic distribution, high drawdown) receives a penalty.

---

## Scoring Model

| Component | Weight | Description |
|---|---|---|
| Trend Quality | 25% | Returns + MA position |
| Trend Acceleration | 20% | Momentum slope + volume expansion + 52w proximity |
| Regime Quality | 20% | Volatility regime, vol constructiveness, vol percentile |
| Risk Control | 15% | Drawdown, VaR, ES tail ratio |
| Liquidity | 10% | Average dollar volume, volume stability |
| Opportunity Bonus | 10% | Market cap discovery bonus |

### Universe-Adjusted Thresholds

Drawdown and VaR tolerance varies by universe:

| Universe | Max Drawdown Threshold | VaR Threshold |
|---|---|---|
| Institutional | 20% | 2.5% daily |
| Emerging | 35% | 4.0% daily |
| Speculative | 55% | 6.0% daily |

A 40% drawdown is catastrophic for an institutional name but within normal range for a small-cap growth stock.

---

## Score Interpretation

| Score | Status |
|---|---|
| 65–100 | Strong candidate — investigate further |
| 50–65  | Developing structure — monitor |
| 30–50  | Weak — pass |
| 0–30   | Avoid |

**The score is not a buy signal.** It identifies names with favorable historical factor structure that warrant further investigation.

---

## Insight Strings

Every scored ticker generates a plain-English explanation of its ranking:

> *"Elevated but constructive volatility; strong sustained uptrend; accelerating momentum (20d outpacing 60d); volume expanding 1.8x above average (accumulation signal); mid-cap opportunity range"*

This is designed to convey:
- **Why this name** — which factors drove the score
- **What kind of momentum** — stable or chaotic
- **What the risk looks like** — drawdown profile, vol quality
- **What the opportunity context is** — cap bucket, vol constructiveness

It is not designed to say "buy this." It is designed to surface names worth further investigation.

---

## Visualisations

Four plots are generated:

| Plot | File | Content |
|---|---|---|
| Score vs Volatility | `score_vs_vol.png` | Scatter: each ticker at its vol and score; coloured by universe |
| Drawdown vs Return | `drawdown_vs_return.png` | Scatter: 60d return vs max drawdown; coloured by opportunity score |
| Regime Distribution | `regime_distribution.png` | Bar chart: regime counts per universe |
| Top Scores | `top_scores.png` | Horizontal bar: top 12 across all universes |

---

## Limitations

**Not a prediction system.** All factors are historical. The trend acceleration that existed yesterday may reverse tomorrow.

**Universe is fixed.** The screener works on a predefined list. Names not in the universe are not screened. This excludes many legitimate emerging opportunities.

**No fundamental analysis.** The screener does not look at revenue, earnings, guidance, or valuation. A high-scoring name could be fundamentally impaired.

**No correlation analysis.** Multiple names from the same sector may all score well simultaneously — but they are highly correlated. A portfolio of the top 5 Emerging names may be effectively a single bet on growth tech.

**Survivorship in universe.** Names in the speculative universe that get delisted or halted are not automatically removed.

---

## How To Run

```bash
# Requires internet connection for yfinance
python examples/demo_opportunity_screener.py

# Output files
cat reports/opportunity_screen.md
open reports/opportunity_screen.csv
ls reports/figures/
```

### Custom Universe

```python
from src.screener.universe import UniverseSpec, CapBucket, VolTolerance
from src.screener import MarketDataFetcher
from src.screener.opportunity import OpportunityPipeline

my_universe = UniverseSpec(
    name="my_picks",
    label="My Custom Universe",
    description="Specific names I want to analyse",
    tickers=["AAPL", "CRWD", "PLTR", "SOFI"],
    vol_tolerance=VolTolerance.MEDIUM,
    ticker_caps={"AAPL": CapBucket.MEGA, "CRWD": CapBucket.LARGE,
                 "PLTR": CapBucket.LARGE, "SOFI": CapBucket.SMALL},
)

fetcher = MarketDataFetcher(lookback_days=365, cache_dir=".cache")
result  = fetcher.fetch(my_universe.tickers)
pipeline = OpportunityPipeline(my_universe)
results  = pipeline.run(result.data)
for r in results:
    print(f"{r.ticker}: {r.opportunity_score:.1f} — {r.insight}")
```

---

## Testing

```bash
pytest tests/test_opportunity_screener.py -v    # 55 tests
pytest tests/ -q                                 # 565 total tests
```

All tests use synthetic data. No network access required.
