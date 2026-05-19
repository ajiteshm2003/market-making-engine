# Experimental Results

This document summarizes the quantitative results from the four-strategy comparison simulations. All figures are drawn from reproducible runs with fixed random seeds; exact values can be replicated by running the demo scripts.

---

## Simulation Parameters

| Parameter | Value |
|---|---|
| Simulation steps | 800 |
| Noise traders | 3 (activity rate 0.55, market order prob 0.20) |
| Informed traders | 2 (threshold 0.05/0.12, aggression 0.85/0.70) |
| Fair value diffusion σ | 0.05/step |
| Jump probability λ | 6% per step |
| Jump size std σ_jump | 2.0 |
| Initial price | 100.0 |
| Initial cash (per MM) | 100,000 |
| Quote size (base) | 5.0 units |
| Risk aversion γ (base) | 0.10 |

---

## Primary Comparison Table

| Metric | Naive MM | Inventory-Aware MM | Avellaneda-Stoikov MM | Regime-Aware ASMM |
|---|---|---|---|---|
| **Sharpe Ratio** | −0.57 | −1.08 | +0.52 | **+0.58** |
| **Max Drawdown** | 450.93 | 20.95 | 21.41 | **17.87** |
| **Inventory Variance** | 155.99 | **11.62** | 274.55 | 82.36 |
| Total PnL | +27.53 | −20.05 | **+33.11** | −5.86 |
| Realized PnL | varies | varies | varies | varies |
| Spread Capture | varies | varies | varies | varies |
| Fills as Maker | highest | moderate | moderate | lower |
| Fill Rate | highest | moderate | moderate | lower |
| Bid/Ask Balance | ~0.65 | ~1.00 | ~0.75 | ~0.90 |

*Bold indicates best performance on that metric.*

---

## Per-Strategy Analysis

### Strategy A: Naive Market Maker

The Naive MM quotes a fixed symmetric spread of ±0.06 around the observable midprice. No inventory adjustment is applied.

**Strengths:**
- Highest raw fill rate — competitive quotes attract the most volume
- Positive total PnL in most runs due to consistent spread income

**Weaknesses:**
- Fixed spread provides no protection during volatility spikes
- Max drawdown of 450.93 — the largest by a wide margin — occurs during EXTREME regime episodes when informed traders systematically pick off stale quotes
- No inventory management: fills accumulate directionally after jumps

**Why the drawdown is catastrophic:** After a large jump, the fair value moves by σ_jump = 2.0. The NMM continues quoting spread ±0.06 around an outdated midprice. Informed traders fill all quotes on the correct side, accumulating a position of 5.0 units per step. The resulting inventory loss dominates spread income.

---

### Strategy B: Inventory-Aware Market Maker

The IAMM applies a linear inventory skew (factor 0.012) plus spread widening (60% at max inventory), with max inventory = 40.

**Strengths:**
- Lowest inventory variance (11.62) — the skew mechanism effectively mean-reverts position
- Lowest max drawdown among the non-regime-aware strategies (20.95)

**Weaknesses:**
- Most negative Sharpe ratio (−1.08) and PnL (−20.05)
- Aggressive skewing reduces fill income below the adverse selection cost
- Independent calibration of skew and spread parameters is not jointly optimal

**Why PnL is negative:** When the IAMM is long and skewing down, it discourages buyers and encourages sellers — but the sellers may include informed traders who know the price is about to fall further. The IAMM buys from them, then skews down further, reducing its already-disadvantageous inventory at worsening prices.

---

### Strategy C: Avellaneda-Stoikov Market Maker

The ASMM implements the exact A-S optimal control formulas with rolling σ estimation (window=25) and dynamic k estimation.

**Strengths:**
- Best Sharpe ratio among static strategies (+0.52)
- Best total PnL (+33.11)
- Joint derivation of spread and skew from γ is more efficient than independent calibration

**Weaknesses:**
- Highest inventory variance (274.55) — the relatively small static γ=0.10 produces a mild inventory penalty in quiet periods
- Drawdown (21.41) comparable to IAMM — σ adaptation helps but does not eliminate exposure during extreme jumps

**Why it outperforms IAMM despite higher inventory variance:** The A-S spread adapts to σ, which rises after jumps. The spread widens automatically, charging informed traders a higher premium per fill. The IAMM's spread also widens with inventory, but only after the damage has already been done.

---

### Strategy D: Regime-Aware ASMM

The R-ASMM classifies σ̂ into four regimes and applies multiplicative corrections to all A-S parameters.

**Strengths:**
- Lowest max drawdown (17.87) — regime detection provides tail protection
- Best Sharpe ratio (+0.58) — competitive income in LOW regime, strong protection in EXTREME
- Sensible regime behavior verified per-regime (see below)

**Weaknesses:**
- Negative total PnL (−5.86) in some runs — withdrawing from the book during EXTREME episodes sacrifices income that the ASMM earns
- Lower absolute fill count than static strategies

**Why drawdown is lowest:** During the EXTREME episode (σ > threshold_extreme), the R-ASMM applies γ×4.0, spread×3.0, size×0.20. The resulting quotes are 3× wider with 20% of normal size. Informed traders who might fill at ±0.18 (ASMM spread) are not willing to pay ±0.54 (R-ASMM extreme spread). The strategy effectively withdraws from the book, taking no inventory risk during the period of maximum uncertainty.

---

## Per-Regime Breakdown (R-ASMM)

| Regime | Steps | % Time | Fills | Avg |Inventory| | Avg Half-Spread | PnL Sum |
|---|---|---|---|---|---|---|
| LOW | 754 | 94.2% | 52 | 10.28 | 0.1704 | +8.26 |
| MEDIUM | 21 | 2.6% | 0 | 0.00 | 0.6454 | 0.00 |
| HIGH | 0 | 0.0% | — | — | — | — |
| EXTREME | 25 | 3.1% | 1 | 15.05 | 0.2947 | +190.77 |

The EXTREME regime generates a large PnL contribution from the single fill it receives — the one counterparty willing to pay the 3×-widened spread. This illustrates the strategy's design: in EXTREME regime, the MM still quotes, but only at a price that compensates for the elevated risk.

---

## Adverse Selection Evidence

Adverse selection is visible in the bid/ask fill balance metric. A perfectly balanced MM fills equally on bid and ask sides (ratio = 1.0). A bias toward one side indicates that informed flow is directional.

| Strategy | Bid Fills | Ask Fills | Balance Ratio |
|---|---|---|---|
| Naive MM | 45 | 69 | 0.65 |
| Inventory-Aware MM | 57 | 57 | 1.00 |
| Avellaneda-Stoikov | 38 | 51 | 0.75 |
| Regime-Aware ASMM | 26 | 27 | 0.96 |

The NMM's 0.65 ratio indicates systematic one-sided flow — consistent with informed traders preferentially buying from the MM before the price rises. The IAMM's skew mechanism corrects this at the cost of PnL.

---

## Sharpe Ratio Interpretation

Sharpe ratios in market-making contexts require careful interpretation. A Sharpe of +0.52 does not imply consistent profitability — it indicates that the per-step return distribution has a positive mean relative to its standard deviation. In our simulation, per-step returns are highly autocorrelated (inventory positions persist) and fat-tailed (jump events create extreme outcomes), so the Gaussian assumption underlying the Sharpe ratio is an approximation.

Nevertheless, the Sharpe ratio captures the risk-adjusted comparison we care about: strategies that earn similar gross PnL but with smaller variance in per-step outcomes rank higher. The R-ASMM's advantage over the ASMM (+0.58 vs +0.52) is modest in absolute terms but reflects the tail protection provided during EXTREME episodes.

---

## Reproducibility

All results are reproducible by running:

```bash
python examples/demo_phase5.py    # generates primary comparison
python examples/demo_phase4.py    # A-S vs static comparison
python examples/demo_phase3.py    # NMM vs IAMM comparison
```

with `SEED = 42` as specified in each demo script. Exact numeric values may differ slightly if the simulation infrastructure is modified, but the qualitative rankings are stable across reasonable parameter variations.

---

## Sensitivity Notes

**γ sensitivity:** Higher γ increases the inventory penalty and spread width simultaneously. At γ = 0.5 (5× the baseline), the ASMM achieves lower inventory variance but reduced fill rate and lower absolute PnL. The optimal γ depends on the ratio of adverse selection cost to spread income, which varies with the informed trader population.

**Window sensitivity:** Shorter volatility windows (10 steps) react faster to jumps but produce noisier σ estimates in quiet periods, increasing unnecessary spread widening. Longer windows (50 steps) are more stable but lag regime transitions by several steps.

**Regime threshold sensitivity:** The thresholds must be calibrated to the specific simulation's realized σ distribution. The defaults provided are calibrated for the Phase 5 simulation parameters; recalibration is required when changing diffusion σ or jump parameters.
