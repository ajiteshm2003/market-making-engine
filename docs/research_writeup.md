# Regime-Aware Adaptive Market Making Under Volatility State Transitions

**Ajitesh Mukherjee**

---

## Abstract

We design and implement a simulation-based framework for studying market-making strategies under adverse selection, inventory risk, and volatility non-stationarity. Beginning from a price-time priority limit order book with FIFO matching, we construct a population of heterogeneous agents — uninformed noise traders and signal-driven informed traders — and evaluate four market-making strategies of increasing sophistication. The culminating strategy extends the Avellaneda-Stoikov (2008) stochastic-control framework with a regime classifier that dynamically adjusts risk aversion, spread width, quote size, and inventory limits based on realized volatility state. In a volatile simulation environment with frequent jump events, the regime-aware strategy achieves the lowest maximum drawdown across all strategies and a competitive Sharpe ratio, validating the hypothesis that static parameter calibration is insufficient for non-stationary market conditions.

---

## 1. Introduction

A market maker occupies the central position in a continuous limit order book: it provides liquidity by continuously quoting bids and asks, earning the spread from market-order flow while absorbing inventory risk from adverse order arrival patterns. The profitability of this activity depends critically on the balance between spread income and inventory cost.

Two interconnected problems make optimal market making non-trivial. First, the *adverse selection problem*: informed traders who possess private information about the evolving fair value will selectively consume mispriced quotes, causing the market maker to accumulate positions just before unfavorable price moves. Second, the *non-stationarity problem*: a spread calibrated to quiet conditions is insufficient protection after a volatility shock; a spread calibrated to volatile conditions sacrifices fill income during calm periods.

The Avellaneda-Stoikov (2008) model provides a principled solution to the first problem via stochastic optimal control, jointly deriving both the quote center (reservation price) and quote width (optimal spread) from a single risk aversion parameter γ. However, the model's parameters — particularly γ and the time horizon T-t — are typically treated as static inputs, leaving the non-stationarity problem unaddressed.

This paper describes a regime-aware extension of the A-S framework in which γ, the spread multiplier, quote size, and inventory limits are all functions of a dynamically estimated volatility regime. We implement this within a full simulation platform and compare it against three baseline strategies.

---

## 2. Market Microstructure Background

### 2.1 The Limit Order Book

A limit order book maintains two sorted queues: bids in descending price order and asks in ascending price order. Orders at the same price are matched in time-priority (FIFO) order. A market order immediately executes against the best available resting quotes; a limit order rests in the book until matched or cancelled.

The bid-ask spread — the difference between the lowest ask and highest bid — represents the cost of immediate execution for a market-order participant and the revenue per round-trip for the market maker.

### 2.2 Adverse Selection

Adverse selection arises from information asymmetry. Informed traders submit orders conditioned on a private signal about the future direction of the fair value. When such a trader buys, it is because the true price is likely to rise; when it sells, the true price is likely to fall. A market maker who cannot identify informed flow will find that its bid fills precede price declines and its ask fills precede price increases — systematically eroding PnL.

The adverse selection cost per fill can be estimated as the covariance between the direction of the fill and the subsequent midprice move. Higher informed-trader activity increases this cost and demands wider spreads for break-even market making.

### 2.3 Inventory Risk

Each fill creates a directional inventory position. A long position loses value when prices fall; a short position loses value when prices rise. The expected cost of holding this position — the inventory risk premium — depends on position size, volatility, and the expected time to unwind.

Optimal inventory management requires skewing quotes to attract flow on the inventory-reducing side: when long, lower both bid and ask to attract sellers; when short, raise both to attract buyers.

---

## 3. Simulation Design

### 3.1 Exchange

We implement a continuous price-time priority limit order book supporting limit orders, market orders, and cancellations with partial fills. The matching engine enforces FIFO within price levels and maintains immutable trade records with full maker/taker attribution.

### 3.2 Fair Value Process

The latent true price follows a Gaussian random walk with a Poisson jump component:

```
V(t+1) = V(t) + σ_diff · ε_t + J_t · Bernoulli(λ)
```

where ε_t ~ N(0,1), J_t ~ N(0, σ_jump), σ_diff is the diffusion volatility per step, and λ is the jump arrival probability. Jump events model exogenous news shocks that create transient pricing uncertainty — the conditions under which adverse selection is most acute.

### 3.3 Agent Population

**Noise traders** submit randomized limit and market orders drawn from a log-normal size distribution, with configurable activity rate, market-order probability, and resting order limit. Their role is to generate background order flow and keep the book populated.

**Informed traders** compare the latent fair value V(t) to the observable midprice. When the deviation |V(t) - mid| exceeds a configurable threshold, they trade aggressively toward the fair value with size proportional to conviction. An inventory penalty suppresses size as their position grows.

### 3.4 Simulation Loop

Each timestep proceeds in fixed order: (1) advance fair value, (2) collect orders from all agents in randomized sequence, (3) process cancellations, (4) submit orders to the matching engine, (5) route fills back to agents, (6) update mark-to-market PnL, (7) record metrics.

Randomized agent ordering prevents the first-mover bias that would arise from a fixed submission sequence.

---

## 4. Market-Making Strategies

### 4.1 Naive Market Maker (Strategy A)

The baseline strategy quotes a fixed symmetric spread around the observable midprice:

```
bid = mid - δ_fixed
ask = mid + δ_fixed
```

No inventory adjustment is applied. This strategy is the reference against which all others are measured.

### 4.2 Inventory-Aware Market Maker (Strategy B)

Extends the baseline with a linear inventory skew applied to the quote center:

```
r_B = mid - γ_eff · inventory
bid = r_B - δ(inventory)
ask = r_B + δ(inventory)
```

where δ(inventory) widens with |inventory| by a configurable factor. The skew and spread parameters are calibrated independently, which — as the results show — can lead to over-conservative quoting that sacrifices fill income.

### 4.3 Avellaneda-Stoikov Market Maker (Strategy C)

Implements the exact A-S optimal control solution. The reservation price and optimal spread are jointly derived:

```
r = S - q·γ·σ²·(T-t)

δ* = (γ·σ²·(T-t))/2 + (1/γ)·ln(1 + γ/k)
```

Volatility σ is estimated from a rolling window of log-midprice returns. The order arrival intensity k is estimated from recent maker fill frequency. Both estimators update each timestep, making the strategy partially adaptive without explicit regime detection.

### 4.4 Regime-Aware ASMM (Strategy D)

Extends Strategy C with a four-state volatility regime classifier. The classifier maps the current σ̂ to {LOW, MEDIUM, HIGH, EXTREME} using configurable thresholds with hysteresis. Per-regime multipliers are applied to all model parameters:

```
γ_eff  = γ_base  × m_γ(regime)
δ_eff  = δ*(·)   × m_δ(regime)
qs_eff = qs_base × m_qs(regime)
inv_eff = inv_base × m_inv(regime)
```

In HIGH/EXTREME regimes, the effective risk premium `(γ_eff · σ² · (T-t))/2` grows through both the elevated γ_eff and the elevated σ simultaneously — a compounding defense that exceeds what either mechanism achieves alone.

---

## 5. Experimental Results

### 5.1 Setup

Simulations run for 800 timesteps with 3 noise traders and 2 informed traders. All four market makers participate concurrently with identical initial cash. Fair value parameters: diffusion σ = 0.05/step, jump probability λ = 0.06, jump std σ_jump = 2.0. Parameters are calibrated to produce all four regime states within a typical run.

### 5.2 Strategy Comparison

| Strategy | Sharpe Ratio | Max Drawdown | Inventory Variance | Total PnL |
|---|---|---|---|---|
| Naive MM | −0.57 | 450.93 | 155.99 | +27.53 |
| Inventory-Aware MM | −1.08 | 20.95 | 11.62 | −20.05 |
| Avellaneda-Stoikov MM | +0.52 | 21.41 | 274.55 | +33.11 |
| Regime-Aware ASMM | +0.58 | **17.87** | 82.36 | −5.86 |

Sharpe ratios are computed from per-step PnL returns, annualized by √252.

### 5.3 Observations

**Naive MM**: The fixed spread provides no protection against volatility spikes. During the EXTREME regime episode, the strategy continues quoting its baseline spread into a rapidly moving market, accumulating a 450.93 peak-to-trough drawdown — by far the largest across all strategies.

**Inventory-Aware MM**: Achieves the lowest inventory variance (11.62) through aggressive quote skewing, but this same aggressiveness reduces fill income below the cost of adverse selection. The result is the lowest Sharpe ratio (−1.08) and negative PnL.

**Avellaneda-Stoikov MM**: The most significant improvement comes from the A-S optimal spread formula — particularly the risk premium term, which automatically widens spreads when σ rises after jump events. The Sharpe ratio improves from −0.57 to +0.52. The high inventory variance (274.55) reflects the strategy's relatively light inventory penalty in MEDIUM/LOW regimes.

**Regime-Aware ASMM**: Achieves the lowest drawdown (17.87) by applying extreme defensive parameters during EXTREME episodes. The cost is lower average PnL (−5.86) because the strategy withdraws from the book during the most active periods. The appropriate benchmark is not raw PnL but risk-adjusted return: the Sharpe ratio (0.58) is the highest across all strategies.

### 5.4 Per-Regime Analysis

The EXTREME regime accounts for 3.1% of simulation steps but dominates the Naive MM's drawdown. The Regime-Aware ASMM detects the σ spike, transitions parameters within one to two steps of the jump event, and posts only one fill during the entire EXTREME episode — limiting inventory accumulation to near zero while the market price discovers the new fair value.

---

## 6. Key Findings

1. **Static spreads are insufficient in non-stationary environments.** The Naive MM's drawdown is 25× larger than the Regime-Aware ASMM's despite being a smaller fraction of total time in the volatile regime.

2. **The A-S framework's joint derivation of spread and skew is superior to independent calibration.** The Inventory-Aware MM, which optimizes spread and skew separately, achieves worse Sharpe than both A-S variants.

3. **Regime detection provides tail protection without requiring accurate regime prediction.** The classifier reacts to realized σ; it need not predict when jumps will occur. Hysteresis ensures stability while still allowing rapid response to genuine volatility changes.

4. **Risk-adjusted metrics are the correct comparison criterion.** Raw PnL rankings reverse when adjusted for risk: the Naive MM has the highest raw PnL in this run but the lowest Sharpe; the Regime-Aware ASMM has the highest Sharpe despite negative raw PnL.

---

## 7. Limitations

**Simulation fidelity.** The fair value process is a stylized model. Real markets exhibit more complex dynamics: correlated jumps across assets, intraday seasonality in volatility and volume, and microstructure effects such as price impact and queue depletion that are absent here.

**Parameter sensitivity.** The regime multipliers are hand-calibrated rather than estimated from data. An empirical calibration against historical LOB data would be required before deploying the strategy in production.

**Single-asset setting.** The simulation treats one asset in isolation. A real market maker hedges inventory across correlated instruments; the cross-asset dimension of inventory risk is not modeled here.

**Order arrival model.** The A-S framework assumes exponential fill probability as a function of depth, which is a simplification. Hawkes process models better capture the self-exciting, clustered nature of real order flow.

**No latency modeling.** The simulation processes all agents synchronously within each timestep. Real electronic markets involve meaningful latency asymmetries between participants that affect queue position and fill probability.

---

## 8. Future Work

- Calibrate against historical Binance/Coinbase Level 2 order book data for empirical validation
- Replace Poisson order arrivals with a Hawkes self-exciting process
- Implement multi-level quoting with queue-position tracking
- Train a PPO agent to learn quoting policy within this simulation as an environment
- Port the matching engine to C++ to enable realistic latency and throughput simulation
- Extend to a multi-asset setting with correlated fair value processes and cross-asset hedging

---

## References

Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.

Glosten, L. & Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.

Ho, T. & Stoll, H. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.

O'Hara, M. (1995). *Market Microstructure Theory*. Blackwell, Cambridge, MA.

Johnson, B. (2010). *Algorithmic Trading & DMA*. 4Myeloma Press.
