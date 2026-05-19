# Future Work

This document describes planned extensions to the engine, ordered from most to least implementation-ready. Each section includes the technical approach, expected difficulty, and the research question it addresses.

---

## 1. Historical L2 Order Book Replay

**Status:** Infrastructure complete; data pipeline not yet built.

**Goal:** Replace the simulated fair value process and agent population with a replay of real Level 2 order book data, enabling empirical validation of strategy performance against actual market microstructure.

**Approach:**
- Ingest Binance or Coinbase L2 snapshots and incremental updates via their public WebSocket APIs
- Reconstruct the full order book state at each timestep from the snapshot + delta stream
- Feed observed midprice sequence into the existing volatility and regime estimators
- Replay historical market orders against the current strategy's live quotes, or implement a more sophisticated counterfactual replay that estimates fill probability from observed depth

**Research questions:**
- How do realized σ distributions compare to the simulated process?
- Do the calibrated regime thresholds transfer across different assets and market conditions?
- What fraction of real fills are attributable to informed flow (via post-trade price move analysis)?

**Difficulty:** Medium. The simulation infrastructure is market-data-agnostic; the main work is in data ingestion and fill simulation.

---

## 2. Hawkes Process Order Arrivals

**Status:** Not implemented.

**Goal:** Replace the Poisson (memoryless) order arrival model with a self-exciting Hawkes process that captures the empirically observed clustering of market orders.

**Motivation:** Real order flow is highly autocorrelated — a large market order predicts a higher rate of further market orders over the next several seconds. The Poisson assumption in the A-S model underestimates this clustering, leading to a k estimate that is too stable. A Hawkes process would produce k estimates that spike after bursts of activity and decay gradually.

**Approach:**

The bivariate Hawkes process models buy and sell market order arrivals with mutual excitation:

```
λ_buy(t)  = μ_buy  + Σ_{t_i < t} α·exp(-β·(t - t_i))
λ_sell(t) = μ_sell + Σ_{t_i < t} α·exp(-β·(t - t_i))
```

Parameters (μ, α, β) are estimated from historical trade data via maximum likelihood.

**Research questions:**
- Does a Hawkes-calibrated k estimator produce better fill-rate predictions than the simple fill-frequency estimator currently used?
- How does the self-exciting nature of order arrivals interact with regime transitions?

**Difficulty:** Medium-high. Requires Hawkes process simulation and MLE calibration.

---

## 3. Multi-Level Quoting

**Status:** Not implemented. Architecture supports it.

**Goal:** Post limit orders at multiple price levels simultaneously, rather than a single bid-ask pair. This is closer to real market maker behavior and enables more sophisticated inventory management through depth management.

**Approach:**
- Extend `BaseMarketMaker._compute_quotes()` to return a list of (price, quantity) pairs per side
- Implement a skewed depth schedule: more quantity at prices closer to mid, less at extremes
- Track queue position at each level (simplified: assume back-of-queue on new orders)
- Cancel and requote only levels that have moved more than a threshold from current mid (partial requoting)

**Research questions:**
- Does multi-level quoting improve fill rate without proportionally increasing inventory risk?
- What is the optimal depth schedule under the A-S framework?
- How should inventory skew be distributed across levels?

**Difficulty:** Medium. The matching engine already handles it; the strategy and metrics layers need extension.

---

## 4. Queue Position Modeling

**Status:** Not implemented.

**Goal:** Model the position of the market maker's limit orders within the FIFO queue at each price level, and use this to estimate fill probability more accurately than the simple exponential model in A-S.

**Motivation:** In real markets, a limit order at the best bid fills only after all orders ahead of it in the queue are consumed. Queue position determines fill probability and is a key determinant of market maker economics in tight-spread environments (e.g., US equity market making at 1-cent spread).

**Approach:**
- Track cumulative quantity ahead of each order in the queue
- Model fill probability as a function of (queue position, order flow rate, volatility)
- Use a simplified LOB model: fill when a market order exceeds the queue ahead of our order

**Research questions:**
- How much does queue position matter for the A-S optimal spread formula?
- In what regimes is queue management more valuable than spread management?

**Difficulty:** High. Requires significant changes to the matching engine and agent fill-probability modeling.

---

## 5. Reinforcement Learning Market Maker

**Status:** Not implemented. Simulation environment is RL-ready.

**Goal:** Train a reinforcement learning agent to learn an optimal quoting policy within the existing simulation environment, using the market simulation as the environment and policy gradient methods (PPO) or value-based methods (DQN) as the learning algorithm.

**Approach:**
- Wrap `MarketSimulation` as an OpenAI Gym environment
- State space: inventory, midprice, σ̂, k̂, regime, time remaining, bid/ask depth
- Action space: (bid_offset, ask_offset, quote_size) — deviations from A-S baseline
- Reward: per-step mark-to-market PnL, with inventory penalty at terminal step
- Use Stable Baselines 3 (PPO) as the training framework

**Key consideration:** This project explicitly avoids RL until the rule-based strategies are properly implemented and understood. An RL agent trained on a poorly designed simulation learns to exploit simulator artifacts rather than real market structure. The existing simulation must first be validated against real data before RL results can be taken seriously.

**Research questions:**
- Does a learned policy outperform the A-S optimal control solution?
- Does the RL agent rediscover the A-S intuition (skew toward inventory zero) or find qualitatively different strategies?
- How sensitive is the learned policy to simulation parameters?

**Difficulty:** Medium. The environment wrapper is straightforward; the training stability is the main challenge.

---

## 6. C++ Matching Engine

**Status:** Not implemented.

**Goal:** Rewrite the matching engine in C++ to enable realistic microsecond-level latency simulation and throughput benchmarking.

**Motivation:** The Python matching engine processes orders sequentially and cannot model the latency asymmetries between market participants that are central to HFT economics. A C++ core would enable:
- Nanosecond-level timestamp resolution
- Realistic network latency modeling between agents
- Order processing throughput of millions of orders per second

**Approach:**
- Implement the order book as a sorted `std::map<price, std::deque<Order>>` in C++
- Expose a Python binding via pybind11 for integration with the existing agent and strategy code
- Model latency by assigning each agent a "processing delay" measured in simulated nanoseconds

**Difficulty:** High. C++ implementation is straightforward; the Python-C++ integration and latency modeling add significant complexity.

---

## 7. Multi-Asset Market Making

**Status:** Not implemented.

**Goal:** Extend the single-asset framework to a correlated multi-asset setting with cross-asset hedging and portfolio-level inventory constraints.

**Motivation:** Real market makers simultaneously quote multiple correlated instruments. A position in one asset can be partially hedged by a position in a correlated asset. The optimal quoting policy in this setting is qualitatively different from the single-asset case.

**Approach:**
- Extend the fair value process to a multivariate Gaussian with correlation matrix Σ
- Extend the A-S framework to the multi-asset case (Cartea & Jaimungal, 2015)
- Add a portfolio-level inventory constraint: the total delta-weighted position must stay within bounds
- Implement cross-asset hedging: when inventory in asset A exceeds a threshold, hedge by trading correlated asset B

**Research questions:**
- How much of single-asset inventory risk can be hedged cross-asset?
- Does regime detection transfer to a correlated multi-asset setting?

**Difficulty:** Very high. The multi-asset A-S solution involves matrix Riccati equations and is significantly more complex than the scalar case.

---

## Implementation Priority

| Extension | Impact | Difficulty | Priority |
|---|---|---|---|
| Historical L2 replay | Validation critical | Medium | High |
| Hawkes process arrivals | Model accuracy | Medium | Medium |
| Multi-level quoting | Realism | Medium | Medium |
| RL market maker | Research novelty | Medium | Lower |
| Queue position modeling | Microstructure depth | High | Lower |
| C++ matching engine | Performance | High | Lower |
| Multi-asset | Scope expansion | Very high | Future |

The highest-priority extension is historical data replay, as it is the only path to empirical validation of the strategies against real market conditions. All other extensions build on or assume a validated baseline.
