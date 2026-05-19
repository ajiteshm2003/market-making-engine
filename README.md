# Regime-Aware Market Microstructure & Adaptive Market Making Engine

A simulation-based research platform implementing a realistic electronic exchange, agent-based market participants, and a progression of market-making strategies culminating in a regime-aware extension of the Avellaneda-Stoikov (2008) stochastic-control framework.

---

## Motivation

Market making is an adversarial statistical estimation problem. A dealer continuously posts limit orders on both sides of the book, earning the bid-ask spread from incoming flow. Three compounding risks make this non-trivial:

**Adverse selection.** Informed traders — agents with private information about future price direction — disproportionately consume quotes mispriced relative to the evolving fair value. A dealer who cannot distinguish informed from uninformed flow will systematically lose money on fills that precede directional moves.

**Inventory risk.** Every fill leaves the dealer directionally exposed. Accumulated inventory must eventually be unwound, often at unfavorable prices. The expected cost of this unwinding is the inventory risk premium.

**Volatility non-stationarity.** A spread calibrated to calm periods provides insufficient protection after a news shock. A spread calibrated to volatile periods is uncompetitive in quiet markets. Static strategies cannot optimally navigate both regimes.

This project constructs a complete simulation environment to study these tradeoffs and to implement and compare strategies of increasing sophistication.

---

## System Overview

```
src/
├── exchange/        Limit order book, matching engine, trade logging
├── agents/          NoiseTrader, InformedTrader, BaseAgent
├── simulation/      MarketSimulation, FairValueProcess, MarketState, metrics
├── strategies/      NaiveMM, InventoryAwareMM, AvellanedaStoikovMM, RegimeAwareMM
└── models/          Volatility estimator, arrival intensity, A-S math, analytics, regime classifier
tests/               327 unit and integration tests
examples/            Runnable demo scripts for each phase
docs/                Research writeup, architecture, results, future work
```

---

## Core Features

### Exchange Infrastructure
- Price-time priority limit order book with sorted bid/ask levels
- Continuous matching engine: limit orders, market orders, cancellations
- FIFO queue discipline within price levels; partial fill support
- Immutable trade records with full execution metadata (maker id, taker id, aggressor side)

### Agent Simulation
- **NoiseTrader** — randomized limit/market orders with log-normal size sampling and resting order management
- **InformedTrader** — acts on latent fair-value deviation; trade size scales with conviction; inventory penalty reduces exposure as position grows
- Event-driven loop with randomized agent ordering per timestep

### Fair Value Process
Latent true price evolves as a Gaussian random walk with Poisson jump component:
```
V(t+1) = V(t) + σ·ε + J·Bernoulli(λ),    ε ~ N(0,1),  J ~ N(0, σ_jump)
```
Jump events model news shocks that create adverse selection for the market maker.

### Market-Making Strategy Progression

| Strategy | Spread | Inventory Skew | Volatility Adaptation |
|---|---|---|---|
| Naive MM | Fixed symmetric | None | None |
| Inventory-Aware MM | Fixed + widening | Linear | None |
| Avellaneda-Stoikov MM | A-S optimal | Stochastic-control | Via σ̂ estimator |
| Regime-Aware ASMM | A-S × regime multiplier | A-S + regime-scaled γ | Via regime classifier |

### Analytics Framework
- Per-step metrics: midprice, spread, order imbalance, volume, cumulative trades
- Per-strategy: Sharpe ratio, max drawdown, inventory variance, spread capture, fill rate, bid/ask balance
- Per-regime: steps, fills, average inventory, average spread, PnL contribution

---

## Mathematical Framework

### Avellaneda-Stoikov (2008)

The model solves a stochastic optimal control problem for a dealer posting in a LOB with Poisson order arrivals. The solution yields:

**Reservation price:**
```
r = S - q·γ·σ²·(T-t)
```

**Optimal half-spread:**
```
δ* = (γ·σ²·(T-t))/2  +  (1/γ)·ln(1 + γ/k)
      └─ risk premium ─┘   └─ liquidity premium ─┘
```

Where `S` = midprice, `q` = inventory, `γ` = risk aversion, `σ` = estimated volatility, `T-t` = horizon, `k` = order arrival intensity.

Final quotes: `bid = r − δ*`,  `ask = r + δ*`

Both the quote center (r) and quote width (δ\*) are jointly derived from the same γ, ensuring coherent risk management. Strategy B optimizes these with independent parameters; A-S derives both from a single objective.

### Volatility Estimation
```
σ̂_t = std(ln(S_t/S_{t-1}), ..., ln(S_{t-n}/S_{t-n-1}))
```
Rolling unbiased sample standard deviation of log-midprice returns over a configurable window. EWM variant available for faster regime adaptation.

### Arrival Intensity Estimation
```
k̂ = fill_scale / avg_maker_fills_per_step
```
Higher observed fill rate → lower k → tighter spread needed to remain competitive.

### Regime-Aware Extension

The regime classifier maps σ̂ to a discrete state:
```
σ < θ_low               → LOW
θ_low  ≤ σ < θ_high    → MEDIUM
θ_high ≤ σ < θ_extreme → HIGH
σ ≥ θ_extreme           → EXTREME
```

Hysteresis prevents rapid switching near boundaries. Each regime applies multiplicative corrections:

| Regime | γ mult | Spread mult | Quote size mult | Max inventory mult |
|---|---|---|---|---|
| LOW | 0.6× | 0.7× | 1.4× | 1.5× |
| MEDIUM | 1.0× | 1.0× | 1.0× | 1.0× |
| HIGH | 2.0× | 1.8× | 0.55× | 0.55× |
| EXTREME | 4.0× | 3.0× | 0.20× | 0.25× |

In HIGH/EXTREME regimes, effective γ and spread multiplier compound: the risk premium `(γ·σ²·(T-t))/2` grows via both σ and the elevated γ simultaneously. This multiplicative defense substantially exceeds what either mechanism achieves alone.

---

## Key Results

Representative 800-step simulation: 3 noise traders, 2 informed traders, all four market makers. Fair value: σ = 0.05/step, jump probability = 6%, jump std = 2.0.

### Strategy Comparison

| Strategy | Sharpe | Max Drawdown | Inventory Variance | Total PnL |
|---|---|---|---|---|
| Naive MM | −0.57 | 450.93 | 155.99 | +27.53 |
| Inventory-Aware MM | −1.08 | 20.95 | 11.62 | −20.05 |
| Avellaneda-Stoikov MM | +0.52 | 21.41 | 274.55 | +33.11 |
| **Regime-Aware ASMM** | **+0.58** | **17.87** | **82.36** | −5.86 |

### Key Observations

- The Naive MM has the largest drawdown (450.93) despite positive PnL — its fixed spread provides no protection when volatility spikes.
- The Inventory-Aware MM achieves low inventory variance (11.62) but negative PnL due to aggressive skewing reducing competitive fill rate.
- A-S improves Sharpe from −0.57 to +0.52 primarily through volatility-adaptive spread widening after jumps.
- The Regime-Aware ASMM achieves the lowest drawdown (17.87) across all strategies; during EXTREME episodes (3.1% of simulation), it applies γ×4.0, spread×3.0, size×0.20.

### Regime Distribution

| Regime | Time % | Avg Half-Spread | Fills |
|---|---|---|---|
| LOW | 94.2% | 0.171 | 52 |
| MEDIUM | 2.6% | 0.645 | 0 |
| EXTREME | 3.1% | 0.295 | 1 |

EXTREME accounts for a small fraction of time but drives the drawdown disparity between static and regime-aware strategies.

---

## Demo Outputs

| Script | Output File | Content |
|---|---|---|
| `demo_phase1.py` | `phase1_demo_output.png` | Trade price series, volume |
| `demo_phase2.py` | `phase2_demo_output.png` | Fair value vs midprice, spread, imbalance |
| `demo_phase3.py` | `phase3_demo_output.png` | Inventory, PnL, spread (NMM vs IAMM) |
| `demo_phase4.py` | `phase4_demo_output.png` | σ̂, spread, reservation price, drawdown |
| `demo_phase5.py` | `phase5_demo_output.png` | Regime timeline, γ, quote size, 4-strategy drawdown |

---

## Repository Structure

```
src/exchange/
  order.py              Order dataclass, enums
  order_book.py         Price-time priority LOB
  matching_engine.py    Matching logic, trade log
  trade.py              Immutable Trade record
  trade_log.py          DataFrame export

src/agents/
  base_agent.py         Abstract agent, AgentMetrics
  noise_trader.py       Randomized order flow
  informed_trader.py    Signal-driven trading

src/simulation/
  fair_value.py         Gaussian + jump fair value process
  market_state.py       Immutable market snapshot
  market_simulation.py  Event loop, fill routing
  metrics.py            Time series collector

src/strategies/
  base_market_maker.py               Quote lifecycle, PnL accounting
  naive_market_maker.py              Fixed symmetric spread
  inventory_aware_market_maker.py    Linear skew + spread widening
  avellaneda_stoikov_market_maker.py A-S optimal quoting
  regime_aware_as_market_maker.py    Regime-adaptive A-S
  mm_metrics.py                      MM-specific metrics

src/models/
  avellaneda_stoikov_math.py  A-S formulas (pure functions)
  volatility.py               Rolling σ estimator
  arrival_intensity.py        Fill-frequency k estimator
  analytics.py                Sharpe, drawdown, comparison
  regime.py                   Classifier, thresholds, parameters
```

---

## Testing

```bash
pytest tests/ -v     # 327 tests, all passing
```

Coverage spans: order book correctness, FIFO enforcement, partial fills, agent behavior, simulation lifecycle, A-S formula numerics, volatility estimator bounds, regime classification, hysteresis, per-regime metrics, drawdown reduction in volatile paths. All tests are deterministic under fixed seeds.

---

## How To Run

```bash
git clone https://github.com/ajiteshm2003/market-making-engine
cd market-making-engine
pip install -r requirements.txt

pytest tests/ -q                     # verify: 327 passed

python examples/demo_phase1.py       # matching engine
python examples/demo_phase2.py       # agent simulation
python examples/demo_phase3.py       # NMM vs IAMM
python examples/demo_phase4.py       # A-S vs static strategies
python examples/demo_phase5.py       # regime-aware ASMM
```

---

## Dependencies

```
numpy, pandas, matplotlib, scipy, statsmodels, pytest
```

No machine learning libraries required. Complexity lives in the modeling.

---

## Future Work

- Historical L2 order book replay (Binance, Coinbase)
- Hawkes process order arrivals for clustered flow modeling
- Multi-level quote management and queue position estimation
- Reinforcement learning policy overlay (PPO/DQN)
- C++ matching engine for microsecond latency simulation
- Multi-asset market making with cross-asset hedging

---

## References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
- Glosten, L. & Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
- Ho, T. & Stoll, H. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.
- O'Hara, M. (1995). *Market Microstructure Theory*. Blackwell.
- Johnson, B. (2010). *Algorithmic Trading & DMA*. 4Myeloma Press.├── src/
│   ├── exchange/
│   ├── market_maker/
│   └── strategies/
├── tests/
└── examples/
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, feel free to open an issue or contact the maintainers.


## Phase 1: Matching Engine Complete

Implemented and verified:

- Limit order book
- FIFO queue priority
- Limit orders
- Market orders
- Cancellations
- Partial fills
- Trade logging
- Execution summary statistics
- Demo simulation with plots
- Pytest validation suite

Current validation:

- 42/42 tests passing
- Demo generated 159 simulated trades
- Trade log and execution summary working

## Phase 2: Agent-Based Market Simulation Complete

Implemented:
- Noise traders
- Informed traders
- Event-driven simulation loop
- Fair value process
- Immutable market state snapshots
- Simulation metrics tracking

Validation:
- 102/102 tests passing

- ## Phase 3: Market Making Strategies Complete

Implemented:
- Naive market maker
- Inventory-aware market maker
- Quote cancellation/reposting
- Inventory, cash, realized/unrealized PnL tracking
- Spread capture and fill metrics
- Strategy comparison demo

Validation:
- 159/159 tests passing

Demo result:
- Inventory variance reduced from 342.8 to 10.6
- IAMM improved PnL by 26.19 versus NMM
- Bid/ask fill ratio improved from 0.65 to 1.00

- ## Phase 4: Avellaneda-Stoikov Market Maker Complete

Implemented:
- Avellaneda-Stoikov reservation price
- Optimal half-spread formula
- Rolling volatility estimator
- Arrival intensity estimator
- Sharpe, drawdown, and strategy comparison analytics
- Three-way comparison: NMM vs IAMM vs ASMM

Validation:
- 254/254 tests passing

Demo result:
- ASMM Sharpe: +1.49
- NMM Sharpe: +0.42
- IAMM Sharpe: -1.17
- ASMM achieved best PnL and lowest max drawdown

- ## Phase 5: Regime-Aware Market Making Complete

Implemented:
- Volatility regime classifier
- LOW / MEDIUM / HIGH / EXTREME regime detection
- Regime-aware Avellaneda-Stoikov market maker
- Dynamic gamma, spread, quote size, and inventory-limit adjustment
- PnL-by-regime and drawdown analysis

Validation:
- 327/327 tests passing

Key result:
- Regime-aware ASMM reduced max drawdown to 17.9 vs 450.9 for naive market making in an extreme-volatility simulation
