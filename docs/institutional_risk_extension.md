# Institutional Risk Extension

This document describes the risk analytics, stress testing, historical replay, and performance profiling infrastructure added in the final phase of the engine. Together these modules transform the project from a simulation platform into infrastructure suitable for deployment in an institutional quantitative research or trading environment.

---

## Overview

The extension adds three new top-level packages:

```
src/risk/        VaR, Expected Shortfall, portfolio exposure, stress testing
src/replay/      Market data abstractions, historical replay engine, exchange loaders
src/performance/ Matching engine benchmarking and latency profiling
```

None of these packages modifies the existing engine. They consume outputs from the simulation (PnL series, inventory trajectories, market maker metrics) and from the replay infrastructure (L2 event streams).

---

## Part 1: Risk Engine (`src/risk/`)

### Value-at-Risk

VaR answers: *"What is the maximum loss we expect not to exceed with probability α?"*

Two estimation methods are implemented:

**Historical VaR** uses the empirical quantile of the P&L distribution. It makes no distributional assumptions and is the method preferred by most sell-side risk managers.

```python
result = historical_var(pnl_returns, confidence=0.95)
# result.var: the 95% loss threshold (positive = loss)
```

**Parametric Gaussian VaR** assumes returns are normally distributed and applies the z-score:

```
VaR = -(μ - z_α · σ)
```

For a standard normal: z_0.95 = 1.645, z_0.99 = 2.326. The approximation is accurate for well-behaved return distributions but underestimates tail risk for fat-tailed (leptokurtic) P&L — which is common in market making due to occasional large inventory losses.

### Expected Shortfall (CVaR)

ES answers: *"Given that losses exceed VaR, what is the expected loss?"*

```python
result = expected_shortfall(pnl_returns, confidence=0.95)
# result.es: average loss in the 5% worst scenarios
# result.es >= result.var always
```

ES is a coherent risk measure (satisfies subadditivity); VaR is not. Basel III/IV regulatory frameworks require ES rather than VaR for this reason. ES is particularly relevant for market makers because their loss distribution has a fat left tail from adverse selection during volatile episodes.

### Rolling VaR

```python
rolling = rolling_var(pnl_series, window=100, confidence=0.95)
# List[Optional[float]]: first 99 entries are None (warm-up)
```

Rolling VaR shows how the risk profile evolves over time. Spikes in rolling VaR correspond to volatile episodes — exactly the periods where regime-aware protection is most valuable.

### P&L Distribution Statistics

```python
stats = pnl_distribution_stats(pnl_returns)
# mean, std, Sharpe, skewness, excess kurtosis, percentiles
```

Excess kurtosis > 0 (leptokurtosis) indicates fat tails. For market makers, leptokurtosis arises from rare but large losses during informed-flow episodes. A kurtosis-aware risk manager knows that Gaussian VaR underestimates tail risk.

### Portfolio Exposure Tracker

```python
tracker = PortfolioExposureTracker(initial_cash=100_000)
snap = tracker.update(timestep=t, inventory=q, mark_price=S, cash=C)

# snap fields:
#   notional_exposure = |inventory| × mark_price
#   net_exposure      = inventory × mark_price (signed)
#   equity            = cash + net_exposure
#   leverage          = notional / equity
#   inv_concentration = |inventory| / max_abs_inventory_seen
```

The leverage metric is particularly important: a leverage > 1.0 means the notional exposure exceeds equity — a position that a risk manager would flag for review. The `time_overleveraged` metric in the summary shows what fraction of the simulation was spent in this state.

---

## Part 2: Stress Testing Engine (`src/risk/stress.py`)

### Stress Scenarios

Six pre-built scenarios are defined, each designed to expose a specific vulnerability in market-making strategies:

| Scenario | Description | Key stress |
|---|---|---|
| baseline | Standard parameters | Reference |
| flash_crash | High jump_std, aggressive informed traders | Adverse selection, inventory |
| volatility_spike | 2× diffusion volatility, frequent jumps | Spread adequacy, VaR exceedances |
| liquidity_drought | 20% noise activity | Adverse selection without noise |
| informed_flow_attack | 4 informed traders, threshold=0.02, aggression=0.99 | Maximum adverse selection |
| spread_collapse | 90% noise activity, very tight volatility | Fill income compression |

### Running Stress Tests

```python
runner = StressTestRunner(n_steps=400)
results = runner.run_all()
StressTestRunner.print_summary(results)
```

Each scenario runs all four market makers (NMM, IAMM, ASMM, RASMM) against the same simulation configuration. Results include:

- `total_pnl`: final mark-to-market P&L
- `max_drawdown`: peak-to-trough decline
- `var_95`, `var_99`: VaR at 95% and 99% confidence
- `es_95`: Expected Shortfall at 95%
- `inv_variance`: variance of inventory trajectory
- `worst_step_loss`: largest single-step P&L loss
- `final_inventory`: position at end of simulation

### Interpreting Stress Results

The key comparison is not which strategy has the best PnL in stress scenarios — it is which strategy has the **most consistent drawdown protection**. The Regime-Aware ASMM is designed to reduce drawdown in high/extreme volatility scenarios by withdrawing from the book during the most dangerous periods.

A stress test that shows the RASMM with lower drawdown than the ASMM in `flash_crash` and `volatility_spike` scenarios, even if the RASMM has lower total PnL in those scenarios, confirms the strategy is working as intended.

---

## Part 3: Historical Replay Engine (`src/replay/`)

### Architecture

The replay engine bridges simulated and real market data by providing a unified event-stream abstraction:

```
ReplayEventStream
├── L2Snapshot     (full book state at a point in time)
├── L2Update       (incremental price level change)
└── TradeEvent     (recorded market execution)
```

The `HistoricalReplayEngine` processes a `ReplayEventStream` and produces:
- Reconstructed top-of-book state (best bid, best ask, midprice, spread) at each timestamp
- Rolling volatility estimate from midprice returns
- Regime classification at each step
- Summary statistics: mean spread, mean/max σ̂, regime distribution, transition count

### Synthetic Data (Offline Mode)

The `SyntheticMarketDataGenerator` produces a fully realistic event stream from a parametric fair value process, requiring no external data:

```python
gen = SyntheticMarketDataGenerator(n_steps=500, volatility=0.05, jump_prob=0.06)
stream = gen.generate()
engine = HistoricalReplayEngine()
result = engine.process(stream)
```

### CSV Persistence

Streams can be saved to and loaded from CSV for reproducibility and data sharing:

```python
save_replay_to_csv(stream, "market_data.csv")
loaded = load_replay_from_csv("market_data.csv")
```

### Exchange Loaders

Both `BinanceMarketDataLoader` and `CoinbaseMarketDataLoader` follow the same interface:

```python
loader = BinanceMarketDataLoader()
stream = loader.load(filepath="BTCUSDT-depth-2024-01-01.csv")  # from file
stream = loader.load(n_steps=500)                               # synthetic fallback
```

Live API and WebSocket methods are stubbed with `NotImplementedError` and documented with the exact endpoint URLs for future implementation. No API keys are required for any offline functionality.

### Connecting to Real Data

To use real Binance data:
1. Download L2 depth snapshots from `https://data.binance.vision/`
2. Pass the filepath to `BinanceMarketDataLoader.load(filepath=...)`
3. The replay engine handles reconstruction automatically

The only change required for live data is providing a real file path or implementing the WebSocket stub methods.

---

## Part 4: Performance Profiling (`src/performance/`)

### Metrics

The profiler measures two distinct performance characteristics:

**Order matching throughput**: how many orders per second the matching engine can process. This is the critical metric for HFT and market making, where the matching engine is on the critical path for every order submission.

```
Current Python baseline (10,000 orders):
  Throughput: ~50,000 orders/second
  Avg latency: ~20 µs
  P99 latency: ~40 µs
```

**Simulation loop throughput**: how many complete simulation steps (including agent decisions, matching, fill routing, and metrics) the system can process per second.

```
Current Python baseline (500 steps, full agent population):
  Throughput: ~1,000 steps/second
  Avg step latency: ~1,000 µs
```

### C++ Optimization Path

The profiler explicitly documents the path to C++ optimization:

1. Port `OrderBook` to C++ `std::map<double, std::deque<Order>>`
2. Expose Python binding via pybind11
3. Replace `Order` dataclass with C++ struct
4. Profile with `perf`/`valgrind`, targeting cache misses from dict key lookups
5. Consider lock-free concurrent queue for multi-threaded use

**Expected speedup**: 50-200× for pure matching operations. Strategy logic remains in Python; the C++ replacement is the matching engine only.

**Primary bottleneck today**: Python object allocation, which accounts for approximately 80% of matching latency. Every `Order`, `Trade`, and `Deque` operation involves Python memory management overhead that disappears in C++.

### Running Benchmarks

```python
from src.performance import run_full_benchmark, print_cpp_path

result = run_full_benchmark(n_orders=10_000, n_sim_steps=500)
result.print_report()
print_cpp_path()
```

---

## Test Coverage

The extension adds 127 new tests across four test files:

| File | Tests | Coverage |
|---|---|---|
| `test_risk.py` | 49 | VaR correctness, ES, rolling VaR, portfolio exposure |
| `test_stress.py` | 15 | Scenario construction, runner output, metric validation |
| `test_replay.py` | 36 | Event structures, stream ordering, engine, CSV round-trip, loaders |
| `test_performance.py` | 27 | Latency profiles, throughput, memory, benchmark suite |

All 327 existing tests continue to pass. Total: **454 tests passing**.

---

## Running Everything

```bash
# Full test suite
pytest tests/ -q                        # 454 passed

# Risk engine demo
python examples/demo_risk_engine.py

# Stress test all scenarios
python examples/demo_stress_testing.py

# Historical replay with synthetic L2 data
python examples/demo_historical_replay.py

# Matching engine performance benchmark
python examples/demo_performance_profile.py
```

---

## What This Adds to the Project

This extension transforms the repository from a standalone simulation study into infrastructure that could be extended to an institutional quant workflow:

- **Risk monitoring**: VaR, ES, and portfolio exposure metrics are the language of institutional risk management teams. Implementing them correctly from scratch demonstrates understanding of risk measurement theory, not just simulation.

- **Stress testing**: The scenario-based stress testing framework is how production trading desks validate strategy behavior under adverse conditions before deployment. Having pre-built scenarios for flash crashes, liquidity droughts, and informed-flow attacks shows awareness of real market risks.

- **Historical replay**: The event-stream abstraction is the standard architecture for market data infrastructure. The offline/synthetic mode enables testing without live data connections; the exchange loader stubs document the exact integration path for production data.

- **Performance profiling**: Knowing the Python baseline and documenting the C++ optimization path demonstrates systems-level thinking expected at firms like Jane Street, HRT, and Citadel. An interviewer who asks "how would you speed this up?" receives a concrete, measured answer rather than a vague "use C++."
