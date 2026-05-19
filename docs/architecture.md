# System Architecture

This document describes the internal architecture of the market making engine: how components are organized, how data flows between them, and the design decisions that govern each layer.

---

## High-Level Structure

The system is organized into five packages with a strict dependency hierarchy:

```
models/         ← no internal dependencies
exchange/       ← no internal dependencies
agents/         ← depends on exchange/
simulation/     ← depends on exchange/, agents/, models/
strategies/     ← depends on exchange/, simulation/, models/
```

Nothing in `exchange/` or `models/` imports from `agents/`, `simulation/`, or `strategies/`. This ensures that the exchange core and mathematical models can be tested in isolation.

---

## Exchange Layer (`src/exchange/`)

The exchange implements a realistic continuous limit order book with price-time priority matching.

### Order Representation

```python
@dataclass
class Order:
    order_id: str
    side: OrderSide          # BUY or SELL
    order_type: OrderType    # LIMIT, MARKET, or CANCEL
    quantity: float
    price: Optional[float]   # None for market orders
    timestamp: float         # used for FIFO ordering within price level
    remaining_quantity: float
    status: OrderStatus
```

Orders are value objects. Status transitions (OPEN → PARTIALLY_FILLED → FILLED) are the only mutations allowed after construction.

### Order Book Structure

```
OrderBook
├── _bids: Dict[price, Deque[Order]]   # sorted descending
├── _asks: Dict[price, Deque[Order]]   # sorted ascending
└── _order_map: Dict[order_id, Order]  # fast lookup for cancellations
```

Each price level holds a `deque` of orders in insertion order, providing O(1) access to the FIFO front. The `_order_map` enables O(1) cancellation without a full book scan.

`best_bid` and `best_ask` are computed as `max(bids.keys())` and `min(asks.keys())`. For real-time systems these would be maintained as sorted structures; in simulation context the dict lookup is acceptable.

### Matching Engine

The matching engine is the only component that mutates order state. Its `submit(order)` method:

1. Validates the order (no duplicates, valid type)
2. For market orders: calls `_consume_asks` or `_consume_bids` until filled or book exhausted
3. For limit orders: first attempts aggressive matching (as above, with price limit); any remainder rests in the book via `book.add_limit_order()`
4. For each fill: calls `_execute(maker, taker, price)` which updates remaining quantities and status, creates a `Trade` record, and appends it to the internal trade log

The execution price is always the **maker's price** — standard in real exchanges.

### Trade Record

```python
@dataclass(frozen=True)
class Trade:
    trade_id: str
    timestamp: float
    price: float
    quantity: float
    aggressor_side: OrderSide
    maker_order_id: str
    taker_order_id: str
```

Frozen dataclass ensures immutability. Trade records are the canonical audit trail.

---

## Agent Layer (`src/agents/`)

Agents interact with the exchange only through the simulation mediator — they never call `engine.submit()` directly. This enforces the correct information barrier.

### BaseAgent

```python
class BaseAgent(ABC):
    def act(self, state: MarketState) -> List[Order]: ...
    def notify_fill(self, trade: Trade, as_maker: bool) -> None: ...
    def update_unrealized_pnl(self, midprice: float) -> None: ...
    def flush_cancels(self) -> List[str]: ...
```

`act()` receives a read-only `MarketState` and returns a list of `Order` objects. The simulation submits these to the engine on the agent's behalf. `notify_fill()` is called by the simulation when one of the agent's orders executes. `flush_cancels()` returns order ids to cancel before this step's submissions.

### Information Barrier

Agents receive `MarketState` — an immutable snapshot containing:
- `best_bid`, `best_ask`, `midprice`, `spread`
- `bid_depth`, `ask_depth` (top N price levels)
- `order_imbalance`
- `fair_value` (only accessible because this is a simulation; real agents do not observe this)
- `volume_this_step`, `trade_count`

They cannot access the engine's internal order map, the book's full depth, or other agents' positions.

---

## Simulation Layer (`src/simulation/`)

### Fair Value Process

```python
class FairValueProcess:
    def step(self) -> float:
        diffusion = drift + volatility * N(0,1)
        jump = N(0, jump_std) if U(0,1) < jump_prob else 0
        self._value = max(min_price, self._value + diffusion + jump)
        return self._value
```

The jump component is the primary mechanism by which informed traders gain an edge: their signal derives from the new fair value; the market maker's quotes reflect the old one.

### Simulation Loop

Each timestep in `MarketSimulation.run()`:

```
1. fv_process.step()                  → new fair value
2. for each agent (randomized order):
       orders = agent.act(state)
       cancel_ids += agent.flush_cancels()
3. for oid in cancel_ids:
       engine.cancel(oid)
4. for (agent, order) in all_orders:
       _order_owner[order.order_id] = agent.agent_id
       trades = engine.submit(order)
5. for trade in trades:
       _notify_agents(trade)           → routes to maker + taker agents
6. for agent in agents:
       agent.update_unrealized_pnl(midprice)
7. metrics.record(...)
8. state = _build_state(...)           → snapshot for next step
```

Agent ordering is shuffled each step using the simulation's private RNG. This prevents any systematic first-mover advantage that would bias strategy comparisons.

### Fill Routing

The `_order_owner` dictionary maps `order_id → agent_id`. When a trade executes, both the maker and taker agents receive `notify_fill()` calls with the trade record and their role (`as_maker=True/False`). This enables each agent to maintain its own inventory and cash accounting without the simulation needing to know the agent's internal structure.

---

## Strategy Layer (`src/strategies/`)

### BaseMarketMaker

Market makers extend `BaseMarketMaker`, which handles the quote lifecycle. Every timestep:

```
act(state):
    1. Schedule old bid/ask for cancellation (via _pending_cancels)
    2. Call _compute_quotes(state)   ← subclass implements this
    3. If (bid, ask) returned:
           create limit orders at those prices with self.quote_size
           record new order ids
           update mm_metrics.snapshot()
    4. Return [bid_order, ask_order]
```

This cancel-and-requote pattern ensures the book never accumulates stale MM orders. At most two MM orders rest in the book at any time (current bid + current ask).

`BaseMarketMaker` does **not** inherit from `BaseAgent`. Market makers have fundamentally different accounting (per-regime fills, spread capture, bid/ask fill counts) that doesn't fit the general `AgentMetrics` structure. A `metrics` property provides a compatibility shim so the simulation's summary printer can access `inventory` and `total_pnl` uniformly.

### Strategy Hierarchy

```
BaseMarketMaker
├── NaiveMarketMaker
├── InventoryAwareMarketMaker
└── AvellanedaStoikovMarketMaker
    └── RegimeAwareAvellanedaStoikovMarketMaker
```

Each subclass overrides only `_compute_quotes(state) → (bid, ask)`. The quote lifecycle, cancellation, fill handling, and PnL accounting are inherited from `BaseMarketMaker`.

### Regime-Aware ASMM Override

`RegimeAwareAvellanedaStoikovMarketMaker._compute_quotes()` additionally:
- Reads `self.quote_size` from `ra_config.base_quote_size × regime_quote_size_mult`
- Writes `self.quote_size` before returning, so `BaseMarketMaker.act()` uses the regime-adjusted quantity
- Records per-step regime, gamma, quote_size, spread_mult, max_inv into history lists
- Appends per-regime inventory snapshots and PnL deltas for post-hoc analysis

---

## Models Layer (`src/models/`)

### Volatility Estimator

```
prices: Deque[float]  (maxlen = window + 1)
returns = [ln(p[i]/p[i-1]) for i in 1..n]
sigma = sample_std(returns)
sigma = clamp(sigma, min_vol, max_vol)
```

The `maxlen` deque automatically discards prices beyond the window, giving O(1) update. During warm-up (fewer than `window` observations), the estimate is blended toward `initial_vol` to prevent spurious volatility readings from small samples.

### Arrival Intensity Estimator

```
k = fill_scale / avg_fills_per_step_in_window
k = clamp(k, k_min, k_max)
```

When `avg_fills_per_step = 0`, k defaults to `k_default` (conservative prior producing moderately wide spread).

### Regime Classifier

The classifier maintains a current regime and applies hysteresis on transitions:
- **Upgrade** (e.g., MEDIUM → HIGH): only if σ > upper_boundary + hysteresis
- **Downgrade** (e.g., HIGH → MEDIUM): only if σ < lower_boundary - hysteresis

This prevents rapid chattering near boundaries while still responding promptly to genuine regime changes.

Transition events are recorded with timestep, from/to regime, and σ at transition — enabling post-hoc analysis of when regime changes occurred relative to jump events.

### Analytics

Pure functions with no state:
- `sharpe_ratio(pnl_series)`: step-to-step returns, annualized by √252
- `max_drawdown(pnl_series)`: peak-to-trough maximum decline
- `strategy_comparison(strategies_dict)`: builds comparison DataFrame from all MM instances

---

## Data Flow Summary

```
FairValueProcess.step()
        │
        ▼
MarketState (immutable snapshot)
        │
        ├──▶ NoiseTrader.act()     → [Order, ...]
        ├──▶ InformedTrader.act()  → [Order, ...]  (uses fair_value from state)
        └──▶ MarketMaker.act()     → [Order, ...]
                  │
                  │  _compute_quotes(state)
                  │       ├──▶ VolatilityEstimator.update(midprice)  → σ
                  │       ├──▶ RegimeClassifier.update(σ)            → regime
                  │       ├──▶ ArrivalIntensityEstimator.update()    → k
                  │       └──▶ AS math: reservation_price(), optimal_half_spread()
                  │
                  ▼
        MatchingEngine.submit(order)
                  │
                  ▼
        Trade records
                  │
                  ├──▶ agent.notify_fill()     → inventory, cash, PnL
                  └──▶ SimulationMetrics.record()
```

---

## Testing Architecture

Each module has dedicated test files:

| Test File | Scope |
|---|---|
| `test_order_book.py` | Order construction, book insertion, cancellation, depth, imbalance |
| `test_matching_engine.py` | Resting, crossing, partial fills, FIFO, market orders, trade log |
| `test_agents.py` | NoiseTrader, InformedTrader, fill notifications, PnL |
| `test_simulation.py` | FairValueProcess, MarketState, simulation lifecycle, metrics |
| `test_strategies.py` | NMM, IAMM quote generation, cancels, fills, integration |
| `test_phase4.py` | A-S math formulas, volatility/k estimators, ASMM behavior |
| `test_phase5.py` | Regime classifier, thresholds, hysteresis, R-ASMM behavior |

All tests are deterministic under fixed random seeds. Integration tests run full simulations (100–500 steps) and assert invariants on aggregate outcomes.
