# Market Making Engine

A simulated electronic exchange and adaptive market-making framework with volatility regime-aware liquidity provision.

## Overview

This project implements a sophisticated market-making system designed to:
- Simulate electronic exchange operations
- Provide adaptive liquidity provisioning strategies
- Respond to different market volatility regimes
- Optimize market-maker profitability while managing risk

## Getting Started

### Prerequisites

- Python 3.8+
- Required dependencies (see `requirements.txt`)

### Installation

```bash
git clone https://github.com/ajiteshm2003/market-making-engine.git
cd market-making-engine
pip install -r requirements.txt
```

### Usage

```python
# Example usage
from market_making_engine import Exchange, MarketMaker

# Initialize exchange and market maker
exchange = Exchange()
market_maker = MarketMaker()

# Run simulation
results = exchange.run(market_maker)
```

## Features

- **Volatility Regime Detection**: Automatically identifies and adapts to different market conditions
- **Adaptive Spread Strategy**: Dynamically adjusts spreads based on market conditions
- **Risk Management**: Built-in position limits and stop-loss mechanisms
- **Performance Analytics**: Detailed metrics on profitability, liquidity provision, and risk

## Project Structure

```
market-making-engine/
├── README.md
├── requirements.txt
├── src/
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
