# Figures

This directory contains output figures from the demo scripts. To generate all figures, run the demo scripts from the repository root:

```bash
python examples/demo_phase1.py   # → phase1_demo_output.png
python examples/demo_phase2.py   # → phase2_demo_output.png
python examples/demo_phase3.py   # → phase3_demo_output.png
python examples/demo_phase4.py   # → phase4_demo_output.png
python examples/demo_phase5.py   # → phase5_demo_output.png
```

Move the generated PNG files into this directory:

```bash
mv phase*_demo_output.png docs/figures/
```

## Figure Descriptions

### phase1_demo_output.png
Matching engine walkthrough. Shows trade price time series and volume colored by aggressor side (green = buy-initiated, red = sell-initiated). Validates that the matching engine correctly records all execution metadata.

### phase2_demo_output.png
Agent simulation overview. Six-panel figure showing:
- Fair value vs midprice (price discovery)
- Bid-ask spread over time
- Order imbalance (± 1 scale)
- Cumulative volume
- Trade price distribution
Demonstrates that noise traders generate realistic two-sided flow and that midprice converges toward fair value through informed trader activity.

### phase3_demo_output.png
Naive MM vs Inventory-Aware MM comparison. Five-panel figure:
- Quote midpoints vs fair value
- Inventory trajectories (NMM variance >> IAMM variance)
- PnL trajectories (solid = total, dotted = realized)
- Quoted spreads
- Cumulative fills as maker
Shows that IAMM reduces inventory variance by ~97% relative to NMM but at the cost of reduced fill income.

### phase4_demo_output.png
Avellaneda-Stoikov vs static strategies. Eight-panel figure:
- Price + quote midpoints
- Rolling σ̂ estimate
- Quoted spreads (ASMM widens dynamically post-jump)
- Inventory trajectories
- PnL with Sharpe ratios
- Drawdown comparison
- Reservation price vs midprice (inventory adjustment visible)
Shows that A-S achieves better Sharpe by jointly optimizing spread and skew.

### phase5_demo_output.png
Regime-Aware ASMM in volatile market. Nine-panel dashboard:
- Price discovery with regime-colored background (green=LOW, blue=MEDIUM, orange=HIGH, red=EXTREME)
- σ̂ with regime threshold lines
- Quoted spread comparison (R-ASMM widens in HIGH/EXTREME)
- Effective γ over time (R-ASMM γ rises in HIGH/EXTREME)
- Effective quote size (R-ASMM size drops in HIGH/EXTREME)
- All four inventory trajectories
- Cumulative PnL with Sharpe labels
- Drawdown comparison (R-ASMM has lowest peak-to-trough)
Primary figure demonstrating the regime-aware strategy's tail protection.
