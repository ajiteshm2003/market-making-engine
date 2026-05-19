"""
examples/demo_historical_replay.py
------------------------------------
Demonstrates the historical replay engine on synthetic L2 data.
Shows how real exchange data would be processed for regime analysis.

Run:
    python examples/demo_historical_replay.py
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.replay import (
    SyntheticMarketDataGenerator, HistoricalReplayEngine,
    BinanceMarketDataLoader, CoinbaseMarketDataLoader,
    save_replay_to_csv, load_replay_from_csv,
)
from src.models.volatility import VolatilityConfig
from src.models.regime import RegimeThresholds

def divider(t=""): print(f"\n{'═'*60}\n  {t}\n{'═'*60}" if t else "\n"+"─"*60)

SEED = 42

# ── 1. Generate synthetic L2 data ────────────────────────────────────────────
divider("1. SYNTHETIC MARKET DATA GENERATION")

gen = SyntheticMarketDataGenerator(
    n_steps=400, initial_price=100.0,
    volatility=0.05, jump_prob=0.06, jump_std=2.0,
    spread_ticks=0.06, depth_levels=5, seed=SEED,
)
stream = gen.generate()
print(f"  Events generated   : {len(stream)}")
print(f"  Snapshots          : {stream.n_snapshots}")
print(f"  Updates            : {stream.n_updates}")
print(f"  Trades             : {stream.n_trades}")
t0, t1 = stream.time_range
print(f"  Time range         : {t0:.0f} → {t1:.0f}")
print(f"  Stream ordered     : {stream.is_ordered()}")

# ── 2. CSV round-trip ────────────────────────────────────────────────────────
divider("2. CSV SAVE AND RELOAD")

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    tmppath = f.name

save_replay_to_csv(stream, tmppath)
loaded_stream = load_replay_from_csv(tmppath)
os.unlink(tmppath)

print(f"  Original events    : {len(stream)}")
print(f"  After CSV round-trip: {len(loaded_stream)}")
print(f"  Trades preserved   : {stream.n_trades == loaded_stream.n_trades}")

# ── 3. Run replay engine ─────────────────────────────────────────────────────
divider("3. HISTORICAL REPLAY ENGINE")

engine = HistoricalReplayEngine(
    vol_config=VolatilityConfig(window=20, initial_vol=0.002, max_vol=1.0),
    thresholds=RegimeThresholds(
        low_threshold=0.0008,
        high_threshold=0.0035,
        extreme_threshold=0.0055,
        hysteresis=0.0002,
    ),
    symbol="SIM",
)
result = engine.process(stream)
result.print_summary()

# ── 4. Volatility clustering ─────────────────────────────────────────────────
divider("4. VOLATILITY CLUSTERING ANALYSIS")

sigmas = [s.sigma for s in result.steps if s.sigma > 0]
if sigmas:
    mean_s = sum(sigmas) / len(sigmas)
    high_vol_steps = sum(1 for s in sigmas if s > 2 * mean_s)
    print(f"  Mean σ̂       : {mean_s:.6f}")
    print(f"  Max σ̂        : {max(sigmas):.6f}")
    print(f"  Steps > 2×mean: {high_vol_steps} ({100*high_vol_steps/len(sigmas):.1f}%)")
    print(f"  Clustering ratio: {max(sigmas)/mean_s:.2f}× (>2 = detectable clustering)")

# ── 5. Binance/Coinbase loader demos ─────────────────────────────────────────
divider("5. EXCHANGE LOADERS (OFFLINE MODE)")

binance = BinanceMarketDataLoader(seed=SEED)
binance_stream = binance.load(n_steps=100)
print(f"  Binance synthetic: {len(binance_stream)} events, ordered={binance_stream.is_ordered()}")

coinbase = CoinbaseMarketDataLoader(seed=SEED)
coinbase_stream = coinbase.load(n_steps=100)
print(f"  Coinbase synthetic: {len(coinbase_stream)} events, ordered={coinbase_stream.is_ordered()}")

# ── 6. Replay with Binance-scale data ────────────────────────────────────────
divider("6. BINANCE-SCALE REPLAY (30,000 BTC price)")

btc_gen = SyntheticMarketDataGenerator(
    n_steps=200, initial_price=30_000.0,
    volatility=0.01, jump_prob=0.03, jump_std=200.0,
    spread_ticks=3.0, depth_levels=5, seed=SEED,
)
btc_stream = btc_gen.generate()
btc_engine = HistoricalReplayEngine(
    vol_config=VolatilityConfig(window=20, initial_vol=0.001),
    symbol="BTCUSDT",
)
btc_result = btc_engine.process(btc_stream)
btc_result.print_summary()

divider("DONE")
print("  Historical replay demo complete.")
print("  To use with real data:")
print("    1. Download Binance L2 CSV from https://data.binance.vision/")
print("    2. loader = BinanceMarketDataLoader()")
print("    3. stream = loader.load(filepath='BTCUSDT-depth-2024-01-01.csv')")
print("    4. result = engine.process(stream)")
