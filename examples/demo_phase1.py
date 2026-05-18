"""
examples/demo_phase1.py
-----------------------
Phase 1 Demo — Matching Engine Walkthrough

Designed to run cell-by-cell in Google Colab.
Copy the full project into Colab, then run:

    !pip install pandas matplotlib
    %run examples/demo_phase1.py

Or run locally:

    python examples/demo_phase1.py

Demonstrates:
1. Basic limit order resting
2. Limit-vs-limit crossing
3. Market order execution
4. FIFO queue priority
5. Partial fills
6. Cancellations
7. Trade log export to DataFrame
8. Spread / midprice / depth snapshot
9. Simple price time-series plot
"""

import sys
import os
import time

# Allow running from repo root OR from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.exchange import (
    MatchingEngine,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    execution_summary,
    trades_to_dataframe,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

_oid = 0

def next_id(prefix="O") -> str:
    global _oid
    _oid += 1
    return f"{prefix}{_oid:04d}"

def limit(side, qty, price) -> Order:
    return Order(next_id("L"), side, OrderType.LIMIT, quantity=qty, price=price)

def market(side, qty) -> Order:
    return Order(next_id("M"), side, OrderType.MARKET, quantity=qty)

def divider(title=""):
    width = 60
    print("\n" + "─" * width)
    if title:
        print(f"  {title}")
        print("─" * width)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic Resting
# ─────────────────────────────────────────────────────────────────────────────

divider("1. BASIC RESTING — NO MATCH")

engine = MatchingEngine()

# Post a few bids and asks that don't cross
bids = [limit(OrderSide.BUY,  10, 99.0),
        limit(OrderSide.BUY,   5, 98.0),
        limit(OrderSide.BUY,   8, 97.0)]

asks = [limit(OrderSide.SELL,  7, 101.0),
        limit(OrderSide.SELL,  4, 102.0),
        limit(OrderSide.SELL,  6, 103.0)]

for o in bids + asks:
    trades = engine.submit(o)
    assert trades == [], f"Unexpected trade: {trades}"

print(engine.book)
bids_snap, asks_snap = engine.book.depth_snapshot()
print("\nBid depth (price, qty):")
for p, q in bids_snap:
    print(f"  {p:.2f}  x  {q:.2f}")
print("\nAsk depth (price, qty):")
for p, q in asks_snap:
    print(f"  {p:.2f}  x  {q:.2f}")
print(f"\nSpread      : {engine.book.spread:.2f}")
print(f"Midprice    : {engine.book.midprice:.2f}")
print(f"Imbalance   : {engine.book.order_imbalance():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Limit-vs-Limit Crossing
# ─────────────────────────────────────────────────────────────────────────────

divider("2. LIMIT BUY CROSSES LIMIT SELL")

engine2 = MatchingEngine()
engine2.submit(limit(OrderSide.SELL, 10, 100.0))  # rests as best ask

buy_order = limit(OrderSide.BUY, 10, 100.0)       # crosses
trades = engine2.submit(buy_order)

print(f"Trades generated: {len(trades)}")
t = trades[0]
print(f"  price={t.price}  qty={t.quantity}  aggressor={t.aggressor_side.value}")
print(f"  maker={t.maker_order_id}  taker={t.taker_order_id}")
print(f"Book after match: {engine2.book}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Market Order
# ─────────────────────────────────────────────────────────────────────────────

divider("3. MARKET ORDER SWEEPS TWO LEVELS")

engine3 = MatchingEngine()
engine3.submit(limit(OrderSide.SELL,  5, 100.0))
engine3.submit(limit(OrderSide.SELL,  5, 101.0))
engine3.submit(limit(OrderSide.SELL,  5, 102.0))

mkt = market(OrderSide.BUY, 12)
trades = engine3.submit(mkt)

print(f"Market BUY 12 → {len(trades)} trades:")
for t in trades:
    print(f"  price={t.price}  qty={t.quantity}")
print(f"Taker remaining: {mkt.remaining_quantity}  status: {mkt.status.value}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIFO Queue Priority
# ─────────────────────────────────────────────────────────────────────────────

divider("4. FIFO PRIORITY AT SAME PRICE")

engine4 = MatchingEngine()

# Three SELL orders at the same price, in time order
s1 = Order("FIFO_S1", OrderSide.SELL, OrderType.LIMIT, quantity=3, price=100.0, timestamp=1.0)
s2 = Order("FIFO_S2", OrderSide.SELL, OrderType.LIMIT, quantity=3, price=100.0, timestamp=2.0)
s3 = Order("FIFO_S3", OrderSide.SELL, OrderType.LIMIT, quantity=3, price=100.0, timestamp=3.0)

for s in [s1, s2, s3]:
    engine4.submit(s)

buy = limit(OrderSide.BUY, 5, 100.0)
trades = engine4.submit(buy)

print(f"BUY 5 matched {len(trades)} trades:")
for t in trades:
    print(f"  maker={t.maker_order_id}  qty={t.quantity}")
print("Expected: FIFO_S1 (full 3) then FIFO_S2 (partial 2)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Partial Fill — Maker Side
# ─────────────────────────────────────────────────────────────────────────────

divider("5. PARTIAL FILL — MAKER STAYS IN BOOK")

engine5 = MatchingEngine()
big_sell = limit(OrderSide.SELL, 20, 100.0)
engine5.submit(big_sell)

small_buy = limit(OrderSide.BUY, 7, 100.0)
trades = engine5.submit(small_buy)

print(f"BUY 7 vs SELL 20 → {trades[0].quantity} filled")
print(f"Maker remaining: {big_sell.remaining_quantity}  status: {big_sell.status.value}")
print(f"Book best_ask  : {engine5.book.best_ask}  (should still be 100.0)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Cancellation
# ─────────────────────────────────────────────────────────────────────────────

divider("6. CANCEL A RESTING ORDER")

engine6 = MatchingEngine()
sell_order = limit(OrderSide.SELL, 10, 105.0)
engine6.submit(sell_order)
print(f"Before cancel: best_ask={engine6.book.best_ask}")

cancelled = engine6.cancel(sell_order.order_id)
print(f"Cancelled: {cancelled.order_id}  status: {cancelled.status.value}")
print(f"After cancel : best_ask={engine6.book.best_ask}")

# Verify a crossing BUY now rests (nothing to match against)
trades = engine6.submit(limit(OrderSide.BUY, 5, 105.0))
print(f"Trades after cancel: {len(trades)}  (expected 0)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Simulated Session — Trade Log to DataFrame
# ─────────────────────────────────────────────────────────────────────────────

divider("7. SIMULATED SESSION — TRADE LOG")

engine7 = MatchingEngine()
import random
random.seed(42)

mid = 100.0
timestamps = []
midprices = []

for step in range(80):
    # Random walk the fair value
    mid += random.gauss(0, 0.05)

    # Post resting limit orders around mid
    spread = 0.10
    b = Order(
        f"B_{step}",
        OrderSide.BUY,
        OrderType.LIMIT,
        quantity=round(random.uniform(1, 10), 2),
        price=round(mid - spread / 2, 2),
    )
    a = Order(
        f"A_{step}",
        OrderSide.SELL,
        OrderType.LIMIT,
        quantity=round(random.uniform(1, 10), 2),
        price=round(mid + spread / 2, 2),
    )
    engine7.submit(b)
    engine7.submit(a)

    # Random market orders (30% chance each side)
    if random.random() < 0.30:
        engine7.submit(Order(f"MB_{step}", OrderSide.BUY, OrderType.MARKET,
                             quantity=round(random.uniform(1, 5), 2)))
    if random.random() < 0.30:
        engine7.submit(Order(f"MS_{step}", OrderSide.SELL, OrderType.MARKET,
                             quantity=round(random.uniform(1, 5), 2)))

    mp = engine7.book.midprice
    if mp is not None:
        timestamps.append(time.time() + step * 0.1)
        midprices.append(mp)

df = trades_to_dataframe(engine7.trade_log)
print(f"\nTrade log ({len(df)} trades):")
print(df.head(10).to_string(index=False))

print("\nExecution summary:")
summary = execution_summary(engine7.trade_log)
for k, v in summary.items():
    print(f"  {k:<24} {v}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

divider("8. PLOTS")

if len(df) == 0:
    print("No trades to plot.")
else:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Phase 1 — Matching Engine Demo", fontsize=14, fontweight="bold")

    # --- Trade price time series ---
    ax1 = axes[0]
    ax1.plot(df["timestamp"], df["price"], marker="o", markersize=3, linewidth=1, color="#1f77b4", label="Trade Price")
    ax1.set_ylabel("Price")
    ax1.set_title("Trade Price Over Time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Volume per trade ---
    ax2 = axes[1]
    colors = ["#2ecc71" if s == "buy" else "#e74c3c" for s in df["aggressor_side"]]
    ax2.bar(range(len(df)), df["quantity"], color=colors, alpha=0.8, label="Volume")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Quantity")
    ax2.set_title("Trade Volume (green=buy-initiated, red=sell-initiated)")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("phase1_demo_output.png", dpi=150)
    print("Plot saved to: phase1_demo_output.png")
    plt.show()

divider("DONE")
print("Phase 1 matching engine verified.")
print(f"Total trades in session: {len(engine7.trade_log)}")
print(f"Resting orders in book : {len(engine7.book)}")
