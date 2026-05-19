"""
examples/demo_performance_profile.py
--------------------------------------
Benchmarks the matching engine and simulation loop.
Prints throughput and latency distribution tables.

Run:
    python examples/demo_performance_profile.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.performance import (
    profile_matching_engine, profile_simulation_loop, profile_memory,
    run_full_benchmark, print_benchmark_table, print_cpp_path,
    BenchmarkConfig, run_benchmark_suite,
)

def divider(t=""): print(f"\n{'═'*62}\n  {t}\n{'═'*62}" if t else "\n"+"─"*62)

divider("PERFORMANCE PROFILING — Market Making Engine")

# ── 1. Matching engine latency ───────────────────────────────────────────────
divider("1. ORDER MATCHING THROUGHPUT")

for n in [1_000, 5_000, 10_000]:
    p = profile_matching_engine(n_orders=n, seed=42)
    print(f"  {n:>7,} orders  →  {p.ops_per_second:>10,.0f}/s  "
          f"avg={p.avg_latency_us:.1f}µs  p99={p.p99_latency_us:.1f}µs")

# ── 2. Latency distribution ──────────────────────────────────────────────────
divider("2. LATENCY DISTRIBUTION (10,000 orders)")

p = profile_matching_engine(n_orders=10_000, seed=42)
print(f"  {'Metric':<20} {'Value':>12}")
print(f"  {'─'*34}")
print(f"  {'N operations':<20} {p.n_operations:>12,}")
print(f"  {'Throughput (ops/s)':<20} {p.ops_per_second:>12,.0f}")
print(f"  {'Avg latency (µs)':<20} {p.avg_latency_us:>12.2f}")
print(f"  {'P50 latency (µs)':<20} {p.p50_latency_us:>12.2f}")
print(f"  {'P95 latency (µs)':<20} {p.p95_latency_us:>12.2f}")
print(f"  {'P99 latency (µs)':<20} {p.p99_latency_us:>12.2f}")
print(f"  {'Min latency (µs)':<20} {p.min_latency_us:>12.2f}")
print(f"  {'Max latency (µs)':<20} {p.max_latency_us:>12.2f}")

# ── 3. Simulation loop ───────────────────────────────────────────────────────
divider("3. SIMULATION LOOP THROUGHPUT")

sp = profile_simulation_loop(n_steps=500, seed=42)
print(f"  {500} simulation steps:")
print(f"  Throughput        : {sp.ops_per_second:,.0f} steps/second")
print(f"  Avg step latency  : {sp.avg_latency_us:.1f} µs")
print(f"  P95 step latency  : {sp.p95_latency_us:.1f} µs")

# ── 4. Memory ────────────────────────────────────────────────────────────────
divider("4. MEMORY PROFILE (5,000 orders)")

m = profile_memory(n_orders=5_000)
print(m)

# ── 5. Benchmark suite ───────────────────────────────────────────────────────
divider("5. BENCHMARK COMPARISON TABLE")

configs = [
    BenchmarkConfig(label="1k",  n_orders=1_000,  n_sim_steps=100),
    BenchmarkConfig(label="5k",  n_orders=5_000,  n_sim_steps=200),
    BenchmarkConfig(label="10k", n_orders=10_000, n_sim_steps=300),
]
results = run_benchmark_suite(configs=configs, verbose=False)
print_benchmark_table(results)

# ── 6. C++ path ──────────────────────────────────────────────────────────────
print_cpp_path()

divider("DONE")
print("  Performance profile complete.")
print("  Python baseline established for C++ comparison.")
