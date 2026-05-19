"""
src/performance/benchmarks.py
------------------------------
Benchmark Suite

Defines standardized benchmark configurations and comparison utilities.
Produces a formatted performance table suitable for documentation and
interview demonstration.

C++ Path
---------
This module documents the intended C++ optimization path clearly.
When the C++ matching engine is implemented:
  1. The same benchmark configurations run against both implementations
  2. The speedup ratio is computed automatically
  3. The bottleneck analysis section is populated from profiling data

Current Python bottlenecks (in order of impact):
  1. Order object allocation (dataclass __init__ overhead)
  2. Dict key lookups for price level access
  3. Deque operations for FIFO queue management
  4. Python interpreter overhead per function call
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .profiler import (
    LatencyProfile, MemoryProfile, BenchmarkResult,
    profile_matching_engine, run_full_benchmark,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    label:       str
    n_orders:    int
    n_sim_steps: int
    seed:        int = 42
    description: str = ""


# Standard benchmark configurations
BENCHMARK_CONFIGS: List[BenchmarkConfig] = [
    BenchmarkConfig(
        label="small",
        n_orders=1_000,
        n_sim_steps=100,
        description="Quick sanity check (CI-friendly)",
    ),
    BenchmarkConfig(
        label="medium",
        n_orders=10_000,
        n_sim_steps=500,
        description="Standard benchmark for development comparison",
    ),
    BenchmarkConfig(
        label="large",
        n_orders=50_000,
        n_sim_steps=1_000,
        description="Near-production scale for throughput measurement",
    ),
]


def run_benchmark_suite(
    configs: Optional[List[BenchmarkConfig]] = None,
    verbose: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Run a suite of benchmarks and return all results.

    Parameters
    ----------
    configs : list of BenchmarkConfig (defaults to all BENCHMARK_CONFIGS)
    verbose : if True, print progress

    Returns
    -------
    dict[label, BenchmarkResult]
    """
    configs = configs or BENCHMARK_CONFIGS
    results = {}

    for cfg in configs:
        if verbose:
            print(f"  Running benchmark: {cfg.label} ({cfg.n_orders:,} orders)...")
        result = run_full_benchmark(
            n_orders=cfg.n_orders,
            n_sim_steps=cfg.n_sim_steps,
            seed=cfg.seed,
            label=cfg.label,
        )
        results[cfg.label] = result

    return results


def print_benchmark_table(results: Dict[str, BenchmarkResult]) -> None:
    """Print a formatted benchmark comparison table."""
    headers = ["Config", "N Orders", "Throughput/s", "Avg µs", "P50 µs", "P95 µs", "P99 µs"]
    col_w = [10, 10, 14, 9, 9, 9, 9]

    header_row = "  " + "".join(f"{h:<{w}}" for h, w in zip(headers, col_w))
    print(header_row)
    print("  " + "─" * sum(col_w))

    for label, result in results.items():
        p = result.order_throughput
        row = "  " + "".join([
            f"{label:<{col_w[0]}}",
            f"{p.n_operations:>{col_w[1]},}",
            f"{p.ops_per_second:>{col_w[2]},.0f}",
            f"{p.avg_latency_us:>{col_w[3]}.2f}",
            f"{p.p50_latency_us:>{col_w[4]}.2f}",
            f"{p.p95_latency_us:>{col_w[5]}.2f}",
            f"{p.p99_latency_us:>{col_w[6]}.2f}",
        ])
        print(row)


def print_cpp_path() -> None:
    """Print the documented C++ optimization path."""
    print("""
  ┌─── C++ Optimization Path ─────────────────────────────────────────┐
  │                                                                     │
  │  Target: <1µs average matching latency (from ~100µs Python)        │
  │                                                                     │
  │  Step 1: Port OrderBook to C++ std::map<double, std::deque<Order>> │
  │  Step 2: Expose Python binding via pybind11                         │
  │  Step 3: Replace Order dataclass with C++ struct                    │
  │  Step 4: Profile with perf/valgrind, target cache misses           │
  │  Step 5: Consider lock-free queue for concurrent access            │
  │                                                                     │
  │  Expected speedup: 50-200× for pure matching operations            │
  │  Strategy logic remains in Python (C++ is matching only)           │
  │                                                                     │
  │  Primary bottleneck today: Python object allocation (~80% of time) │
  └─────────────────────────────────────────────────────────────────────┘
""")
