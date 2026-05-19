"""
tests/test_performance.py
--------------------------
Tests for the performance profiling infrastructure.
"""

import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.performance.profiler import (
    profile_matching_engine, profile_simulation_loop, profile_memory,
    run_full_benchmark, LatencyProfile, MemoryProfile, BenchmarkResult,
)
from src.performance.benchmarks import (
    BenchmarkConfig, BENCHMARK_CONFIGS, run_benchmark_suite,
    print_benchmark_table, print_cpp_path,
)


class TestLatencyProfile:

    def _get_profile(self):
        return profile_matching_engine(n_orders=500, seed=42)

    def test_returns_latency_profile(self):
        p = self._get_profile()
        assert isinstance(p, LatencyProfile)

    def test_n_operations_correct(self):
        p = profile_matching_engine(n_orders=500)
        assert p.n_operations == 500

    def test_throughput_positive(self):
        p = self._get_profile()
        assert p.ops_per_second > 0

    def test_latencies_non_negative(self):
        p = self._get_profile()
        assert p.avg_latency_us >= 0
        assert p.p50_latency_us >= 0
        assert p.p95_latency_us >= 0
        assert p.p99_latency_us >= 0

    def test_percentile_ordering(self):
        """p50 <= p95 <= p99."""
        p = self._get_profile()
        assert p.p50_latency_us <= p.p95_latency_us
        assert p.p95_latency_us <= p.p99_latency_us

    def test_avg_between_min_and_max(self):
        p = self._get_profile()
        assert p.min_latency_us <= p.avg_latency_us <= p.max_latency_us

    def test_str_representation(self):
        p = self._get_profile()
        s = str(p)
        assert "µs" in s
        assert "throughput" in s


class TestSimulationProfile:

    def test_simulation_profile_runs(self):
        p = profile_simulation_loop(n_steps=50, seed=42)
        assert isinstance(p, LatencyProfile)

    def test_simulation_ops_per_second_positive(self):
        p = profile_simulation_loop(n_steps=50)
        assert p.ops_per_second > 0

    def test_simulation_n_operations_correct(self):
        p = profile_simulation_loop(n_steps=50)
        assert p.n_operations == 50

    def test_simulation_percentiles_ordered(self):
        p = profile_simulation_loop(n_steps=100)
        assert p.p50_latency_us <= p.p95_latency_us <= p.p99_latency_us


class TestMemoryProfile:

    def test_returns_memory_profile(self):
        m = profile_memory(n_orders=500)
        assert isinstance(m, MemoryProfile)

    def test_peak_non_negative(self):
        m = profile_memory(n_orders=500)
        assert m.peak_kb >= 0

    def test_delta_non_negative(self):
        m = profile_memory(n_orders=500)
        assert m.delta_kb >= 0

    def test_method_is_string(self):
        m = profile_memory(n_orders=500)
        assert isinstance(m.method, str)
        assert len(m.method) > 0

    def test_str_representation(self):
        m = profile_memory(n_orders=500)
        s = str(m)
        assert "KB" in s


class TestFullBenchmark:

    def test_run_full_benchmark(self):
        result = run_full_benchmark(n_orders=500, n_sim_steps=50, label="test")
        assert isinstance(result, BenchmarkResult)

    def test_benchmark_has_all_components(self):
        result = run_full_benchmark(n_orders=500, n_sim_steps=50)
        assert result.order_throughput is not None
        assert result.simulation_profile is not None
        assert result.memory is not None

    def test_benchmark_label(self):
        result = run_full_benchmark(n_orders=200, n_sim_steps=20, label="unit_test")
        assert result.label == "unit_test"

    def test_print_report_runs(self, capsys):
        result = run_full_benchmark(n_orders=200, n_sim_steps=20, label="test")
        result.print_report()
        captured = capsys.readouterr()
        assert "Benchmark" in captured.out


class TestBenchmarkSuite:

    def test_benchmark_configs_exist(self):
        assert len(BENCHMARK_CONFIGS) >= 3

    def test_benchmark_config_fields(self):
        for cfg in BENCHMARK_CONFIGS:
            assert cfg.n_orders > 0
            assert cfg.n_sim_steps > 0
            assert isinstance(cfg.label, str)

    def test_run_small_benchmark(self):
        small = BenchmarkConfig(label="test_small", n_orders=200, n_sim_steps=30)
        results = run_benchmark_suite(configs=[small], verbose=False)
        assert "test_small" in results

    def test_print_benchmark_table_runs(self, capsys):
        small = BenchmarkConfig(label="tbl_test", n_orders=200, n_sim_steps=20)
        results = run_benchmark_suite(configs=[small], verbose=False)
        print_benchmark_table(results)
        captured = capsys.readouterr()
        assert "tbl_test" in captured.out

    def test_print_cpp_path_runs(self, capsys):
        print_cpp_path()
        captured = capsys.readouterr()
        assert "C++" in captured.out
