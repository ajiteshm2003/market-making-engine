"""
examples/demo_stress_testing.py
--------------------------------
Runs all stress scenarios against all four market makers and prints
a comprehensive comparison table.

Run:
    python examples/demo_stress_testing.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.risk.stress import StressTestRunner, SCENARIOS

def divider(t=""): print(f"\n{'═'*65}\n  {t}\n{'═'*65}" if t else "\n"+"─"*65)

divider("STRESS TEST SUITE — All Scenarios × All Strategies")
print(f"  Scenarios: {list(SCENARIOS.keys())}")
print(f"  Strategies: NMM, IAMM, ASMM, RASMM")
print(f"  Steps per scenario: 300")

runner = StressTestRunner(n_steps=300, base_gamma=0.10, base_quote_size=5.0)
all_results = runner.run_all()

StressTestRunner.print_summary(all_results)

# ── Key comparisons ─────────────────────────────────────────────────────────
divider("KEY FINDINGS")

for scenario_name, strat_results in all_results.items():
    drawdowns = {s: r.max_drawdown for s, r in strat_results.items()}
    best = min(drawdowns, key=drawdowns.get)
    worst = max(drawdowns, key=drawdowns.get)
    print(f"  {scenario_name:<22}: "
          f"best DD={best} ({drawdowns[best]:.2f})  "
          f"worst DD={worst} ({drawdowns[worst]:.2f})")

divider("DONE")
print("  Stress test demo complete.")
