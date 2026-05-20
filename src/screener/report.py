"""
src/screener/report.py
-----------------------
Report Generator

Produces three output formats from a list of ScoredTicker objects:
1. CSV: machine-readable data for further analysis
2. Markdown: human-readable report with tables and notes
3. Terminal: compact ranked table for quick inspection

Allocation framework note
--------------------------
The allocation guidance in this report is for experimental validation
of the screening methodology only.  It is based on simple position-sizing
principles (score-weighted, capped, not leveraged) and is NOT financial advice.

Position sizing shown here:
    - Only tickers with total_score >= 65 and regime != EXTREME
    - Max allocation per ticker: 10% of test portfolio
    - Score-weighted within the eligible set
    - Minimum cash reserve: 30% of test portfolio
    - No options, no leverage, no margin

This is appropriate only for paper trading or tiny real-money validation
of the screening approach.  All trading involves risk of loss.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .scoring import ScoredTicker
from ..models.regime import VolatilityRegime

DISCLAIMER = (
    "⚠ NOT FINANCIAL ADVICE. This report is for educational and experimental purposes "
    "only. Past performance does not guarantee future results. All trading involves "
    "risk of loss. Position sizes shown are illustrative only."
)

REPORT_DATE = datetime.today().strftime("%Y-%m-%d")
MAX_POSITION_PCT = 0.10     # 10% cap per ticker
MIN_CASH_RESERVE = 0.30     # keep 30% in cash
ELIGIBILITY_SCORE = 65.0    # minimum score for allocation consideration


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def save_csv(
    scored: List[ScoredTicker],
    filepath: str,
) -> str:
    """Save ranked ticker data to CSV. Returns the filepath written."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    rows = [s.summary_row() for s in scored]
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return filepath


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def save_markdown(
    scored: List[ScoredTicker],
    filepath: str,
    test_portfolio_size: float = 1000.0,
) -> str:
    """
    Save a full Markdown report with ranking tables, regime notes,
    and a suggested allocation framework.

    Parameters
    ----------
    scored              : ranked list of ScoredTicker (score descending)
    filepath            : output file path
    test_portfolio_size : size of hypothetical test portfolio in USD
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    eligible = [s for s in scored if not s.is_avoid and s.total_score >= ELIGIBILITY_SCORE]
    avoid    = [s for s in scored if s.is_avoid]
    watch    = [s for s in scored if not s.is_avoid and s.total_score < ELIGIBILITY_SCORE]

    lines: List[str] = []

    # Header
    lines += [
        f"# Stock Screener Report",
        f"",
        f"**Generated:** {REPORT_DATE}  ",
        f"**Universe:** {len(scored)} tickers  ",
        f"**Eligible:** {len(eligible)}  |  **Watch:** {len(watch)}  |  **Avoid:** {len(avoid)}",
        f"",
        f"> {DISCLAIMER}",
        f"",
        "---",
        "",
    ]

    # Top candidates table
    lines += [
        "## Top Candidates",
        "",
        _md_table(eligible[:10] if eligible else []),
        "",
    ]

    # Full ranking
    lines += [
        "## Full Rankings",
        "",
        _md_table(scored[:20]),
        "",
        "---",
        "",
    ]

    # Avoid list
    if avoid:
        lines += [
            "## Avoid List",
            "",
            "The following tickers are flagged for avoidance due to EXTREME volatility regime,",
            "excessive drawdown, or very low total score:",
            "",
        ]
        for s in avoid:
            lines.append(f"- **{s.ticker}** (score={s.total_score:.0f}, regime={s.regime_label}): {s.reason}")
        lines += ["", "---", ""]

    # Regime summary
    lines += [
        "## Regime Distribution",
        "",
    ]
    regime_counts = {}
    for s in scored:
        label = s.regime_label
        regime_counts[label] = regime_counts.get(label, 0) + 1
    for label, count in sorted(regime_counts.items()):
        lines.append(f"- **{label}**: {count} tickers")
    lines += ["", "---", ""]

    # Allocation framework
    lines += [
        "## Suggested Tiny Test Allocation (Experimental Only)",
        "",
        f"> Test portfolio size: **${test_portfolio_size:,.0f}** — for paper trading or validation only.",
        f"> {DISCLAIMER}",
        "",
        _allocation_table(eligible, test_portfolio_size),
        "",
        "**Rules applied:**",
        f"- Only tickers with score ≥ {ELIGIBILITY_SCORE:.0f} and not in EXTREME regime",
        f"- Maximum {MAX_POSITION_PCT:.0%} of portfolio per ticker",
        f"- Minimum {MIN_CASH_RESERVE:.0%} held in cash",
        "- Score-weighted allocation within eligible set",
        "- No leverage, no options, no margin",
        "",
        "---",
        "",
    ]

    # Risk notes
    lines += [
        "## Risk Notes",
        "",
        "### Volatility Regimes",
        "Regime classification uses a 20-day rolling realized volatility window,",
        "annualised (×√252) and mapped to four states:",
        "",
        "| Regime | Annualised Vol Range | Interpretation |",
        "|---|---|---|",
        "| LOW | < 12% | Calm conditions, spread income favorable |",
        "| MEDIUM | 12–25% | Normal equity volatility |",
        "| HIGH | 25–45% | Elevated risk, reduce sizing |",
        "| EXTREME | > 45% | Dislocated market, avoid new entries |",
        "",
        "### VaR and Expected Shortfall",
        "VaR 95% shows the daily loss not expected to be exceeded on 95% of trading days.",
        "ES 95% shows the average loss on the worst 5% of days — always larger than VaR.",
        "ES is the preferred tail risk measure (coherent, Basel III standard).",
        "",
        "### Score Interpretation",
        "| Score Range | Interpretation |",
        "|---|---|",
        "| 70–100 | Strong candidate — favorable trend, stable regime |",
        "| 55–70  | Monitor — mixed signals |",
        "| 35–55  | Weak — avoid new entries |",
        "| 0–35   | Avoid — poor risk/reward or extreme volatility |",
        "",
        "---",
        "",
        f"*Report generated by the Regime-Aware Market Microstructure Engine — {REPORT_DATE}*",
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def print_terminal_table(
    scored: List[ScoredTicker],
    top_n: int = 10,
    show_avoid: bool = True,
) -> None:
    """Print a compact ranked table to stdout."""
    print(f"\n  {'─'*95}")
    print(f"  STOCK SCREENER REPORT — {REPORT_DATE}")
    print(f"  {'─'*95}")
    print(f"  {'#':<3} {'Ticker':<7} {'Score':>6} {'Regime':<8} "
          f"{'Vol20d':>7} {'Ret20d':>7} {'Ret60d':>7} "
          f"{'MaxDD':>7} {'VaR95':>7} {'Reason'}")
    print(f"  {'─'*95}")

    for i, s in enumerate(scored[:top_n], 1):
        avoid_flag = " ⚠" if s.is_avoid else ""
        vol = f"{s.vol_20d:.0%}" if s.vol_20d else "N/A"
        r20 = f"{s.ret_20d:+.1%}" if s.ret_20d else "N/A"
        r60 = f"{s.ret_60d:+.1%}" if s.ret_60d else "N/A"
        dd  = f"{s.max_drawdown:.0%}" if s.max_drawdown else "N/A"
        var = f"{s.var_95:.2%}" if s.var_95 else "N/A"
        reason_short = s.reason[:40] + "..." if len(s.reason) > 40 else s.reason

        print(f"  {i:<3} {s.ticker:<7}{avoid_flag:<2} {s.total_score:>5.1f}  "
              f"{s.regime_label:<8} {vol:>7} {r20:>7} {r60:>7} "
              f"{dd:>7} {var:>7}  {reason_short}")

    if show_avoid:
        avoid_list = [s for s in scored if s.is_avoid]
        if avoid_list:
            print(f"\n  AVOID: {', '.join(s.ticker for s in avoid_list)}")

    print(f"  {'─'*95}")
    print(f"  ⚠ {DISCLAIMER}")
    print(f"  {'─'*95}\n")


# ---------------------------------------------------------------------------
# Allocation table
# ---------------------------------------------------------------------------

def _allocation_table(
    eligible: List[ScoredTicker],
    portfolio_size: float,
) -> str:
    if not eligible:
        return "_No tickers meet eligibility criteria for this test portfolio._"

    # Score-weighted allocations
    capped = eligible[:8]   # limit to top 8 for concentration
    total_score = sum(s.total_score for s in capped)
    investable = portfolio_size * (1 - MIN_CASH_RESERVE)

    rows = ["| Ticker | Score | Weight | Allocation $ | Notes |",
            "|---|---|---|---|---|"]

    for s in capped:
        raw_weight = s.total_score / total_score
        capped_weight = min(raw_weight, MAX_POSITION_PCT / (1 - MIN_CASH_RESERVE))
        alloc = capped_weight * investable
        note = f"{s.regime_label} regime"
        rows.append(f"| {s.ticker} | {s.total_score:.0f} | {capped_weight:.0%} | ${alloc:,.0f} | {note} |")

    rows.append(f"| **Cash** | — | ≥{MIN_CASH_RESERVE:.0%} | ${portfolio_size * MIN_CASH_RESERVE:,.0f} | Required reserve |")
    return "\n".join(rows)


def _md_table(scored: List[ScoredTicker]) -> str:
    """Generate a Markdown table from a list of ScoredTicker."""
    if not scored:
        return "_No tickers in this category._"

    header = "| # | Ticker | Score | Regime | Vol20d | Ret20d | Ret60d | MaxDD | VaR95 | ES95 |"
    sep    = "|---|---|---|---|---|---|---|---|---|---|"
    rows   = [header, sep]

    for i, s in enumerate(scored, 1):
        vol = f"{s.vol_20d:.0%}" if s.vol_20d else "—"
        r20 = f"{s.ret_20d:+.1%}" if s.ret_20d else "—"
        r60 = f"{s.ret_60d:+.1%}" if s.ret_60d else "—"
        dd  = f"{s.max_drawdown:.0%}" if s.max_drawdown else "—"
        var = f"{s.var_95:.2%}" if s.var_95 else "—"
        es  = f"{s.es_95:.2%}" if s.es_95 else "—"
        flag = " ⚠" if s.is_avoid else ""
        rows.append(f"| {i} | **{s.ticker}**{flag} | {s.total_score:.1f} | {s.regime_label} "
                    f"| {vol} | {r20} | {r60} | {dd} | {var} | {es} |")

    return "\n".join(rows)
