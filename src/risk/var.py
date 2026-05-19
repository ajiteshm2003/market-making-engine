"""
src/risk/var.py
----------------
Value-at-Risk (VaR) and Expected Shortfall (CVaR) Engine

Financial Risk Interpretation
------------------------------
VaR(α) answers: "What is the maximum loss we expect to not exceed with
probability α over a given time horizon?"

If the 1-day 95% VaR is $1,000, it means:
  In 95% of trading days, losses will be no worse than $1,000.
  In the remaining 5% of days (the tail), losses may exceed this amount.

Expected Shortfall (ES), also called CVaR, answers a stronger question:
  "Given that losses DO exceed the VaR threshold, what is the expected loss?"

ES is always >= VaR at the same confidence level.  For fat-tailed
distributions (which real P&L often is), ES captures tail risk that VaR
misses.  Most modern regulatory frameworks (Basel III/IV) prefer ES over VaR
for exactly this reason.

Three estimation methods are implemented:

1. Historical VaR: use the empirical return distribution directly.
   Pro: no distributional assumptions.
   Con: requires sufficient history; tail estimates are noisy.

2. Parametric Gaussian VaR: assume returns ~ N(μ, σ²), apply z-score.
   Pro: closed-form, fast, smooth.
   Con: underestimates tail risk for leptokurtic (fat-tailed) distributions.

3. Expected Shortfall: average of returns below the VaR threshold.
   Always computed empirically from the tail of the distribution.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VaRResult:
    """
    Output of a single VaR calculation.

    Attributes
    ----------
    confidence : float      Confidence level (e.g. 0.95 for 95% VaR)
    var        : float      Value-at-Risk (positive = loss threshold)
    method     : str        Estimation method used
    n_obs      : int        Number of observations used
    mean_return: float      Mean of the input distribution
    std_return : float      Standard deviation of the input distribution
    """
    confidence:  float
    var:         float
    method:      str
    n_obs:       int
    mean_return: float
    std_return:  float

    def __str__(self) -> str:
        return (
            f"VaR({self.confidence:.0%}, {self.method}): "
            f"{self.var:.4f}  [n={self.n_obs}, μ={self.mean_return:.4f}, σ={self.std_return:.4f}]"
        )


@dataclass(frozen=True)
class ESResult:
    """
    Output of an Expected Shortfall (CVaR) calculation.

    Attributes
    ----------
    confidence : float      Confidence level
    var        : float      VaR threshold at this confidence level
    es         : float      Expected Shortfall (average of tail losses)
    tail_obs   : int        Number of observations in the tail
    n_obs      : int        Total observations
    """
    confidence: float
    var:        float
    es:         float
    tail_obs:   int
    n_obs:      int

    def __str__(self) -> str:
        return (
            f"ES({self.confidence:.0%}): {self.es:.4f}  "
            f"[VaR={self.var:.4f}, tail_n={self.tail_obs}/{self.n_obs}]"
        )


@dataclass
class PnLDistributionStats:
    """
    Descriptive statistics for a P&L distribution.
    """
    n:              int
    mean:           float
    std:            float
    skewness:       float
    excess_kurtosis: float
    min:            float
    p5:             float
    p25:            float
    median:         float
    p75:            float
    p95:            float
    p99:            float
    max:            float
    sharpe:         float       # annualized (× √252)
    positive_days:  float       # fraction of positive-return observations

    def __str__(self) -> str:
        lines = [
            f"P&L Distribution ({self.n} obs)",
            f"  Mean={self.mean:.4f}  Std={self.std:.4f}  Sharpe={self.sharpe:.3f}",
            f"  Skew={self.skewness:.3f}  ExKurt={self.excess_kurtosis:.3f}",
            f"  Min={self.min:.4f}  P5={self.p5:.4f}  P95={self.p95:.4f}  Max={self.max:.4f}",
            f"  Positive days: {self.positive_days:.1%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core VaR functions
# ---------------------------------------------------------------------------

def _validate(pnl: Sequence[float], min_obs: int = 5) -> List[float]:
    """Clean and validate a P&L sequence."""
    cleaned = [x for x in pnl if x is not None and not math.isnan(x)]
    if len(cleaned) < min_obs:
        raise ValueError(
            f"Insufficient observations: {len(cleaned)} < {min_obs} required."
        )
    return cleaned


def _percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile (0–100) of a sorted or unsorted list."""
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    idx = (p / 100.0) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


def historical_var(
    pnl: Sequence[float],
    confidence: float = 0.95,
) -> VaRResult:
    """
    Historical (non-parametric) Value-at-Risk.

    Uses the empirical quantile of the P&L distribution.
    VaR is reported as a positive loss magnitude.

    Parameters
    ----------
    pnl        : sequence of per-step P&L values (positive = profit)
    confidence : confidence level in (0, 1)

    Returns
    -------
    VaRResult
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")

    data = _validate(pnl)
    n = len(data)
    mean_r = sum(data) / n
    std_r = statistics.stdev(data) if n > 1 else 0.0

    # VaR = negative of the (1-confidence) quantile of P&L
    # e.g. at 95% confidence: take the 5th percentile of P&L (a loss)
    loss_quantile = _percentile(data, (1 - confidence) * 100)
    var = -loss_quantile  # positive = loss threshold

    return VaRResult(
        confidence=confidence,
        var=var,
        method="historical",
        n_obs=n,
        mean_return=round(mean_r, 8),
        std_return=round(std_r, 8),
    )


def parametric_var(
    returns: Sequence[float],
    confidence: float = 0.95,
) -> VaRResult:
    """
    Parametric Gaussian Value-at-Risk.

    Assumes returns are normally distributed.
    Applies the z-score corresponding to the confidence level.

    Parameters
    ----------
    returns    : sequence of return values (per-step)
    confidence : confidence level in (0, 1)

    Returns
    -------
    VaRResult

    Notes
    -----
    The Gaussian z-scores used here:
      90%  → z = 1.282
      95%  → z = 1.645
      99%  → z = 2.326
      99.9%→ z = 3.090
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")

    data = _validate(returns)
    n = len(data)
    mean_r = sum(data) / n
    std_r = statistics.stdev(data) if n > 1 else 0.0

    # Rational approximation for the Gaussian inverse CDF (Abramowitz & Stegun)
    z = _normal_ppf(confidence)
    var = -(mean_r - z * std_r)

    return VaRResult(
        confidence=confidence,
        var=var,
        method="parametric_gaussian",
        n_obs=n,
        mean_return=round(mean_r, 8),
        std_return=round(std_r, 8),
    )


def expected_shortfall(
    pnl: Sequence[float],
    confidence: float = 0.95,
) -> ESResult:
    """
    Expected Shortfall (CVaR) — average loss beyond the VaR threshold.

    ES is always >= VaR at the same confidence level.
    ES satisfies the subadditivity property required for a coherent
    risk measure; VaR does not.

    Parameters
    ----------
    pnl        : sequence of per-step P&L values
    confidence : confidence level in (0, 1)

    Returns
    -------
    ESResult
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")

    data = _validate(pnl)
    n = len(data)

    # VaR threshold (loss magnitude)
    var_result = historical_var(data, confidence)
    var_threshold = var_result.var

    # Tail: observations where P&L < -VaR (i.e., losses worse than VaR)
    tail = [x for x in data if x < -var_threshold]

    if not tail:
        # No observations in tail — use VaR as ES
        es = var_threshold
    else:
        # ES = average of tail losses (as positive magnitude)
        es = -sum(tail) / len(tail)

    return ESResult(
        confidence=confidence,
        var=round(var_threshold, 8),
        es=round(es, 8),
        tail_obs=len(tail),
        n_obs=n,
    )


def rolling_var(
    pnl_series: Sequence[float],
    window: int = 100,
    confidence: float = 0.95,
    method: str = "historical",
) -> List[Optional[float]]:
    """
    Rolling Value-at-Risk over a sliding window.

    Returns a list of length len(pnl_series).
    The first (window - 1) entries are None (insufficient history).

    Parameters
    ----------
    pnl_series : sequence of per-step P&L values
    window     : lookback period for each VaR calculation
    confidence : confidence level
    method     : 'historical' or 'parametric_gaussian'

    Returns
    -------
    List[Optional[float]] : VaR estimate at each step (or None)
    """
    data = list(pnl_series)
    n = len(data)
    result: List[Optional[float]] = [None] * n

    fn = parametric_var if method == "parametric_gaussian" else historical_var

    for i in range(window - 1, n):
        window_data = data[i - window + 1 : i + 1]
        try:
            r = fn(window_data, confidence)
            result[i] = r.var
        except ValueError:
            result[i] = None

    return result


def pnl_distribution_stats(pnl: Sequence[float]) -> PnLDistributionStats:
    """
    Compute comprehensive descriptive statistics for a P&L series.

    Parameters
    ----------
    pnl : sequence of per-step P&L values

    Returns
    -------
    PnLDistributionStats
    """
    data = _validate(pnl, min_obs=4)
    n = len(data)
    s = sorted(data)

    mean_r = sum(data) / n
    std_r  = statistics.stdev(data) if n > 1 else 0.0
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

    # Skewness and excess kurtosis
    if std_r > 0:
        skew = sum((x - mean_r) ** 3 for x in data) / (n * std_r ** 3)
        kurt = sum((x - mean_r) ** 4 for x in data) / (n * std_r ** 4) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    pos_frac = sum(1 for x in data if x > 0) / n

    return PnLDistributionStats(
        n=n,
        mean=round(mean_r, 6),
        std=round(std_r, 6),
        skewness=round(skew, 4),
        excess_kurtosis=round(kurt, 4),
        min=s[0],
        p5=_percentile(data, 5),
        p25=_percentile(data, 25),
        median=_percentile(data, 50),
        p75=_percentile(data, 75),
        p95=_percentile(data, 95),
        p99=_percentile(data, 99),
        max=s[-1],
        sharpe=round(sharpe, 4),
        positive_days=round(pos_frac, 4),
    )


# ---------------------------------------------------------------------------
# Internal: Gaussian inverse CDF approximation
# ---------------------------------------------------------------------------

def _normal_ppf(p: float) -> float:
    """
    Rational approximation to the standard normal quantile (Abramowitz & Stegun 26.2.17).
    Accurate to ~|error| < 4.5e-4 for all p in (0,1).
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0,1), got {p}")

    # For p > 0.5, use symmetry
    flip = p > 0.5
    q = 1.0 - p if flip else p

    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    z = t - num / den

    return z if flip else -z
