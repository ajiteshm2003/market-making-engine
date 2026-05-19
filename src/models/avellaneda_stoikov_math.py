"""
src/models/avellaneda_stoikov_math.py
---------------------------------------
Avellaneda-Stoikov Model — Core Mathematical Functions

This module contains PURE FUNCTIONS implementing the A-S framework.
No state, no side effects.  Every function is independently testable.

The Avellaneda-Stoikov (2008) model
-------------------------------------
Avellaneda and Stoikov solve a stochastic optimal control problem:

    maximize E[ W_T - γ × q_T² × S_T ]

Where:
    W_T = terminal wealth
    q_T = terminal inventory
    S_T = terminal asset price
    γ   = absolute risk aversion (CARA utility)

The optimal strategy is to quote a RESERVATION PRICE:

    r(S, q, t) = S - q × γ × σ² × (T - t)

And an OPTIMAL HALF-SPREAD:

    δ*(t) = (γ × σ² × (T-t)) / 2  +  (1/γ) × ln(1 + γ/k)

Derivation intuition
--------------------
The reservation price has two components:
1. S               → the current midprice (observable reference)
2. -q × γ × σ²(T-t) → inventory penalty
   - If long (q > 0): shift price DOWN to incentivize selling
   - If short (q < 0): shift price UP to incentivize buying
   - Penalty grows with: inventory (q), risk aversion (γ),
     variance (σ²), and time remaining (T-t)

The optimal spread has two components:
1. (γ × σ²(T-t)) / 2  → RISK PREMIUM
   Compensates the MM for the uncertainty in price movement during
   the time it takes to get filled.  Wider when volatile or long horizon.
   
2. (1/γ) × ln(1 + γ/k) → LIQUIDITY PREMIUM
   The profit per unit the MM earns from the bid-ask spread.
   This is a closed-form result from solving the Poisson fill-rate
   optimization.  Wider when k is small (selective takers).

Final quotes:
    bid = r - δ*
    ask = r + δ*

Economic interpretation of γ
------------------------------
γ controls the trade-off between:
    - Spread income (larger γ → wider spreads → more income per fill)
    - Inventory risk (larger γ → heavier inventory penalty → tighter control)

In practice γ is calibrated to match observed spread behavior or target
inventory statistics.  Typical values: 0.01 – 1.0.

References
----------
Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit
order book." Quantitative Finance, 8(3), 217-224.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Core A-S formulas — exact implementations matching the paper
# ---------------------------------------------------------------------------

def reservation_price(
    midprice: float,
    inventory: float,
    gamma: float,
    sigma: float,
    time_remaining: float,
) -> float:
    """
    Compute the Avellaneda-Stoikov reservation price.

        r = S - q × γ × σ² × (T-t)

    Parameters
    ----------
    midprice : float
        Current mid price S.
    inventory : float
        Current inventory q (positive = long, negative = short).
    gamma : float
        Risk aversion coefficient γ > 0.
    sigma : float
        Per-step price volatility σ > 0.
    time_remaining : float
        Remaining time horizon (T-t) ≥ 0.

    Returns
    -------
    float : reservation price r

    Notes
    -----
    - When inventory = 0, r = S (no adjustment).
    - When long (q > 0), r < S (shift down to attract sellers).
    - When short (q < 0), r > S (shift up to attract buyers).
    - The adjustment grows linearly with inventory, quadratically
      with volatility, and linearly with remaining time.
    """
    if time_remaining < 0:
        raise ValueError(f"time_remaining must be >= 0, got {time_remaining}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    inventory_penalty = inventory * gamma * (sigma ** 2) * time_remaining
    return midprice - inventory_penalty


def optimal_half_spread(
    gamma: float,
    sigma: float,
    time_remaining: float,
    k: float,
) -> float:
    """
    Compute the Avellaneda-Stoikov optimal half-spread.

        δ* = (γ × σ² × (T-t)) / 2  +  (1/γ) × ln(1 + γ/k)

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient γ > 0.
    sigma : float
        Per-step price volatility σ > 0.
    time_remaining : float
        Remaining time horizon (T-t) ≥ 0.
    k : float
        Order arrival intensity k > 0.

    Returns
    -------
    float : optimal half-spread δ* ≥ 0

    Decomposition
    -------------
    risk_premium      = (γ × σ² × (T-t)) / 2
    liquidity_premium = (1/γ) × ln(1 + γ/k)
    δ*                = risk_premium + liquidity_premium

    Notes
    -----
    - When T-t = 0, only the liquidity premium remains.
    - As k → ∞, ln(1 + γ/k) → γ/k → 0, so spread → risk_premium.
    - As σ → 0, spread → (1/γ) × ln(1 + γ/k) (pure liquidity premium).
    - The spread is always non-negative.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if time_remaining < 0:
        raise ValueError(f"time_remaining must be >= 0, got {time_remaining}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    risk_premium      = (gamma * sigma**2 * time_remaining) / 2.0
    liquidity_premium = (1.0 / gamma) * math.log(1.0 + gamma / k)
    return risk_premium + liquidity_premium


def compute_quotes(
    midprice: float,
    inventory: float,
    gamma: float,
    sigma: float,
    time_remaining: float,
    k: float,
    min_half_spread: float = 1e-4,
) -> Tuple[float, float, float, float]:
    """
    Compute A-S bid and ask prices from model parameters.

    Combines reservation_price() and optimal_half_spread() into a
    single call returning the complete quote.

    Parameters
    ----------
    midprice : float
    inventory : float
    gamma : float
    sigma : float
    time_remaining : float
    k : float
    min_half_spread : float
        Floor on δ* to prevent zero-spread quoting.

    Returns
    -------
    (bid, ask, reservation, half_spread) : tuple[float, float, float, float]
        bid          = r - δ*
        ask          = r + δ*
        reservation  = r   (for diagnostics)
        half_spread  = δ*  (for diagnostics)
    """
    r    = reservation_price(midprice, inventory, gamma, sigma, time_remaining)
    delta = max(min_half_spread, optimal_half_spread(gamma, sigma, time_remaining, k))

    bid = r - delta
    ask = r + delta
    return bid, ask, r, delta


def spread_decomposition(
    gamma: float,
    sigma: float,
    time_remaining: float,
    k: float,
) -> Tuple[float, float, float]:
    """
    Decompose the optimal half-spread into its two components.

    Returns
    -------
    (total_half_spread, risk_premium, liquidity_premium)
    """
    risk_premium      = (gamma * sigma**2 * time_remaining) / 2.0
    liquidity_premium = (1.0 / gamma) * math.log(1.0 + gamma / k)
    total             = risk_premium + liquidity_premium
    return total, risk_premium, liquidity_premium


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def implied_k_from_spread(
    observed_half_spread: float,
    gamma: float,
    sigma: float,
    time_remaining: float,
) -> Optional[float]:
    """
    Back out the implied k from an observed market half-spread.

    Inverts the liquidity premium component:
        liquidity_premium = δ_observed - risk_premium
        (1/γ) × ln(1 + γ/k) = liquidity_premium
        k = γ / (exp(γ × liquidity_premium) - 1)

    Returns None if no valid k exists (e.g., spread too small).
    """
    risk_premium = (gamma * sigma**2 * time_remaining) / 2.0
    liquidity_premium = observed_half_spread - risk_premium

    if liquidity_premium <= 0:
        return None

    try:
        exp_term = math.exp(gamma * liquidity_premium)
        if exp_term <= 1.0:
            return None
        return gamma / (exp_term - 1.0)
    except OverflowError:
        return None


def sensitivity_analysis(
    midprice: float = 100.0,
    inventory: float = 0.0,
    gamma: float = 0.1,
    sigma: float = 0.05,
    time_remaining: float = 1.0,
    k: float = 1.5,
) -> dict:
    """
    Compute A-S quotes and decompose all components.
    Useful for understanding model behavior and for unit tests.
    """
    bid, ask, r, delta = compute_quotes(
        midprice, inventory, gamma, sigma, time_remaining, k
    )
    total, rp, lp = spread_decomposition(gamma, sigma, time_remaining, k)
    return {
        "midprice":          midprice,
        "inventory":         inventory,
        "gamma":             gamma,
        "sigma":             sigma,
        "time_remaining":    time_remaining,
        "k":                 k,
        "reservation_price": r,
        "half_spread":       delta,
        "full_spread":       2 * delta,
        "bid":               bid,
        "ask":               ask,
        "risk_premium":      rp,
        "liquidity_premium": lp,
        "inventory_penalty": midprice - r,
    }
