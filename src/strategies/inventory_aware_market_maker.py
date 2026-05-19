"""
src/strategies/inventory_aware_market_maker.py
------------------------------------------------
InventoryAwareMarketMaker — Strategy B

Extends the naive strategy by dynamically skewing quotes based on
current inventory to control directional risk.

Core intuition
--------------
If the market maker is LONG (bought more than sold):
    → She wants to SELL to reduce inventory.
    → Lower both bid AND ask to make her ask more attractive.
    → This discourages further buying and encourages selling to her.

If the market maker is SHORT (sold more than bought):
    → She wants to BUY to reduce inventory.
    → Raise both bid AND ask to make her bid more attractive.

This is inventory skew. The adjustment is proportional to current inventory:

    skew = inventory_skew_factor × inventory / max_inventory

    bid = reference - half_spread + skew   (negative skew shifts down when long)
    ask = reference + half_spread + skew

Wait — that's not right. Let me be precise:

    When LONG (inventory > 0):
        We shift quotes DOWN → skew is NEGATIVE
        bid = ref - half_spread - |skew|
        ask = ref + half_spread - |skew|

    When SHORT (inventory < 0):
        We shift quotes UP → skew is POSITIVE
        bid = ref - half_spread + |skew|
        ask = ref + half_spread + |skew|

Unified formula:
    reservation_price = ref - skew_factor × inventory   (inventory penalty)
    bid = reservation_price - half_spread
    ask = reservation_price + half_spread

This is the discrete, simplified version of the Avellaneda-Stoikov
reservation price concept. Phase 3 uses a linear penalty; Phase 4 will
implement the full stochastic-control derivation with γσ²(T-t).

Spread widening (optional)
--------------------------
Optionally widens the spread as |inventory| grows:
    effective_spread = base_half_spread × (1 + spread_widening × |inventory| / max_inventory)

This means the market maker demands more compensation per unit risk
as her position grows — consistent with diminishing inventory tolerance.

Parameters
----------
half_spread : float
    Base half-spread at zero inventory.
inventory_skew_factor : float
    How aggressively to shift the reservation price per unit of inventory.
    Units: price per unit of inventory.
    Larger = more aggressive skew, faster mean-reversion of inventory.
max_inventory : float
    Normalization constant. Skew is capped at half_spread when
    |inventory| >= max_inventory (prevents inverting the book).
spread_widening : float
    Factor by which spread expands as |inventory| grows.
    0.0 = no widening (fixed spread regardless of inventory).
    1.0 = spread doubles at max_inventory.
use_fair_value : bool
    Use latent fair_value as reference (True) or observable mid (False).

Academic note
-------------
The reservation price adjustment here is:
    r = S - q × γ   (simplified, without the σ²(T-t) term)

Where:
    S = midprice
    q = inventory
    γ = inventory_skew_factor (plays the role of risk aversion × variance)

This is pedagogically useful: you can see how the full A-S formula
(Phase 4) generalizes this by making γ time-varying via volatility.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

from .base_market_maker import BaseMarketMaker

if TYPE_CHECKING:
    from ..simulation.market_state import MarketState


class InventoryAwareMarketMaker(BaseMarketMaker):
    """
    Inventory-skewing market maker.

    Shifts both bid and ask based on current inventory to
    mean-revert position toward zero.

    Parameters
    ----------
    agent_id : str
    half_spread : float
        Base half-spread at zero inventory.
    inventory_skew_factor : float
        Price shift per unit of inventory (risk aversion proxy).
    max_inventory : float
        Position at which skew reaches its maximum (clamp).
    spread_widening : float
        Spread expansion rate with |inventory|. 0 = fixed spread.
    use_fair_value : bool
        Center on fair_value (True) or midprice (False).
    quote_size : float
    initial_cash : float
    min_spread : float
        Minimum allowed half_spread after widening.
    """

    def __init__(
        self,
        agent_id: str,
        half_spread: float = 0.05,
        inventory_skew_factor: float = 0.01,
        max_inventory: float = 50.0,
        spread_widening: float = 0.5,
        use_fair_value: bool = False,
        quote_size: float = 5.0,
        initial_cash: float = 100_000.0,
        min_spread: float = 0.001,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            quote_size=quote_size,
            initial_cash=initial_cash,
        )

        if half_spread < min_spread:
            raise ValueError(
                f"half_spread ({half_spread}) must be >= min_spread ({min_spread})"
            )
        if inventory_skew_factor < 0:
            raise ValueError("inventory_skew_factor must be >= 0")
        if max_inventory <= 0:
            raise ValueError("max_inventory must be > 0")
        if spread_widening < 0:
            raise ValueError("spread_widening must be >= 0")

        self.half_spread = half_spread
        self.inventory_skew_factor = inventory_skew_factor
        self.max_inventory = max_inventory
        self.spread_widening = spread_widening
        self.use_fair_value = use_fair_value
        self.min_spread = min_spread

    def _compute_quotes(
        self, state: "MarketState"
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute inventory-skewed quotes.

        Steps:
          1. Get reference price (mid or fair_value).
          2. Compute inventory skew → reservation price.
          3. Compute effective half-spread (widen if needed).
          4. Return bid = reservation - eff_half, ask = reservation + eff_half.
        """
        # ── Reference price ─────────────────────────────────────────────
        if self.use_fair_value:
            ref = state.fair_value
        else:
            ref = state.midprice

        if ref is None:
            ref = state.fair_value
        if ref is None:
            return None, None

        # ── Inventory skew ───────────────────────────────────────────────
        inventory = self.mm_metrics.inventory

        # Clamp inventory to [-max_inventory, +max_inventory] for skew calc
        clamped_inv = max(-self.max_inventory, min(self.max_inventory, inventory))

        # Reservation price: shift ref away from current inventory direction
        # Long (inv > 0) → r < ref  (lower our prices to sell)
        # Short (inv < 0) → r > ref (raise our prices to buy)
        skew = self.inventory_skew_factor * clamped_inv
        reservation_price = ref - skew

        # ── Spread widening ──────────────────────────────────────────────
        inv_fraction = abs(clamped_inv) / self.max_inventory   # in [0, 1]
        effective_half = self.half_spread * (1.0 + self.spread_widening * inv_fraction)
        effective_half = max(self.min_spread, effective_half)

        # ── Final quotes ─────────────────────────────────────────────────
        bid = reservation_price - effective_half
        ask = reservation_price + effective_half

        return bid, ask

    def reservation_price(self, ref: float) -> float:
        """
        Compute the current reservation price given a reference price.

        Exposed as a public method for testing and analytics.
        """
        inventory = self.mm_metrics.inventory
        clamped = max(-self.max_inventory, min(self.max_inventory, inventory))
        return ref - self.inventory_skew_factor * clamped

    def effective_half_spread(self) -> float:
        """
        Current effective half-spread (base + widening from inventory).
        """
        inventory = self.mm_metrics.inventory
        clamped = max(-self.max_inventory, min(self.max_inventory, inventory))
        inv_fraction = abs(clamped) / self.max_inventory
        return max(self.min_spread, self.half_spread * (1.0 + self.spread_widening * inv_fraction))
