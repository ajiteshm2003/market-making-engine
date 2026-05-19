"""
src/strategies/naive_market_maker.py
--------------------------------------
NaiveMarketMaker — Strategy A

The simplest possible market-making strategy.
Posts a fixed, symmetric spread around the current midprice (or fair value).

    bid = reference_price - half_spread
    ask = reference_price + half_spread

Where reference_price is:
    - fair_value  if use_fair_value=True  (uses the latent signal directly)
    - midprice    if use_fair_value=False (uses only observable book data)

In a real exchange, market makers do NOT observe the fair value directly.
Setting use_fair_value=False is the realistic mode.
Setting use_fair_value=True gives an upper bound on naive strategy performance.

Why this is a baseline, not a good strategy
-------------------------------------------
- The spread is static; it never adjusts to volatility or flow toxicity.
- The strategy ignores inventory entirely. If filled repeatedly on one side,
  the maker accumulates directional risk with no corrective mechanism.
- Informed traders will systematically pick off quotes that lag fair value.
- This makes it a useful LOWER BOUND: if the inventory-aware MM doesn't
  beat this, something is wrong.

In the academic literature, this corresponds to a dealer with no
inventory adjustment (Garman 1976, Ho-Stoll 1981 baseline case).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from .base_market_maker import BaseMarketMaker

if TYPE_CHECKING:
    from ..simulation.market_state import MarketState


class NaiveMarketMaker(BaseMarketMaker):
    """
    Fixed symmetric spread market maker.

    Parameters
    ----------
    agent_id : str
    half_spread : float
        Half the quoted bid-ask spread (in price units).
        Full spread = 2 × half_spread.
        E.g., half_spread=0.05 → bid = mid - 0.05, ask = mid + 0.05.
    use_fair_value : bool
        If True, center quotes on fair_value (state.fair_value).
        If False, center quotes on observable midprice (state.midprice).
        Default False = realistic (no privileged information).
    quote_size : float
        Quantity for each quote.
    initial_cash : float
    min_spread : float
        Minimum allowed half_spread (prevents zero-spread quoting).
    """

    def __init__(
        self,
        agent_id: str,
        half_spread: float = 0.05,
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

        self.half_spread = half_spread
        self.use_fair_value = use_fair_value
        self.min_spread = min_spread

    def _compute_quotes(
        self, state: "MarketState"
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute symmetric fixed-spread quotes.

        Returns (None, None) if no reference price is available.
        """
        # Pick the reference price
        if self.use_fair_value:
            ref = state.fair_value
        else:
            ref = state.midprice

        # Fall back: if observable mid is unavailable but fair_value is, use it
        if ref is None:
            ref = state.fair_value
        if ref is None:
            return None, None

        bid = ref - self.half_spread
        ask = ref + self.half_spread

        return bid, ask

    @property
    def quoted_spread(self) -> float:
        """Full bid-ask spread being quoted."""
        return 2.0 * self.half_spread
