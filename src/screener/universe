"""
src/screener/universe.py
-------------------------
Universe Definitions

Three screening universes, each calibrated for a different risk/opportunity profile.

Design principle
----------------
The separation prevents mega-cap names (SPY, AAPL, MSFT) from crowding out
smaller names that may have stronger trend structure. A $3T company cannot
meaningfully compete on "emerging momentum" with a $5B company — the dynamics
are entirely different.

Each universe has:
  - tickers: the candidate list
  - label: display name
  - description: what it represents
  - cap_bucket: used for the market-cap opportunity bonus in scoring
  - vol_tolerance: MEDIUM names are penalised less for moderate vol,
                   HIGH_BETA names are allowed higher vol before penalising

Universe 1 — Institutional Stability
  Large-cap liquid names. Broad market ETFs plus Mag-7.
  Screener rewards regime stability and drawdown control.
  This is the "quality anchor" of the portfolio.

Universe 2 — Emerging Leaders
  $2B–$100B market cap. Names with genuine revenue and institutional ownership
  but not yet fully priced in. Screener rewards trend acceleration and
  improving momentum structure. Moderate volatility is acceptable.

Universe 3 — Speculative High Beta
  Smaller-cap, high-growth, high-volatility. Requires strict liquidity floor.
  Screener penalises EXTREME regime and unstable drawdowns.
  Only the best risk/reward names should survive.
  This universe is NOT for position sizing — it is for watchlist awareness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class CapBucket(Enum):
    """Market-cap classification for opportunity bonus."""
    MEGA      = "mega"       # > $500B
    LARGE     = "large"      # $100B – $500B
    MID       = "mid"        # $10B – $100B
    SMALL     = "small"      # $2B – $10B
    MICRO     = "micro"      # < $2B
    ETF       = "etf"        # no cap applicable
    UNKNOWN   = "unknown"


class VolTolerance(Enum):
    """How aggressively the scorer penalises elevated volatility."""
    LOW    = "low"     # institutional: penalise HIGH vol heavily
    MEDIUM = "medium"  # emerging: allow MEDIUM/HIGH, penalise EXTREME
    HIGH   = "high"    # spec: allow HIGH, penalise only EXTREME + chaotic


@dataclass
class UniverseSpec:
    """Specification for a screening universe."""
    name:          str
    label:         str
    description:   str
    tickers:       List[str]
    cap_bucket:    CapBucket        = CapBucket.UNKNOWN
    vol_tolerance: VolTolerance     = VolTolerance.MEDIUM

    # Per-ticker cap bucket overrides (ticker → CapBucket)
    ticker_caps:   Dict[str, CapBucket] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Universe 1 — Institutional Stability
# ---------------------------------------------------------------------------

INSTITUTIONAL = UniverseSpec(
    name="institutional",
    label="Institutional Stability",
    description=(
        "Broad market ETFs and mega-cap quality names. "
        "Rewards regime stability, drawdown control, and deep liquidity. "
        "Penalises elevated volatility. Reference universe for risk anchoring."
    ),
    tickers=[
        "SPY", "QQQ", "IWM", "DIA",         # broad ETFs
        "XLK", "XLF", "XLE",                 # sector ETFs
        "MSFT", "AAPL", "AMZN", "META",      # Mag-7
        "GOOGL", "NVDA", "AVGO",             # large-cap tech
        "JPM", "V", "MA", "BRK-B",           # financial / quality
        "JNJ", "UNH", "LLY",                 # healthcare quality
        "GLD", "TLT", "USO",                 # macro / hedges
    ],
    vol_tolerance=VolTolerance.LOW,
    ticker_caps={
        "SPY": CapBucket.ETF, "QQQ": CapBucket.ETF, "IWM": CapBucket.ETF,
        "DIA": CapBucket.ETF, "XLK": CapBucket.ETF, "XLF": CapBucket.ETF,
        "XLE": CapBucket.ETF, "GLD": CapBucket.ETF, "TLT": CapBucket.ETF,
        "USO": CapBucket.ETF,
        "MSFT": CapBucket.MEGA, "AAPL": CapBucket.MEGA,
        "AMZN": CapBucket.MEGA, "META": CapBucket.MEGA,
        "GOOGL": CapBucket.MEGA, "NVDA": CapBucket.MEGA,
        "AVGO": CapBucket.LARGE, "JPM": CapBucket.MEGA,
        "V": CapBucket.MEGA, "MA": CapBucket.MEGA, "BRK-B": CapBucket.MEGA,
        "JNJ": CapBucket.MEGA, "UNH": CapBucket.MEGA, "LLY": CapBucket.MEGA,
    },
)

# ---------------------------------------------------------------------------
# Universe 2 — Emerging Leaders
# ---------------------------------------------------------------------------

EMERGING = UniverseSpec(
    name="emerging",
    label="Emerging Leaders",
    description=(
        "Mid-to-small-cap names with genuine revenue, institutional ownership, "
        "and strong recent trend structure. Rewards trend acceleration and "
        "improving momentum quality. Moderate volatility is acceptable — "
        "extreme drawdowns and chaotic vol are penalised."
    ),
    tickers=[
        # Software / Cloud
        "CRWD", "NET", "TTD", "DDOG", "SNOW", "MDB", "ZS",
        # Consumer / Lifestyle
        "ONON", "DUOL", "CAVA", "CELH",
        # Fintech
        "HOOD", "SOFI", "AFRM", "NU",
        # AI / Data
        "PLTR", "APP", "RDDT", "TEM",
        # Biotech / Health
        "HIMS",
        # Aerospace / Defence
        "RKLB",
        # Semiconductors mid-cap
        "MRVL", "SMCI",
    ],
    vol_tolerance=VolTolerance.MEDIUM,
    ticker_caps={
        "CRWD": CapBucket.LARGE, "NET": CapBucket.MID, "TTD": CapBucket.MID,
        "DDOG": CapBucket.MID,   "SNOW": CapBucket.LARGE, "MDB": CapBucket.MID,
        "ZS":   CapBucket.MID,   "ONON": CapBucket.MID,  "DUOL": CapBucket.MID,
        "CAVA": CapBucket.SMALL, "CELH": CapBucket.SMALL,
        "HOOD": CapBucket.SMALL, "SOFI": CapBucket.SMALL,
        "AFRM": CapBucket.SMALL, "NU":   CapBucket.MID,
        "PLTR": CapBucket.LARGE, "APP":  CapBucket.LARGE,
        "RDDT": CapBucket.MID,   "TEM":  CapBucket.MID,
        "HIMS": CapBucket.SMALL, "RKLB": CapBucket.SMALL,
        "MRVL": CapBucket.LARGE, "SMCI": CapBucket.MID,
    },
)

# ---------------------------------------------------------------------------
# Universe 3 — Speculative High Beta
# ---------------------------------------------------------------------------

SPECULATIVE = UniverseSpec(
    name="speculative",
    label="Speculative High Beta",
    description=(
        "Small-cap, high-growth, high-volatility names in emerging sectors. "
        "Strict liquidity floor applies. Only survives screening if trend structure "
        "is coherent (not chaotic) and drawdown is survivable. "
        "Watchlist awareness only — not for meaningful position sizing."
    ),
    tickers=[
        # Quantum / AI hardware
        "IONQ", "RGTI", "QUBT",
        # Space / Defence small-cap
        "ASTS", "LUNR",
        # Biotech / genomics
        "RXRX",
        # Crypto-adjacent
        "MSTR", "COIN",
        # Small-cap software momentum
        "SOUN", "BBAI",
        # Clean energy speculative
        "PLUG", "BE",
    ],
    vol_tolerance=VolTolerance.HIGH,
    ticker_caps={
        t: CapBucket.SMALL for t in [
            "IONQ", "RGTI", "QUBT", "ASTS", "LUNR",
            "RXRX", "SOUN", "BBAI", "PLUG", "BE",
        ]
    },
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_UNIVERSES: Dict[str, UniverseSpec] = {
    "institutional": INSTITUTIONAL,
    "emerging":      EMERGING,
    "speculative":   SPECULATIVE,
}


def get_universe(name: str) -> UniverseSpec:
    """Retrieve a universe by name. Raises KeyError if not found."""
    if name not in ALL_UNIVERSES:
        raise KeyError(f"Unknown universe '{name}'. Available: {list(ALL_UNIVERSES.keys())}")
    return ALL_UNIVERSES[name]


def all_tickers() -> List[str]:
    """Return deduplicated list of all tickers across all universes."""
    seen = set()
    result = []
    for u in ALL_UNIVERSES.values():
        for t in u.tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result
