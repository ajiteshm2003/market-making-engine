"""
src/screener/factors.py
------------------------
Factor Engine

Computes all screening factors for a single ticker's price series.
Uses the existing risk modules (VaR, ES) and regime classifier from the
market making engine rather than reimplementing them.

Factors computed
----------------

Trend
  ret_20d        : 20-day log return
  ret_60d        : 60-day log return
  price_vs_ma20  : close / 20-day SMA - 1  (positive = above moving average)
  price_vs_ma50  : close / 50-day SMA - 1

Volatility / Regime
  vol_20d        : annualised realized volatility over 20 days
  vol_60d        : annualised realized volatility over 60 days
  vol_pct        : percentile of current vol_20d in the historical vol distribution
  regime         : LOW / MEDIUM / HIGH / EXTREME

Risk
  max_drawdown   : max peak-to-trough decline over the lookback
  var_95         : 95% historical VaR on daily log-returns
  es_95          : 95% Expected Shortfall on daily log-returns
  downside_vol   : annualised std of negative daily returns only

Liquidity
  avg_dollar_vol : average daily $ volume
  vol_stability  : 1 - coefficient_of_variation(volume)  → higher = more stable

All volatility figures are annualised (× √252).
All return figures are total log-returns over the period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from ..risk.var import historical_var, expected_shortfall
from ..models.regime import (
    RegimeClassifier, RegimeThresholds, VolatilityRegime,
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TickerFactors:
    """All computed factors for a single ticker."""
    ticker: str

    # Trend
    ret_20d:       Optional[float] = None
    ret_60d:       Optional[float] = None
    price_vs_ma20: Optional[float] = None
    price_vs_ma50: Optional[float] = None

    # Volatility
    vol_20d:       Optional[float] = None   # annualised
    vol_60d:       Optional[float] = None
    vol_pct:       Optional[float] = None   # 0-100 percentile
    regime:        Optional[VolatilityRegime] = None

    # Risk
    max_drawdown:  Optional[float] = None   # positive = loss magnitude
    var_95:        Optional[float] = None
    es_95:         Optional[float] = None
    downside_vol:  Optional[float] = None

    # Liquidity
    avg_dollar_vol:  Optional[float] = None
    vol_stability:   Optional[float] = None   # 0-1

    # Raw data for downstream use
    n_days:        int = 0
    error:         Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """True if all core factors were computed successfully."""
        return (
            self.error is None
            and self.ret_20d is not None
            and self.vol_20d is not None
            and self.max_drawdown is not None
        )

    @property
    def regime_label(self) -> str:
        return self.regime.value if self.regime else "unknown"

    def __str__(self) -> str:
        return (
            f"{self.ticker:<6}  ret20={self.ret_20d:+.2%}  "
            f"ret60={self.ret_60d:+.2%}  "
            f"vol20={self.vol_20d:.1%}  "
            f"regime={self.regime_label:<8}  "
            f"dd={self.max_drawdown:.2%}  "
            f"VaR95={self.var_95:.3%}"
            if self.is_valid else f"{self.ticker}: {self.error}"
        )


# ---------------------------------------------------------------------------
# Regime thresholds calibrated for annualised equity volatility
# Daily log-return std for equities: ~1% = annualised 16%
# We classify in annualised vol terms
# ---------------------------------------------------------------------------

EQUITY_THRESHOLDS = RegimeThresholds(
    low_threshold     = 0.12,   # <12% annualised → LOW
    high_threshold    = 0.25,   # >25% annualised → HIGH
    extreme_threshold = 0.45,   # >45% annualised → EXTREME
    hysteresis        = 0.02,
)


# ---------------------------------------------------------------------------
# Factor engine
# ---------------------------------------------------------------------------

class FactorEngine:
    """
    Computes all screening factors from a TickerData object.

    Parameters
    ----------
    thresholds : RegimeThresholds
        Volatility thresholds for regime classification.
        Defaults are calibrated for annualised equity realized volatility.
    """

    ANNUALISE = math.sqrt(252)

    def __init__(
        self,
        thresholds: Optional[RegimeThresholds] = None,
    ) -> None:
        self.thresholds = thresholds or EQUITY_THRESHOLDS

    def compute(self, ticker_data) -> TickerFactors:
        """
        Compute all factors for a single TickerData.

        Parameters
        ----------
        ticker_data : TickerData (from market_data module)

        Returns
        -------
        TickerFactors
        """
        f = TickerFactors(ticker=ticker_data.ticker, n_days=ticker_data.n_days)

        try:
            price = ticker_data.adj_close
            vol   = ticker_data.volume

            if len(price) < 21:
                f.error = f"Insufficient data: {len(price)} days"
                return f

            log_ret = _log_returns(price)

            # ── Trend ──────────────────────────────────────────────────────
            f.ret_20d = _total_log_return(price, 20)
            f.ret_60d = _total_log_return(price, 60)

            if len(price) >= 20:
                ma20 = price.rolling(20).mean().iloc[-1]
                f.price_vs_ma20 = price.iloc[-1] / ma20 - 1.0 if ma20 > 0 else None

            if len(price) >= 50:
                ma50 = price.rolling(50).mean().iloc[-1]
                f.price_vs_ma50 = price.iloc[-1] / ma50 - 1.0 if ma50 > 0 else None

            # ── Volatility / Regime ────────────────────────────────────────
            f.vol_20d = _realized_vol(log_ret, 20) * self.ANNUALISE
            f.vol_60d = _realized_vol(log_ret, 60) * self.ANNUALISE if len(log_ret) >= 60 else None

            # Volatility percentile: where does current vol sit in history?
            rolling_vol = (
                log_ret.rolling(20).std() * self.ANNUALISE
            ).dropna()
            if len(rolling_vol) >= 5:
                current_vol = rolling_vol.iloc[-1]
                f.vol_pct = round(
                    100 * (rolling_vol < current_vol).mean(), 1
                )

            # Regime: classify annualised vol_20d
            f.regime = self._classify_regime(f.vol_20d)

            # ── Risk ───────────────────────────────────────────────────────
            f.max_drawdown = _max_drawdown(price)

            daily_returns = list(log_ret.dropna())
            if len(daily_returns) >= 10:
                try:
                    f.var_95 = historical_var(daily_returns, 0.95).var
                    f.es_95  = expected_shortfall(daily_returns, 0.95).es
                except ValueError:
                    pass

            neg_ret = [r for r in daily_returns if r < 0]
            if len(neg_ret) >= 5:
                mean_neg = sum(neg_ret) / len(neg_ret)
                var_neg = sum((r - mean_neg) ** 2 for r in neg_ret) / (len(neg_ret) - 1)
                f.downside_vol = math.sqrt(var_neg) * self.ANNUALISE

            # ── Liquidity ──────────────────────────────────────────────────
            f.avg_dollar_vol = ticker_data.avg_dollar_volume

            if len(vol) >= 10:
                vol_mean = vol.mean()
                vol_std  = vol.std()
                cv = vol_std / vol_mean if vol_mean > 0 else 1.0
                f.vol_stability = max(0.0, min(1.0, 1.0 - cv))

        except Exception as e:
            f.error = str(e)[:200]

        return f

    def compute_batch(self, ticker_data_dict: dict) -> dict:
        """Compute factors for all tickers in a FetchResult.data dict."""
        return {ticker: self.compute(td) for ticker, td in ticker_data_dict.items()}

    def _classify_regime(self, annualised_vol: float) -> VolatilityRegime:
        """Classify annualised volatility into a regime."""
        t = self.thresholds
        if annualised_vol >= t.extreme_threshold:
            return VolatilityRegime.EXTREME
        elif annualised_vol >= t.high_threshold:
            return VolatilityRegime.HIGH
        elif annualised_vol >= t.low_threshold:
            return VolatilityRegime.MEDIUM
        else:
            return VolatilityRegime.LOW


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _log_returns(price: pd.Series) -> pd.Series:
    """Compute daily log returns."""
    import numpy as np
    return np.log(price / price.shift(1)).dropna()


def _total_log_return(price: pd.Series, n_days: int) -> Optional[float]:
    """Log return over the last n_days trading days."""
    if len(price) < n_days + 1:
        return None
    p_end   = price.iloc[-1]
    p_start = price.iloc[-n_days - 1]
    if p_start <= 0:
        return None
    import math
    return math.log(p_end / p_start)


def _realized_vol(log_ret: pd.Series, n_days: int) -> float:
    """Per-step std of last n_days log returns (not yet annualised)."""
    window = log_ret.iloc[-n_days:]
    if len(window) < 2:
        return 0.0
    return float(window.std())


def _max_drawdown(price: pd.Series) -> float:
    """Maximum peak-to-trough drawdown over the full series. Returns positive magnitude."""
    rolling_max = price.cummax()
    drawdowns = (rolling_max - price) / rolling_max.replace(0, float("nan"))
    return float(drawdowns.max()) if not drawdowns.empty else 0.0
