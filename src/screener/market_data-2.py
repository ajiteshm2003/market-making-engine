"""
src/screener/market_data.py
-----------------------------
Market Data Fetcher

Downloads historical OHLCV data via yfinance with:
- per-ticker error isolation (one failure doesn't stop others)
- optional on-disk CSV cache to reduce repeat downloads
- graceful handling of missing/delisted tickers
- data quality validation before returning

This module is intentionally decoupled from the factor engine.
All downstream code works on plain pandas DataFrames with columns:
    Open, High, Low, Close, Volume, Adj Close

The screener uses 'Adj Close' for all return calculations to account
for splits and dividends.
"""

from __future__ import annotations

import os
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Default universe
# ---------------------------------------------------------------------------

DEFAULT_UNIVERSE: List[str] = [
    # Broad market ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Large-cap tech
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "AVGO", "AMD", "TSLA",
    # Sector ETFs
    "XLE", "XLF", "XLK",
    # Commodity / bond
    "USO", "GLD", "TLT",
]

MINIMUM_TRADING_DAYS = 60       # reject tickers with fewer observations
MINIMUM_ADV_USD      = 10_000_000  # reject tickers with < $10M avg daily volume


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TickerData:
    """Validated OHLCV data for a single ticker."""
    ticker:       str
    df:           pd.DataFrame        # columns: Open High Low Close Volume Adj_Close
    start_date:   str
    end_date:     str
    n_days:       int
    avg_dollar_volume: float

    @property
    def adj_close(self) -> pd.Series:
        return self.df["Adj_Close"]

    @property
    def volume(self) -> pd.Series:
        return self.df["Volume"]

    @property
    def close(self) -> pd.Series:
        return self.df["Close"]


@dataclass
class FetchResult:
    """Summary of a download batch."""
    successful:  List[str]            = field(default_factory=list)
    failed:      Dict[str, str]       = field(default_factory=dict)   # ticker → reason
    rejected:    Dict[str, str]       = field(default_factory=dict)   # ticker → reason
    data:        Dict[str, TickerData] = field(default_factory=dict)

    def print_summary(self) -> None:
        print(f"  Fetched    : {len(self.successful)} tickers")
        print(f"  Failed     : {len(self.failed)}")
        for t, reason in self.failed.items():
            print(f"    {t}: {reason}")
        print(f"  Rejected   : {len(self.rejected)}")
        for t, reason in self.rejected.items():
            print(f"    {t}: {reason}")


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class MarketDataFetcher:
    """
    Downloads and caches historical OHLCV data for a list of tickers.

    Parameters
    ----------
    lookback_days : int
        Calendar days of history to request (default 365 = ~252 trading days).
    cache_dir : str, optional
        Directory to store cached CSV files. None = no cache.
    min_trading_days : int
        Minimum observations required to keep a ticker.
    min_adv_usd : float
        Minimum average dollar volume to keep a ticker.
    request_delay : float
        Seconds to sleep between yfinance requests (rate-limit courtesy).
    """

    def __init__(
        self,
        lookback_days: int = 365,
        cache_dir: Optional[str] = None,
        min_trading_days: int = MINIMUM_TRADING_DAYS,
        min_adv_usd: float = MINIMUM_ADV_USD,
        request_delay: float = 0.3,
    ) -> None:
        self.lookback_days    = lookback_days
        self.cache_dir        = cache_dir
        self.min_trading_days = min_trading_days
        self.min_adv_usd      = min_adv_usd
        self.request_delay    = request_delay

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def fetch(
        self,
        tickers: List[str],
        end_date: Optional[str] = None,
    ) -> FetchResult:
        """
        Download data for all tickers.

        Parameters
        ----------
        tickers  : list of ticker symbols
        end_date : YYYY-MM-DD string (default = today)

        Returns
        -------
        FetchResult
        """
        result = FetchResult()

        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()
        start_dt = end_dt - timedelta(days=self.lookback_days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str   = end_dt.strftime("%Y-%m-%d")

        for ticker in tickers:
            try:
                df = self._load_or_download(ticker, start_str, end_str)
                if df is None or df.empty:
                    result.failed[ticker] = "No data returned"
                    continue

                # Standardise column names
                df = self._standardise(df)

                # Quality checks
                ok, reason = self._validate(ticker, df)
                if not ok:
                    result.rejected[ticker] = reason
                    continue

                adv = (df["Close"] * df["Volume"]).mean()
                td_data = TickerData(
                    ticker=ticker,
                    df=df,
                    start_date=df.index[0].strftime("%Y-%m-%d"),
                    end_date=df.index[-1].strftime("%Y-%m-%d"),
                    n_days=len(df),
                    avg_dollar_volume=round(adv, 0),
                )
                result.data[ticker] = td_data
                result.successful.append(ticker)

            except Exception as e:
                result.failed[ticker] = str(e)[:120]

            time.sleep(self.request_delay)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_or_download(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Return cached data if fresh enough, else download."""
        if self.cache_dir:
            path = self._cache_path(ticker, start, end)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                    if not df.empty:
                        return df
                except Exception:
                    pass

        df = self._download(ticker, start, end)

        if self.cache_dir and df is not None and not df.empty:
            df.to_csv(self._cache_path(ticker, start, end))

        return df

    def _download(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download from yfinance. Returns None on failure."""
        try:
            import yfinance as yf
            tkr = yf.Ticker(ticker)
            df = tkr.history(start=start, end=end, auto_adjust=False)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            return None

    def _standardise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and drop unnecessary columns."""
        df = df.copy()
        # yfinance may return 'Adj Close' or 'Adj_Close'
        rename = {}
        for col in df.columns:
            if "adj" in col.lower() and "close" in col.lower():
                rename[col] = "Adj_Close"
        df = df.rename(columns=rename)

        # Keep standard OHLCV + Adj_Close
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj_Close"]
                if c in df.columns]
        df = df[keep].copy()

        # If no Adj_Close, fall back to Close
        if "Adj_Close" not in df.columns and "Close" in df.columns:
            df["Adj_Close"] = df["Close"]

        df = df.dropna(subset=["Close", "Adj_Close"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def _validate(
        self, ticker: str, df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Run data quality checks. Returns (ok, reason)."""
        if len(df) < self.min_trading_days:
            return False, f"Only {len(df)} trading days (min {self.min_trading_days})"

        if "Volume" in df.columns and df["Volume"].mean() > 0:
            adv = (df["Close"] * df["Volume"]).mean()
            if adv < self.min_adv_usd:
                return False, f"ADV ${adv:,.0f} < ${self.min_adv_usd:,.0f} minimum"

        if df["Close"].isna().mean() > 0.05:
            return False, "More than 5% missing close prices"

        if (df["Close"] <= 0).any():
            return False, "Non-positive prices found"

        return True, ""

    def _cache_path(self, ticker: str, start: str, end: str) -> str:
        key = f"{ticker}_{start}_{end}"
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"{ticker}_{h}.csv")
