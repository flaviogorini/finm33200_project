"""
Load daily stock prices (PX_LAST) from `_data/US_Companies_Hist_Data.parquet`
and compute weekly / monthly simple returns.

The long-form parquet has columns: ``ticker``, ``date``, ``field``, ``value``.
We filter to a single ticker and the ``PX_LAST`` field, then delegate
resampling + percent-change to ``forecast_utils.compute_returns``.
"""

from __future__ import annotations

import pandas as pd

from forecast_utils import compute_returns
from pull_manual_companies import load_manual_companies_hist


def load_stock_prices(ticker: str) -> pd.Series:
    """Return a date-indexed daily ``PX_LAST`` series for `ticker`.

    Accepts either the short symbol (``AAPL``) or the full Bloomberg
    identifier (``AAPL US Equity``).
    """
    hist = load_manual_companies_hist()
    sub = hist[(hist["ticker"] == ticker) & (hist["field"] == "PX_LAST")]
    if sub.empty:
        full = f"{ticker} US Equity"
        sub = hist[(hist["ticker"] == full) & (hist["field"] == "PX_LAST")]
    if sub.empty:
        available = sorted(hist["ticker"].unique())
        raise ValueError(
            f"no PX_LAST rows for ticker={ticker!r}. "
            f"available tickers: {available}"
        )
    s = (
        sub[["date", "value"]]
        .dropna()
        .sort_values("date")
        .set_index("date")["value"]
        .astype(float)
    )
    s.name = "PX_LAST"
    s.index.name = "date"
    return s


def compute_stock_returns(ticker: str, freq: str = "M") -> pd.DataFrame:
    """Return DataFrame with columns ``date`` and ``ret`` for `ticker`."""
    prices = load_stock_prices(ticker)
    return compute_returns(prices, freq=freq)
