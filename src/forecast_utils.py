"""
Shared utilities for the stock return-forecasting pipeline.

Two responsibilities:
    1. Resample a daily price series to weekly / monthly closes and compute
       simple returns.
    2. Yield (train, target_date, target_return) tuples for both forecasting
       schemes:
         - 'expanding': train = all returns strictly before target_date
         - 'rolling':   train = last `window_years` years before target_date
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd
from dateutil.relativedelta import relativedelta

TEST_START = "2010-01-01"
ROLLING_WINDOW_YEARS = 5

FREQ_RULE = {"W": "W-FRI", "M": "ME"}


def compute_returns(prices: pd.Series, freq: str = "M") -> pd.DataFrame:
    """Resample a daily price series to last-of-period close, then compute
    simple returns.

    Parameters
    ----------
    prices : pd.Series
        Date-indexed daily prices.
    freq : {'W', 'M'}
        Period frequency. ``W`` → Friday close, ``M`` → calendar month-end.

    Returns
    -------
    pd.DataFrame
        Columns ``date`` and ``ret``, sorted by date, no NaNs.
    """
    if freq not in FREQ_RULE:
        raise ValueError(f"freq must be one of {list(FREQ_RULE)}, got {freq!r}")

    s = prices.dropna().sort_index()
    closes = s.resample(FREQ_RULE[freq]).last().dropna()
    ret = closes.pct_change().dropna()
    return ret.rename("ret").reset_index().rename(columns={s.index.name or "index": "date"})


def iter_windows(
    returns: pd.DataFrame,
    *,
    scheme: str,
    window_years: int | None = ROLLING_WINDOW_YEARS,
    test_start: str | pd.Timestamp = TEST_START,
) -> Iterator[tuple[pd.Series, pd.Timestamp, float]]:
    """Yield (train_returns, target_date, target_return) per test period.

    Parameters
    ----------
    returns : pd.DataFrame with columns ['date', 'ret'].
    scheme : 'expanding' | 'rolling'.
    window_years : rolling window size in years (ignored when expanding).
    test_start : first target date to forecast.

    Yields
    ------
    train_returns : pd.Series indexed by date — returns strictly before
        target_date.
    target_date : pd.Timestamp.
    target_return : float — the realised return at target_date.
    """
    if scheme not in {"expanding", "rolling"}:
        raise ValueError(f"unknown scheme {scheme!r}")
    if scheme == "rolling" and (window_years is None or window_years <= 0):
        raise ValueError("rolling scheme requires window_years > 0")

    test_start = pd.Timestamp(test_start)
    s = returns.set_index("date")["ret"].sort_index()
    test_dates = s.index[s.index >= test_start]

    for target_date in test_dates:
        if scheme == "expanding":
            train = s.loc[s.index < target_date]
        else:
            window_start = target_date - relativedelta(years=window_years)
            train = s.loc[(s.index >= window_start) & (s.index < target_date)]

        if len(train) < 10:
            continue

        yield train, target_date, float(s.loc[target_date])
