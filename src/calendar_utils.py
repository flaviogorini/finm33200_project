"""Single source of truth for date and horizon conventions.

Every script that touches dates in this project imports from this module so
that the trading-day calendar, the forward-return horizon, the execution
gap, and the rebalance date definition stay identical across phases.

Conventions:
    - Trading day:  pandas ``BDay`` (Mon-Fri, no holiday filter). Coarse but
      consistent. If a refined NYSE calendar is needed later, swap to
      ``pandas_market_calendars`` *only* by editing this module.
    - Rebalance date:  last *business* day of each calendar month
      (``freq='BM'``), NOT calendar month-end.
    - Forward-return horizon:  21 trading days from the signal date.
    - Execution gap:  T+1 trading day entry. Combined with the 21-day
      holding period, the exit is at T+22.

Calendar-day uses (the *only* places calendar days appear in this project):
    - ``days_since_earnings`` feature for ridge regression (spec section
      4.2 explicitly says "days").
    - 60-day stale-call robustness filter (spec section 8).
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay

HOLDING_BDAYS = 21
EXEC_GAP_BDAYS = 1
REVISION_LOOKBACK_BDAYS = 21
STALE_THRESHOLD_CDAYS = 60


def month_end_bd(start, end) -> pd.DatetimeIndex:
    """Last business day of each month in [start, end], inclusive."""
    return pd.date_range(start=start, end=end, freq="BME")


def bday_shift(date, n: int) -> pd.Timestamp:
    """Shift a date by n trading days (signed)."""
    return pd.Timestamp(date) + n * BDay()


def fwd_ret_bd(
    prices: pd.Series,
    date,
    h: int = HOLDING_BDAYS,
    gap: int = EXEC_GAP_BDAYS,
) -> float:
    """Forward return from execution at T+gap to exit at T+gap+h.

    ``prices`` must be a ``Series`` indexed by date. If either the entry
    or exit date is missing from the index, the nearest *prior* available
    price is used (``ffill``-style lookup via ``asof``).

    Returns ``float('nan')`` if either side is unavailable (e.g. tail of
    series).
    """
    t = pd.Timestamp(date)
    entry_date = t + gap * BDay()
    exit_date = t + (gap + h) * BDay()

    last_available = prices.index.max()
    if pd.isna(last_available) or exit_date > last_available:
        return float("nan")

    px_entry = prices.asof(entry_date)
    px_exit = prices.asof(exit_date)

    if pd.isna(px_entry) or pd.isna(px_exit) or px_entry == 0:
        return float("nan")
    return float(px_exit / px_entry - 1.0)
