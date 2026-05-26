"""Single source of truth for date and horizon conventions (v3).

v3 switch: the project moved from 21-trading-day forward-return windows to
**calendar-month** windows so that strategy returns can be regressed directly
against Ken French's calendar-month FF5 factors without the timestamp-shift
hack that v2 had in §5.6. The 21-BDay constants from v1/v2 are deleted on
this branch — v1/v2 live on main.

Conventions:
    - Trading day: pandas ``BDay`` (Mon-Fri, no holiday filter). Coarse but
      consistent. Used only for the per-call CAR3 window
      ``[event_date − 1 BDay, event_date + 1 BDay]`` (literature-standard PEAD
      announcement window). The monthly backtest cadence uses BME, not BDay.
    - Rebalance date: last *business* day of each calendar month
      (``freq='BME'``). Same as v2.
    - Forward-return window (monthly backtest): BME(T) close → BME(T+1) close.
      Same as Ken French's monthly FF5 returns. No execution gap; trades are
      assumed to execute at the close of the signal date (standard academic
      convention for monthly backtests, matches FF5 / Fama-MacBeth practice).
    - Holding period: exactly one calendar month (BME → next BME).

Calendar-day uses (the *only* places calendar days appear in the project):
    - ``days_since_earnings`` feature for ridge regression and the
      stale-call carry-forward window (60 calendar days).
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay, BMonthEnd

# Calendar-day windows (unchanged from v2).
STALE_THRESHOLD_CDAYS = 60

# Per-call CAR3 announcement-window definition (unchanged from v2 — this is a
# call-level event-window measurement, not a monthly cadence concept).
CAR3_GAP_BDAYS = -1
CAR3_H_BDAYS = 2


def month_end_bd(start, end) -> pd.DatetimeIndex:
    """Last business day of each month in [start, end], inclusive."""
    return pd.date_range(start=start, end=end, freq="BME")


def bday_shift(date, n: int) -> pd.Timestamp:
    """Shift a date by n trading days (signed). Kept for CAR3 event-window
    arithmetic and ad-hoc per-call use; the monthly backtest cadence is BME."""
    return pd.Timestamp(date) + n * BDay()


def next_bme(date) -> pd.Timestamp:
    """Next business-month-end strictly after ``date``."""
    t = pd.Timestamp(date)
    # BMonthEnd(0) is "this BME if t is a BME, else next BME"; we want strictly next.
    bme0 = t + BMonthEnd(0)
    return bme0 if bme0 > t else (t + BMonthEnd(1))


def fwd_ret_calmonth(prices: pd.Series, date) -> float:
    """Calendar-month forward return: close at BME(T) → close at BME(T+1).

    ``prices`` must be a ``Series`` indexed by date. If either endpoint is
    missing from the index, the nearest *prior* available price is used
    (``ffill``-style lookup via ``asof``). The signal date itself must
    already be a BME — that's the caller's responsibility (the project's
    rebalance dates are always BME).

    Returns ``float('nan')`` if either side is unavailable (e.g. tail of
    series, or the next BME extends beyond the price history).
    """
    t = pd.Timestamp(date)
    exit_date = next_bme(t)

    last_available = prices.index.max()
    if pd.isna(last_available) or exit_date > last_available:
        return float("nan")

    px_entry = prices.asof(t)
    px_exit = prices.asof(exit_date)

    if pd.isna(px_entry) or pd.isna(px_exit) or px_entry == 0:
        return float("nan")
    return float(px_exit / px_entry - 1.0)


def fwd_ret_event_calmonth(prices: pd.Series, event_date) -> float:
    """Forward-return target for the ridge model, calendar-month variant.

    Used by ``train_ridge.py``. Measures the return from the close of
    ``event_date`` (the earnings-call date) to the close of the next BME
    that lies at least one calendar month after the event. This is the
    natural calendar-month analog of v2's 21-trading-day target, anchored
    to per-call event dates rather than month-end rebalance dates.
    """
    t = pd.Timestamp(event_date)
    target_after = t + pd.DateOffset(months=1)
    # Snap up to the BME at-or-after target_after.
    exit_date = target_after + BMonthEnd(0)
    if exit_date < target_after:
        exit_date = target_after + BMonthEnd(1)

    last_available = prices.index.max()
    if pd.isna(last_available) or exit_date > last_available:
        return float("nan")

    px_entry = prices.asof(t)
    px_exit = prices.asof(exit_date)

    if pd.isna(px_entry) or pd.isna(px_exit) or px_entry == 0:
        return float("nan")
    return float(px_exit / px_entry - 1.0)


def fwd_ret_bd(prices: pd.Series, date, h: int, gap: int = 0) -> float:
    """Forward return over a generic BDay window. Kept for the per-call CAR3
    measurement (``gap=-1, h=2`` → px[t-1] → px[t+1]). NOT used for the
    monthly backtest cadence in v3 — that uses ``fwd_ret_calmonth``."""
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
