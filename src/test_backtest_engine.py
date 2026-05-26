"""Unit tests for the backtest engine + IC computation + calendar utils (v3).

Run via ``pytest src/test_backtest_engine.py`` (or as part of
``doit run_pytest``). The tests use synthetic data and avoid any on-disk
parquets so they pass in CI even before the WRDS bulk pull completes.

v3 change vs v2: the forward-return column is ``fwd_ret_1m`` (calendar
month) instead of ``fwd_ret_21d`` (21 BDays). The ``HOLDING_BDAYS`` /
``EXEC_GAP_BDAYS`` / ``REVISION_LOOKBACK_BDAYS`` constants are deleted
on the v3 branch; tests that asserted on them are removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import (
    MIN_OBS_PER_MONTH,
    compute_ic,
    ic_summary,
    run_backtest,
)
from calendar_utils import (
    STALE_THRESHOLD_CDAYS,
    bday_shift,
    fwd_ret_bd,
    fwd_ret_calmonth,
    fwd_ret_event_calmonth,
    month_end_bd,
    next_bme,
)


def test_calendar_utils_constants():
    # v3 keeps only the calendar-day stale threshold.
    assert STALE_THRESHOLD_CDAYS == 60


def test_month_end_bd_returns_business_month_ends():
    dates = month_end_bd("2024-01-01", "2024-03-31")
    assert len(dates) == 3
    # 2024-01-31 was a Wednesday (BD), 2024-02-29 Thu, 2024-03-29 Fri (Mar 31 is Sun)
    assert all(d.weekday() < 5 for d in dates)
    assert dates[-1] == pd.Timestamp("2024-03-29")


def test_bday_shift_skips_weekends():
    # Friday + 1 BD == Monday
    friday = pd.Timestamp("2024-01-05")
    assert bday_shift(friday, 1) == pd.Timestamp("2024-01-08")


def test_next_bme_strictly_after():
    # Jan 31 2024 is a BME (Wednesday). next_bme should be Feb 29 2024 (Thursday).
    assert next_bme(pd.Timestamp("2024-01-31")) == pd.Timestamp("2024-02-29")
    # A mid-month date returns this month's BME.
    assert next_bme(pd.Timestamp("2024-02-15")) == pd.Timestamp("2024-02-29")


def test_fwd_ret_calmonth_matches_manual():
    """fwd_ret_calmonth at a BME returns price ratio to next BME."""
    idx = pd.bdate_range("2024-01-02", "2024-04-30")
    prices = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx, name="px")

    signal_date = pd.Timestamp("2024-01-31")  # BME
    expected_entry = prices.asof(pd.Timestamp("2024-01-31"))
    expected_exit = prices.asof(pd.Timestamp("2024-02-29"))
    expected = expected_exit / expected_entry - 1.0

    got = fwd_ret_calmonth(prices, signal_date)
    assert got == pytest.approx(expected, rel=1e-9)


def test_fwd_ret_calmonth_returns_nan_past_data_end():
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    # The last BME within the series can't produce a next-BME forward return
    # if there isn't enough data after it.
    bmes = month_end_bd(idx[0], idx[-1])
    if len(bmes):
        assert np.isnan(fwd_ret_calmonth(prices, bmes[-1]))


def test_fwd_ret_event_calmonth_matches_manual():
    """fwd_ret_event_calmonth at a non-BME event date measures close-to-BME-one-month-out."""
    idx = pd.bdate_range("2024-01-02", "2024-04-30")
    prices = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx, name="px")

    event_date = pd.Timestamp("2024-01-15")  # mid-Jan
    # target_after = Feb 15; BME at-or-after = Feb 29
    expected_entry = prices.asof(event_date)
    expected_exit = prices.asof(pd.Timestamp("2024-02-29"))
    expected = expected_exit / expected_entry - 1.0

    got = fwd_ret_event_calmonth(prices, event_date)
    assert got == pytest.approx(expected, rel=1e-9)


def test_fwd_ret_bd_car3_window():
    """fwd_ret_bd at gap=-1, h=2 is the CAR3 announcement window."""
    idx = pd.bdate_range("2024-01-02", "2024-04-30")
    prices = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx, name="px")

    event_date = idx[10]  # T
    expected_entry = prices.iloc[9]   # T - 1 BD
    expected_exit = prices.iloc[11]   # T + 1 BD
    expected = expected_exit / expected_entry - 1.0

    got = fwd_ret_bd(prices, event_date, h=2, gap=-1)
    assert got == pytest.approx(expected, rel=1e-9)


def _make_synthetic_panel(n_months: int = 24, n_tickers: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic monthly panel with a planted signal positively correlated
    with the realised forward return."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=n_months, freq="BME")
    rows = []
    for d in dates:
        sig = rng.uniform(-1.0, 1.0, size=n_tickers)
        noise = rng.normal(0, 0.03, size=n_tickers)
        fwd = 0.02 * sig + noise
        for t in range(n_tickers):
            rows.append({
                "date": d,
                "ticker": f"T{t:03d}",
                "sig": float(sig[t]),
                "fwd_ret_1m": float(fwd[t]),
            })
    return pd.DataFrame(rows)


def test_run_backtest_synthetic_planted_signal():
    panel = _make_synthetic_panel()
    res = run_backtest(panel, "sig")
    assert not res.monthly.empty
    assert res.metrics["n_months"] == 24
    assert res.metrics["ann_ret"] > 0
    assert res.metrics["hit_rate"] > 0.6


def test_run_backtest_no_signal_random_returns():
    """A signal uncorrelated with returns has ann_ret within sampling noise."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-31", periods=60, freq="BME")
    rows = []
    for d in dates:
        for t in range(100):
            rows.append({
                "date": d, "ticker": f"T{t:03d}",
                "sig": rng.normal(),
                "fwd_ret_1m": rng.normal(0, 0.05),
            })
    panel = pd.DataFrame(rows)
    res = run_backtest(panel, "sig")
    assert abs(res.metrics["ann_ret"]) < 0.10


def test_run_backtest_filters_post_start_date():
    panel = _make_synthetic_panel(n_months=36)
    res_main = run_backtest(panel, "sig")
    res_post = run_backtest(panel, "sig", filters={"start_date": panel["date"].iloc[-12 * 100]})
    assert res_main.metrics["n_months"] > res_post.metrics["n_months"]


def test_run_backtest_skips_months_below_min_obs():
    panel = _make_synthetic_panel(n_months=6, n_tickers=MIN_OBS_PER_MONTH - 5)
    res = run_backtest(panel, "sig")
    assert res.monthly.empty


def test_compute_ic_planted_signal_is_positive():
    panel = _make_synthetic_panel()
    ic = compute_ic(panel, "sig")
    summary = ic_summary(ic)
    assert summary["ic_mean"] > 0
    assert summary["ic_ir"] > 0.5


def test_compute_ic_random_signal_centred_at_zero():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2018-01-31", periods=60, freq="BME")
    rows = []
    for d in dates:
        for t in range(80):
            rows.append({
                "date": d, "ticker": f"T{t:03d}",
                "sig": rng.normal(), "fwd_ret_1m": rng.normal(),
            })
    panel = pd.DataFrame(rows)
    ic = compute_ic(panel, "sig")
    summary = ic_summary(ic)
    assert abs(summary["ic_mean"]) < 0.05


def test_stale_filter_only_applies_to_transcript_strategies():
    """Per spec, drop_stale_gt is for per-call signal strategies only.
    Momentum and revisions should see the same panel under stale_excl as under main.
    """
    from run_backtests import _filters_for, TRANSCRIPT_STRATEGIES

    base = {"drop_stale_gt": 60}
    for label in TRANSCRIPT_STRATEGIES:
        assert _filters_for(label, "stale_excl", base) == base
    for label in ["momentum", "revisions"]:
        assert "drop_stale_gt" not in _filters_for(label, "stale_excl", base)

    # Other specs pass through untouched
    base_post = {"start_date": "2018-12-31"}
    for label in ["anchor", "ridge", "lm", "momentum", "revisions", "car3"]:
        assert _filters_for(label, "post2018", base_post) == base_post
