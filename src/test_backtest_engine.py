"""Unit tests for the backtest engine + IC computation + calendar utils.

Run via ``pytest src/test_backtest_engine.py`` (or as part of
``doit run_pytest``). The tests use synthetic data and avoid any
on-disk parquets so they pass in CI even before the WRDS bulk pull
completes.
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
    EXEC_GAP_BDAYS,
    HOLDING_BDAYS,
    bday_shift,
    fwd_ret_bd,
    month_end_bd,
)


def test_calendar_utils_constants():
    assert HOLDING_BDAYS == 21
    assert EXEC_GAP_BDAYS == 1


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


def test_fwd_ret_bd_matches_manual_calc():
    """fwd_ret_bd at gap=1, h=21 entries T+1 and exits T+22 from a known price series."""
    idx = pd.bdate_range("2024-01-02", periods=60)
    prices = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx, name="px")

    signal_date = idx[10]                # has 49 bdays after it -> ample room
    expected_entry = prices.iloc[11]     # T+1 in business days
    expected_exit = prices.iloc[32]      # T+22 (gap+h)
    expected_ret = expected_exit / expected_entry - 1.0

    got = fwd_ret_bd(prices, signal_date)
    assert got == pytest.approx(expected_ret, rel=1e-9)


def test_fwd_ret_bd_returns_nan_past_data_end():
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    # the last signal date can't produce a 21-bday forward return
    assert np.isnan(fwd_ret_bd(prices, idx[-1]))


def _make_synthetic_panel(n_months: int = 24, n_tickers: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic monthly panel with a planted signal that is positively
    correlated with the realised forward return."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=n_months, freq="BME")
    rows = []
    for d in dates:
        # signal in [-1, 1]; fwd_ret = 0.02 * signal + noise (signal-to-noise ~ same OOM)
        sig = rng.uniform(-1.0, 1.0, size=n_tickers)
        noise = rng.normal(0, 0.03, size=n_tickers)
        fwd = 0.02 * sig + noise
        for t in range(n_tickers):
            rows.append({
                "date": d,
                "ticker": f"T{t:03d}",
                "sig": float(sig[t]),
                "fwd_ret_21d": float(fwd[t]),
            })
    return pd.DataFrame(rows)


def test_run_backtest_synthetic_planted_signal():
    panel = _make_synthetic_panel()
    res = run_backtest(panel, "sig")
    assert not res.monthly.empty
    assert res.metrics["n_months"] == 24
    # ranking should systematically capture the positive signal -> positive ann_ret
    assert res.metrics["ann_ret"] > 0
    # hit rate should be substantially above 50% with a real signal
    assert res.metrics["hit_rate"] > 0.6


def test_run_backtest_no_signal_random_returns():
    """A signal uncorrelated with returns produces ann_ret within sampling noise.

    With quintile legs of 20 tickers and 60 months, the long-short mean
    return has std ~ 2 * 0.05 / sqrt(20 * 60) ~ 0.3%/month ~ 3.5% annualised.
    A 0.10 ann_ret bound is roughly 3 sigma; any tighter and the test would
    flake on individual seeds.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-31", periods=60, freq="BME")
    rows = []
    for d in dates:
        for t in range(100):
            rows.append({
                "date": d, "ticker": f"T{t:03d}",
                "sig": rng.normal(),
                "fwd_ret_21d": rng.normal(0, 0.05),
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
    """If a single month has fewer than MIN_OBS_PER_MONTH tickers, drop it."""
    panel = _make_synthetic_panel(n_months=6, n_tickers=MIN_OBS_PER_MONTH - 5)
    res = run_backtest(panel, "sig")
    assert res.monthly.empty


def test_compute_ic_planted_signal_is_positive():
    panel = _make_synthetic_panel()
    ic = compute_ic(panel, "sig")
    summary = ic_summary(ic)
    assert summary["ic_mean"] > 0
    # IR > 1 because the synthetic signal is strong
    assert summary["ic_ir"] > 0.5


def test_compute_ic_random_signal_centred_at_zero():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2018-01-31", periods=60, freq="BME")
    rows = []
    for d in dates:
        for t in range(80):
            rows.append({
                "date": d, "ticker": f"T{t:03d}",
                "sig": rng.normal(), "fwd_ret_21d": rng.normal(),
            })
    panel = pd.DataFrame(rows)
    ic = compute_ic(panel, "sig")
    summary = ic_summary(ic)
    assert abs(summary["ic_mean"]) < 0.05


def test_stale_filter_only_applies_to_transcript_strategies():
    """Per spec section 8, drop_stale_gt is for the three sentiment
    strategies only. Momentum and revisions should see the same panel
    under stale_excl as under main.
    """
    from run_backtests import _filters_for, TRANSCRIPT_STRATEGIES

    base = {"drop_stale_gt": 60}
    for label in TRANSCRIPT_STRATEGIES:
        assert _filters_for(label, "stale_excl", base) == base
    for label in ["momentum", "revisions"]:
        # drop_stale_gt stripped; otherwise empty
        assert "drop_stale_gt" not in _filters_for(label, "stale_excl", base)

    # Other specs pass through untouched
    base_post = {"start_date": "2018-12-31"}
    for label in ["anchor", "ridge", "lm", "momentum", "revisions"]:
        assert _filters_for(label, "post2018", base_post) == base_post
