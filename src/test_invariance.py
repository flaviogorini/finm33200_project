"""Invariance regression tests: v2 CAR3 cascade must not perturb v1 outputs.

After the v2 changes (sig_car3 added to signal panel, CAR3 added to
run_backtests STRATEGIES, sig_car3 added to joint_regression ALL_SIGNALS,
plus new factor_baseline / factor_regression modules), the original 5
strategies' backtest results must be byte-identical to the pre-CAR3
baseline.

These tests require a snapshot at ``_data_baseline/`` created BEFORE any
CAR3 code changes were merged. If the snapshot is missing, the tests are
skipped — they only run when a regression baseline exists.

What we check:

- ``metrics_main.json`` / ``metrics_stale_excl.json`` /
  ``metrics_post2018.json``: every numeric field for keys
  {anchor, ridge, lm, momentum, revisions} matches the snapshot exactly.
- ``ic_summary.json``: same keys, byte-identical metric values.
- ``results_main.parquet`` / ``results_stale_excl.parquet`` /
  ``results_post2018.parquet``: rows filtered to ``strategy != 'car3'``
  match the snapshot frame-equal.
- ``signal_panel_monthly.parquet``: every column except ``sig_car3``
  matches the snapshot.

What we do NOT check (expected to differ):

- ``fm_results.json``: the cross-sectional FM regression itself has a
  new 6th column (sig_car3), so coefficients for ALL signals legitimately
  shift. This is the new test result, not a regression bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
BASELINE_DIR = DATA_DIR.parent / "_data_baseline"

ORIGINAL_STRATEGIES = ["anchor", "ridge", "lm", "momentum", "revisions"]
INVARIANT_PANEL_COLS = [
    "date", "ticker", "px_eom", "fwd_ret_21d",
    "sig_anchor", "sig_ridge", "sig_lm", "sig_mom", "sig_rev",
    "days_since_earnings",
]
RESULTS_FILES = ["results_main.parquet", "results_stale_excl.parquet", "results_post2018.parquet"]
METRICS_FILES = ["metrics_main.json", "metrics_stale_excl.json", "metrics_post2018.json"]


def _require_snapshot():
    if not BASELINE_DIR.exists():
        pytest.skip(
            f"No invariance baseline at {BASELINE_DIR}. Snapshot before merging CAR3 "
            "changes to enable these tests."
        )


@pytest.mark.parametrize("fname", METRICS_FILES)
def test_metrics_unchanged_for_original_strategies(fname: str) -> None:
    """Per-strategy metrics for the 5 originals must be byte-identical."""
    _require_snapshot()
    base_path = BASELINE_DIR / fname
    curr_path = DATA_DIR / fname
    if not base_path.exists():
        pytest.skip(f"Baseline {fname} missing — partial snapshot")
    base = json.loads(base_path.read_text())
    curr = json.loads(curr_path.read_text())
    for strat in ORIGINAL_STRATEGIES:
        if strat not in base:
            continue
        assert strat in curr, f"{fname}: strategy {strat!r} disappeared"
        assert base[strat] == curr[strat], (
            f"{fname}: metrics drifted for {strat}:\n"
            f"  baseline: {base[strat]}\n"
            f"  current:  {curr[strat]}"
        )


def test_ic_summary_unchanged_for_original_strategies() -> None:
    """IC summary for the 5 originals must be byte-identical."""
    _require_snapshot()
    base_path = BASELINE_DIR / "ic_summary.json"
    curr_path = DATA_DIR / "ic_summary.json"
    if not base_path.exists():
        pytest.skip("Baseline ic_summary.json missing")
    base = json.loads(base_path.read_text())
    curr = json.loads(curr_path.read_text())
    for strat in ORIGINAL_STRATEGIES:
        if strat not in base:
            continue
        assert strat in curr, f"ic_summary.json: strategy {strat!r} disappeared"
        assert base[strat] == curr[strat], (
            f"ic_summary.json: IC summary drifted for {strat}:\n"
            f"  baseline: {base[strat]}\n"
            f"  current:  {curr[strat]}"
        )


@pytest.mark.parametrize("fname", RESULTS_FILES)
def test_results_unchanged_for_original_strategies(fname: str) -> None:
    """results_*.parquet rows for the 5 originals must match the baseline frame-equal."""
    _require_snapshot()
    base_path = BASELINE_DIR / fname
    curr_path = DATA_DIR / fname
    if not base_path.exists():
        pytest.skip(f"Baseline {fname} missing")
    base = pd.read_parquet(base_path)
    curr = pd.read_parquet(curr_path)
    base_filtered = base[base["strategy"].isin(ORIGINAL_STRATEGIES)].reset_index(drop=True)
    curr_filtered = (
        curr[curr["strategy"].isin(ORIGINAL_STRATEGIES)]
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )
    base_filtered = base_filtered.sort_values(["strategy", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(base_filtered, curr_filtered, check_like=True)


def test_signal_panel_invariant_columns() -> None:
    """All signal-panel columns EXCEPT sig_car3 must match the baseline frame-equal."""
    _require_snapshot()
    base_path = BASELINE_DIR / "signal_panel_monthly.parquet"
    curr_path = DATA_DIR / "signal_panel_monthly.parquet"
    if not base_path.exists():
        pytest.skip("Baseline signal_panel_monthly.parquet missing")
    base = pd.read_parquet(base_path).sort_values(["date", "ticker"]).reset_index(drop=True)
    curr = pd.read_parquet(curr_path).sort_values(["date", "ticker"]).reset_index(drop=True)
    cols = [c for c in INVARIANT_PANEL_COLS if c in base.columns and c in curr.columns]
    pd.testing.assert_frame_equal(base[cols], curr[cols], check_like=False)
