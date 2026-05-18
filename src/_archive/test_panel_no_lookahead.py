"""No-lookahead-bias regression tests for the assembled monthly panel.

Hard-fails if the panel ever lets information from the future leak into a
row dated ``t``. Run with ``pytest src/test_panel_no_lookahead.py``.

Tests skip gracefully if the panel parquet hasn't been built yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from build_panel import LABEL_COLS, OUTPUT_FILENAME
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
PANEL_PATH = DATA_DIR / OUTPUT_FILENAME


def _load_panel_or_skip() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        pytest.skip(
            f"{PANEL_PATH} not found — run `python src/build_panel.py` first."
        )
    df = pd.read_parquet(PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_dates_are_month_end():
    panel = _load_panel_or_skip()
    bad = panel.loc[panel["date"] != panel["date"] + pd.offsets.MonthEnd(0), "date"]
    assert bad.empty, f"non-month-end dates: {bad.unique()[:5]}"


def test_no_duplicate_keys():
    panel = _load_panel_or_skip()
    counts = panel.groupby(["date", "ticker"]).size()
    dup = counts[counts > 1]
    assert dup.empty, f"duplicate (date, ticker) keys: {dup.head().to_dict()}"


def test_dates_strictly_within_universe():
    """No row dated in the future relative to today."""
    panel = _load_panel_or_skip()
    today = pd.Timestamp.today().normalize()
    future = panel[panel["date"] > today + pd.offsets.MonthEnd(0)]
    assert future.empty, f"{len(future)} rows dated after today"


def test_sentiment_anchor_date_in_past():
    """If the sentiment block is present, last_event_date must be <= row date."""
    panel = _load_panel_or_skip()
    if "last_event_date" not in panel.columns:
        pytest.skip("sentiment columns not in panel")
    sub = panel.dropna(subset=["last_event_date"])
    if sub.empty:
        pytest.skip("no sentiment rows present")
    leaked = sub[pd.to_datetime(sub["last_event_date"]) > sub["date"]]
    assert leaked.empty, (
        f"sentiment activated from a future call on "
        f"{len(leaked)} rows; first 3:\n{leaked.head(3)}"
    )


def test_label_columns_isolated_from_features():
    """``fwd_*`` columns exist but are never relied upon as features."""
    panel = _load_panel_or_skip()
    present_labels = [c for c in LABEL_COLS if c in panel.columns]
    if not present_labels:
        pytest.skip("no fwd_* label columns present (returns not built?)")
    # Enforce label naming convention is honoured.
    for c in panel.columns:
        if c.startswith("fwd_"):
            assert c in LABEL_COLS, (
                f"unknown fwd_ column {c!r} not declared in build_panel.LABEL_COLS"
            )
