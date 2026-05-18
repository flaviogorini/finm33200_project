"""Smoke test for the point-in-time 10-Q monthly panel."""

import pandas as pd

from build_10q_monthly_panel import FEATURES_PATH, build_monthly_panel


def test_monthly_panel_has_no_lookahead_when_features_exist():
    if not FEATURES_PATH.exists():
        return
    panel = build_monthly_panel(FEATURES_PATH)
    valid = panel.dropna(subset=["filing_date"])
    assert (pd.to_datetime(valid["filing_date"]) <= pd.to_datetime(valid["date"])).all()
