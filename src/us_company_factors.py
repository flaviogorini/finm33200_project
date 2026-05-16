"""
Build the per-ticker monthly panel for the US-company experiment.

For each of the 13 US tickers in the Bloomberg manual workbook:
  * resample daily ``PX_LAST`` to month-end → ``ret`` (simple monthly return)
  * resample analyst consensus levels to month-end and compute
    month-on-month percent changes:
      - ``F_pe``     = pct_change(BEST_PE_RATIO)
      - ``F_marg``   = pct_change(BEST_NET_INCOME / BEST_SALES)
      - ``F_sales``  = pct_change(BEST_SALES)

All four series are aligned on month-end dates. The output parquet is
long-form with one row per (ticker, month-end):

    ticker, date, ret, F_pe, F_marg, F_sales

The first few months for each ticker are NaN due to the pct_change lag,
and rows fully NaN on the factor columns are dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pull_manual_companies import (
    load_manual_companies_forecast,
    load_manual_companies_hist,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

US_PANEL_PARQUET = "US_Company_Panel.parquet"

US_TEST_START = "2010-01-01"
US_ROLLING_WINDOW_YEARS = 5
FACTOR_COLS = ("F_pe", "F_marg", "F_sales")


def _wide_monthly(long_df: pd.DataFrame, value_field: str) -> pd.DataFrame:
    """Pivot the long-form parquet to wide (date × ticker) at month-end."""
    sub = long_df[long_df["field"] == value_field]
    wide = (
        sub.pivot(index="date", columns="ticker", values="value")
        .sort_index()
        .astype(float)
    )
    return wide.resample("ME").last()


def build_panel() -> pd.DataFrame:
    """Construct the long-form (ticker, date) panel with returns + 3 factors."""
    hist = load_manual_companies_hist()
    fc = load_manual_companies_forecast()

    px = _wide_monthly(hist, "PX_LAST")
    pe = _wide_monthly(fc, "BEST_PE_RATIO")
    sales = _wide_monthly(fc, "BEST_SALES")
    ni = _wide_monthly(fc, "BEST_NET_INCOME")

    ret = px.pct_change()
    f_pe = pe.pct_change()
    f_sales = sales.pct_change()
    margin = ni / sales
    f_marg = margin.pct_change()

    # Replace inf (from divide-by-near-zero on margin) with NaN
    for df in (ret, f_pe, f_sales, f_marg):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    panel = (
        pd.concat(
            {"ret": ret, "F_pe": f_pe, "F_marg": f_marg, "F_sales": f_sales},
            axis=1,
            names=["field", "ticker"],
        )
        .stack("ticker", future_stack=True)
        .reset_index()
        .rename(columns={"level_0": "date"})
    )
    panel = panel.dropna(subset=list(FACTOR_COLS))
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel[["ticker", "date", "ret", *FACTOR_COLS]]


def load_us_panel(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / US_PANEL_PARQUET)


if __name__ == "__main__":
    panel = build_panel()
    out = DATA_DIR / US_PANEL_PARQUET
    panel.to_parquet(out, index=False)
    print(f"wrote {len(panel):,} rows → {out}")
