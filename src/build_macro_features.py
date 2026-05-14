"""Build monthly macro features from the manual Bloomberg macro pull.

Consumes:
    _data/Macro_Data_US.parquet   (cols: date, ticker, field, value)

The macro file's "tickers" are Bloomberg macro indices (VIX Index, GT10 Govt,
DXY Curncy, …) and the only field is PX_LAST. We pivot wide on ticker and
resample to month-end.

Output:
    _data/features_macro_monthly.parquet

Schema (one column per macro series, names normalised to snake_case):
    date            month-end Timestamp
    vix             CBOE VIX, end-of-month close
    treas_10y       US 10Y Treasury yield, EOM
    treas_2y        US 2Y, EOM
    treas_30y       US 30Y, EOM
    treas_5y        US 5Y, EOM
    yield_curve_2s10s   2s10s spread (bps), EOM
    yield_curve_2s30s   2s30s spread, EOM
    breakeven_10y   USGGT10Y inflation breakeven, EOM
    dxy             Dollar Index, EOM
    eurusd_iv_3m    EUR/USD 3m implied vol, EOM
    wti_front       WTI front-month futures price, EOM
    us_cds_5y       US sovereign 5Y CDS spread, EOM

Real-time series — every row at date ``t`` uses only data observable on day
``t``. No lag needed.

This module supersedes the empty-stub plan: the manual macro file already
covers Flavio's expected scope (VIX + treasuries + FX + commodities). When
Flavio swaps in a richer FF5 + RF source, just add columns to the rename map.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "Macro_Data_US.parquet"
OUTPUT_FILENAME = "features_macro_monthly.parquet"

# Bloomberg ticker → our column name. Anything not listed here is dropped.
TICKER_RENAME = {
    "VIX Index": "vix",
    "GT10 Govt": "treas_10y",
    "GT2 Govt": "treas_2y",
    "GT30 Govt": "treas_30y",
    "GT5 Govt": "treas_5y",
    "USYC2Y10 Index": "yield_curve_2s10s",
    "USYC2Y30 Index": "yield_curve_2s30s",
    "USGGT10Y Index": "breakeven_10y",
    "DXY Curncy": "dxy",
    "EURUSDV3M Curncy": "eurusd_iv_3m",
    "CL1 Comdty": "wti_front",
    "US CDS EUR SR 5Y D14 Corp": "us_cds_5y",
}


def load_macro_long(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pull_manual_macro.py` first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def to_wide_monthly(long: pd.DataFrame) -> pd.DataFrame:
    # Keep only known macro series; pivot ticker → column.
    keep = long[long["ticker"].isin(TICKER_RENAME)].copy()
    if keep.empty:
        raise ValueError(
            "No known macro tickers in the input. "
            f"Expected any of {list(TICKER_RENAME)}; got {sorted(long['ticker'].unique())}."
        )

    wide = (
        keep.pivot_table(
            index="date", columns="ticker", values="value", aggfunc="last"
        )
        .rename(columns=TICKER_RENAME)
    )

    monthly = wide.resample("ME").last().reset_index()
    monthly = monthly.dropna(how="all", subset=[c for c in monthly.columns if c != "date"])

    # Stable column order: date first, then known macros in TICKER_RENAME order.
    cols = ["date"] + [c for c in TICKER_RENAME.values() if c in monthly.columns]
    return monthly[cols]


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    long = load_macro_long(data_dir)
    return to_wide_monthly(long)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    print(f"Wrote {len(panel):,} rows x {len(panel.columns)} cols -> {out}")
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
