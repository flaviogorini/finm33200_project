"""Build monthly fundamentals features from the manual Bloomberg historical pull.

Consumes the long-format parquet produced by ``pull_manual_companies.py``:

    _data/US_Companies_Hist_Data.parquet   (cols: date, ticker, field, value)

Bloomberg's daily ``Hist_Data`` values are already point-in-time: at any date
``t``, ``SALES_REV_TURN`` is the most-recently-reported quarterly revenue
known on day ``t``, and ``PE_RATIO`` updates with the daily price. So the only
transform needed is wide-pivot + month-end resample (last value in month).

Output:
    _data/features_fundamentals_monthly.parquet

Schema:
    date            month-end Timestamp
    ticker          str  (Bloomberg suffix " US Equity" stripped)
    px_last         float
    pe_ratio        float
    revenue         float  (SALES_REV_TURN — trailing-period as of date)
    net_income      float
    net_debt        float
    ebitda          float

Activation rule: every value at row ``date`` is the last observation in
[start_of_month, date]; nothing from the future leaks in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "US_Companies_Hist_Data.parquet"
OUTPUT_FILENAME = "features_fundamentals_monthly.parquet"

FIELD_RENAME = {
    "PX_LAST": "px_last",
    "PE_RATIO": "pe_ratio",
    "SALES_REV_TURN": "revenue",
    "NET_INCOME": "net_income",
    "NET_DEBT": "net_debt",
    "EBITDA": "ebitda",
}


def _strip_bbg_suffix(ticker: str) -> str:
    """'AAPL US Equity' → 'AAPL'. Pass through anything that doesn't match."""
    if isinstance(ticker, str) and ticker.endswith(" US Equity"):
        return ticker[: -len(" US Equity")]
    return ticker


def load_hist_long(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the long-format historical fundamentals parquet."""
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pull_manual_companies.py` first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].map(_strip_bbg_suffix)
    return df


def to_wide_monthly(long: pd.DataFrame) -> pd.DataFrame:
    """Long [date, ticker, field, value] → wide monthly [date, ticker, *fields]."""
    keep = long[long["field"].isin(FIELD_RENAME)].copy()
    if keep.empty:
        raise ValueError(
            f"No rows match expected fields {list(FIELD_RENAME)}. "
            f"Got fields: {sorted(long['field'].unique())}"
        )

    wide = (
        keep.pivot_table(
            index=["date", "ticker"], columns="field", values="value", aggfunc="last"
        )
        .rename(columns=FIELD_RENAME)
        .reset_index()
    )

    # Resample to month-end per ticker — last available value in the month.
    out_frames: list[pd.DataFrame] = []
    feature_cols = list(FIELD_RENAME.values())
    for tk, grp in wide.groupby("ticker", sort=True):
        ts = grp.set_index("date").sort_index()
        monthly = ts[feature_cols].resample("ME").last()
        monthly["ticker"] = tk
        monthly = monthly.reset_index()
        # Drop months where every feature is NaN (before any data exists).
        monthly = monthly.dropna(subset=feature_cols, how="all")
        out_frames.append(monthly)

    if not out_frames:
        raise ValueError("After pivoting, no ticker had any non-null data.")

    out = pd.concat(out_frames, ignore_index=True)
    return out[["date", "ticker", *feature_cols]]


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    long = load_hist_long(data_dir)
    return to_wide_monthly(long)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    print(f"Wrote {len(panel):,} rows × {len(panel.columns)} cols → {out}")
    print(f"Tickers ({panel['ticker'].nunique()}):", sorted(panel["ticker"].unique()))
    print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
