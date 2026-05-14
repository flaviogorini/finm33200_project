"""Build monthly analyst-consensus features (Bloomberg ``BEST_*``).

Consumes the long-format parquet produced by ``pull_manual_companies.py``:

    _data/US_Companies_Forecast.parquet   (cols: date, ticker, field, value)

Bloomberg's ``BEST_*`` series are the IBES-equivalent: at any date ``t``, the
value reflects the consensus estimate as of that date. Daily frequency.
We resample to month-end (last value in month) — that's the consensus an
investor would have seen at the close of month ``t``.

Output:
    _data/features_consensus_monthly.parquet

Schema:
    date              month-end Timestamp
    ticker            str  (' US Equity' suffix stripped)
    best_pe_ratio     float  consensus forward P/E
    best_sales        float  consensus forward revenue
    best_net_income   float  consensus forward net income
    best_net_debt     float  consensus forward net debt
    best_ebitda       float  consensus forward EBITDA

These columns are the comparison targets when evaluating Chronos-2 forecasts:
``best_sales`` ↔ Chronos forecast of revenue, ``best_net_income`` ↔ forecast of
net income, etc.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_fundamentals_features import _strip_bbg_suffix
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "US_Companies_Forecast.parquet"
OUTPUT_FILENAME = "features_consensus_monthly.parquet"

FIELD_RENAME = {
    "BEST_PE_RATIO": "best_pe_ratio",
    "BEST_SALES": "best_sales",
    "BEST_NET_INCOME": "best_net_income",
    "BEST_NET_DEBT": "best_net_debt",
    "BEST_EBITDA": "best_ebitda",
}


def load_forecast_long(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the long-format consensus-forecast parquet."""
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

    feature_cols = list(FIELD_RENAME.values())
    out_frames: list[pd.DataFrame] = []
    for tk, grp in wide.groupby("ticker", sort=True):
        ts = grp.set_index("date").sort_index()
        monthly = ts[feature_cols].resample("ME").last()
        monthly["ticker"] = tk
        monthly = monthly.reset_index()
        monthly = monthly.dropna(subset=feature_cols, how="all")
        out_frames.append(monthly)

    if not out_frames:
        raise ValueError("After pivoting, no ticker had any non-null data.")

    out = pd.concat(out_frames, ignore_index=True)
    return out[["date", "ticker", *feature_cols]]


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    long = load_forecast_long(data_dir)
    return to_wide_monthly(long)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    print(f"Wrote {len(panel):,} rows x {len(panel.columns)} cols -> {out}")
    print(f"Tickers ({panel['ticker'].nunique()}):", sorted(panel["ticker"].unique()))
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
