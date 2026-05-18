"""Build the monthly analyst-revisions panel from Bloomberg BEst Net Income.

Reads ``_data/US_Companies_Forecast.parquet`` (long format, the output of
``pull_manual_companies.py``), filters to ``BEST_NET_INCOME``, snaps to
the business-month-end rebalance calendar, and writes the
21-trading-day change in consensus FY1 net income normalised by the
absolute value of the prior estimate.

The spec (section 4.5) phrases the lookback as "30 days". This project
operationalises that as **21 trading days** (one trading month) so the
revision horizon matches every other monthly quantity in the pipeline —
in particular, the forward-return horizon. The column name
``rev_30d`` keeps the spec wording but the value is the 21-bday delta.

Output:
    _data/revisions_monthly.parquet

Schema:
    date     last business day of the calendar month
    ticker   upper-case, ' US Equity' suffix stripped
    rev_30d  (NI_t - NI_{t-21bd}) / |NI_{t-21bd}|, NaN if denom=0 or missing
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

from calendar_utils import REVISION_LOOKBACK_BDAYS, month_end_bd
from build_returns_monthly import _strip_bbg_suffix
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "US_Companies_Forecast.parquet"
OUTPUT_FILENAME = "revisions_monthly.parquet"
BBG_FIELD = "BEST_NET_INCOME"
FFILL_LIMIT_BDAYS = 5


def load_best_ni(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read BEST_NET_INCOME from the forecast parquet, long format."""
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pull_manual_companies.py` first."
        )
    df = pd.read_parquet(path)
    df = df[df["field"] == BBG_FIELD].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].map(_strip_bbg_suffix).astype(str).str.upper()
    df = df.rename(columns={"value": "best_ni"})
    return df[["date", "ticker", "best_ni"]].dropna(subset=["best_ni"]).reset_index(drop=True)


def _ticker_panel(rows: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute the 21-bday revision panel for one ticker."""
    series = rows.set_index("date")["best_ni"].sort_index()
    series = series[~series.index.duplicated(keep="last")]

    bdays = pd.bdate_range(series.index.min(), series.index.max())
    daily = series.reindex(bdays).ffill(limit=FFILL_LIMIT_BDAYS)

    current = daily.reindex(rebalance_dates, method="ffill")
    prior_dates = rebalance_dates - REVISION_LOOKBACK_BDAYS * BDay()
    prior = daily.reindex(prior_dates, method="ffill")
    prior.index = rebalance_dates

    denom = prior.abs()
    rev = (current - prior) / denom
    rev = rev.where(denom > 0, other=float("nan"))

    out = pd.DataFrame({"date": rebalance_dates, "rev_30d": rev.to_numpy()})
    out["ticker"] = rows["ticker"].iloc[0]
    return out[["date", "ticker", "rev_30d"]]


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    rows = load_best_ni(data_dir)
    start = rows["date"].min()
    end = rows["date"].max()
    rebalance_dates = month_end_bd(start, end)

    frames = [
        _ticker_panel(grp, rebalance_dates)
        for _, grp in rows.groupby("ticker", sort=True)
    ]
    panel = pd.concat(frames, ignore_index=True)
    return panel.dropna(subset=["rev_30d"]).reset_index(drop=True)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    print(f"Wrote {len(panel):,} rows -> {out}")
    print(f"Tickers ({panel['ticker'].nunique()})")
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
