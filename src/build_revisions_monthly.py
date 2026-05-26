"""Build the monthly analyst-revisions panel from Bloomberg BEst Net Income.

v3: calendar-month convention. Reads ``_data/US_Companies_Forecast.parquet``,
filters to ``BEST_NET_INCOME``, snaps to BME, and writes the one-calendar-month
percentage change in consensus blended-forward net income normalised by the
absolute value of the prior estimate.

v3 change vs v2: lookback is now 1 calendar month (previous BME) instead of
21 trading days. Column renamed ``rev_30d`` → ``rev_1m`` to make the
convention obvious. This aligns the revision horizon with the project's
single forward-return horizon (``fwd_ret_1m``) and with Ken French's
calendar-month FF5 returns.

Output:
    _data/revisions_monthly.parquet

Schema:
    date     last business day of the calendar month (BME)
    ticker   upper-case, ' US Equity' suffix stripped
    rev_1m   (NI_t - NI_{t-1m}) / |NI_{t-1m}|, NaN if denom=0 or missing
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_returns_monthly import _strip_bbg_suffix
from calendar_utils import month_end_bd
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "US_Companies_Forecast.parquet"
OUTPUT_FILENAME = "revisions_monthly.parquet"
BBG_FIELD = "BEST_NET_INCOME"
# Calendar-day forward-fill cap for sparse Bloomberg observations.
FFILL_LIMIT_CDAYS = 7


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
    """Compute the 1-calendar-month revision panel for one ticker.

    Lookup is asof (most-recent observation at-or-before the BME) with a
    calendar-day ffill cap. Prior value at BME(T) is the asof lookup at
    BME(T-1) — i.e. the previous BME in the rebalance index.
    """
    series = rows.set_index("date")["best_ni"].sort_index()
    series = series[~series.index.duplicated(keep="last")]

    # Daily-ffill on calendar days so asof lookups respect the FFILL cap.
    cdays = pd.date_range(series.index.min(), series.index.max(), freq="D")
    daily = series.reindex(cdays).ffill(limit=FFILL_LIMIT_CDAYS)

    current = daily.reindex(rebalance_dates, method="ffill")
    # Prior = previous BME. Shift by one position in the rebalance index.
    prior = current.shift(1)

    denom = prior.abs()
    rev = (current - prior) / denom
    rev = rev.where(denom > 0, other=float("nan"))

    out = pd.DataFrame({"date": rebalance_dates, "rev_1m": rev.to_numpy()})
    out["ticker"] = rows["ticker"].iloc[0]
    return out[["date", "ticker", "rev_1m"]]


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
    return panel.dropna(subset=["rev_1m"]).reset_index(drop=True)


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
