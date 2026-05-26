"""Build the point-in-time Nasdaq-100 membership panel for v3.

Reads:
    data_manual/_meta/nasdaq100_historical_constituents.csv
        (output of src/pull_historical_nasdaq100.py)

Writes:
    data_manual/_meta/nasdaq100_pit_panel.parquet

Schema:
    date        BME (business-month-end) timestamp
    ticker      uppercase ticker
    in_universe  bool — was this ticker in Nasdaq-100 at this BME?

Construction:
- Compute a BME index spanning [BACKTEST_START, BACKTEST_END].
- For each (ticker, from_date, thru_date) interval in the constituent CSV,
  mark every BME in `[from_date, thru_date]` (or `>= from_date` if thru is
  null) as `in_universe = True` for that ticker.
- A ticker that has multiple intervals (e.g. joined, left, rejoined) gets
  all of them stitched into the panel.

Downstream consumers:
- `build_signal_panel.py` joins this onto the monthly signal panel so
  `backtest.py` can apply an `in_universe` filter at each rebalance.
- `embed_transcripts.py` uses it (joined with the per-call event_date
  panel) to determine which calls qualify for embedding.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calendar_utils import month_end_bd
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
META_DIR = MANUAL_DATA_DIR / "_meta"

INPUT_FILENAME = "nasdaq100_historical_constituents.csv"
OUTPUT_FILENAME = "nasdaq100_pit_panel.parquet"

BACKTEST_START = "2006-01-01"
BACKTEST_END = "2026-12-31"


def load_constituents(meta_dir: Path = META_DIR) -> pd.DataFrame:
    path = meta_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pull_historical_nasdaq100.py` first."
        )
    df = pd.read_csv(path, parse_dates=["from_date", "thru_date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def build(
    constituents: pd.DataFrame,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
) -> pd.DataFrame:
    rebalance_dates = month_end_bd(start, end)

    # For each interval, generate (date, ticker) rows for every BME in range.
    rows: list[dict] = []
    end_ts = pd.Timestamp(end)
    for r in constituents.itertuples(index=False):
        ticker = r.ticker
        from_ts = pd.Timestamp(r.from_date)
        thru_ts = pd.Timestamp(r.thru_date) if pd.notna(r.thru_date) else end_ts
        mask = (rebalance_dates >= from_ts) & (rebalance_dates <= thru_ts)
        for d in rebalance_dates[mask]:
            rows.append({"date": d, "ticker": ticker})

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "in_universe"])

    panel = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    panel["in_universe"] = True
    return panel


def write(panel: pd.DataFrame, meta_dir: Path = META_DIR) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / OUTPUT_FILENAME
    panel.to_parquet(out, index=False)
    return out


def main() -> None:
    constituents = load_constituents()
    panel = build(constituents)
    out = write(panel)

    print(f"Wrote {len(panel):,} (date, ticker) in-universe rows -> {out}")
    print(f"  unique tickers ever in panel: {panel['ticker'].nunique()}")
    print(f"  unique BME dates: {panel['date'].nunique()}")
    if len(panel):
        sizes = panel.groupby("date").size()
        q = sizes.quantile([0.05, 0.50, 0.95]).round(0).astype(int).to_dict()
        print(f"  tickers per BME — q05/q50/q95: {q[0.05]}/{q[0.50]}/{q[0.95]}")
        print(f"  min cross-section: {sizes.min()} on {sizes.idxmin().date()}")
        print(f"  max cross-section: {sizes.max()} on {sizes.idxmax().date()}")
        # Sanity: most recent BME should be close to 100 (Nasdaq-100 size).
        latest = sizes.index.max()
        print(f"  latest BME ({latest.date()}): {sizes.loc[latest]} tickers")


if __name__ == "__main__":
    main()
