"""Per-call delta vectors and days_since_earnings.

For each ticker, sort calls by ``event_date`` and compute the change in
the per-call embedding versus the previous call (the immediately
preceding earnings call for the same ticker, regardless of calendar
gap). Used as the input feature matrix for Strategy 2 (learned ridge
regression on Δ call vectors, spec section 4.2).

Per spec section 5.1, the first call per ticker is dropped (no prior to
difference against).

Input:
    _data/call_vectors.parquet

Output:
    _data/delta_vectors.parquet

Schema:
    transcript_id        int
    ticker               str  upper-case
    event_date           date
    prev_event_date      date
    days_since_earnings  int  calendar days since previous call
    delta_vector         list[float]  1536-D (current minus previous)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
INPUT_FILENAME = "call_vectors.parquet"
OUTPUT_FILENAME = "delta_vectors.parquet"


def load_call_vectors(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/build_call_vectors.py` first."
        )
    df = pd.read_parquet(path)
    df["embedding"] = df["embedding"].map(lambda x: np.asarray(x, dtype=np.float32))
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.sort_values(["ticker", "event_date"]).reset_index(drop=True)


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    calls = load_call_vectors(data_dir)

    out_rows: list[dict] = []
    for ticker, grp in calls.groupby("ticker", sort=True):
        grp = grp.sort_values("event_date").reset_index(drop=True)
        for i in range(1, len(grp)):
            prev = grp.iloc[i - 1]
            cur = grp.iloc[i]
            delta = cur["embedding"] - prev["embedding"]
            days_since = (cur["event_date"] - prev["event_date"]).days
            out_rows.append(
                {
                    "transcript_id": int(cur["transcript_id"]),
                    "ticker": ticker,
                    "event_date": cur["event_date"].date(),
                    "prev_event_date": prev["event_date"].date(),
                    "days_since_earnings": int(days_since),
                    "delta_vector": delta.tolist(),
                }
            )
    return pd.DataFrame(out_rows)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    print(f"Wrote {len(panel):,} delta vectors -> {out}")
    if not panel.empty:
        print(f"Tickers: {panel['ticker'].nunique()}")
        print(f"Date range: {panel['event_date'].min()} -> {panel['event_date'].max()}")
        print(f"days_since_earnings: median={int(panel['days_since_earnings'].median())}  "
              f"95th pct={int(panel['days_since_earnings'].quantile(0.95))}")


if __name__ == "__main__":
    main()
