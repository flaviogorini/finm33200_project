"""Build monthly sentiment features by carrying per-call sentiment forward.

Reads ``_data/sentiment_transcripts.parquet`` (one row per earnings call)
and emits a monthly panel where each row uses the *most recently announced*
earnings-call sentiment as of that month-end.

Output:
    _data/features_sentiment_monthly.parquet

Schema:
    date                    month-end Timestamp
    ticker                  str
    sentiment_pos           float — last available cosine to positive anchors
    sentiment_neg           float — last available cosine to negative anchors
    sentiment_diff          float — sentiment_pos - sentiment_neg
    sentiment_diff_qoq      float — sentiment_diff(t) - sentiment_diff(prev call)
    days_since_earnings     int   — days between event_date and month-end
    last_event_date         date  — anchor date for the active sentiment

Activation rule: at month-end ``t``, only earnings calls with
``event_date <= t`` are visible. Calls announced after ``t`` are masked. This
guarantees no lookahead bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "sentiment_transcripts.parquet"
OUTPUT_FILENAME = "features_sentiment_monthly.parquet"


def load_sentiment(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/score_transcript_sentiment.py [--synthetic]` first."
        )
    df = pd.read_parquet(path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df.sort_values(["ticker", "event_date"]).reset_index(drop=True)


def _build_monthly_for_ticker(calls: pd.DataFrame) -> pd.DataFrame:
    """Carry per-call sentiment forward to a monthly index for one ticker."""
    calls = calls.sort_values("event_date").reset_index(drop=True).copy()
    calls["sentiment_diff_qoq"] = calls["sentiment_diff"].diff()

    first = calls["event_date"].min()
    # Cover from the first month with sentiment to today (or last call + 1Y, whichever larger).
    end = max(pd.Timestamp.today().normalize(), calls["event_date"].max() + pd.DateOffset(months=12))
    months = pd.date_range(
        start=(first - pd.offsets.MonthEnd(0)), end=end, freq="ME"
    )

    # For each month-end, find the last call with event_date <= month_end.
    event_dates = calls["event_date"].to_numpy()
    idx = np.searchsorted(event_dates, months.to_numpy(), side="right") - 1
    valid = idx >= 0

    out = pd.DataFrame({"date": months})
    out["ticker"] = calls["ticker"].iloc[0]

    out["sentiment_pos"] = np.where(valid, calls["sentiment_pos"].to_numpy()[np.clip(idx, 0, None)], np.nan)
    out["sentiment_neg"] = np.where(valid, calls["sentiment_neg"].to_numpy()[np.clip(idx, 0, None)], np.nan)
    out["sentiment_diff"] = np.where(valid, calls["sentiment_diff"].to_numpy()[np.clip(idx, 0, None)], np.nan)
    out["sentiment_diff_qoq"] = np.where(
        valid, calls["sentiment_diff_qoq"].to_numpy()[np.clip(idx, 0, None)], np.nan
    )
    last_event = np.where(
        valid, calls["event_date"].to_numpy()[np.clip(idx, 0, None)], np.datetime64("NaT")
    )
    out["last_event_date"] = pd.to_datetime(last_event)
    out["days_since_earnings"] = (
        (out["date"] - out["last_event_date"]).dt.days.astype("Int64")
    )

    # Drop the lead-in months before any earnings call existed.
    return out.dropna(subset=["sentiment_diff"]).reset_index(drop=True)


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    calls = load_sentiment(data_dir)
    frames = [
        _build_monthly_for_ticker(grp) for _, grp in calls.groupby("ticker", sort=True)
    ]
    return pd.concat(frames, ignore_index=True)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    print(f"Wrote {len(panel):,} monthly sentiment rows → {out}")
    print(f"Tickers ({panel['ticker'].nunique()}):", sorted(panel["ticker"].unique()))
    print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
