"""
1-step Chronos2 forecasts of weekly and monthly simple returns for a single
US stock loaded from `_data/US_Companies_Hist_Data.parquet`.

For each (frequency, scheme) and each test target_date in
``forecast_utils.iter_windows``, the pipeline is conditioned on the
training window's returns and asked to predict ``prediction_length=1``.

Output (long-form) → DATA_DIR / {TICKER}_Chronos_Forecasts.parquet:
    date, freq, scheme, forecast, q10, q90, actual, window_years
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

from forecast_utils import ROLLING_WINDOW_YEARS, TEST_START, iter_windows
from settings import config
from stock_returns import compute_stock_returns

DATA_DIR = Path(config("DATA_DIR"))

DEFAULT_FREQS = ("W", "M")
SCHEMES = ("expanding", "rolling")
MODEL_ID = "amazon/chronos-2"
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def _build_context_df(train: pd.Series, series_id: str) -> pd.DataFrame:
    """Shape a train series into the DataFrame Chronos2 expects."""
    return pd.DataFrame(
        {
            "series": series_id,
            "timestamp": train.index,
            "target": np.asarray(train, dtype=float),
        }
    )


def run_chronos_forecasts(
    pipeline: Chronos2Pipeline,
    ticker: str,
    freqs: tuple[str, ...] = DEFAULT_FREQS,
    test_start: str = TEST_START,
    window_years: int = ROLLING_WINDOW_YEARS,
    log_every: int = 50,
) -> pd.DataFrame:
    """Run all (freq × scheme) Chronos forecasts for `ticker`; return long DataFrame."""
    series_id = f"{ticker.lower()}_ret"
    rows: list[dict] = []
    for freq in freqs:
        returns = compute_stock_returns(ticker, freq=freq)
        for scheme in SCHEMES:
            t0 = time.perf_counter()
            n = 0
            for train, target_date, actual in iter_windows(
                returns,
                scheme=scheme,
                window_years=window_years,
                test_start=test_start,
            ):
                ctx = _build_context_df(train, series_id)
                pred = pipeline.predict_df(
                    ctx,
                    prediction_length=1,
                    quantile_levels=QUANTILE_LEVELS,
                    id_column="series",
                    timestamp_column="timestamp",
                    target="target",
                )
                row = pred.iloc[0]
                rows.append(
                    {
                        "date": target_date,
                        "freq": freq,
                        "scheme": scheme,
                        "forecast": float(row["predictions"]),
                        "q10": float(row["0.1"]),
                        "q90": float(row["0.9"]),
                        "actual": actual,
                        "window_years": (
                            window_years if scheme == "rolling" else np.nan
                        ),
                    }
                )
                n += 1
                if log_every and n % log_every == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[stock_chronos] {ticker} {freq}/{scheme}: "
                        f"{n} forecasts in {elapsed:.1f}s "
                        f"({elapsed / n:.2f}s/forecast)",
                        file=sys.stderr,
                        flush=True,
                    )
            elapsed = time.perf_counter() - t0
            print(
                f"[stock_chronos] done {ticker} {freq}/{scheme}: "
                f"{n} forecasts in {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    return pd.DataFrame(rows)


def load_chronos_forecasts(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{ticker}_Chronos_Forecasts.parquet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default="AAPL", help="ticker symbol (default: AAPL)")
    p.add_argument(
        "--freqs",
        nargs="+",
        default=list(DEFAULT_FREQS),
        choices=["W", "M"],
        help="frequencies to run (default: W M)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")
    df = run_chronos_forecasts(pipeline, args.ticker, freqs=tuple(args.freqs))
    out = DATA_DIR / f"{args.ticker}_Chronos_Forecasts.parquet"
    df.to_parquet(out)
    print(f"wrote {len(df):,} rows → {out}")
