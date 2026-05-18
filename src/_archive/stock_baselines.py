"""
Baseline 1-step forecasts of weekly and monthly simple returns for a single
US stock loaded from `_data/US_Companies_Hist_Data.parquet`.

Four baselines per (frequency, scheme):
    - mean : forecast = train.mean()
    - ar1  : statsmodels AutoReg(p=1)
    - arima: statsmodels ARIMA(1,0,1)
    - zero : forecast = 0.0

Schemes per (frequency, baseline):
    - expanding : train = all returns strictly before target_date
    - rolling   : train = last ROLLING_WINDOW_YEARS years before target_date

Output (long-form) → DATA_DIR / {TICKER}_Baseline_Forecasts.parquet:
    date, freq, scheme, model, forecast, actual, window_years
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from forecast_utils import ROLLING_WINDOW_YEARS, TEST_START, iter_windows
from settings import config
from stock_returns import compute_stock_returns

DATA_DIR = Path(config("DATA_DIR"))

DEFAULT_FREQS = ("W", "M")
SCHEMES = ("expanding", "rolling")
MODELS = ("mean", "ar1", "arima", "zero")


def _forecast_mean(train: pd.Series) -> float:
    return float(train.mean())


def _forecast_zero(train: pd.Series) -> float:
    return 0.0


def _forecast_ar1(train: pd.Series) -> float:
    arr = np.asarray(train, dtype=float)
    model = AutoReg(arr, lags=1, old_names=False).fit()
    return float(model.forecast(steps=1)[-1])


def _forecast_arima(train: pd.Series) -> float:
    arr = np.asarray(train, dtype=float)
    try:
        model = ARIMA(arr, order=(1, 0, 1)).fit()
        return float(model.forecast(steps=1)[-1])
    except Exception as exc:
        print(
            f"[stock_baselines] ARIMA fit failed ({exc.__class__.__name__}); "
            f"falling back to mean for this window",
            file=sys.stderr,
        )
        return _forecast_mean(train)


_FORECASTERS = {
    "mean": _forecast_mean,
    "ar1": _forecast_ar1,
    "arima": _forecast_arima,
    "zero": _forecast_zero,
}


def run_baseline_forecasts(
    ticker: str,
    freqs: tuple[str, ...] = DEFAULT_FREQS,
    test_start: str = TEST_START,
    window_years: int = ROLLING_WINDOW_YEARS,
) -> pd.DataFrame:
    """Run all (model × freq × scheme) baselines for `ticker`; return long DataFrame."""
    rows: list[dict] = []
    for freq in freqs:
        returns = compute_stock_returns(ticker, freq=freq)
        for scheme in SCHEMES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train, target_date, actual in iter_windows(
                    returns,
                    scheme=scheme,
                    window_years=window_years,
                    test_start=test_start,
                ):
                    for model_name in MODELS:
                        forecast = _FORECASTERS[model_name](train)
                        rows.append(
                            {
                                "date": target_date,
                                "freq": freq,
                                "scheme": scheme,
                                "model": model_name,
                                "forecast": forecast,
                                "actual": actual,
                                "window_years": (
                                    window_years if scheme == "rolling" else np.nan
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def load_baseline_forecasts(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{ticker}_Baseline_Forecasts.parquet")


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
    df = run_baseline_forecasts(args.ticker, freqs=tuple(args.freqs))
    out = DATA_DIR / f"{args.ticker}_Baseline_Forecasts.parquet"
    df.to_parquet(out)
    print(f"wrote {len(df):,} rows → {out}")
