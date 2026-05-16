"""
Per-ticker 1-step monthly forecasts of US-company returns.

Two methods, both run by default (use ``--only=regression`` or
``--only=chronos`` to run one of them):

1. **Factor regression** — per ``(ticker, scheme, target_date)``, fit
   ``r_{s+1} = α + β_pe·F_pe_s + β_marg·F_marg_s + β_sales·F_sales_s + ε``
   on the training window, then predict using factor values observed at
   the end of month ``t`` (i.e., the lag-0 factor changes known at
   decision time).

   Also emits a ``mean`` baseline per ticker (mean of training returns).

   Output → ``_data/US_Regression_Forecasts.parquet`` with columns:
       ticker, date, scheme, model, forecast, actual

   where ``model ∈ {"factor", "mean"}``.

2. **Chronos2 with past covariates** — per ``(scheme, target_date)``,
   condition the model on each ticker's own past returns plus the three
   analyst-factor changes as **past covariates** on matching timestamps
   (no lag). No ``future_df`` is supplied — we are not providing any
   future covariate information; Chronos uses only the joint history
   ``(ret, F_pe, F_marg, F_sales)`` per ticker to forecast the next
   return. All 13 tickers passed in a single ``predict_df`` call with
   ``cross_learning=False`` so each ticker is forecasted independently
   (no cross-ticker information).

   Output → ``_data/US_Chronos_Forecasts.parquet`` with columns:
       ticker, date, scheme, forecast, q10, q90, actual

Both pipelines reuse the existing iteration helper in
``forecast_utils.iter_windows`` for their (expanding | rolling) schemes,
with the start date and rolling window controlled by
``US_TEST_START`` and ``US_ROLLING_WINDOW_YEARS`` from
``us_company_factors``.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from forecast_utils import iter_windows
from settings import config
from us_company_factors import (
    FACTOR_COLS,
    US_ROLLING_WINDOW_YEARS,
    US_TEST_START,
    load_us_panel,
)

DATA_DIR = Path(config("DATA_DIR"))

REG_PARQUET = "US_Regression_Forecasts.parquet"
CHRONOS_PARQUET = "US_Chronos_Forecasts.parquet"

SCHEMES = ("expanding", "rolling")
CHRONOS_MODEL_ID = "amazon/chronos-2"
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# Per-ticker regression
# ---------------------------------------------------------------------------


def _per_ticker_pred_frame(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build a (date-indexed) DataFrame for one ticker with the training
    rows that the regression sees: realized return ``ret`` at date `s+1`
    plus the factor values from date `s` (``F_*_lag1``).

    Concretely: we want to regress ``r_{s+1}`` on ``F_s``. Shifting factors
    forward by one month means at row ``s+1`` we have ``r_{s+1}`` *and*
    ``F_pe_lag1 = F_pe_s``. The very first month per ticker drops out (its
    factor lag is NaN).
    """
    df = (
        panel[panel["ticker"] == ticker]
        .sort_values("date")
        .set_index("date")
    )
    out = pd.DataFrame(
        {
            "ret": df["ret"],
            "F_pe_lag1": df["F_pe"].shift(1),
            "F_marg_lag1": df["F_marg"].shift(1),
            "F_sales_lag1": df["F_sales"].shift(1),
            "F_pe_now": df["F_pe"],
            "F_marg_now": df["F_marg"],
            "F_sales_now": df["F_sales"],
        }
    )
    return out.dropna(subset=["F_pe_lag1", "F_marg_lag1", "F_sales_lag1"])


def run_regression_forecasts(
    panel: pd.DataFrame,
    test_start: str = US_TEST_START,
    window_years: int = US_ROLLING_WINDOW_YEARS,
) -> pd.DataFrame:
    """Per-ticker rolling/expanding 3-factor regressions + per-ticker mean."""
    rows: list[dict] = []
    factor_lag_cols = ["F_pe_lag1", "F_marg_lag1", "F_sales_lag1"]
    factor_now_cols = ["F_pe_now", "F_marg_now", "F_sales_now"]
    for ticker in sorted(panel["ticker"].unique()):
        td = _per_ticker_pred_frame(panel, ticker)
        ret_series = td["ret"]
        for scheme in SCHEMES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train_ret, target_date, actual in iter_windows(
                    ret_series.reset_index().rename(columns={"ret": "ret"})
                    .rename(columns={"date": "date", "ret": "ret"}),
                    scheme=scheme,
                    window_years=window_years,
                    test_start=test_start,
                ):
                    train_dates = train_ret.index
                    X_train = td.loc[train_dates, factor_lag_cols].to_numpy(
                        dtype=float
                    )
                    y_train = train_ret.to_numpy(dtype=float)
                    # The factor values at decision time (= end of month t)
                    # are the *current-month* factors at target_date, since
                    # F_pe_lag1 at target_date corresponds to F_pe_{t-1}.
                    # We want F_t — that's stored in F_*_now at the row
                    # immediately BEFORE target_date in `td`.
                    prev_row = td.loc[td.index < target_date].iloc[-1]
                    x_pred = prev_row[factor_now_cols].to_numpy(dtype=float)

                    # Factor regression
                    X_train_c = sm.add_constant(X_train, has_constant="add")
                    try:
                        beta = sm.OLS(y_train, X_train_c).fit().params
                        forecast_factor = float(
                            beta[0]
                            + beta[1] * x_pred[0]
                            + beta[2] * x_pred[1]
                            + beta[3] * x_pred[2]
                        )
                    except Exception as exc:
                        print(
                            f"[us_company_forecasts] OLS failed for "
                            f"{ticker}/{scheme}/{target_date}: {exc}",
                            file=sys.stderr,
                        )
                        forecast_factor = float(np.mean(y_train))

                    rows.append(
                        {
                            "ticker": ticker,
                            "date": target_date,
                            "scheme": scheme,
                            "model": "factor",
                            "forecast": forecast_factor,
                            "actual": actual,
                        }
                    )
                    rows.append(
                        {
                            "ticker": ticker,
                            "date": target_date,
                            "scheme": scheme,
                            "model": "mean",
                            "forecast": float(np.mean(y_train)),
                            "actual": actual,
                        }
                    )
    return pd.DataFrame(rows)


def load_us_regression_forecasts(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / REG_PARQUET)


# ---------------------------------------------------------------------------
# Per-ticker Chronos2 with covariates
# ---------------------------------------------------------------------------


def _train_slice(
    panel: pd.DataFrame,
    target_date: pd.Timestamp,
    scheme: str,
    window_years: int,
) -> pd.DataFrame:
    """Return the slice of `panel` used as context for forecasting
    `target_date`. Includes all tickers' rows strictly before target_date,
    bounded below by the rolling window if scheme=='rolling'.
    """
    cutoff = target_date
    df = panel[panel["date"] < cutoff]
    if scheme == "rolling":
        lower = cutoff - pd.DateOffset(years=window_years)
        df = df[df["date"] >= lower]
    return df


def run_chronos_forecasts(
    panel: pd.DataFrame,
    test_start: str = US_TEST_START,
    window_years: int = US_ROLLING_WINDOW_YEARS,
    log_every: int = 25,
) -> pd.DataFrame:
    """Per-ticker monthly Chronos2 forecasts. The 3 analyst-factor changes
    are fed as **past covariates** alongside each ticker's return history,
    aligned on the same timestamps (no lag, no future_df). Chronos sees the
    joint past dynamics of (ret, F_pe, F_marg, F_sales) per ticker and
    forecasts the next return.

    One predict_df call per (scheme, target_date), batched over all tickers
    with cross_learning=False (so each ticker is independent).
    """
    from chronos import Chronos2Pipeline  # import here to keep dependency

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID, device_map="cpu"
    )

    test_start_ts = pd.Timestamp(test_start)
    target_dates = sorted(
        panel.loc[panel["date"] >= test_start_ts, "date"].unique()
    )

    actual_lookup = panel.set_index(["ticker", "date"])["ret"].to_dict()

    rows: list[dict] = []
    for scheme in SCHEMES:
        t0 = time.perf_counter()
        n = 0
        for target_date in target_dates:
            train_slice = _train_slice(
                panel, target_date, scheme=scheme, window_years=window_years
            )
            if train_slice.empty:
                continue
            # Keep only tickers with at least 10 train obs to keep
            # Chronos happy.
            counts = train_slice.groupby("ticker").size()
            keep_tickers = counts[counts >= 10].index.tolist()
            if not keep_tickers:
                continue
            train_slice = train_slice[train_slice["ticker"].isin(keep_tickers)]

            ctx_df = train_slice.rename(columns={"date": "timestamp"})[
                [
                    "ticker",
                    "timestamp",
                    "ret",
                    "F_pe",
                    "F_marg",
                    "F_sales",
                ]
            ]

            try:
                pred = pipeline.predict_df(
                    ctx_df,
                    prediction_length=1,
                    quantile_levels=QUANTILE_LEVELS,
                    id_column="ticker",
                    timestamp_column="timestamp",
                    target="ret",
                    cross_learning=False,
                )
            except Exception as exc:
                print(
                    f"[us_company_forecasts] Chronos failed for "
                    f"{scheme}/{target_date}: {exc}",
                    file=sys.stderr,
                )
                continue

            for _, r in pred.iterrows():
                tk = r["ticker"]
                rows.append(
                    {
                        "ticker": tk,
                        "date": pd.Timestamp(r["timestamp"]),
                        "scheme": scheme,
                        "forecast": float(r["predictions"]),
                        "q10": float(r["0.1"]),
                        "q90": float(r["0.9"]),
                        "actual": float(
                            actual_lookup.get((tk, pd.Timestamp(r["timestamp"])), np.nan)
                        ),
                    }
                )

            n += 1
            if log_every and n % log_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[us_company_forecasts] {scheme}: {n} target dates "
                    f"in {elapsed:.1f}s ({elapsed / n:.2f}s/date)",
                    file=sys.stderr,
                    flush=True,
                )
        elapsed = time.perf_counter() - t0
        print(
            f"[us_company_forecasts] done {scheme}: {n} target dates "
            f"in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["actual"])  # drop tickers that lack a real return on the target date
    return out


def load_us_chronos_forecasts(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / CHRONOS_PARQUET)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("regression", "chronos"),
        default=None,
        help="Run only one of the two forecast pipelines (default: both).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    panel = load_us_panel()

    if args.only in (None, "regression"):
        reg = run_regression_forecasts(panel)
        path = DATA_DIR / REG_PARQUET
        reg.to_parquet(path, index=False)
        print(f"wrote {len(reg):,} rows → {path}")

    if args.only in (None, "chronos"):
        ch = run_chronos_forecasts(panel)
        path = DATA_DIR / CHRONOS_PARQUET
        ch.to_parquet(path, index=False)
        print(f"wrote {len(ch):,} rows → {path}")
