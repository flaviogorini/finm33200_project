"""Assemble the unified monthly signal panel for backtests + joint regression.

Joins, on (date, ticker), the canonical returns panel with whatever
signal panels are available on disk. The canonical date x ticker grid
comes from ``returns_monthly.parquet`` (the project's only forward-return
source). Signal panels are left-joined so the grid is preserved even if
a strategy's panel is missing or partial.

Inputs (all optional except returns):
    _data/returns_monthly.parquet          (REQUIRED) -> fwd_ret_21d, px_eom
    _data/features_sentiment_monthly.parquet -> sig_anchor (sentiment_diff_qoq),
                                                days_since_earnings
    _data/lm_scores_transcripts.parquet    -> sig_lm (lm_delta, carried fwd)
    _data/momentum_monthly.parquet         -> sig_mom (mom_12_1)
    _data/revisions_monthly.parquet        -> sig_rev (rev_30d)
    _data/ridge_predictions.parquet        -> sig_ridge (carried fwd from event_date)

Output:
    _data/signal_panel_monthly.parquet

Schema:
    date, ticker,
    fwd_ret_21d, px_eom,
    sig_anchor, sig_lm, sig_mom, sig_rev, sig_ridge,
    days_since_earnings   (calendar days since most-recent earnings call;
                           only place calendar days appear in the project)

Missing signals are kept as NaN columns so downstream code can detect
them via ``panel[col].notna().any()``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

OUTPUT_FILENAME = "signal_panel_monthly.parquet"


def _load_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [missing] {path.name} — column will be NaN in panel")
        return None
    df = pd.read_parquet(path)
    print(f"  [loaded ] {path.name}: {len(df):,} rows")
    return df


def _carry_forward_per_call(
    per_call: pd.DataFrame,
    value_col: str,
    rebalance_dates: pd.DatetimeIndex,
    date_col: str = "event_date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Per ticker, at each rebalance date use the most recent value with
    ``date_col <= rebalance_date``. Returns long [date, ticker, value_col,
    days_since_earnings]."""
    out_rows: list[pd.DataFrame] = []
    rebal = np.sort(rebalance_dates.to_numpy())
    for ticker, grp in per_call.groupby(ticker_col, sort=True):
        grp = grp.sort_values(date_col)
        ev = pd.to_datetime(grp[date_col]).to_numpy()
        idx = np.searchsorted(ev, rebal, side="right") - 1
        valid = idx >= 0
        vals = np.where(valid, grp[value_col].to_numpy()[np.clip(idx, 0, None)], np.nan)
        last_ev = np.where(valid, ev[np.clip(idx, 0, None)], np.datetime64("NaT"))
        days_since = (rebal - last_ev) / np.timedelta64(1, "D")
        out_rows.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(rebal),
                    "ticker": ticker,
                    value_col: vals,
                    "days_since_earnings": days_since,
                }
            )
        )
    if not out_rows:
        return pd.DataFrame(columns=["date", "ticker", value_col, "days_since_earnings"])
    return pd.concat(out_rows, ignore_index=True)


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    returns_path = data_dir / "returns_monthly.parquet"
    if not returns_path.exists():
        raise FileNotFoundError(
            f"{returns_path} not found. Run `python src/build_returns_monthly.py` first."
        )
    panel = pd.read_parquet(returns_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    rebalance_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    sentiment = _load_optional(data_dir / "features_sentiment_monthly.parquet")
    if sentiment is not None:
        sentiment = sentiment.rename(columns={"sentiment_diff_qoq": "sig_anchor"})
        sentiment["date"] = pd.to_datetime(sentiment["date"])
        sentiment["ticker"] = sentiment["ticker"].astype(str).str.upper()
        keep = ["date", "ticker", "sig_anchor", "days_since_earnings"]
        panel = panel.merge(sentiment[keep], on=["date", "ticker"], how="left")
    else:
        panel["sig_anchor"] = np.nan
        panel["days_since_earnings"] = np.nan

    lm_per_call = _load_optional(data_dir / "lm_scores_transcripts.parquet")
    if lm_per_call is not None and not lm_per_call.empty:
        lm_per_call = lm_per_call.copy()
        lm_per_call["ticker"] = lm_per_call["ticker"].astype(str).str.upper()
        lm_per_call["event_date"] = pd.to_datetime(lm_per_call["event_date"])
        lm_monthly = _carry_forward_per_call(
            lm_per_call.dropna(subset=["lm_delta"]),
            value_col="lm_delta",
            rebalance_dates=rebalance_dates,
        )
        lm_monthly = lm_monthly.rename(columns={"lm_delta": "sig_lm"}).drop(
            columns="days_since_earnings"
        )
        panel = panel.merge(lm_monthly, on=["date", "ticker"], how="left")
    else:
        panel["sig_lm"] = np.nan

    momentum = _load_optional(data_dir / "momentum_monthly.parquet")
    if momentum is not None:
        momentum = momentum.rename(columns={"mom_12_1": "sig_mom"})
        momentum["date"] = pd.to_datetime(momentum["date"])
        momentum["ticker"] = momentum["ticker"].astype(str).str.upper()
        panel = panel.merge(momentum[["date", "ticker", "sig_mom"]], on=["date", "ticker"], how="left")
    else:
        panel["sig_mom"] = np.nan

    revisions = _load_optional(data_dir / "revisions_monthly.parquet")
    if revisions is not None:
        revisions = revisions.rename(columns={"rev_30d": "sig_rev"})
        revisions["date"] = pd.to_datetime(revisions["date"])
        revisions["ticker"] = revisions["ticker"].astype(str).str.upper()
        panel = panel.merge(revisions[["date", "ticker", "sig_rev"]], on=["date", "ticker"], how="left")
    else:
        panel["sig_rev"] = np.nan

    ridge = _load_optional(data_dir / "ridge_predictions.parquet")
    if ridge is not None and not ridge.empty:
        ridge = ridge.copy()
        ridge["ticker"] = ridge["ticker"].astype(str).str.upper()
        ridge["event_date"] = pd.to_datetime(ridge["event_date"])
        ridge_monthly = _carry_forward_per_call(
            ridge.dropna(subset=["y_pred"]),
            value_col="y_pred",
            rebalance_dates=rebalance_dates,
        ).rename(columns={"y_pred": "sig_ridge"}).drop(columns="days_since_earnings")
        panel = panel.merge(ridge_monthly, on=["date", "ticker"], how="left")
    else:
        panel["sig_ridge"] = np.nan

    ordered = [
        "date", "ticker", "px_eom", "fwd_ret_21d",
        "sig_anchor", "sig_ridge", "sig_lm", "sig_mom", "sig_rev",
        "days_since_earnings",
    ]
    return panel[[c for c in ordered if c in panel.columns]].sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    n = len(panel)
    print(f"\nWrote {n:,} rows -> {out}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    for col in ["sig_anchor", "sig_ridge", "sig_lm", "sig_mom", "sig_rev", "fwd_ret_21d"]:
        if col in panel.columns:
            print(f"  {col:>12s} non-null: {panel[col].notna().sum():>6,d} / {n:,}")


if __name__ == "__main__":
    main()
