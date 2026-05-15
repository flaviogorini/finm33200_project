"""Chronos-2 fundamentals forecast backtest vs Bloomberg consensus + naive YoY.

For each (ticker, as_of) in a small backtest grid, this:
1. Generates a 4-quarter Chronos forecast for revenue and net_income using
   only data with quarter-end <= as_of (point-in-time guard, enforced inside
   :func:`forecast_chronos2.forecast_for_ticker`).
2. Computes a naive YoY baseline: the realized value 4 quarters prior to the
   target quarter (always strictly earlier than as_of, so PIT-safe).
3. Attaches Bloomberg consensus (BEST_*) at as_of as a reference. NOTE: the
   panel's ``best_*`` columns are next-fiscal-year analyst estimates, not
   per-quarter point forecasts. We attach the same as_of consensus to every
   horizon row — the per-horizon Chronos-vs-consensus comparison is therefore
   a rough apples-to-oranges and is reported with that caveat in the writeup.
   The apples-to-apples comparison is Chronos-vs-naive_yoy.
4. Looks up the realized value at the target quarter end if present in the panel.
5. Writes ``_output/chronos2_backtest.parquet`` and ``chronos2_backtest_summary.json``.

Grid (matches the digest grid in :mod:`generate_digest`):
    5 tickers x 4 as_of dates x 2 targets x 4 horizons = 160 rows.

Usage:
    python src/backtest_chronos2.py                          # full grid
    python src/backtest_chronos2.py --tickers AAPL --as-of 2024-09-30   # one cell
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import load_panel
from forecast_chronos2 import (
    DEFAULT_TARGETS,
    _load_chronos_pipeline,
    forecast,
    quarterly_history,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

BACKTEST_TICKERS = ["AAPL", "MSFT", "JPM", "KO", "NVDA"]
BACKTEST_AS_OF_DATES = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
HORIZON_Q = 4

CONSENSUS_FOR = {
    "revenue": "best_sales",
    "net_income": "best_net_income",
    "ebitda": "best_ebitda",
}


def lookup_realized(history: pd.Series, quarter_end: pd.Timestamp) -> float:
    """Return the realized value at ``quarter_end`` if present, else NaN."""
    qe = pd.Timestamp(quarter_end)
    match = history.loc[history.index == qe]
    return float(match.iloc[0]) if not match.empty else float("nan")


def assemble_cell_rows(
    panel: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    forecasts: pd.DataFrame,
) -> list[dict]:
    """Expand one forecast cell into output rows with realized/naive/error fields."""
    rows: list[dict] = []
    for target in forecasts["target"].unique():
        history = quarterly_history(panel, ticker, target)
        cell = forecasts[forecasts["target"] == target].sort_values("horizon_q")
        for _, fr in cell.iterrows():
            target_qe = as_of + pd.offsets.QuarterEnd(int(fr["horizon_q"]))
            naive_qe = target_qe - pd.offsets.QuarterEnd(4)
            realized = lookup_realized(history, target_qe)
            naive_yoy = lookup_realized(history, naive_qe)
            chronos_q50 = float(fr["forecast_q50"])
            chronos_q10 = float(fr["forecast_q10"])
            chronos_q90 = float(fr["forecast_q90"])
            consensus = float(fr["consensus"]) if pd.notna(fr["consensus"]) else float("nan")
            rows.append(
                dict(
                    ticker=ticker,
                    target=target,
                    as_of_date=as_of,
                    target_quarter_end=target_qe,
                    horizon_q=int(fr["horizon_q"]),
                    chronos_q10=chronos_q10,
                    chronos_q50=chronos_q50,
                    chronos_q90=chronos_q90,
                    consensus=consensus,
                    naive_yoy=naive_yoy,
                    realized=realized,
                    chronos_abs_err=(
                        abs(realized - chronos_q50) if not np.isnan(realized) else float("nan")
                    ),
                    consensus_abs_err=(
                        abs(realized - consensus)
                        if not (np.isnan(realized) or np.isnan(consensus))
                        else float("nan")
                    ),
                    naive_abs_err=(
                        abs(realized - naive_yoy)
                        if not (np.isnan(realized) or np.isnan(naive_yoy))
                        else float("nan")
                    ),
                    chronos_in_band=(
                        bool(chronos_q10 <= realized <= chronos_q90)
                        if not np.isnan(realized)
                        else False
                    ),
                )
            )
    return rows


def summarize(backtest: pd.DataFrame) -> dict:
    """Aggregate per-cell rows into headline metrics."""
    summary: dict = {"per_target_horizon": {}, "overall": {}}

    def _mape(err: pd.Series, denom: pd.Series) -> float:
        ok = denom != 0
        return float((err[ok] / denom[ok].abs()).mean()) if ok.any() else float("nan")

    for (target, horizon), grp in backtest.groupby(["target", "horizon_q"]):
        n = len(grp)
        n_realized = grp["realized"].notna().sum()
        chronos_mae = float(grp["chronos_abs_err"].mean(skipna=True))
        naive_mae = float(grp["naive_abs_err"].mean(skipna=True))
        consensus_mae = float(grp["consensus_abs_err"].mean(skipna=True))
        chronos_mape = _mape(grp["chronos_abs_err"], grp["realized"])
        naive_mape = _mape(grp["naive_abs_err"], grp["realized"])
        consensus_mape = _mape(grp["consensus_abs_err"], grp["realized"])
        # Win rates only on rows with both numerators present.
        both_cn = grp.dropna(subset=["chronos_abs_err", "naive_abs_err"])
        wr_cn = (
            float((both_cn["chronos_abs_err"] < both_cn["naive_abs_err"]).mean())
            if len(both_cn)
            else float("nan")
        )
        both_cc = grp.dropna(subset=["chronos_abs_err", "consensus_abs_err"])
        wr_cc = (
            float((both_cc["chronos_abs_err"] < both_cc["consensus_abs_err"]).mean())
            if len(both_cc)
            else float("nan")
        )
        # Calibration: fraction of realized values inside Chronos [q10, q90].
        realized_rows = grp.dropna(subset=["realized"])
        calib = (
            float(realized_rows["chronos_in_band"].mean()) if len(realized_rows) else float("nan")
        )
        summary["per_target_horizon"][f"{target}__h{horizon}"] = {
            "n_cells": int(n),
            "n_realized": int(n_realized),
            "chronos_mae": chronos_mae,
            "naive_mae": naive_mae,
            "consensus_mae": consensus_mae,
            "chronos_mape": chronos_mape,
            "naive_mape": naive_mape,
            "consensus_mape": consensus_mape,
            "chronos_beats_naive_win_rate": wr_cn,
            "chronos_beats_consensus_win_rate": wr_cc,
            "chronos_q10q90_coverage_rate": calib,
        }

    realized_rows = backtest.dropna(subset=["realized"])
    if len(realized_rows):
        summary["overall"] = {
            "n_cells_total": int(len(backtest)),
            "n_cells_realized": int(len(realized_rows)),
            "chronos_q10q90_coverage_rate": float(realized_rows["chronos_in_band"].mean()),
            "chronos_beats_naive_overall": float(
                (
                    realized_rows.dropna(subset=["naive_abs_err"])["chronos_abs_err"]
                    < realized_rows.dropna(subset=["naive_abs_err"])["naive_abs_err"]
                ).mean()
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=BACKTEST_TICKERS)
    parser.add_argument("--as-of", nargs="+", default=BACKTEST_AS_OF_DATES)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    args = parser.parse_args()

    panel = load_panel(DATA_DIR)
    pipeline = _load_chronos_pipeline(args.device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for ticker in args.tickers:
        if not (panel["ticker"] == ticker).any():
            print(f"  skip {ticker}: not in panel")
            continue
        for as_of_str in args.as_of:
            as_of = pd.Timestamp(as_of_str)
            print(f"  forecasting {ticker} as_of={as_of.date()}")
            fc = forecast(
                ticker,
                as_of,
                panel=panel,
                pipeline=pipeline,
                targets=args.targets,
                horizon_q=HORIZON_Q,
            )
            if fc.empty:
                print(f"    no forecasts (insufficient history)")
                continue
            all_rows.extend(assemble_cell_rows(panel, ticker, as_of, fc))

    if not all_rows:
        print("no rows produced; check inputs")
        return

    backtest = pd.DataFrame(all_rows)
    parquet_path = OUTPUT_DIR / "chronos2_backtest.parquet"
    backtest.to_parquet(parquet_path, index=False)
    print(f"wrote {len(backtest)} rows -> {parquet_path}")

    summary = summarize(backtest)
    summary_path = OUTPUT_DIR / "chronos2_backtest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote summary -> {summary_path}")

    realized = backtest.dropna(subset=["realized"])
    if len(realized):
        print(
            f"\n{len(realized)}/{len(backtest)} cells have realized values "
            f"(chronos overall MAE = ${realized['chronos_abs_err'].mean():.2f}M, "
            f"naive overall MAE = ${realized.dropna(subset=['naive_abs_err'])['naive_abs_err'].mean():.2f}M)"
        )


if __name__ == "__main__":
    main()
