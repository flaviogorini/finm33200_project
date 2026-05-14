"""Zero-shot fundamentals forecast with Amazon Chronos-2.

Loads ``amazon/chronos-2`` from Hugging Face via the ``chronos-forecasting``
library and produces probabilistic 4-quarter-ahead forecasts of revenue and
net income for each ticker in the panel, then aligns them against Bloomberg
``BEST_*`` analyst consensus at the same as-of date.

Output:
    _output/chronos2_forecast_{TICKER}_{ASOF}.parquet

Schema:
    ticker          str
    target          str  in {"revenue", "net_income"}
    as_of_date      Timestamp
    horizon_q       int  in {1, 2, 3, 4}
    forecast_q10    float
    forecast_q50    float  (median — primary point forecast)
    forecast_q90    float
    consensus       float  Bloomberg BEST_* value at as_of_date  (NaN if missing)

Usage:
    python forecast_chronos2.py AAPL                       # default as_of = latest panel date
    python forecast_chronos2.py AAPL --as-of 2024-12-31
    python forecast_chronos2.py AAPL MSFT --as-of 2024-09-30 --horizon 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import load_panel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

DEFAULT_TARGETS = ["revenue", "net_income"]
DEFAULT_HORIZON_Q = 4

CONSENSUS_FOR = {
    "revenue": "best_sales",
    "net_income": "best_net_income",
    "ebitda": "best_ebitda",
}


def quarterly_history(panel: pd.DataFrame, ticker: str, target: str) -> pd.Series:
    """Extract a quarterly time series for one ticker / target column.

    The monthly panel carries the most-recently-reported quarterly value
    forward; we subsample to quarter-ends to avoid feeding 3 identical
    monthly observations of the same quarterly fundamental into Chronos.
    """
    sub = (
        panel[panel["ticker"] == ticker]
        .set_index("date")[target]
        .sort_index()
        .dropna()
    )
    quarterly = sub.resample("QE").last().dropna()
    return quarterly


def _load_chronos_pipeline(device: str = "cpu"):
    """Lazy-import chronos so the module is usable for I/O without torch installed."""
    try:
        from chronos import Chronos2Pipeline
    except ImportError as e:
        raise ImportError(
            "chronos-forecasting is not installed. "
            "Run `pip install chronos-forecasting` (also installs torch + transformers)."
        ) from e

    print(f"  loading amazon/chronos-2 onto {device}…")
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)


def _predict_univariate(
    pipeline, context: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single univariate Chronos-2 forecast.

    Returns three arrays of shape (horizon,) for q10, q50, q90.
    """
    import torch

    ctx_tensor = torch.tensor(context, dtype=torch.float32)
    quantiles, _mean = pipeline.predict_quantiles(
        [ctx_tensor],
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    pred = quantiles[0]
    if hasattr(pred, "cpu"):
        pred = pred.cpu().numpy()
    pred = np.asarray(pred)[0]  # (n_variates=1, horizon, 3) → (horizon, 3)
    return pred[:, 0], pred[:, 1], pred[:, 2]


def forecast_for_ticker(
    panel: pd.DataFrame,
    pipeline,
    ticker: str,
    targets: list[str],
    as_of: pd.Timestamp,
    horizon_q: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for target in targets:
        history = quarterly_history(panel, ticker, target)
        history = history[history.index <= as_of]
        if len(history) < 8:
            print(f"  skip {ticker}/{target}: only {len(history)} historical quarters")
            continue

        q10, q50, q90 = _predict_univariate(pipeline, history.to_numpy(), horizon_q)

        consensus_col = CONSENSUS_FOR.get(target)
        consensus_at_asof = (
            panel.loc[
                (panel["ticker"] == ticker) & (panel["date"] == as_of),
                consensus_col,
            ].iloc[0]
            if consensus_col and consensus_col in panel.columns
            and not panel.loc[
                (panel["ticker"] == ticker) & (panel["date"] == as_of), consensus_col
            ].empty
            else np.nan
        )

        for h in range(horizon_q):
            rows.append(
                dict(
                    ticker=ticker,
                    target=target,
                    as_of_date=as_of,
                    horizon_q=h + 1,
                    forecast_q10=float(q10[h]),
                    forecast_q50=float(q50[h]),
                    forecast_q90=float(q90[h]),
                    consensus=float(consensus_at_asof),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; default = latest panel date for ticker")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_Q)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    args = parser.parse_args()

    panel = load_panel(DATA_DIR)
    pipeline = _load_chronos_pipeline(args.device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        sub = panel[panel["ticker"] == ticker]
        if sub.empty:
            print(f"  skip {ticker}: not in panel")
            continue
        as_of = pd.Timestamp(args.as_of) if args.as_of else sub["date"].max()
        out = forecast_for_ticker(panel, pipeline, ticker, args.targets, as_of, args.horizon)
        if out.empty:
            print(f"  no forecasts produced for {ticker}")
            continue
        path = OUTPUT_DIR / f"chronos2_forecast_{ticker}_{as_of.strftime('%Y%m%d')}.parquet"
        out.to_parquet(path, index=False)
        print(f"  {ticker}: {len(out)} forecast rows -> {path}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
