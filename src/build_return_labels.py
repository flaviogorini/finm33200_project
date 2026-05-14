"""Build monthly return features and forward-return labels.

Consumes the long-format historical parquet produced by
``pull_manual_companies.py`` (the PX_LAST column inside it) and emits a
monthly panel:

    _data/labels_returns_monthly.parquet

Schema:
    date            month-end Timestamp
    ticker          str  (' US Equity' suffix stripped)
    px_eom          float, end-of-month last price
    ret_1m          float, trailing 1-month return  (px_t / px_{t-1} - 1)
    ret_3m          float, trailing 3-month return
    ret_6m          float, trailing 6-month return
    ret_12m         float, trailing 12-month return
    fwd_ret_1m      float, FORWARD 1-month return  (px_{t+1} / px_t - 1)
    fwd_ret_3m      float, FORWARD 3-month return
    fwd_ret_6m      float, FORWARD 6-month return
    fwd_ret_12m     float, FORWARD 12-month return

`fwd_*` columns are LABELS, not features. Downstream models with target date
``t`` MUST exclude them from the input matrix.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_fundamentals_features import _strip_bbg_suffix
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "US_Companies_Hist_Data.parquet"
OUTPUT_FILENAME = "labels_returns_monthly.parquet"

LOOKBACK_WINDOWS = {"ret_1m": 1, "ret_3m": 3, "ret_6m": 6, "ret_12m": 12}
FORWARD_WINDOWS = {"fwd_ret_1m": 1, "fwd_ret_3m": 3, "fwd_ret_6m": 6, "fwd_ret_12m": 12}


def load_prices(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read PX_LAST from the historical parquet.

    Returns a long frame [date, ticker, px_last] indexed by row.
    """
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pull_manual_companies.py` first."
        )
    df = pd.read_parquet(path)
    df = df[df["field"] == "PX_LAST"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].map(_strip_bbg_suffix)
    df = df.rename(columns={"value": "px_last"})
    return df[["date", "ticker", "px_last"]].dropna()


def build_returns_for_ticker(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute monthly trailing and forward returns for one ticker."""
    series = (
        prices[prices["ticker"] == ticker]
        .set_index("date")["px_last"]
        .sort_index()
    )
    monthly_px = series.resample("ME").last().dropna()

    out = pd.DataFrame({"px_eom": monthly_px})
    for col, n in LOOKBACK_WINDOWS.items():
        out[col] = monthly_px.pct_change(n)
    for col, n in FORWARD_WINDOWS.items():
        out[col] = monthly_px.shift(-n) / monthly_px - 1

    out = out.reset_index()
    out["ticker"] = ticker
    return out[["date", "ticker", "px_eom", *LOOKBACK_WINDOWS, *FORWARD_WINDOWS]]


def build(
    tickers: list[str] | None = None, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    prices = load_prices(data_dir)
    available = sorted(prices["ticker"].unique())
    use_tickers = tickers if tickers else available
    missing = set(use_tickers) - set(available)
    if missing:
        print(f"  warning: no PX_LAST for {sorted(missing)}; skipping")
    use_tickers = [t for t in use_tickers if t in available]
    if not use_tickers:
        raise RuntimeError(
            f"No requested tickers have PX_LAST in {INPUT_FILENAME}. "
            f"Available: {available}"
        )

    frames = [build_returns_for_ticker(prices, t) for t in use_tickers]
    return pd.concat(frames, ignore_index=True)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    import sys

    tickers = sys.argv[1:] or None
    panel = build(tickers)
    out = write(panel)
    print(f"Wrote {len(panel):,} rows x {len(panel.columns)} cols -> {out}")
    print(f"Tickers ({panel['ticker'].nunique()}):", sorted(panel["ticker"].unique()))
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
