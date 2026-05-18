"""Build the monthly returns panel from Bloomberg PX_LAST.

Reads ``_data/US_Companies_Hist_Data.parquet`` (long format, the output of
``pull_manual_companies.py``), filters to ``PX_LAST``, snaps to the
business-month-end rebalance calendar, and writes the per-ticker monthly
price + 21-trading-day forward return panel.

Output:
    _data/returns_monthly.parquet

Schema:
    date          last business day of the calendar month
    ticker        upper-case, ' US Equity' suffix stripped
    px_eom        PX_LAST at that rebalance date (or last available <=)
    fwd_ret_21d   forward 21-bday return: enter T+1, exit T+22

Forward return uses :func:`calendar_utils.fwd_ret_bd` so the value is
exactly the same quantity used as the ridge regression target and the
backtest holding-period return. There is no other forward-horizon
definition anywhere in the project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calendar_utils import EXEC_GAP_BDAYS, HOLDING_BDAYS, fwd_ret_bd, month_end_bd
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "US_Companies_Hist_Data.parquet"
OUTPUT_FILENAME = "returns_monthly.parquet"


def _strip_bbg_suffix(ticker: str) -> str:
    """'AAPL US Equity' -> 'AAPL'. Pass through anything that doesn't match."""
    if isinstance(ticker, str) and ticker.endswith(" US Equity"):
        return ticker[: -len(" US Equity")]
    return ticker


def load_prices(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read PX_LAST from the Bloomberg long-format parquet.

    Returns a long frame ``[date, ticker, px_last]`` with the
    ' US Equity' suffix stripped from tickers.
    """
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pull_manual_companies.py` first."
        )
    df = pd.read_parquet(path)
    df = df[df["field"] == "PX_LAST"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].map(_strip_bbg_suffix).astype(str).str.upper()
    df = df.rename(columns={"value": "px_last"})
    return df[["date", "ticker", "px_last"]].dropna(subset=["px_last"]).reset_index(drop=True)


def _ticker_panel(prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the monthly panel for one ticker."""
    series = prices.set_index("date")["px_last"].sort_index()
    series = series[~series.index.duplicated(keep="last")]

    px_eom = series.reindex(rebalance_dates, method="ffill")
    fwd_ret = pd.Series(
        [fwd_ret_bd(series, d, h=HOLDING_BDAYS, gap=EXEC_GAP_BDAYS) for d in rebalance_dates],
        index=rebalance_dates,
    )

    out = pd.DataFrame({"date": rebalance_dates, "px_eom": px_eom.to_numpy(), "fwd_ret_21d": fwd_ret.to_numpy()})
    out["ticker"] = prices["ticker"].iloc[0]
    return out[["date", "ticker", "px_eom", "fwd_ret_21d"]]


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    prices = load_prices(data_dir)
    start = prices["date"].min()
    end = prices["date"].max()
    rebalance_dates = month_end_bd(start, end)

    frames = [
        _ticker_panel(grp, rebalance_dates)
        for _, grp in prices.groupby("ticker", sort=True)
    ]
    panel = pd.concat(frames, ignore_index=True)
    return panel.dropna(subset=["px_eom"]).reset_index(drop=True)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    print(f"Wrote {len(panel):,} rows -> {out}")
    print(f"Tickers ({panel['ticker'].nunique()})")
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    n_fwd = panel["fwd_ret_21d"].notna().sum()
    print(f"Non-null fwd_ret_21d: {n_fwd:,} / {len(panel):,}")
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
