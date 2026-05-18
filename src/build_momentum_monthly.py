"""Build the monthly 12-1 price momentum panel.

12-1 momentum is the cumulative total return from 12 months ago through 1
month ago, skipping the most recent month to avoid short-term reversal
contamination. Implemented as a simple ratio on the rebalance-date price
series produced by :mod:`build_returns_monthly`:

    Mom_{i, m} = px_eom_{m-1} / px_eom_{m-12} - 1

Output:
    _data/momentum_monthly.parquet

Schema:
    date       last business day of the calendar month (matches returns panel)
    ticker     upper-case
    mom_12_1   trailing 12-1 momentum (NaN if not enough history)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

INPUT_FILENAME = "returns_monthly.parquet"
OUTPUT_FILENAME = "momentum_monthly.parquet"


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/build_returns_monthly.py` first."
        )
    returns = pd.read_parquet(path).sort_values(["ticker", "date"])

    wide = returns.pivot(index="date", columns="ticker", values="px_eom").sort_index()
    mom = wide.shift(1) / wide.shift(12) - 1.0

    long = mom.stack().rename("mom_12_1").reset_index()
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


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
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
