"""Parse the manual Bloomberg macro Excel export into a tidy parquet file.

Reads `data_manual/Macro_Data_US.xlsx` (a Bloomberg-style banded sheet with a
3-row header: ticker / human field name / Bloomberg field code) and writes
`_data/Macro_Data_US.parquet` in long format with columns
`date, ticker, field, value`.

The `_parse_banded_excel` helper is shared with `pull_manual_companies.py`
which uses the same header convention.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))

MACRO_FILENAME = "Macro_Data_US.xlsx"
MACRO_SHEET = "Macro_Data"
MACRO_PARQUET = "Macro_Data_US.parquet"

description_macro = {
    "GT5 Govt": "US Treasury 5-Year benchmark yield",
    "GT2 Govt": "US Treasury 2-Year benchmark yield",
    "GT30 Govt": "US Treasury 30-Year benchmark yield",
    "GT10 Govt": "US Treasury 10-Year benchmark yield",
    "DXY Curncy": "US Dollar Index (DXY)",
    "USGGT10Y Index": "US 10-Year breakeven inflation (TIPS-implied)",
    "US CDS EUR SR 5Y D14 Corp": "US sovereign 5Y CDS (EUR, senior)",
    "USYC2Y10 Index": "2s10s US Treasury yield curve spread (bps)",
    "USYC2Y30 Index": "2s30s US Treasury yield curve spread (bps)",
    "EURUSDV3M Curncy": "EUR/USD 3-month implied volatility",
    "VIX Index": "CBOE Volatility Index (VIX)",
    "CL1 Comdty": "WTI crude oil front-month futures",
}


def _parse_banded_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    """Parse a Bloomberg-style banded Excel sheet into long format.

    Layout assumed:
      row 0  : ticker name in the first column of each block, blanks elsewhere
      row 1  : human-readable field name per column
      row 2  : Bloomberg field code per column ("Dates" in column 0)
      row 3+ : data — column 0 is the date, the rest are numeric values
               (Bloomberg N/A sentinels like "#N/A N/A" are coerced to NaN).

    Returns a DataFrame with columns: date, ticker, field, value.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    tickers = raw.iloc[0, 1:].ffill().astype(str).tolist()
    fields = raw.iloc[2, 1:].astype(str).tolist()

    data = raw.iloc[3:].reset_index(drop=True)
    dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")

    values = data.iloc[:, 1:].copy()
    values.columns = pd.MultiIndex.from_arrays(
        [tickers, fields], names=["ticker", "field"]
    )
    values = values.apply(pd.to_numeric, errors="coerce")
    values.insert(0, ("date", ""), dates)

    long = (
        values.set_index(("date", ""))
        .stack(["ticker", "field"], future_stack=True)
        .rename("value")
        .reset_index()
        .rename(columns={("date", ""): "date"})
    )
    long.columns = ["date", "ticker", "field", "value"]

    long = long.dropna(subset=["date", "value"])
    long = long.sort_values(["ticker", "field", "date"]).reset_index(drop=True)
    long["value"] = long["value"].astype(np.float64)
    return long


def pull_manual_macro(manual_data_dir: Path = MANUAL_DATA_DIR) -> pd.DataFrame:
    """Parse the macro Excel and return long-format DataFrame."""
    path = Path(manual_data_dir) / MACRO_FILENAME
    return _parse_banded_excel(path, MACRO_SHEET)


def load_manual_macro(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the parsed macro parquet."""
    return pd.read_parquet(Path(data_dir) / MACRO_PARQUET)


if __name__ == "__main__":
    macro = pull_manual_macro(manual_data_dir=MANUAL_DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(DATA_DIR / MACRO_PARQUET, index=False)
