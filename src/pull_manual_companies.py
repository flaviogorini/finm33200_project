"""Parse the manual Bloomberg per-company prediction Excel into parquet files.

Reads `data_manual/US_Companies_Prediction_Data.xlsx` and writes two long-format
parquet files in `_data/`:

  - `US_Companies_Forecast.parquet`  (sheet `Forecast`)
  - `US_Companies_Hist_Data.parquet` (sheet `Hist_Data`)

Both share the same banded 3-row header convention as the macro file, so we
reuse `_parse_banded_excel` from `pull_manual_macro`. The `Info` sheet in the
workbook is free-text Bloomberg field documentation and is not parsed here.
"""

from pathlib import Path

import pandas as pd

from pull_manual_macro import _parse_banded_excel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))

COMPANIES_FILENAME = "US_Companies_Prediction_Data.xlsx"
FORECAST_SHEET = "Forecast"
HIST_SHEET = "Hist_Data"
FORECAST_PARQUET = "US_Companies_Forecast.parquet"
HIST_PARQUET = "US_Companies_Hist_Data.parquet"

description_forecast = {
    "BEST_PE_RATIO": "Bloomberg consensus forward P/E ratio",
    "BEST_SALES": "Bloomberg consensus forward sales/revenue estimate",
    "BEST_NET_INCOME": "Bloomberg consensus forward net income estimate",
    "BEST_NET_DEBT": "Bloomberg consensus forward net debt estimate",
    "BEST_EBITDA": "Bloomberg consensus forward EBITDA estimate",
}

description_hist = {
    "PX_LAST": "Last price",
    "PE_RATIO": "Trailing price / earnings ratio",
    "TRAIL_12M_NET_SALES": "Trailing 12-month net sales",
    "TRAIL_12M_NET_INC": "Trailing 12-month net income",
    "TRAIL_12M_EBITDA": "Trailing 12-month EBITDA",
}


def pull_manual_companies_forecast(
    manual_data_dir: Path = MANUAL_DATA_DIR,
) -> pd.DataFrame:
    """Parse the Forecast sheet and return long-format DataFrame.

    The Forecast sheet carries a single "Start Date" preamble row above the
    banded 3-row header, so we skip it via `header_offset=1`.
    """
    path = Path(manual_data_dir) / COMPANIES_FILENAME
    return _parse_banded_excel(path, FORECAST_SHEET, header_offset=1)


def pull_manual_companies_hist(
    manual_data_dir: Path = MANUAL_DATA_DIR,
) -> pd.DataFrame:
    """Parse the Hist_Data sheet and return long-format DataFrame."""
    path = Path(manual_data_dir) / COMPANIES_FILENAME
    return _parse_banded_excel(path, HIST_SHEET)


def load_manual_companies_forecast(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the parsed forecast parquet."""
    return pd.read_parquet(Path(data_dir) / FORECAST_PARQUET)


def load_manual_companies_hist(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the parsed historical-data parquet."""
    return pd.read_parquet(Path(data_dir) / HIST_PARQUET)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    forecast = pull_manual_companies_forecast(manual_data_dir=MANUAL_DATA_DIR)
    forecast.to_parquet(DATA_DIR / FORECAST_PARQUET, index=False)

    hist = pull_manual_companies_hist(manual_data_dir=MANUAL_DATA_DIR)
    hist.to_parquet(DATA_DIR / HIST_PARQUET, index=False)
