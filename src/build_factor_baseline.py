"""Build factor baseline data for §5.4-§5.6 of writeup_v2.

Two outputs:

1. ``_data/ff5_monthly.parquet`` — Ken French 5-factor monthly returns
   (Mkt-RF, SMB, HML, RMW, CMA, RF) in decimal form. Downloaded over HTTP
   from the public Ken French data library; no auth required.

2. ``_data/car3_per_call.parquet`` — CAR3 = px[t+1] / px[t-1] - 1 for each
   earnings call. Uses existing daily Bloomberg PX_LAST history. Window
   chosen to be the standard [-1, +1] announcement-return CAR (literature
   convention).

Downstream consumers:
- ``build_signal_panel.py`` loads ``car3_per_call.parquet`` and adds a
  ``sig_car3`` column via the standard 60-day carry-forward mechanism
  (same code path as ``sig_anchor`` / ``sig_ridge`` / ``sig_lm``).
- ``factor_regression.py`` loads ``ff5_monthly.parquet`` as RHS factors
  for the §5.6 nested α progression.

The CAR3 long-short backtest portfolio is NOT built here. It comes out of
``run_backtests.py`` once ``sig_car3`` is in the unified signal panel,
exactly the same way every other strategy's LS portfolio is built.

Per-call source: ``lm_scores_transcripts.parquet`` provides the cleanest
(ticker, event_date) per-call frame and is already a standard pipeline
output (from ``score_transcript_lm.py``). We only read those two columns.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from build_returns_monthly import load_prices
from calendar_utils import fwd_ret_bd
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# Ken French data library — FF5 monthly, 2x3 sort. Free, no auth.
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
FF5_CSV_NAME = "F-F_Research_Data_5_Factors_2x3.csv"
FF5_FILENAME = "ff5_monthly.parquet"

# CAR3 window: entry at event_date - 1 BDay, exit at event_date + 1 BDay.
# Standard PEAD literature uses [-1, +1] for the announcement-window return.
CAR3_GAP = -1
CAR3_H = 2  # entry at t+gap = t-1; exit at t+gap+h = t+1
CAR3_FILENAME = "car3_per_call.parquet"

# Per-call (ticker, event_date) source. Built by score_transcript_lm.py.
PER_CALL_SOURCE = "lm_scores_transcripts.parquet"


def fetch_ff5_monthly() -> pd.DataFrame:
    """Download and parse Ken French FF5 monthly factor returns.

    Returns
    -------
    DataFrame with columns ``date, mkt_rf, smb, hml, rmw, cma, rf``.
    Dates are business-month-end timestamps; returns are in DECIMAL form
    (Ken French publishes percent; we divide by 100).
    """
    req = Request(FF5_URL, headers={"User-Agent": "finm33200_project research"})
    with urlopen(req, timeout=60) as resp:
        raw = resp.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(FF5_CSV_NAME) as f:
            text = f.read().decode("utf-8", errors="replace")

    lines = text.splitlines()

    # Find the column-header line. Format is something like:
    #     "      ,Mkt-RF,SMB,HML,RMW,CMA,RF"
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.replace(" ", "").startswith(",Mkt-RF") or ln.strip().startswith("Mkt-RF"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"FF5 download missing column header; got {len(lines)} lines"
        )

    # Parse rows after header. Stop at blank line or non-monthly format
    # (annual section follows the monthly section in the same file).
    monthly_rows: list[list] = []
    for ln in lines[header_idx + 1:]:
        s = ln.strip()
        if not s:
            break
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 7:
            break
        date_str = parts[0]
        if not (date_str.isdigit() and len(date_str) == 6):
            break
        try:
            vals = [float(p) / 100.0 for p in parts[1:]]
        except ValueError:
            break
        monthly_rows.append([date_str, *vals])

    if not monthly_rows:
        raise ValueError("FF5 download contained no monthly rows")

    df = pd.DataFrame(
        monthly_rows,
        columns=["yyyymm", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"],
    )
    df["date"] = (
        pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.BMonthEnd(0)
    )
    return (
        df[["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def compute_car3_per_call(
    prices: pd.DataFrame,
    per_call: pd.DataFrame,
) -> pd.DataFrame:
    """CAR3 = px[event_date + 1 BD] / px[event_date - 1 BD] - 1 per call.

    Args
    ----
    prices : long DataFrame ``[date, ticker, px_last]``
    per_call : long DataFrame with at least ``[ticker, event_date]``;
        other columns ignored.

    Returns
    -------
    DataFrame ``[ticker, event_date, car3]``. Rows where the price window
    cannot be evaluated (e.g. event near series end) are dropped.
    """
    per_call = per_call.copy()
    per_call["ticker"] = per_call["ticker"].astype(str).str.upper()
    per_call["event_date"] = pd.to_datetime(per_call["event_date"])

    rows: list[dict] = []
    for ticker, grp in per_call.groupby("ticker", sort=True):
        sub = prices[prices["ticker"] == ticker]
        if sub.empty:
            continue
        px_series = sub.set_index("date")["px_last"].sort_index()
        if px_series.empty:
            continue
        for event_date in grp["event_date"].drop_duplicates():
            car3 = fwd_ret_bd(px_series, event_date, h=CAR3_H, gap=CAR3_GAP)
            rows.append({
                "ticker": ticker,
                "event_date": pd.Timestamp(event_date),
                "car3": car3,
            })

    out = pd.DataFrame(rows)
    return (
        out.dropna(subset=["car3"])
        .sort_values(["ticker", "event_date"])
        .reset_index(drop=True)
    )


def build(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff5 = fetch_ff5_monthly()

    per_call_path = data_dir / PER_CALL_SOURCE
    if not per_call_path.exists():
        raise FileNotFoundError(
            f"{per_call_path} not found. Run "
            f"`python src/score_transcript_lm.py` first "
            f"(or `python -m doit build_signals:lm`)."
        )
    per_call = pd.read_parquet(per_call_path)

    prices = load_prices(data_dir)
    car3 = compute_car3_per_call(prices, per_call)
    return ff5, car3


def write(
    ff5: pd.DataFrame,
    car3: pd.DataFrame,
    data_dir: Path = DATA_DIR,
) -> tuple[Path, Path]:
    ff5_path = data_dir / FF5_FILENAME
    car3_path = data_dir / CAR3_FILENAME
    ff5.to_parquet(ff5_path, index=False)
    car3.to_parquet(car3_path, index=False)
    return ff5_path, car3_path


def main() -> None:
    ff5, car3 = build()
    ff5_path, car3_path = write(ff5, car3)

    print(f"Wrote {len(ff5):,} monthly FF5 rows -> {ff5_path}")
    print(
        f"  date range: {ff5['date'].min().date()} -> {ff5['date'].max().date()}"
    )
    print(
        f"  mean Mkt-RF: {ff5['mkt_rf'].mean()*12*100:+.2f}% ann | "
        f"mean RF: {ff5['rf'].mean()*12*100:+.2f}% ann"
    )

    print(f"\nWrote {len(car3):,} per-call CAR3 rows -> {car3_path}")
    print(f"  tickers: {car3['ticker'].nunique()}")
    print(
        f"  date range: {car3['event_date'].min().date()} -> "
        f"{car3['event_date'].max().date()}"
    )
    q = car3["car3"].quantile([0.1, 0.5, 0.9]).round(4).to_list()
    print(f"  CAR3 quantiles (10/50/90): {q}")
    extreme = (car3["car3"].abs() > 0.20).sum()
    print(f"  |CAR3| > 0.20: {extreme} rows (potential outliers)")


if __name__ == "__main__":
    main()
