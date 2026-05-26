"""Pull historical Nasdaq-100 constituent membership from WRDS Compustat.

Replaces v2's `pull_nasdaq100_universe.py` (Wikipedia → current-only) with
the point-in-time historical record needed for v3.

Source: `comp.idxcst_his` is Compustat's index-constituent-history table.
Each row is one (gvkey, iid) membership interval `[from, thru]` in some
index `gvkeyx`. The Nasdaq-100 has `gvkeyx = '000208'` (confirmed via
`comp.idx_index` lookup: `conm = 'Nasdaq 100'`, `tic = 'I0028'`,
`indextype = 'LGCAP'`, `indexgeo = 'USA'`).

Filters applied:
- `gvkeyx = '000208'` — Nasdaq-100 only.
- Interval overlaps `[BACKTEST_START, BACKTEST_END]` — drop intervals
  that ended entirely before the backtest window or started entirely
  after it.

Resolution: joined with `comp.security` (gvkey, iid → ticker, CUSIP) and
`comp.company` (gvkey → conm) for human-readable names. The (gvkey, iid)
join is needed because a single gvkey can have multiple share classes
(e.g. Alphabet GOOG / GOOGL with iids 01 / 02).

Output:
    data_manual/_meta/nasdaq100_historical_constituents.csv

Schema:
    gvkey         Compustat firm identifier
    iid           Issue identifier (share class)
    ticker        Primary US ticker (uppercase)
    cusip         9-character CUSIP
    company_name  Compustat conm
    from_date     YYYY-MM-DD inclusive — when this share class entered the index
    thru_date     YYYY-MM-DD inclusive — when it left (blank if still in)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import wrds
from sqlalchemy import text

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
META_DIR = MANUAL_DATA_DIR / "_meta"

WRDS_USERNAME = config("WRDS_USERNAME")
WRDS_PASSWORD = config("WRDS_PASSWORD", default=None)

NASDAQ_100_GVKEYX = "000208"
BACKTEST_START = "2006-01-01"
BACKTEST_END = "2026-12-31"

OUTPUT_FILENAME = "nasdaq100_historical_constituents.csv"


def connect_wrds() -> wrds.Connection:
    """Same auth pattern as src/pull_wrds_earning_transcripts.py."""
    kwargs: dict[str, str] = {"wrds_username": WRDS_USERNAME}
    if WRDS_PASSWORD:
        kwargs["wrds_password"] = WRDS_PASSWORD
    return wrds.Connection(**kwargs)


def fetch_constituents(db: wrds.Connection) -> pd.DataFrame:
    """Return the historical constituent intervals overlapping the backtest
    window, with ticker + name resolved."""
    q = text(
        """
        SELECT h.gvkey, h.iid, h.from AS from_date, h.thru AS thru_date,
               s.tic AS ticker, s.cusip, s.excntry,
               c.conm AS company_name
        FROM comp.idxcst_his h
        LEFT JOIN comp.security s ON h.gvkey = s.gvkey AND h.iid = s.iid
        LEFT JOIN comp.company  c ON h.gvkey = c.gvkey
        WHERE h.gvkeyx = :gvkeyx
          AND h.from <= :end
          AND (h.thru IS NULL OR h.thru >= :start)
        ORDER BY h.from, h.gvkey, h.iid
        """
    )
    with db.engine.connect() as conn:
        df = pd.read_sql(
            q,
            conn,
            params={
                "gvkeyx": NASDAQ_100_GVKEYX,
                "start": BACKTEST_START,
                "end": BACKTEST_END,
            },
        )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise ticker case, drop non-US issues, sort canonically."""
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out = out[out["excntry"] == "USA"].drop(columns="excntry")
    out["from_date"] = pd.to_datetime(out["from_date"]).dt.date
    out["thru_date"] = pd.to_datetime(out["thru_date"]).dt.date
    return (
        out[["gvkey", "iid", "ticker", "cusip", "company_name",
             "from_date", "thru_date"]]
        .sort_values(["ticker", "from_date"])
        .reset_index(drop=True)
    )


def main() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    db = connect_wrds()
    try:
        raw = fetch_constituents(db)
    finally:
        db.close()

    df = clean(raw)

    out_path = META_DIR / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)

    n_ever = df["ticker"].nunique()
    n_current = df["thru_date"].isna().sum()
    date_min = df["from_date"].min()
    date_max = pd.Series(df["thru_date"].dropna()).max() if df["thru_date"].notna().any() else None
    print(f"Wrote {len(df):,} interval rows -> {out_path}")
    print(f"  unique tickers ever in Nasdaq-100 during "
          f"{BACKTEST_START}..{BACKTEST_END}: {n_ever}")
    print(f"  currently in (thru null): {n_current}")
    print(f"  earliest from_date: {date_min}")
    print(f"  latest non-null thru_date: {date_max}")
    print(f"  multi-iid gvkeys (share-class splits): "
          f"{(df.groupby('gvkey')['iid'].nunique() > 1).sum()}")


if __name__ == "__main__":
    main()
