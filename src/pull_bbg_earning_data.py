"""Pull Bloomberg financial ratios, quarterly history, and price data for any ticker.

Fetches valuation, profitability, leverage, growth data via BQL/BDP/BDH,
writes output CSVs to the project DATA_DIR (_data/ by default).

Usage:
    python pull_bbg_earning_data.py                      # defaults to LITE US Equity
    python pull_bbg_earning_data.py QCOM                 # appends " US Equity"
    python pull_bbg_earning_data.py "QCOM US Equity"     # full Bloomberg ticker
    python pull_bbg_earning_data.py QCOM AAPL MSFT       # multiple tickers
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from blp import blp

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# ── BDP fields: current snapshot ────────────────────────────────────────────

BDP_FIELDS = [
    # Price & market cap
    "PX_LAST",
    "CUR_MKT_CAP",
    "ENTERPRISE_VALUE",
    # Valuation multiples
    "PE_RATIO",
    "EV_TO_T12M_EBITDA",
    "PX_TO_SALES_RATIO",
    "EV_TO_T12M_SALES",
    "PX_TO_BOOK_RATIO",
    "PX_TO_FREE_CASH_FLOW",
    # Profitability
    "GROSS_MARGIN",
    "EBITDA_MARGIN",
    "OPER_MARGIN",
    "RETURN_ON_ASSET",
    "RETURN_COM_EQY",
    # Growth (trailing)
    "SALES_GROWTH",
    "EBITDA_GROWTH",
    # Leverage & liquidity
    "NET_DEBT_TO_EBITDA",
    "TOT_DEBT_TO_TOT_EQY",
    "INTEREST_COVERAGE_RATIO",
    "TOT_DEBT_TO_TOT_ASSET",
    # Per-share
    "BOOK_VAL_PER_SH",
    "TRAIL_12M_EPS",
    "FREE_CASH_FLOW_PER_SH",
    "DVD_SH_12M",
    "EQY_DVD_YLD_IND",
]

# BEST_* consensus fields — pulled with BEST_FPERIOD_OVERRIDE='1FQ' to force
# next-fiscal-quarter consensus. Without the override Bloomberg's default can
# drift between annual (1FY) and quarterly depending on terminal config and
# ticker; '1FY' returns annual values (e.g. AAPL BEST_EPS≈$8.7) which is what
# we're avoiding. '1BQ'/'1Q' don't work for PE/EV-EBITDA/EPS fields — '1FQ'
# is the syntax that BDP accepts for all four.
BEST_QUARTERLY_FIELDS = [
    "BEST_PE_RATIO",
    "BEST_EV_TO_BEST_EBITDA",
    "BEST_EPS",
    "BEST_SALES",
]
BEST_QUARTERLY_OVERRIDE = [("BEST_FPERIOD_OVERRIDE", "1FQ")]

# ── BDH config ───────────────────────────────────────────────────────────────

BDH_START = "20050101"
BDH_FIELDS = ["PX_LAST", "VOLUME"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _normalise_ticker(arg: str) -> str:
    return arg if " " in arg else f"{arg} US Equity"


def _build_bql_queries(ticker: str) -> dict[str, str]:
    return {
        "quarterly_eps": f"""
            get(is_eps(fpt=q, fpo=range(-79,0), ae=a).value)
            for(['{ticker}'])
        """,
        "quarterly_revenue": f"""
            get(sales_rev_turn(fpt=q, fpo=range(-79,0)).value)
            for(['{ticker}'])
        """,
        "quarterly_ebitda": f"""
            get(ebitda(fpt=q, fpo=range(-79,0)).value)
            for(['{ticker}'])
        """,
        "quarterly_gross_margin": f"""
            get(gross_margin(fpt=q, fpo=range(-79,0)).value)
            for(['{ticker}'])
        """,
        # Forward consensus (next 4 quarters). Dropping ae=a tells BQL to
        # return estimates instead of actuals.
        "quarterly_consensus_eps": f"""
            get(is_eps(fpt=q, fpo=range(1,4)).value)
            for(['{ticker}'])
        """,
        "quarterly_consensus_revenue": f"""
            get(sales_rev_turn(fpt=q, fpo=range(1,4)).value)
            for(['{ticker}'])
        """,
        "quarterly_consensus_ebitda": f"""
            get(ebitda(fpt=q, fpo=range(1,4)).value)
            for(['{ticker}'])
        """,
    }


def _probe_fields(bq, ticker: str, fields: list[str]) -> tuple[list[str], list[str]]:
    valid, invalid = [], []
    for field in fields:
        try:
            bq.bdp([ticker], [field])
            valid.append(field)
        except TypeError:
            invalid.append(field)
    return valid, invalid


# ── Pull functions ────────────────────────────────────────────────────────────


def pull_bbg_snapshot(bq, ticker: str) -> pd.DataFrame:
    try:
        df = bq.bdp([ticker], BDP_FIELDS)
    except TypeError:
        print("  Bad field detected — probing each field individually...")
        valid, invalid = _probe_fields(bq, ticker, BDP_FIELDS)
        print(f"  Valid ({len(valid)}): {valid}")
        print(f"  Invalid ({len(invalid)}): {invalid}")
        df = bq.bdp([ticker], valid)

    # Pull BEST_* consensus with quarterly override (defaults to annual FY1
    # without override — that's the bug this branch fixes).
    best_df = bq.bdp(
        [ticker], BEST_QUARTERLY_FIELDS, overrides=BEST_QUARTERLY_OVERRIDE
    )
    df = df.merge(best_df, on="security", how="left")

    df.insert(0, "as_of", datetime.today().strftime("%Y-%m-%d"))
    return df


def pull_bbg_quarterly_history(bq, ticker: str) -> dict[str, pd.DataFrame]:
    results = {}
    for name, query in _build_bql_queries(ticker).items():
        try:
            raw = bq.bql(query)
            df = raw[0] if raw else pd.DataFrame()
            if not df.empty:
                df.insert(0, "series", name)
                results[name] = df
        except Exception as e:
            print(f"  {name}: skipped — {e}")
    return results


def pull_bbg_price_history(
    bq, ticker: str, start_date: str = BDH_START
) -> pd.DataFrame:
    end_date = datetime.today().strftime("%Y%m%d")
    df = bq.bdh([ticker], BDH_FIELDS, start_date, end_date)
    df["date"] = pd.to_datetime(df["date"])
    return (
        df.drop(columns="security")
        .set_index("date")
        .resample("W")
        .last()
        .reset_index()
    )


# ── Load functions ────────────────────────────────────────────────────────────


def load_bbg_snapshot(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    prefix = ticker.split()[0]
    files = sorted(Path(data_dir).glob(f"{prefix}_snapshot_*.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshot CSVs found for {prefix} in {data_dir}")
    return pd.read_csv(files[-1])


def load_bbg_quarterly_history(
    ticker: str, name: str, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    prefix = ticker.split()[0]
    files = sorted(Path(data_dir).glob(f"{prefix}_{name}_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No quarterly history CSVs found for {prefix}/{name} in {data_dir}"
        )
    return pd.read_csv(files[-1])


def load_bbg_prices(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    prefix = ticker.split()[0]
    files = sorted(Path(data_dir).glob(f"{prefix}_prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No price CSVs found for {prefix} in {data_dir}")
    return pd.read_csv(files[-1], index_col="date", parse_dates=True)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = sys.argv[1:] or ["LITE"]
    tickers = [_normalise_ticker(a) for a in args]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime("%Y%m%d")

    print(f"Connecting to Bloomberg...")
    bq = blp.BlpQuery().start()

    for ticker in tickers:
        prefix = ticker.split()[0]
        print(f"\n── {ticker} ──")

        print("  Fetching snapshot (BDP)...")
        snapshot = pull_bbg_snapshot(bq, ticker)
        snap_path = DATA_DIR / f"{prefix}_snapshot_{today}.csv"
        snapshot.to_csv(snap_path, index=False)
        print(f"  Saved → {snap_path}")
        print(snapshot.to_string(index=False))

        print("\n  Fetching quarterly history (BQL)...")
        history = pull_bbg_quarterly_history(bq, ticker)
        for name, df in history.items():
            out = DATA_DIR / f"{prefix}_{name}_{today}.csv"
            df.to_csv(out, index=False)
            print(f"  {name}: {len(df)} rows → {out}")

        print("\n  Fetching weekly price history (BDH)...")
        prices = pull_bbg_price_history(bq, ticker)
        price_path = DATA_DIR / f"{prefix}_prices_{today}.csv"
        prices.to_csv(price_path)
        print(f"  {len(prices)} weeks → {price_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
