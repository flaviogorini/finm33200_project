"""Pull WRDS Capital IQ earnings-call transcripts for the full 13-ticker panel.

Iterates over the ticker→companyId map below and invokes the existing
single-ticker extractor (``pull_wrds_earning_transcripts.main``) per ticker.
Skips tickers whose outputs already exist on disk (cheap idempotency).

Capital IQ companyIds were resolved against ``ciq_common.wrds_ciqsymbol_primary``
joined on SEC CIK.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

TICKER_COMPANY_IDS: dict[str, int] = {
    "AAPL": 24937,
    "AMZN": 18749,
    "BA": 370857,
    "CVX": 98506,
    "GS": 398625,
    "HD": 278679,
    "IBM": 112350,
    "JPM": 658776,
    "KO": 26642,
    "MSFT": 21835,
    "NKE": 291981,
    "NVDA": 32307,
    "VZ": 415798,
}


def _has_existing_output(ticker: str) -> bool:
    out_dir = TRANSCRIPTS_DIR / ticker
    calls_csv = out_dir / f"{ticker.lower()}_earnings_calls.csv"
    return calls_csv.exists() and calls_csv.stat().st_size > 0


def main() -> None:
    script = Path(__file__).with_name("pull_wrds_earning_transcripts.py")
    for ticker, company_id in TICKER_COMPANY_IDS.items():
        if _has_existing_output(ticker):
            print(f"[skip] {ticker}: existing output found")
            continue
        cmd = [
            sys.executable,
            str(script),
            "--ticker",
            ticker,
            "--company-id",
            str(company_id),
            "--skip-availability-check",
        ]
        print(f"\n=== {ticker} (companyId={company_id}) ===")
        print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[warn] {ticker} failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
