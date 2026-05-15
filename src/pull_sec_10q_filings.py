"""Fetch 10-Q filings directly from SEC EDGAR using edgartools.

EDGAR is the canonical source for SEC filings; WRDS just mirrors it. By
hitting EDGAR directly we eliminate the WRDS SFTP requirement entirely
(no SSH keys, no WRDS_PASSWORD), at the cost of a single SEC-mandated
contact identifier in the HTTP User-Agent (`SEC_EDGAR_USER_AGENT`).

The single combined metadata file `_data/sec_10q/_meta/filing_index.csv`
is preserved as the canonical input for the rest of the 10-Q pipeline
(`clean_sec_10q_text.py`, `score_sec_10q_text.py`,
`build_10q_monthly_panel.py`). Schema and on-disk layout are unchanged.
"""

from pathlib import Path

import pandas as pd
from edgar import Company, set_identity

from settings import (
    DEFAULT_TICKERS,
    SEC_10Q_DIR,
    SEC_10Q_END_DATE,
    SEC_10Q_META_DIR,
    SEC_10Q_START_DATE,
    SEC_EDGAR_USER_AGENT,
    TICKERS,
    USE_CACHE,
    cik_for,
    clean_filings_dir,
    create_sec_10q_dirs,
    raw_filings_dir,
)

FORM_TYPE = "10-Q"
FILING_INDEX_PATH = SEC_10Q_META_DIR / "filing_index.csv"


# ---- Path helpers (unchanged from WRDS-era; downstream scripts depend on these) ----


def _rel_to_sec10q(absolute: Path) -> str:
    """Return ``absolute`` as a string path relative to ``SEC_10Q_DIR``."""
    return str(Path(absolute).resolve().relative_to(SEC_10Q_DIR.resolve()))


def resolve_sec10q_path(rel_or_abs: str) -> Path:
    """Resolve a path stored in an index CSV back to an absolute path."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (SEC_10Q_DIR / p).resolve()


# ---- EDGAR access -------------------------------------------------------


_edgar_initialised = False


def _init_edgar() -> None:
    """Set the SEC-required User-Agent identifier exactly once per process."""
    global _edgar_initialised
    if not _edgar_initialised:
        set_identity(SEC_EDGAR_USER_AGENT)
        _edgar_initialised = True


def _filing_local_basename(accession_number: str) -> str:
    """Deterministic local filename: ``{accession}.htm``."""
    safe = str(accession_number).replace("/", "_")
    return f"{safe}.htm"


def pull_filing_metadata(tickers: list[str]) -> pd.DataFrame:
    """Hit EDGAR for 10-Q metadata in the same schema the WRDS path produced.

    Returns a DataFrame with the same columns the previous WRDS-routed
    version emitted, PLUS a private ``_filing`` column holding the
    edgartools ``Filing`` object (dropped before the CSV is written;
    used by ``download_filing`` so we don't re-query EDGAR per file).
    """
    _init_edgar()
    start = pd.Timestamp(SEC_10Q_START_DATE)
    end = pd.Timestamp(SEC_10Q_END_DATE)

    rows: list[dict] = []
    for ticker in tickers:
        cik_padded = cik_for(ticker)  # zero-padded 10-digit
        cik_int = int(cik_padded)
        company = Company(cik_int)
        coname = getattr(company, "name", ticker)

        filings = company.get_filings(form=FORM_TYPE)
        kept = 0
        for f in filings:
            # Strict 10-Q only — exclude 10-Q/A amendments. EDGAR's form
            # filter is prefix-based and would include /A amendments,
            # which have a different (much shorter) narrative shape and
            # break downstream cleaner-word-count assumptions.
            if str(getattr(f, "form", "")).upper() != FORM_TYPE:
                continue

            fdate = pd.Timestamp(f.filing_date)
            if not (start <= fdate <= end):
                continue

            basename = _filing_local_basename(f.accession_number)
            raw_path = _rel_to_sec10q(raw_filings_dir(ticker) / basename)
            clean_path = _rel_to_sec10q(clean_filings_dir(ticker) / basename)

            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik_padded,
                    "coname": coname,
                    "form": FORM_TYPE,
                    "filing_date": fdate,
                    "secdate": fdate,  # legacy alias
                    "accession_number": f.accession_number,
                    "sec_url": getattr(f, "homepage_url", "")
                    or getattr(f, "filing_url", ""),
                    "edgar_path": getattr(f, "primary_document_url", "")
                    or getattr(f, "homepage_url", ""),
                    "wrds_relpath": basename,  # legacy field; just the local filename
                    "raw_local_path": raw_path,
                    "clean_local_path": clean_path,
                    "_filing": f,
                }
            )
            kept += 1
        print(f"  {ticker}: {kept} 10-Qs in [{SEC_10Q_START_DATE}, {SEC_10Q_END_DATE}]")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    return df


# ---- Download -----------------------------------------------------------


def local_file_is_valid(path: Path, min_bytes: int = 100) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def download_filing(filing, raw_path: Path, clean_path: Path) -> bool:
    """Fetch the primary 10-Q document from EDGAR and write to both paths.

    EDGAR doesn't provide a separate "pre-cleaned" version, so both
    ``raw_local_path`` and ``clean_local_path`` get the same HTML —
    the downstream cleaner (``clean_sec_10q_text.py:clean_one_file``)
    runs ``html_to_text`` on ``clean_local_path`` itself. The two-path
    layout is kept for backwards compatibility with the WRDS-era
    ``filing_index.csv`` schema and the cleaner's existing column lookups.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    if USE_CACHE and local_file_is_valid(raw_path) and local_file_is_valid(clean_path):
        return True

    try:
        html = filing.html()
    except Exception as exc:
        print(f"  warn: EDGAR fetch failed for {filing.accession_number}: {exc}")
        return False

    if not html or len(html) < 100:
        print(f"  warn: empty/short payload for {filing.accession_number}")
        return False

    raw_path.write_text(html, encoding="utf-8")
    clean_path.write_text(html, encoding="utf-8")
    return True


def fetch_10q_filings(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    unknown = [t for t in tickers if t not in TICKERS]
    if unknown:
        raise KeyError(f"Tickers not registered in settings.TICKERS: {unknown}")
    create_sec_10q_dirs(tickers)

    metadata = pull_filing_metadata(tickers)
    if metadata.empty:
        raise RuntimeError(f"No 10-Q filings found from EDGAR for tickers={tickers}.")

    n = len(metadata)
    for i, row in metadata.iterrows():
        print(
            f"[{i + 1}/{n}] {row['ticker']} "
            f"{row['filing_date'].date()} {row['accession_number']}"
        )
        download_filing(
            row["_filing"],
            resolve_sec10q_path(row["raw_local_path"]),
            resolve_sec10q_path(row["clean_local_path"]),
        )

    SEC_10Q_META_DIR.mkdir(parents=True, exist_ok=True)
    out = metadata.drop(columns="_filing")
    out.to_csv(FILING_INDEX_PATH, index=False)
    print(f"Wrote {FILING_INDEX_PATH}  ({len(out)} filings across {len(tickers)} tickers)")
    return out


def load_filing_index() -> pd.DataFrame:
    """Read the cached 10-Q filing index produced by ``fetch_10q_filings()``."""
    return pd.read_csv(FILING_INDEX_PATH, parse_dates=["filing_date", "secdate"])


if __name__ == "__main__":
    fetch_10q_filings()
