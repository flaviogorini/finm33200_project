"""Fetch 10-Q filings from WRDS SEC Analytics Suite for one or more tickers.

Queries filing metadata from wrdssec_all.dforms, then downloads raw and
cleaned filing text via SFTP. The single combined metadata file
`_data/sec_10q/_meta/filing_index.csv` is the canonical input for the rest
of the 10-Q pipeline.
"""

from pathlib import Path

import pandas as pd
import paramiko
import wrds

from settings import (
    DEFAULT_TICKERS,
    SEC_10Q_DIR,
    SEC_10Q_END_DATE,
    SEC_10Q_META_DIR,
    SEC_10Q_START_DATE,
    TICKERS,
    USE_CACHE,
    WRDS_PASSWORD,
    WRDS_USERNAME,
    cik_for,
    clean_filings_dir,
    create_sec_10q_dirs,
    raw_filings_dir,
)

WRDS_RAW_BASE = "/wrds/sec/warchives"
WRDS_CLEAN_BASE = "/wrds/sec/wrds_clean_filings"
FORM_TYPE = "10-Q"

FILING_INDEX_PATH = SEC_10Q_META_DIR / "filing_index.csv"


def _rel_to_sec10q(absolute: Path) -> str:
    """Return `absolute` as a string path relative to SEC_10Q_DIR.

    Paths stored in `filing_index.csv` / `cleaned_index.csv` are relative
    so the indices are portable across machines and users. Call
    `resolve_sec10q_path()` (below) to get the absolute path back.
    """
    return str(Path(absolute).resolve().relative_to(SEC_10Q_DIR.resolve()))


def resolve_sec10q_path(rel_or_abs: str) -> Path:
    """Resolve a path stored in an index CSV back to an absolute path.

    Accepts either the new relative form (relative to SEC_10Q_DIR) or an
    already-absolute path (backwards compatibility for indices that
    pre-date the relative-path change).
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (SEC_10Q_DIR / p).resolve()


def edgar_to_wrds_path(fname: str) -> str:
    """Convert EDGAR filename paths into the WRDS mirror path convention."""
    parts = fname.split("/")
    cik = parts[2]
    filename = parts[3]
    cik_padded = cik.zfill(10)
    return f"{cik_padded[:6]}/{cik}/{filename}"


def pull_filing_metadata(tickers: list[str]) -> pd.DataFrame:
    if WRDS_USERNAME is None:
        raise RuntimeError("Set WRDS_USERNAME in .env or the environment.")

    # wrdssec_all.dforms.cik is VARCHAR(10) zero-padded (e.g. '0000320193'),
    # so we keep the padded form when querying and mapping back to ticker.
    cik_to_ticker = {cik_for(t): t for t in tickers}
    cik_list = ",".join(f"'{c}'" for c in cik_to_ticker)

    query = f"""
        SELECT cik, coname, form, fdate, secdate, fname
        FROM wrdssec_all.dforms
        WHERE form = '{FORM_TYPE}'
          AND cik IN ({cik_list})
          AND fdate >= '{SEC_10Q_START_DATE}'
          AND fdate <= '{SEC_10Q_END_DATE}'
          AND fname IS NOT NULL
        ORDER BY cik, fdate
    """
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    df = db.raw_sql(query, date_cols=["fdate", "secdate"])
    db.close()

    if df.empty:
        return df

    df = df.rename(columns={"fdate": "filing_date", "fname": "edgar_path"})
    df["cik"] = df["cik"].astype(str)
    df["ticker"] = df["cik"].map(cik_to_ticker)
    df["accession_number"] = df["edgar_path"].str.extract(r"([^/]+)\.txt$")[0]
    df["sec_url"] = df["edgar_path"].map(
        lambda x: f"https://www.sec.gov/Archives/{x}" if pd.notna(x) else None
    )
    df["wrds_relpath"] = df["edgar_path"].map(edgar_to_wrds_path)

    def _raw_path(row):
        return _rel_to_sec10q(raw_filings_dir(row["ticker"]) / row["wrds_relpath"])

    def _clean_path(row):
        return _rel_to_sec10q(clean_filings_dir(row["ticker"]) / row["wrds_relpath"])

    df["raw_local_path"] = df.apply(_raw_path, axis=1)
    df["clean_local_path"] = df.apply(_clean_path, axis=1)

    cols = [
        "ticker", "cik", "coname", "form",
        "filing_date", "secdate",
        "accession_number", "sec_url", "edgar_path", "wrds_relpath",
        "raw_local_path", "clean_local_path",
    ]
    return df[cols]


def get_sftp_connection():
    if WRDS_USERNAME is None or WRDS_PASSWORD is None:
        raise RuntimeError("Set WRDS_USERNAME and WRDS_PASSWORD in .env or the environment.")
    transport = paramiko.Transport(("wrds-cloud.wharton.upenn.edu", 22))
    transport.connect(username=WRDS_USERNAME, password=WRDS_PASSWORD)
    return transport, paramiko.SFTPClient.from_transport(transport)


def local_file_is_valid(path: Path, min_bytes: int = 100) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def download_one(sftp, edgar_path: str, local_path: Path, version: str) -> Path | None:
    relpath = edgar_to_wrds_path(edgar_path)
    remote_base = WRDS_RAW_BASE if version == "raw" else WRDS_CLEAN_BASE
    remote_path = f"{remote_base}/{relpath}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if USE_CACHE and local_file_is_valid(local_path):
        return local_path

    try:
        sftp.get(remote_path, str(local_path))
        if not local_file_is_valid(local_path):
            local_path.unlink(missing_ok=True)
            return None
        return local_path
    except Exception as exc:
        print(f"Warning: could not download {version} {edgar_path}: {exc}")
        local_path.unlink(missing_ok=True)
        return None


def fetch_10q_filings(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    unknown = [t for t in tickers if t not in TICKERS]
    if unknown:
        raise KeyError(f"Tickers not registered in settings.TICKERS: {unknown}")
    create_sec_10q_dirs(tickers)

    metadata = pull_filing_metadata(tickers)
    if metadata.empty:
        raise RuntimeError(f"No 10-Q filings found from WRDS for tickers={tickers}.")

    transport, sftp = get_sftp_connection()
    try:
        for i, row in metadata.iterrows():
            print(
                f"[{i + 1}/{len(metadata)}] {row['ticker']} "
                f"{row['filing_date'].date()} {row['accession_number']}"
            )
            download_one(sftp, row["edgar_path"], resolve_sec10q_path(row["raw_local_path"]), "raw")
            download_one(sftp, row["edgar_path"], resolve_sec10q_path(row["clean_local_path"]), "cleaned")
    finally:
        sftp.close()
        transport.close()

    SEC_10Q_META_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(FILING_INDEX_PATH, index=False)
    print(f"Wrote {FILING_INDEX_PATH}  ({len(metadata)} filings across {len(tickers)} tickers)")
    return metadata


def load_filing_index() -> pd.DataFrame:
    """Read the cached 10-Q filing index produced by `fetch_10q_filings()`."""
    return pd.read_csv(FILING_INDEX_PATH, parse_dates=["filing_date", "secdate"])


if __name__ == "__main__":
    fetch_10q_filings()
