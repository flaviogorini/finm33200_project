"""Pull Capital IQ earnings-call transcripts from WRDS by ticker.

Looks up the Capital IQ ``companyid`` for a given ticker via the
Compustat-CIQ bridge (``comp.security`` -> ``ciq.wrds_gvkey``), pulls all
transcript components for that company, and writes one markdown file per
call to ``{DATA_DIR}/transcripts/{TICKER}/{YYYY}Q{N}.md``.

Run as a CLI:

    python src/pull_transcripts.py --ticker AAPL --year 2015

Output files use calendar-quarter naming derived from the call date.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"


def ticker_to_company_id(ticker: str, db: wrds.Connection) -> int:
    """Resolve a ticker to its Capital IQ ``companyid``.

    Uses the WRDS-curated Compustat <-> Capital IQ bridge. ``tpci='0'``
    restricts ``comp.security`` to common stock to avoid dual-class duplicates.
    Raises ``ValueError`` if no match is found; warns and picks the first
    match if more than one company id maps to the ticker.
    """
    query = """
        SELECT DISTINCT g.companyid, c.conm
        FROM comp.security AS s
        JOIN comp.company AS c ON s.gvkey = c.gvkey
        JOIN ciq.wrds_gvkey AS g ON s.gvkey = g.gvkey
        WHERE UPPER(s.tic) = UPPER(%(ticker)s)
          AND s.tpci = '0'
    """
    df = db.raw_sql(query, params={"ticker": ticker})

    if df.empty:
        raise ValueError(
            f"Ticker {ticker!r} not found in Compustat-CIQ bridge "
            "(comp.security x ciq.wrds_gvkey)."
        )

    if len(df) > 1:
        candidates = ", ".join(
            f"{int(row.companyid)} ({row.conm})" for row in df.itertuples()
        )
        warnings.warn(
            f"Ticker {ticker!r} maps to multiple companyids: {candidates}. "
            f"Using the first ({int(df['companyid'].iloc[0])}).",
            stacklevel=2,
        )

    return int(df["companyid"].iloc[0])


def get_company_name(company_id: int, db: wrds.Connection) -> str:
    """Look up the Capital IQ company name for ``company_id``."""
    query = """
        SELECT companyname
        FROM ciq.ciqcompany
        WHERE companyid = %(cid)s
    """
    df = db.raw_sql(query, params={"cid": company_id})
    if df.empty:
        return "Unknown Company"
    return df["companyname"].iloc[0]


def pull_transcripts(
    company_id: int, year: int | None, db: wrds.Connection
) -> pd.DataFrame:
    """Fetch all transcript components for ``company_id``.

    Filters to ``keydeveventtypename = 'Earnings Calls'`` to drop investor
    conferences and similar non-earnings events. WRDS stores multiple
    revisions of each call (preliminary, final, reordered, ...) under
    different ``transcriptid`` values that share a ``mostimportantdateutc``;
    we keep only the latest revision (max ``transcriptid``) per call date.

    If ``year`` is provided, restrict to calls whose ``mostimportantdateutc``
    falls in that calendar year.
    """
    year_filter = (
        "AND date_part('year', mostimportantdateutc) = %(year)s" if year else ""
    )
    query = f"""
        SELECT a.transcriptid, a.companyid, a.companyname, a.mostimportantdateutc,
               a.keydeveventtypename,
               b.transcriptpersonname, b.speakertypename, b.componentorder,
               c.componenttext
        FROM (
            SELECT *
            FROM ciq.wrds_transcript_detail
            WHERE companyid = %(cid)s
              AND keydeveventtypename = 'Earnings Calls'
            {year_filter}
        ) AS a,
        ciq.wrds_transcript_person AS b,
        ciq.ciqtranscriptcomponent AS c
        WHERE a.transcriptid = b.transcriptid
          AND b.transcriptcomponentid = c.transcriptcomponentid
        ORDER BY a.transcriptid, b.componentorder
    """
    params = {"cid": company_id}
    if year:
        params["year"] = year
    df = db.raw_sql(query, params=params)
    if df.empty:
        return df

    latest_per_date = df.groupby("mostimportantdateutc")["transcriptid"].transform(
        "max"
    )
    return df[df["transcriptid"] == latest_per_date].reset_index(drop=True)


def _build_markdown(group: pd.DataFrame, company_name: str) -> str:
    """Format a single transcript group as markdown."""
    call_date = pd.to_datetime(group["mostimportantdateutc"].iloc[0])
    date_str = call_date.strftime("%Y-%m-%d")

    event_title = group["keydeveventtypename"].iloc[0]
    if not event_title or pd.isna(event_title):
        event_title = "Earnings Call"

    lines = [f"# Transcript of {company_name} - {event_title} ({date_str})", ""]
    prev_person = None
    for _, row in group.iterrows():
        person_name = row.get("transcriptpersonname", "Unknown")
        speaker_type = row.get("speakertypename", "")
        text = row.get("componenttext", "")

        if pd.isna(text) or not str(text).strip():
            continue

        speaker_label = (
            f"**{person_name} ({speaker_type}):**"
            if speaker_type
            else f"**{person_name}:**"
        )
        if person_name != prev_person:
            lines.append(f"{speaker_label} {text}")
        else:
            lines.append(text)
        prev_person = person_name

    lines.append("")
    return "\n".join(lines)


def _quarter_filename(call_date: pd.Timestamp) -> str:
    """Return ``YYYYQ{N}`` for the calendar quarter of ``call_date``."""
    quarter = ((call_date.month - 1) // 3) + 1
    return f"{call_date.year}Q{quarter}"


def write_transcripts(
    df: pd.DataFrame,
    ticker: str,
    company_name: str,
    out_root: Path = TRANSCRIPTS_DIR,
) -> list[Path]:
    """Write one markdown file per transcript under ``out_root/TICKER/``.

    Filenames are ``{YYYY}Q{N}.md`` from the call's calendar quarter; if two
    calls share a quarter, ``_2``, ``_3``, ... are appended.
    """
    ticker_dir = out_root / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    seen: dict[str, int] = {}

    for _, group in df.groupby("transcriptid", sort=True):
        group = group.sort_values("componentorder")
        call_date = pd.to_datetime(group["mostimportantdateutc"].iloc[0])
        base = _quarter_filename(call_date)

        seen[base] = seen.get(base, 0) + 1
        suffix = "" if seen[base] == 1 else f"_{seen[base]}"
        filepath = ticker_dir / f"{base}{suffix}.md"

        filepath.write_text(_build_markdown(group, company_name))
        paths.append(filepath)

    return paths


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download Capital IQ earnings-call transcripts from WRDS by ticker."
    )
    p.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help="Calendar year filter on the call date (optional; default: all years)",
    )
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    ticker = args.ticker.upper()

    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:
        company_id = ticker_to_company_id(ticker, db)
        company_name = get_company_name(company_id, db)
        print(
            f"Resolved {ticker} -> companyid={company_id} ({company_name}). Pulling transcripts..."
        )
        df = pull_transcripts(company_id, args.year, db)
    finally:
        db.close()

    print(f"Got {len(df)} component rows.")
    if df.empty:
        print(
            f"No transcripts found for {ticker} (companyid={company_id}, year={args.year})."
        )
        raise SystemExit(0)

    paths = write_transcripts(df, ticker, company_name)
    for p in paths:
        print(f"Saved: {p}")
    print("Done.")
