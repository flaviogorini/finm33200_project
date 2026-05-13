"""Enhanced mapping QC and transcript metadata availability checks.

This script is limited to pre-extraction validation. It reads the existing
Nasdaq-100 universe and Capital IQ company mapping, then queries WRDS Capital
IQ transcript metadata only. It does not download transcript component text,
clean transcripts, create processed datasets, or generate embeddings.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import wrds

from settings import config


DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
META_DIR = DATA_DIR / "transcripts" / "_meta"
QC_DIR = OUTPUT_DIR / "transcripts" / "qc"

UNIVERSE_PATH = META_DIR / "nasdaq100_constituents.csv"
MAPPING_PATH = META_DIR / "ciq_company_mapping.csv"
ENHANCED_QC_CSV = QC_DIR / "enhanced_company_mapping_qc.csv"
ENHANCED_QC_MD = QC_DIR / "enhanced_company_mapping_qc.md"
AVAILABILITY_CSV = QC_DIR / "transcript_availability_by_company.csv"
AVAILABILITY_MD = QC_DIR / "transcript_availability_summary.md"
AVAILABILITY_SCHEMA_JSON = META_DIR / "ciq_transcript_availability_schema_inspection.json"

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"
HIGH_CANDIDATE_COUNT_THRESHOLD = 25
LOW_COVERAGE_THRESHOLD = 20
EXPECTED_YEARS = set(range(2005, 2026))

SAMPLE_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "COST", "PEP", "ADBE", "AMGN"]


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def fqtn(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def normalise_name(value: Any) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value or ""))
    text = re.sub(
        r"\b(inc|inc\.|corp|corp\.|corporation|co|co\.|company|plc|ltd|limited|"
        r"holdings|holding|class|cl|ordinary|ads|adr|nv|sa|se)\b",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().upper()


def name_similarity(left: Any, right: Any) -> float:
    a = normalise_name(left)
    b = normalise_name(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def parse_candidate_count(notes: Any) -> int | None:
    match = re.search(r"candidate_count=(\d+)", str(notes or ""))
    return int(match.group(1)) if match else None


def connect_wrds() -> wrds.Connection:
    username = config("WRDS_USERNAME")
    password = config("WRDS_PASSWORD", default=None)
    kwargs: dict[str, str] = {"wrds_username": username}
    if password:
        kwargs["wrds_password"] = password
    return wrds.Connection(**kwargs)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(UNIVERSE_PATH).fillna("")
    mapping = pd.read_csv(MAPPING_PATH).fillna("")
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    mapping["ticker"] = mapping["ticker"].astype(str).str.upper().str.strip()
    mapping["ciq_company_id"] = pd.to_numeric(
        mapping["ciq_company_id"], errors="coerce"
    ).astype("Int64")
    return universe, mapping


def build_enhanced_mapping_qc(
    universe: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    ticker_counts = mapping["ticker"].value_counts()
    company_counts = mapping["ciq_company_id"].value_counts(dropna=True)
    related_company_nunique = (
        mapping[mapping["related_tickers"].astype(str) != ""]
        .groupby("related_tickers")["ciq_company_id"]
        .nunique(dropna=True)
    )
    related_split_groups = set(related_company_nunique[related_company_nunique > 1].index)

    rows: list[dict[str, Any]] = []
    for row in mapping.itertuples(index=False):
        record = row._asdict()
        ticker = record["ticker"]
        ciq_company_id = record["ciq_company_id"]
        related_tickers = str(record.get("related_tickers") or "")
        sim = name_similarity(record["company_name"], record["ciq_company_name"])
        candidate_count = parse_candidate_count(record.get("notes"))
        company_id_ticker_count = (
            int(company_counts.loc[ciq_company_id]) if pd.notna(ciq_company_id) else 0
        )
        flags: list[str] = []
        review_reasons: list[str] = []

        if ticker_counts.loc[ticker] > 1:
            flags.append("duplicate_ticker")
            review_reasons.append("ticker appears more than once")
        if company_id_ticker_count > 1 and not related_tickers:
            flags.append("company_id_multiple_tickers_without_related_group")
            review_reasons.append("company_id maps to multiple tickers without related_tickers")
        if related_tickers and related_tickers in related_split_groups:
            flags.append("related_tickers_split_across_company_ids")
            review_reasons.append("related ticker group maps to multiple company IDs")
        if sim < 0.60:
            flags.append("company_name_mismatch")
            review_reasons.append(f"low company-name similarity {sim:.3f}")
        if candidate_count is not None and candidate_count > HIGH_CANDIDATE_COUNT_THRESHOLD:
            flags.append("high_candidate_count")
        if record.get("match_method") == "ticker_exchange_match":
            exchange_check = "confirmed_by_wrds_ticker_exchange_match"
        else:
            exchange_check = "not_confirmed_by_ticker_exchange_match"
            flags.append("exchange_not_confirmed")
            review_reasons.append("match_method did not confirm exchange")

        recommended_status = (
            "needs_review"
            if any(flag in flags for flag in [
                "duplicate_ticker",
                "company_id_multiple_tickers_without_related_group",
                "related_tickers_split_across_company_ids",
                "company_name_mismatch",
                "exchange_not_confirmed",
            ])
            else "verified"
        )
        rows.append(
            {
                **record,
                "ticker_is_unique": ticker_counts.loc[ticker] == 1,
                "company_id_ticker_count": company_id_ticker_count,
                "is_multi_ticker_company_id": company_id_ticker_count > 1,
                "multi_ticker_is_declared_related_group": bool(related_tickers),
                "company_name_similarity": round(sim, 3),
                "candidate_count": candidate_count,
                "candidate_count_high": (
                    candidate_count is not None
                    and candidate_count > HIGH_CANDIDATE_COUNT_THRESHOLD
                ),
                "match_confidence_all_same": mapping["match_confidence"].nunique() == 1,
                "exchange_check": exchange_check,
                "qc_flags": "; ".join(flags),
                "manual_review_reasons": "; ".join(review_reasons),
                "recommended_mapping_status": recommended_status,
            }
        )
    return pd.DataFrame(rows)


def inspect_transcript_schema(db: wrds.Connection) -> dict[str, Any]:
    query = """
        SELECT table_schema, table_name, column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_name ILIKE '%%transcript%%'
           OR table_name ILIKE '%%keydev%%'
           OR table_name ILIKE '%%event%%'
        ORDER BY table_schema, table_name, ordinal_position
    """
    columns = db.raw_sql(query)
    if columns.empty:
        raise RuntimeError("No transcript/keydev/event metadata found.")

    def has_columns(schema: str, table: str, required: set[str]) -> bool:
        cols = set(
            columns[
                (columns["table_schema"] == schema)
                & (columns["table_name"] == table)
            ]["column_name"]
            .str.lower()
            .tolist()
        )
        return required.issubset(cols)

    tables = columns[["table_schema", "table_name"]].drop_duplicates()
    detail_required = {
        "companyid",
        "companyname",
        "transcriptid",
        "keydevid",
        "headline",
        "mostimportantdateutc",
        "keydeveventtypeid",
        "keydeveventtypename",
    }
    detail_matches = []
    for row in tables.itertuples(index=False):
        if has_columns(row.table_schema, row.table_name, detail_required):
            detail_matches.append((row.table_schema, row.table_name))
    if not detail_matches:
        raise RuntimeError("Could not find WRDS transcript detail metadata table.")
    detail_matches = sorted(
        detail_matches,
        key=lambda x: (
            0 if x[0].lower() == "ciq" else 1,
            0 if x[1].lower() == "wrds_transcript_detail" else 1,
            x[0],
            x[1],
        ),
    )
    schema, table = detail_matches[0]
    detail_columns = (
        columns[(columns["table_schema"] == schema) & (columns["table_name"] == table)]
        .sort_values("ordinal_position")["column_name"]
        .tolist()
    )
    event_type_columns = [
        col
        for col in detail_columns
        if any(token in col.lower() for token in ["eventtype", "event_type", "typename", "headline"])
    ]
    version_columns = [
        col
        for col in detail_columns
        if any(
            token in col.lower()
            for token in ["presentation", "collection", "status", "version", "creation"]
        )
    ]
    return {
        "inspection_timestamp": datetime.now().isoformat(timespec="seconds"),
        "transcript_detail": {"schema": schema, "table": table},
        "event_type_related_columns": event_type_columns,
        "version_status_related_columns": version_columns,
        "all_transcript_detail_columns": detail_columns,
    }


def sql_id_list(values: pd.Series) -> str:
    ids = sorted({int(v) for v in values.dropna().tolist()})
    return ", ".join(str(v) for v in ids)


def query_transcript_metadata(
    db: wrds.Connection, schema_info: dict[str, Any], company_ids: pd.Series
) -> pd.DataFrame:
    detail = schema_info["transcript_detail"]
    table = fqtn(detail["schema"], detail["table"])
    id_list = sql_id_list(company_ids)
    query = f"""
        SELECT
            companyid::bigint AS ciq_company_id,
            companyname AS ciq_company_name,
            keydevid::bigint AS key_devid,
            transcriptid::bigint AS transcript_id,
            mostimportantdateutc::date AS transcript_date,
            headline,
            keydeveventtypeid AS event_type_id,
            keydeveventtypename AS event_type_name,
            transcriptpresentationtypeid AS transcript_presentation_type_id,
            transcriptpresentationtypename AS transcript_presentation_type_name,
            transcriptcollectiontypeid AS transcript_collection_type_id,
            transcriptcollectiontypename AS transcript_collection_type_name,
            transcriptcreationdate_utc AS transcript_creation_date_utc,
            audiolengthsec AS audio_length_sec
        FROM {table}
        WHERE companyid IN ({id_list})
          AND mostimportantdateutc::date BETWEEN %(start_date)s AND %(end_date)s
    """
    return db.raw_sql(query, params={"start_date": START_DATE, "end_date": END_DATE})


def build_availability(
    mapping: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    company_map = (
        mapping.sort_values(["ciq_company_id", "is_primary_share_class"], ascending=[True, False])
        .groupby("ciq_company_id", dropna=True)
        .agg(
            ciq_company_name=("ciq_company_name", "first"),
            primary_ticker=("primary_ticker", "first"),
            related_tickers=("related_tickers", "first"),
            tickers=("ticker", lambda x: "|".join(sorted(set(map(str, x))))),
            sector=("sector", "first"),
            industry=("industry", "first"),
        )
        .reset_index()
    )
    if metadata.empty:
        meta = metadata.copy()
    else:
        meta = metadata.copy()
        meta["transcript_date"] = pd.to_datetime(meta["transcript_date"])
        meta["event_year"] = meta["transcript_date"].dt.year
        meta["headline_has_earnings_call"] = meta["headline"].fillna("").str.contains(
            "Earnings Call", case=False, regex=False
        )
        meta["headline_has_quarter"] = meta["headline"].fillna("").str.contains(
            r"\bQ[1-4]\b", case=False, regex=True
        )
        meta["is_earnings_call"] = (
            pd.to_numeric(meta["event_type_id"], errors="coerce").eq(48)
            | meta["event_type_name"].fillna("").str.contains(
                "Earnings", case=False, regex=False
            )
            | meta["headline_has_earnings_call"]
        )

    rows: list[dict[str, Any]] = []
    for row in company_map.itertuples(index=False):
        cid = int(row.ciq_company_id)
        grp = meta[meta["ciq_company_id"] == cid] if not meta.empty else pd.DataFrame()
        earn = grp[grp["is_earnings_call"]] if not grp.empty else pd.DataFrame()
        years = sorted(set(earn["event_year"].dropna().astype(int).tolist())) if not earn.empty else []
        missing_years = sorted(EXPECTED_YEARS.difference(years))
        duplicates = (
            int(earn.duplicated(["key_devid"]).sum())
            if not earn.empty and "key_devid" in earn
            else 0
        )
        if grp.empty:
            status = "no_transcripts"
        elif earn.empty:
            status = "needs_review"
        elif len(earn) < LOW_COVERAGE_THRESHOLD:
            status = "low_coverage"
        else:
            status = "ready"
        rows.append(
            {
                "ciq_company_id": cid,
                "ciq_company_name": row.ciq_company_name,
                "primary_ticker": row.primary_ticker,
                "related_tickers": row.related_tickers,
                "tickers": row.tickers,
                "sector": row.sector,
                "industry": row.industry,
                "number_of_transcripts": len(grp),
                "number_of_earnings_call_transcripts": len(earn),
                "first_transcript_date": (
                    "" if grp.empty else grp["transcript_date"].min().date().isoformat()
                ),
                "last_transcript_date": (
                    "" if grp.empty else grp["transcript_date"].max().date().isoformat()
                ),
                "years_covered": "|".join(map(str, years)),
                "years_covered_count": len(years),
                "missing_years": "|".join(map(str, missing_years)),
                "event_type_distribution": (
                    ""
                    if grp.empty
                    else json.dumps(
                        grp["event_type_name"].fillna("UNKNOWN").value_counts().to_dict(),
                        sort_keys=True,
                    )
                ),
                "transcript_version_distribution": (
                    ""
                    if grp.empty
                    else json.dumps(
                        grp["transcript_collection_type_name"]
                        .fillna("UNKNOWN")
                        .value_counts()
                        .to_dict(),
                        sort_keys=True,
                    )
                ),
                "duplicate_event_candidates": duplicates,
                "availability_status": status,
            }
        )
    availability = pd.DataFrame(rows)
    event_distribution = (
        meta["event_type_name"].fillna("UNKNOWN").value_counts().reset_index()
        if not meta.empty
        else pd.DataFrame(columns=["event_type_name", "count"])
    )
    version_distribution = (
        meta["transcript_collection_type_name"].fillna("UNKNOWN").value_counts().reset_index()
        if not meta.empty
        else pd.DataFrame(columns=["transcript_collection_type_name", "count"])
    )
    return availability, event_distribution, version_distribution


def write_enhanced_qc_report(qc: pd.DataFrame, universe: pd.DataFrame, mapping: pd.DataFrame) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    qc.to_csv(ENHANCED_QC_CSV, index=False)
    needs_review = qc[qc["recommended_mapping_status"] == "needs_review"]
    high_candidates = qc[qc["candidate_count_high"]]
    multi = qc[qc["is_multi_ticker_company_id"]][
        ["ticker", "company_name", "ciq_company_id", "ciq_company_name", "primary_ticker", "related_tickers", "qc_flags"]
    ].copy()
    if not multi.empty:
        multi["related_tickers"] = multi["related_tickers"].str.replace("|", "\\|", regex=False)
    match_conf_unique = sorted(map(str, mapping["match_confidence"].dropna().unique()))
    report = f"""# Enhanced Company Mapping QC

Generated: {datetime.now().isoformat(timespec="seconds")}

## Inputs

- Universe: `{UNIVERSE_PATH}`
- Mapping: `{MAPPING_PATH}`
- Universe as-of dates: {', '.join(sorted(map(str, universe['universe_as_of_date'].unique())))}
- Universe source: {universe['source'].iloc[0]}

## Summary

- Universe ticker rows: {len(universe)}
- Mapping rows: {len(mapping)}
- Unique tickers in mapping: {mapping['ticker'].nunique()}
- Unique `ciq_company_id`: {mapping['ciq_company_id'].nunique()}
- Company IDs with multiple tickers: {int((qc['company_id_ticker_count'] > 1).sum())} ticker rows
- Match methods: `{json.dumps(mapping['match_method'].value_counts().to_dict(), sort_keys=True)}`
- Match confidence values: {', '.join(match_conf_unique)}
- Rows with high candidate_count > {HIGH_CANDIDATE_COUNT_THRESHOLD}: {len(high_candidates)}
- Recommended needs_review rows: {len(needs_review)}

## Multi-Ticker Company IDs

{multi.to_markdown(index=False) if not multi.empty else 'None detected.'}

## High Candidate Count Rows

{high_candidates[['ticker', 'company_name', 'ciq_company_id', 'ciq_company_name', 'candidate_count', 'qc_flags']].to_markdown(index=False) if not high_candidates.empty else 'None.'}

## Needs Manual Review

{needs_review[['ticker', 'company_name', 'ciq_company_id', 'ciq_company_name', 'qc_flags', 'manual_review_reasons']].to_markdown(index=False) if not needs_review.empty else 'None.'}

## Interpretation

All mapping rows may share the same `match_method` and `match_confidence` because
the first-pass mapper selected WRDS Capital IQ ticker + exchange matches for every
current Nasdaq-100 ticker. This enhanced QC treats high `candidate_count` as a
monitoring flag, not an automatic failure, when the company name and declared
share-class grouping are otherwise consistent.
"""
    ENHANCED_QC_MD.write_text(report, encoding="utf-8")


def write_availability_report(
    availability: pd.DataFrame,
    schema_info: dict[str, Any],
    event_distribution: pd.DataFrame,
    version_distribution: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    availability.to_csv(AVAILABILITY_CSV, index=False)
    status_counts = availability["availability_status"].value_counts().to_dict()
    no_transcripts = availability.loc[
        availability["availability_status"] == "no_transcripts", "primary_ticker"
    ].tolist()
    low_coverage = availability.loc[
        availability["availability_status"] == "low_coverage", "primary_ticker"
    ].tolist()
    needs_review = availability.loc[
        availability["availability_status"] == "needs_review", "primary_ticker"
    ].tolist()
    sample_check = mapping[mapping["ticker"].isin(SAMPLE_TICKERS)][
        ["ticker", "ciq_company_id", "ciq_company_name", "mapping_status"]
    ].sort_values("ticker")
    ready_sample = availability[
        availability["primary_ticker"].isin(SAMPLE_TICKERS)
        | availability["tickers"].map(lambda x: any(t in str(x).split("|") for t in SAMPLE_TICKERS))
    ]
    report = f"""# Transcript Availability Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

## Scope

- Date range: {START_DATE} to {END_DATE}
- Input mapping: `{MAPPING_PATH}`
- Availability output: `{AVAILABILITY_CSV}`
- Unique `ciq_company_id` checked: {len(availability)}

This is a metadata-level check only. It did not download component text, clean
transcripts, build processed datasets, or create embeddings.

## Schema Inspection

- Transcript detail table: `{schema_info['transcript_detail']['schema']}.{schema_info['transcript_detail']['table']}`
- Event type related fields: {', '.join(schema_info['event_type_related_columns'])}
- Version/status related fields: {', '.join(schema_info['version_status_related_columns'])}

## Earnings Call Identification Rule

Earnings call metadata rows are identified when at least one of the following is true:

- `keydeveventtypeid = 48`
- `keydeveventtypename` contains `Earnings`
- `headline` contains `Earnings Call`

The headline condition is kept as an auxiliary guard because transcript metadata
can contain presentation/collection variants and event naming differences. The
main event-type fields available in the inspected table are listed above.

## Coverage Summary

- Companies with any transcript metadata: {int((availability['number_of_transcripts'] > 0).sum())}
- Companies with earnings call transcript metadata: {int((availability['number_of_earnings_call_transcripts'] > 0).sum())}
- Availability status counts: `{json.dumps(status_counts, sort_keys=True)}`
- Earliest transcript date: {availability.loc[availability['first_transcript_date'] != '', 'first_transcript_date'].min()}
- Latest transcript date: {availability.loc[availability['last_transcript_date'] != '', 'last_transcript_date'].max()}

## Exception Lists

- No transcripts: {', '.join(no_transcripts) if no_transcripts else 'none'}
- Low coverage: {', '.join(low_coverage) if low_coverage else 'none'}
- Needs review: {', '.join(needs_review) if needs_review else 'none'}

## Event Type Distribution

{event_distribution.to_markdown(index=False)}

## Transcript Version / Collection Distribution

{version_distribution.to_markdown(index=False)}

## Suggested Small Sample Check

Requested sample tickers:

{sample_check.to_markdown(index=False)}

Sample company availability:

{ready_sample[['primary_ticker', 'tickers', 'ciq_company_id', 'ciq_company_name', 'number_of_earnings_call_transcripts', 'years_covered_count', 'availability_status']].to_markdown(index=False)}

## Recommendation

Use `{AVAILABILITY_CSV}` to choose the next small raw extraction sample. Do not
run full Nasdaq-100 extraction before reviewing any `low_coverage` or
`needs_review` rows. For extraction, use unique `ciq_company_id`, not ticker.
"""
    AVAILABILITY_MD.write_text(report, encoding="utf-8")


def main() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    universe, mapping = load_inputs()
    enhanced_qc = build_enhanced_mapping_qc(universe, mapping)
    write_enhanced_qc_report(enhanced_qc, universe, mapping)

    db = connect_wrds()
    try:
        schema_info = inspect_transcript_schema(db)
        AVAILABILITY_SCHEMA_JSON.write_text(
            json.dumps(schema_info, indent=2), encoding="utf-8"
        )
        metadata = query_transcript_metadata(db, schema_info, mapping["ciq_company_id"])
    finally:
        db.close()

    availability, event_distribution, version_distribution = build_availability(
        mapping, metadata
    )
    write_availability_report(
        availability, schema_info, event_distribution, version_distribution, mapping
    )

    print(f"Wrote enhanced mapping QC: {ENHANCED_QC_CSV}")
    print(f"Wrote enhanced mapping QC report: {ENHANCED_QC_MD}")
    print(f"Wrote availability: {AVAILABILITY_CSV}")
    print(f"Wrote availability summary: {AVAILABILITY_MD}")
    print("Mapping recommended statuses:")
    print(enhanced_qc["recommended_mapping_status"].value_counts().to_string())
    print("Availability statuses:")
    print(availability["availability_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
