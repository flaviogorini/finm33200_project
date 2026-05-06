"""Pull and clean WRDS Capital IQ earnings-call transcripts.

This script intentionally inspects WRDS PostgreSQL metadata before querying the
Capital IQ tables. WRDS table and column names are often lowercase, and access
can vary by subscription, so the extraction query is built only after the
expected tables and fields are found in ``information_schema``.

Default target:
    Apple Inc. / AAPL / Capital IQ companyId 24937, 2005-01-01 through
    2025-12-31, quarterly earnings calls only.

Run:
    python src/pull_wrds_earning_transcripts.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import wrds
from sqlalchemy.exc import ProgrammingError

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
WRDS_PASSWORD = config("WRDS_PASSWORD", default=None)
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

DEFAULT_TICKER = "AAPL"
DEFAULT_COMPANY_ID = 24937
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_END_DATE = "2025-12-31"
EARNINGS_CALL_EVENT_TYPE_ID = 48
FINAL_PRESENTATION_TYPE_ID = 5
PROOFED_COPY_COLLECTION_TYPE_ID = 1
EDITED_COPY_COLLECTION_TYPE_ID = 2

COMPONENT_OUTPUT_COLUMNS = [
    "ticker",
    "company_id",
    "company_name",
    "key_devid",
    "transcript_id",
    "event_date",
    "headline",
    "component_order",
    "transcript_component_type_id",
    "transcript_component_type_name",
    "transcript_person_id",
    "speaker_name",
    "speaker_company_name",
    "speaker_type_id",
    "component_text_clean",
    "transcript_creation_date_utc",
    "audio_length_sec",
]

CALL_OUTPUT_COLUMNS = [
    "ticker",
    "company_id",
    "company_name",
    "key_devid",
    "transcript_id",
    "event_date",
    "fiscal_year",
    "fiscal_quarter",
    "fiscal_year_from_headline",
    "fiscal_quarter_from_headline",
    "headline",
    "transcript_creation_date_utc",
    "audio_length_sec",
    "full_text",
    "word_count",
    "component_count",
]

TABLE_CANDIDATES = {
    "transcript": ["ciqtranscript"],
    "component": ["ciqtranscriptcomponent"],
    "person": ["ciqtranscriptperson"],
    "component_type": ["ciqtranscriptcomponenttype"],
    "collection_type": ["ciqtranscriptcollectiontype"],
    "presentation_type": ["ciqtranscriptpresentationtype"],
    "event": ["ciqevent", "ciqkeydev"],
    "event_object_type": [
        "ciqeventtoobjecttoeventtype",
        "ciqkeydevtoobjecttoeventtype",
    ],
    "event_type": ["ciqeventtype", "ciqkeydevcategorytype"],
    "company": ["ciqcompany"],
    "wrds_transcript_detail": ["wrds_transcript_detail"],
    "wrds_transcript_person": ["wrds_transcript_person"],
}

KEY_COLUMN_CANDIDATES = {
    "transcript": [
        "keydevid",
        "transcriptid",
        "transcriptcollectiontypeid",
        "transcriptpresentationtypeid",
        "transcriptcreationdateutc",
        "audiolengthsec",
    ],
    "component": [
        "transcriptcomponentid",
        "transcriptid",
        "componentorder",
        "transcriptcomponenttypeid",
        "transcriptpersonid",
        "componenttext",
    ],
    "person": [
        "transcriptpersonid",
        "transcriptpersonname",
        "proid",
        "companyname",
        "speakertypeid",
    ],
    "component_type": [
        "transcriptcomponenttypeid",
        "transcriptcomponenttypename",
    ],
    "collection_type": [
        "transcriptcollectiontypeid",
        "transcriptcollectiontypename",
    ],
    "presentation_type": [
        "transcriptpresentationtypeid",
        "transcriptpresentationtypename",
    ],
    "event": ["keydevid", "announceddateutc", "mostimportantdateutc", "headline"],
    "event_object_type": ["keydevid", "objectid", "keydeveventtypeid"],
    "event_type": ["keydeveventtypeid", "keydeveventtypename"],
    "company": ["companyid", "companyname"],
    "wrds_transcript_detail": [
        "companyid",
        "keydevid",
        "transcriptid",
        "headline",
        "mostimportantdateutc",
        "keydeveventtypeid",
        "keydeveventtypename",
        "companyname",
        "transcriptcollectiontypeid",
        "transcriptpresentationtypeid",
        "transcriptcreationdate_utc",
        "audiolengthsec",
    ],
    "wrds_transcript_person": [
        "transcriptid",
        "transcriptcomponentid",
        "componentorder",
        "transcriptcomponenttypeid",
        "transcriptcomponenttypename",
        "transcriptpersonid",
        "transcriptpersonname",
        "companyofperson",
        "speakertypeid",
    ],
}

EVENT_DATE_CANDIDATES = [
    "announceddateutc",
    "mostimportantdateutc",
    "eventdateutc",
    "keydeveventdateutc",
]
HEADLINE_CANDIDATES = ["headline", "eventheadline", "keydeveventheadline"]
NON_QUARTERLY_HEADLINE_PATTERNS = [
    r"Investor Day",
    r"Conference Presentation",
    r"M&A Call",
    r"Guidance\s*/?\s*Update Call",
    r"Special Call",
    r"Shareholder\s*/?\s*Analyst Call",
    r"Operating Results Call",
]
QUARTER_PATTERN = re.compile(r"\b(Q[1-4])\b", flags=re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class TableRef:
    """Actual WRDS table metadata discovered from information_schema."""

    logical_name: str
    schema_name: str
    table_name: str
    columns_by_lower: dict[str, str]

    @property
    def fqtn(self) -> str:
        return f"{quote_ident(self.schema_name)}.{quote_ident(self.table_name)}"

    def col(self, lower_name: str) -> str:
        return quote_ident(self.columns_by_lower[lower_name.lower()])

    def has_col(self, lower_name: str) -> bool:
        return lower_name.lower() in self.columns_by_lower


def quote_ident(identifier: str) -> str:
    """Safely quote a PostgreSQL identifier discovered from metadata."""
    return '"' + identifier.replace('"', '""') + '"'


def connect_wrds() -> wrds.Connection:
    """Connect to WRDS using credentials from .env, env vars, or .pgpass."""
    kwargs: dict[str, str] = {"wrds_username": WRDS_USERNAME}
    if WRDS_PASSWORD:
        kwargs["wrds_password"] = WRDS_PASSWORD
    return wrds.Connection(**kwargs)


def clean_component_text(value: Any) -> str:
    """Normalize whitespace without altering financial content."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def extract_fiscal_quarter(headline: Any) -> str | None:
    match = QUARTER_PATTERN.search(str(headline or ""))
    return match.group(1).upper() if match else None


def extract_fiscal_year(headline: Any) -> int | None:
    years = [
        int(match.group(1)) for match in YEAR_PATTERN.finditer(str(headline or ""))
    ]
    return years[0] if years else None


def build_component_line(row: pd.Series) -> str:
    component_type = row.get("transcript_component_type_name") or "Unknown Component"
    speaker = row.get("speaker_name") or "Unknown"
    text = row.get("component_text_clean") or ""
    return f"[{component_type}] {speaker}: {text}"


def inspect_ciq_schema(db: wrds.Connection) -> tuple[dict[str, TableRef], pd.DataFrame]:
    """Find the actual Capital IQ transcript/event tables and columns."""
    metadata_query = """
        SELECT table_schema, table_name, column_name, ordinal_position
        FROM information_schema.columns
        WHERE table_schema ILIKE '%%ciq%%'
           OR table_name ILIKE '%%transcript%%'
           OR table_name ILIKE '%%event%%'
           OR table_name ILIKE '%%company%%'
        ORDER BY table_schema, table_name, ordinal_position
    """
    columns = db.raw_sql(metadata_query)
    if columns.empty:
        raise RuntimeError("No Capital IQ/transcript/event/company metadata found.")

    table_refs: dict[str, TableRef] = {}
    for logical_name, candidates in TABLE_CANDIDATES.items():
        match = find_table(columns, candidates)
        if match is None:
            raise RuntimeError(
                f"Could not find WRDS table for {logical_name}. "
                f"Tried: {', '.join(candidates)}"
            )
        schema_name, table_name = match
        table_cols = columns[
            (columns["table_schema"] == schema_name)
            & (columns["table_name"] == table_name)
        ]["column_name"].tolist()
        ref = TableRef(
            logical_name=logical_name,
            schema_name=schema_name,
            table_name=table_name,
            columns_by_lower={col.lower(): col for col in table_cols},
        )
        validate_key_columns(ref, KEY_COLUMN_CANDIDATES[logical_name])
        table_refs[logical_name] = ref

    return table_refs, columns


def find_table(columns: pd.DataFrame, candidates: list[str]) -> tuple[str, str] | None:
    tables = (
        columns[["table_schema", "table_name"]]
        .drop_duplicates()
        .assign(table_name_lower=lambda df: df["table_name"].str.lower())
        .sort_values(["table_schema", "table_name"])
    )
    for candidate in candidates:
        matches = tables[tables["table_name_lower"] == candidate.lower()]
        if not matches.empty:
            ciq_matches = matches[matches["table_schema"].str.lower() == "ciq"]
            chosen = ciq_matches.iloc[0] if not ciq_matches.empty else matches.iloc[0]
            return str(chosen["table_schema"]), str(chosen["table_name"])
    return None


def validate_key_columns(ref: TableRef, expected_lower_columns: list[str]) -> None:
    missing = [
        column
        for column in expected_lower_columns
        if column not in ref.columns_by_lower
    ]
    if missing:
        raise RuntimeError(
            f"{ref.schema_name}.{ref.table_name} is missing expected columns: "
            f"{', '.join(missing)}"
        )


def choose_column(ref: TableRef, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if ref.has_col(candidate):
            return candidate
    raise RuntimeError(
        f"Could not find a usable {label} column in "
        f"{ref.schema_name}.{ref.table_name}. Tried: {', '.join(candidates)}"
    )


def print_schema_report(table_refs: dict[str, TableRef]) -> None:
    print("Actual WRDS schema/table names used:")
    for logical_name, ref in table_refs.items():
        key_cols = [
            ref.columns_by_lower[col]
            for col in KEY_COLUMN_CANDIDATES[logical_name]
            if col in ref.columns_by_lower
        ]
        print(
            f"  {logical_name}: {ref.schema_name}.{ref.table_name} "
            f"({', '.join(key_cols)})"
        )


def run_availability_checks(
    db: wrds.Connection,
    table_refs: dict[str, TableRef],
    company_id: int,
    start_date: str,
    end_date: str,
) -> None:
    """Print step-by-step Apple earnings-call transcript availability checks."""
    detail = table_refs["wrds_transcript_detail"]
    params = {
        "company_id": company_id,
        "event_type_id": EARNINGS_CALL_EVENT_TYPE_ID,
        "presentation_type_id": FINAL_PRESENTATION_TYPE_ID,
        "collection_type_id": PROOFED_COPY_COLLECTION_TYPE_ID,
        "start_date": start_date,
        "end_date": end_date,
        "earnings_call_pattern": "%Earnings Call%",
        "quarter_regex": r"\mQ[1-4]\M",
    }
    step_query = f"""
        WITH base AS (
            SELECT *,
                CASE
                    WHEN {detail.col("mostimportantdateutc")}::date
                         BETWEEN %(start_date)s AND %(end_date)s
                    THEN 1 ELSE 0
                END AS in_range,
                CASE
                    WHEN {detail.col("headline")} ILIKE %(earnings_call_pattern)s
                    THEN 1 ELSE 0
                END AS has_earnings_call,
                CASE
                    WHEN {detail.col("headline")} ~* %(quarter_regex)s
                    THEN 1 ELSE 0
                END AS has_q,
                CASE
                    WHEN {detail.col("transcriptpresentationtypeid")} = %(presentation_type_id)s
                    THEN 1 ELSE 0
                END AS is_final,
                CASE
                    WHEN {detail.col("transcriptcollectiontypeid")} = %(collection_type_id)s
                    THEN 1 ELSE 0
                END AS is_proofed
            FROM {detail.fqtn}
            WHERE {detail.col("companyid")} = %(company_id)s
              AND {detail.col("keydeveventtypeid")} = %(event_type_id)s
        )
        SELECT 'Step 1: Apple + keyDevEventTypeId = 48' AS step,
               COUNT(*) AS rows,
               COUNT(DISTINCT {detail.col("keydevid")}) AS key_devids,
               COUNT(DISTINCT {detail.col("transcriptid")}) AS transcript_ids
        FROM base
        UNION ALL
        SELECT 'Step 2: add event_date range', COUNT(*),
               COUNT(DISTINCT {detail.col("keydevid")}),
               COUNT(DISTINCT {detail.col("transcriptid")})
        FROM base
        WHERE in_range = 1
        UNION ALL
        SELECT 'Step 3: add headline Earnings Call', COUNT(*),
               COUNT(DISTINCT {detail.col("keydevid")}),
               COUNT(DISTINCT {detail.col("transcriptid")})
        FROM base
        WHERE in_range = 1 AND has_earnings_call = 1
        UNION ALL
        SELECT 'Step 4: add headline Q1-Q4', COUNT(*),
               COUNT(DISTINCT {detail.col("keydevid")}),
               COUNT(DISTINCT {detail.col("transcriptid")})
        FROM base
        WHERE in_range = 1 AND has_earnings_call = 1 AND has_q = 1
        UNION ALL
        SELECT 'Step 5: add Final presentation', COUNT(*),
               COUNT(DISTINCT {detail.col("keydevid")}),
               COUNT(DISTINCT {detail.col("transcriptid")})
        FROM base
        WHERE in_range = 1 AND has_earnings_call = 1 AND has_q = 1
          AND is_final = 1
        UNION ALL
        SELECT 'Step 6: add Proofed Copy', COUNT(*),
               COUNT(DISTINCT {detail.col("keydevid")}),
               COUNT(DISTINCT {detail.col("transcriptid")})
        FROM base
        WHERE in_range = 1 AND has_earnings_call = 1 AND has_q = 1
          AND is_final = 1 AND is_proofed = 1
        UNION ALL
        SELECT 'Step 7: latest per keyDevId within Final + Proofed', COUNT(*),
               COUNT(DISTINCT {detail.col("keydevid")}),
               COUNT(DISTINCT {detail.col("transcriptid")})
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY {detail.col("keydevid")}
                       ORDER BY
                           {detail.col("transcriptcreationdate_utc")} DESC NULLS LAST,
                           {detail.col("transcriptid")} DESC
                   ) AS rn
            FROM base
            WHERE in_range = 1 AND has_earnings_call = 1 AND has_q = 1
              AND is_final = 1 AND is_proofed = 1
        ) strict_latest
        WHERE rn = 1
    """
    print("\nAvailability check using inspected WRDS transcript detail view:")
    print(db.raw_sql(step_query, params=params).to_string(index=False))

    coverage_query = f"""
        SELECT
            DATE_PART('year', {detail.col("mostimportantdateutc")})::int AS event_year,
            SUBSTRING({detail.col("headline")} FROM '(Q[1-4])') AS fiscal_quarter,
            COUNT(*) AS rows,
            COUNT(DISTINCT {detail.col("keydevid")}) AS key_devids,
            COUNT(DISTINCT {detail.col("transcriptid")}) AS transcript_ids
        FROM {detail.fqtn}
        WHERE {detail.col("companyid")} = %(company_id)s
          AND {detail.col("keydeveventtypeid")} = %(event_type_id)s
          AND {detail.col("mostimportantdateutc")}::date
              BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    print("\nApple keyDevEventTypeId=48 transcript availability by event year/quarter:")
    print(db.raw_sql(coverage_query, params=params).to_string(index=False))

    no_proofed_query = f"""
        WITH final_events AS (
            SELECT
                {detail.col("keydevid")} AS key_devid,
                MIN({detail.col("mostimportantdateutc")}::date) AS event_date,
                MIN({detail.col("headline")}) AS headline,
                BOOL_OR(
                    {detail.col("transcriptcollectiontypeid")} = %(collection_type_id)s
                ) AS has_final_proofed,
                STRING_AGG(
                    DISTINCT
                    {detail.col("transcriptcollectiontypeid")}::text || ': ' ||
                    {detail.col("transcriptcollectiontypename")},
                    '; '
                    ORDER BY
                    {detail.col("transcriptcollectiontypeid")}::text || ': ' ||
                    {detail.col("transcriptcollectiontypename")}
                ) AS available_final_collection_types
            FROM {detail.fqtn}
            WHERE {detail.col("companyid")} = %(company_id)s
              AND {detail.col("keydeveventtypeid")} = %(event_type_id)s
              AND {detail.col("transcriptpresentationtypeid")} = %(presentation_type_id)s
              AND {detail.col("mostimportantdateutc")}::date
                  BETWEEN %(start_date)s AND %(end_date)s
              AND {detail.col("headline")} ILIKE %(earnings_call_pattern)s
              AND {detail.col("headline")} ~* %(quarter_regex)s
            GROUP BY {detail.col("keydevid")}
        )
        SELECT key_devid, event_date, headline, available_final_collection_types
        FROM final_events
        WHERE NOT has_final_proofed
        ORDER BY event_date, key_devid
    """
    no_proofed = db.raw_sql(no_proofed_query, params=params)
    if no_proofed.empty:
        print("\nEvery Final Apple quarterly earnings call has a Proofed Copy.")
    else:
        print(
            "\nFinal Apple quarterly earnings-call events without a Final + "
            "Proofed Copy transcript:"
        )
        print(no_proofed.to_string(index=False))


def build_latest_calls_query(
    table_refs: dict[str, TableRef],
    event_date_col: str,
    headline_col: str,
    require_proofed_copy: bool,
) -> str:
    t = table_refs["transcript"]
    e = table_refs["event"]
    ete = table_refs["event_object_type"]
    et = table_refs["event_type"]
    comp = table_refs["company"]
    ct = table_refs["collection_type"]
    pt = table_refs["presentation_type"]
    proofed_filter = (
        f"AND t.{t.col('transcriptcollectiontypeid')} = %(collection_type_id)s"
        if require_proofed_copy
        else ""
    )
    return f"""
        WITH candidate_calls AS (
            SELECT
                t.{t.col("keydevid")} AS key_devid,
                t.{t.col("transcriptid")} AS transcript_id,
                t.{t.col("transcriptcreationdateutc")} AS transcript_creation_date_utc,
                t.{t.col("audiolengthsec")} AS audio_length_sec,
                t.{t.col("transcriptpresentationtypeid")} AS transcript_presentation_type_id,
                pt.{pt.col("transcriptpresentationtypename")} AS transcript_presentation_type_name,
                t.{t.col("transcriptcollectiontypeid")} AS transcript_collection_type_id,
                ct.{ct.col("transcriptcollectiontypename")} AS transcript_collection_type_name,
                e.{e.col(event_date_col)} AS event_date,
                e.{e.col(headline_col)} AS headline,
                comp.{comp.col("companyid")} AS company_id,
                comp.{comp.col("companyname")} AS company_name,
                et.{et.col("keydeveventtypeid")} AS key_dev_event_type_id,
                ROW_NUMBER() OVER (
                    PARTITION BY t.{t.col("keydevid")}
                    ORDER BY
                        CASE
                            WHEN t.{t.col("transcriptcollectiontypeid")} = %(collection_type_id)s
                            THEN 0 ELSE 1
                        END,
                        t.{t.col("transcriptcreationdateutc")} DESC NULLS LAST,
                        t.{t.col("transcriptid")} DESC
                ) AS rn
            FROM {t.fqtn} AS t
            JOIN {e.fqtn} AS e
              ON e.{e.col("keydevid")} = t.{t.col("keydevid")}
            JOIN {ete.fqtn} AS ete
              ON ete.{ete.col("keydevid")} = t.{t.col("keydevid")}
            JOIN {comp.fqtn} AS comp
              ON comp.{comp.col("companyid")} = ete.{ete.col("objectid")}
            JOIN {et.fqtn} AS et
              ON et.{et.col("keydeveventtypeid")} = ete.{ete.col("keydeveventtypeid")}
            LEFT JOIN {ct.fqtn} AS ct
              ON ct.{ct.col("transcriptcollectiontypeid")} = t.{t.col("transcriptcollectiontypeid")}
            LEFT JOIN {pt.fqtn} AS pt
              ON pt.{pt.col("transcriptpresentationtypeid")} = t.{t.col("transcriptpresentationtypeid")}
            WHERE comp.{comp.col("companyid")} = %(company_id)s
              AND et.{et.col("keydeveventtypeid")} = %(event_type_id)s
              AND t.{t.col("transcriptpresentationtypeid")} = %(presentation_type_id)s
              {proofed_filter}
              AND e.{e.col(event_date_col)}::date BETWEEN %(start_date)s AND %(end_date)s
              AND e.{e.col(headline_col)} ILIKE %(earnings_call_pattern)s
              AND e.{e.col(headline_col)} ~* %(quarter_regex)s
              AND NOT (
                  e.{e.col(headline_col)} ~* %(exclude_regex)s
              )
        )
        SELECT *
        FROM candidate_calls
        WHERE rn = 1
        ORDER BY event_date, key_devid
    """


def build_component_query(
    table_refs: dict[str, TableRef],
    event_date_col: str,
    headline_col: str,
    require_proofed_copy: bool,
) -> str:
    latest_calls_query = build_latest_calls_query(
        table_refs, event_date_col, headline_col, require_proofed_copy
    )
    c = table_refs["component"]
    p = table_refs["person"]
    ct = table_refs["component_type"]
    return f"""
        WITH latest_calls AS (
            {latest_calls_query}
        )
        SELECT
            latest_calls.key_devid,
            latest_calls.transcript_id,
            latest_calls.transcript_creation_date_utc,
            latest_calls.audio_length_sec,
            latest_calls.transcript_presentation_type_id,
            latest_calls.transcript_presentation_type_name,
            latest_calls.transcript_collection_type_id,
            latest_calls.transcript_collection_type_name,
            latest_calls.event_date,
            latest_calls.headline,
            latest_calls.company_id,
            latest_calls.company_name,
            c.{c.col("componentorder")} AS component_order,
            c.{c.col("transcriptcomponenttypeid")} AS transcript_component_type_id,
            ct.{ct.col("transcriptcomponenttypename")} AS transcript_component_type_name,
            c.{c.col("transcriptpersonid")} AS transcript_person_id,
            p.{p.col("transcriptpersonname")} AS speaker_name,
            p.{p.col("companyname")} AS speaker_company_name,
            p.{p.col("speakertypeid")} AS speaker_type_id,
            c.{c.col("componenttext")} AS component_text
        FROM latest_calls
        JOIN {c.fqtn} AS c
          ON c.{c.col("transcriptid")} = latest_calls.transcript_id
        LEFT JOIN {p.fqtn} AS p
          ON p.{p.col("transcriptpersonid")} = c.{c.col("transcriptpersonid")}
        LEFT JOIN {ct.fqtn} AS ct
          ON ct.{ct.col("transcriptcomponenttypeid")} = c.{c.col("transcriptcomponenttypeid")}
        ORDER BY latest_calls.event_date, latest_calls.key_devid,
                 latest_calls.transcript_id, c.{c.col("componentorder")}
    """


def query_components(
    db: wrds.Connection,
    table_refs: dict[str, TableRef],
    company_id: int,
    start_date: str,
    end_date: str,
    require_proofed_copy: bool,
) -> pd.DataFrame:
    event_ref = table_refs["event"]
    event_date_col = choose_column(event_ref, EVENT_DATE_CANDIDATES, "event date")
    headline_col = choose_column(event_ref, HEADLINE_CANDIDATES, "headline")
    query = build_component_query(
        table_refs, event_date_col, headline_col, require_proofed_copy
    )
    exclude_regex = "|".join(
        f"({pattern})" for pattern in NON_QUARTERLY_HEADLINE_PATTERNS
    )
    params = {
        "company_id": company_id,
        "event_type_id": EARNINGS_CALL_EVENT_TYPE_ID,
        "presentation_type_id": FINAL_PRESENTATION_TYPE_ID,
        "collection_type_id": PROOFED_COPY_COLLECTION_TYPE_ID,
        "start_date": start_date,
        "end_date": end_date,
        "earnings_call_pattern": "%Earnings Call%",
        "quarter_regex": r"\mQ[1-4]\M",
        "exclude_regex": exclude_regex,
    }
    try:
        return db.raw_sql(query, params=params)
    except ProgrammingError as error:
        if "permission denied" not in str(error).lower():
            raise
        print(
            "\nBase Capital IQ event tables are visible in metadata but not "
            "selectable with this WRDS account. Falling back to inspected WRDS "
            "Capital IQ helper views."
        )
        return query_components_from_wrds_helpers(
            db=db,
            table_refs=table_refs,
            company_id=company_id,
            start_date=start_date,
            end_date=end_date,
            require_proofed_copy=require_proofed_copy,
        )


def query_components_from_wrds_helpers(
    db: wrds.Connection,
    table_refs: dict[str, TableRef],
    company_id: int,
    start_date: str,
    end_date: str,
    require_proofed_copy: bool,
) -> pd.DataFrame:
    calls = query_latest_calls_from_wrds_helpers(
        db=db,
        table_refs=table_refs,
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        require_proofed_copy=require_proofed_copy,
    )
    if calls.empty:
        return empty_raw_component_frame()

    component_frames = []
    transcript_ids = [int(value) for value in calls["transcript_id"].dropna().unique()]
    for start in range(0, len(transcript_ids), 25):
        id_chunk = transcript_ids[start : start + 25]
        component_frames.append(
            query_component_chunk_from_wrds_helpers(
                db=db,
                table_refs=table_refs,
                calls=calls[calls["transcript_id"].isin(id_chunk)],
                transcript_ids=id_chunk,
            )
        )
    return pd.concat(component_frames, ignore_index=True)


def query_latest_calls_from_wrds_helpers(
    db: wrds.Connection,
    table_refs: dict[str, TableRef],
    company_id: int,
    start_date: str,
    end_date: str,
    require_proofed_copy: bool,
) -> pd.DataFrame:
    detail = table_refs["wrds_transcript_detail"]
    proofed_filter = (
        f"AND d.{detail.col('transcriptcollectiontypeid')} = %(collection_type_id)s"
        if require_proofed_copy
        else ""
    )
    exclude_regex = "|".join(
        f"({pattern})" for pattern in NON_QUARTERLY_HEADLINE_PATTERNS
    )
    query = f"""
        WITH candidate_calls AS (
            SELECT
                d.{detail.col("keydevid")} AS key_devid,
                d.{detail.col("transcriptid")} AS transcript_id,
                d.{detail.col("transcriptcreationdate_utc")} AS transcript_creation_date_utc,
                d.{detail.col("audiolengthsec")} AS audio_length_sec,
                d.{detail.col("transcriptpresentationtypeid")} AS transcript_presentation_type_id,
                d.{detail.col("transcriptpresentationtypename")} AS transcript_presentation_type_name,
                d.{detail.col("transcriptcollectiontypeid")} AS transcript_collection_type_id,
                d.{detail.col("transcriptcollectiontypename")} AS transcript_collection_type_name,
                d.{detail.col("mostimportantdateutc")} AS event_date,
                d.{detail.col("headline")} AS headline,
                d.{detail.col("companyid")} AS company_id,
                d.{detail.col("companyname")} AS company_name,
                d.{detail.col("keydeveventtypeid")} AS key_dev_event_type_id,
                ROW_NUMBER() OVER (
                    PARTITION BY d.{detail.col("keydevid")}
                    ORDER BY
                        CASE
                            WHEN d.{detail.col("transcriptcollectiontypeid")} = %(collection_type_id)s
                            THEN 0 ELSE 1
                        END,
                        d.{detail.col("transcriptcreationdate_utc")} DESC NULLS LAST,
                        d.{detail.col("transcriptid")} DESC
                ) AS rn
            FROM {detail.fqtn} AS d
            WHERE d.{detail.col("companyid")} = %(company_id)s
              AND d.{detail.col("keydeveventtypeid")} = %(event_type_id)s
              AND d.{detail.col("transcriptpresentationtypeid")} = %(presentation_type_id)s
              {proofed_filter}
              AND d.{detail.col("mostimportantdateutc")}::date BETWEEN %(start_date)s AND %(end_date)s
              AND d.{detail.col("headline")} ILIKE %(earnings_call_pattern)s
              AND d.{detail.col("headline")} ~* %(quarter_regex)s
              AND NOT (
                  d.{detail.col("headline")} ~* %(exclude_regex)s
              )
        ),
        latest_calls AS (
            SELECT *
            FROM candidate_calls
            WHERE rn = 1
        )
        SELECT *
        FROM latest_calls
        ORDER BY event_date, key_devid
    """
    params = {
        "company_id": company_id,
        "event_type_id": EARNINGS_CALL_EVENT_TYPE_ID,
        "presentation_type_id": FINAL_PRESENTATION_TYPE_ID,
        "collection_type_id": PROOFED_COPY_COLLECTION_TYPE_ID,
        "start_date": start_date,
        "end_date": end_date,
        "earnings_call_pattern": "%Earnings Call%",
        "quarter_regex": r"\mQ[1-4]\M",
        "exclude_regex": exclude_regex,
    }
    return db.raw_sql(query, params=params)


def query_component_chunk_from_wrds_helpers(
    db: wrds.Connection,
    table_refs: dict[str, TableRef],
    calls: pd.DataFrame,
    transcript_ids: list[int],
) -> pd.DataFrame:
    person = table_refs["wrds_transcript_person"]
    component = table_refs["component"]
    id_list = ", ".join(str(transcript_id) for transcript_id in transcript_ids)
    query = f"""
        SELECT
            p.{person.col("transcriptid")} AS transcript_id,
            p.{person.col("componentorder")} AS component_order,
            p.{person.col("transcriptcomponenttypeid")} AS transcript_component_type_id,
            p.{person.col("transcriptcomponenttypename")} AS transcript_component_type_name,
            p.{person.col("transcriptpersonid")} AS transcript_person_id,
            p.{person.col("transcriptpersonname")} AS speaker_name,
            p.{person.col("companyofperson")} AS speaker_company_name,
            p.{person.col("speakertypeid")} AS speaker_type_id,
            c.{component.col("componenttext")} AS component_text
        FROM {person.fqtn} AS p
        JOIN {component.fqtn} AS c
          ON c.{component.col("transcriptcomponentid")} = p.{person.col("transcriptcomponentid")}
        WHERE p.{person.col("transcriptid")} IN ({id_list})
        ORDER BY p.{person.col("transcriptid")}, p.{person.col("componentorder")}
    """
    components = db.raw_sql(query)
    return components.merge(calls.drop(columns=["rn"]), on="transcript_id", how="left")


def empty_raw_component_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "key_devid",
            "transcript_id",
            "transcript_creation_date_utc",
            "audio_length_sec",
            "transcript_presentation_type_id",
            "transcript_presentation_type_name",
            "transcript_collection_type_id",
            "transcript_collection_type_name",
            "event_date",
            "headline",
            "company_id",
            "company_name",
            "component_order",
            "transcript_component_type_id",
            "transcript_component_type_name",
            "transcript_person_id",
            "speaker_name",
            "speaker_company_name",
            "speaker_type_id",
            "component_text",
        ]
    )


def clean_components(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = df.copy()
    cleaned["ticker"] = ticker.upper()
    integer_columns = [
        "company_id",
        "key_devid",
        "transcript_id",
        "transcript_presentation_type_id",
        "transcript_collection_type_id",
        "component_order",
        "transcript_component_type_id",
        "transcript_person_id",
        "speaker_type_id",
    ]
    for column in integer_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype(
            "Int64"
        )
    cleaned["event_date"] = pd.to_datetime(cleaned["event_date"]).dt.date.astype(str)
    cleaned["component_text_clean"] = cleaned["component_text"].map(
        clean_component_text
    )
    cleaned = cleaned[cleaned["component_text_clean"] != ""].copy()

    internal_columns = COMPONENT_OUTPUT_COLUMNS + [
        "transcript_presentation_type_id",
        "transcript_presentation_type_name",
        "transcript_collection_type_id",
        "transcript_collection_type_name",
    ]
    return cleaned[internal_columns].sort_values(
        ["event_date", "key_devid", "transcript_id", "component_order"]
    )


def build_call_level(components: pd.DataFrame) -> pd.DataFrame:
    if components.empty:
        return components
    working = components.copy()
    working["line_text"] = working.apply(build_component_line, axis=1)
    grouped = (
        working.groupby(["key_devid", "transcript_id"], sort=True)
        .agg(
            ticker=("ticker", "first"),
            company_id=("company_id", "first"),
            company_name=("company_name", "first"),
            event_date=("event_date", "first"),
            headline=("headline", "first"),
            transcript_creation_date_utc=("transcript_creation_date_utc", "first"),
            audio_length_sec=("audio_length_sec", "first"),
            transcript_presentation_type_id=(
                "transcript_presentation_type_id",
                "first",
            ),
            transcript_presentation_type_name=(
                "transcript_presentation_type_name",
                "first",
            ),
            transcript_collection_type_id=("transcript_collection_type_id", "first"),
            transcript_collection_type_name=(
                "transcript_collection_type_name",
                "first",
            ),
            full_text=("line_text", "\n".join),
            component_count=("component_order", "count"),
        )
        .reset_index()
    )
    grouped["fiscal_year"] = grouped["headline"].map(extract_fiscal_year)
    grouped["fiscal_quarter"] = grouped["headline"].map(extract_fiscal_quarter)
    grouped["fiscal_year_from_headline"] = grouped["fiscal_year"]
    grouped["fiscal_quarter_from_headline"] = grouped["fiscal_quarter"]
    grouped["word_count"] = grouped["full_text"].str.split().str.len()

    internal_columns = CALL_OUTPUT_COLUMNS + [
        "transcript_presentation_type_id",
        "transcript_presentation_type_name",
        "transcript_collection_type_id",
        "transcript_collection_type_name",
    ]
    return grouped[internal_columns].sort_values(
        ["event_date", "key_devid", "transcript_id"]
    )


def write_outputs(
    components: pd.DataFrame,
    calls: pd.DataFrame,
    ticker: str,
    out_root: Path = TRANSCRIPTS_DIR,
) -> dict[str, Path]:
    ticker_dir = out_root / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    component_path = ticker_dir / f"{ticker.lower()}_transcript_components.csv"
    call_path = ticker_dir / f"{ticker.lower()}_earnings_calls.csv"
    jsonl_path = ticker_dir / f"{ticker.lower()}_earnings_calls_llm.jsonl"

    components[COMPONENT_OUTPUT_COLUMNS].to_csv(component_path, index=False)
    calls[CALL_OUTPUT_COLUMNS].to_csv(call_path, index=False)
    write_jsonl(calls, jsonl_path)

    return {
        "components": component_path,
        "calls": call_path,
        "jsonl": jsonl_path,
    }


def write_jsonl(calls: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in calls.itertuples(index=False):
            record = {
                "ticker": row.ticker,
                "company_id": int(row.company_id),
                "event_date": row.event_date,
                "fiscal_year": (
                    None if pd.isna(row.fiscal_year) else int(row.fiscal_year)
                ),
                "fiscal_quarter": row.fiscal_quarter,
                "headline": row.headline,
                "transcript_id": int(row.transcript_id),
                "text": row.full_text,
                "metadata": {
                    "source": "WRDS Capital IQ Transcripts",
                    "event_type": "Earnings Call",
                    "presentation_type": row.transcript_presentation_type_name,
                    "collection_type": row.transcript_collection_type_name,
                    "presentation_type_id": (
                        None
                        if pd.isna(row.transcript_presentation_type_id)
                        else int(row.transcript_presentation_type_id)
                    ),
                    "collection_type_id": (
                        None
                        if pd.isna(row.transcript_collection_type_id)
                        else int(row.transcript_collection_type_id)
                    ),
                    "key_devid": int(row.key_devid),
                    "company_name": row.company_name,
                },
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_and_report(
    calls: pd.DataFrame,
    components: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    call_count = len(calls)
    duplicate_key_devids = (
        int(calls["key_devid"].duplicated().sum()) if call_count else 0
    )
    empty_full_text = (
        int((calls["full_text"].str.strip() == "").sum()) if call_count else 0
    )

    print(f"Final Apple earnings-call count: {call_count}")
    if call_count == 0:
        return

    print("\nCounts by fiscal_year and fiscal_quarter:")
    count_table = (
        calls.groupby(["fiscal_year", "fiscal_quarter"], dropna=False)
        .size()
        .reset_index(name="call_count")
        .sort_values(["fiscal_year", "fiscal_quarter"])
    )
    print(count_table.to_string(index=False))

    print(f"\nDuplicate key_devid rows in call-level file: {duplicate_key_devids}")
    print(f"Calls with empty full_text: {empty_full_text}")
    print(f"Earliest event_date: {calls['event_date'].min()}")
    print(f"Latest event_date: {calls['event_date'].max()}")
    print(f"Component rows after cleaning: {len(components)}")

    collection_counts = (
        calls.groupby(
            ["transcript_collection_type_id", "transcript_collection_type_name"],
            dropna=False,
        )
        .size()
        .reset_index(name="call_count")
        .sort_values(
            ["transcript_collection_type_id", "transcript_collection_type_name"]
        )
    )
    print("\nSelected transcript collection types:")
    print(collection_counts.to_string(index=False))

    component_counts = (
        components.groupby(["key_devid", "transcript_id"])
        .size()
        .reset_index(name="component_count_from_components")
    )
    component_check = calls[["key_devid", "transcript_id", "component_count"]].merge(
        component_counts, on=["key_devid", "transcript_id"], how="left"
    )
    component_mismatches = component_check[
        component_check["component_count"]
        != component_check["component_count_from_components"]
    ]
    print(
        "Component count mismatches between call-level and component-level files: "
        f"{len(component_mismatches)}"
    )
    if not component_mismatches.empty:
        print(component_mismatches.to_string(index=False))

    print("\nSample calls:")
    sample = calls[
        [
            "event_date",
            "fiscal_year",
            "fiscal_quarter",
            "key_devid",
            "transcript_id",
            "headline",
            "word_count",
        ]
    ].head()
    print(sample.to_string(index=False))

    available_start = pd.to_datetime(calls["event_date"]).min().date()
    requested_start = pd.to_datetime(start_date).date()
    if available_start > requested_start:
        print(
            "\nNote: no qualifying WRDS Capital IQ transcript was returned before "
            f"{available_start}. The requested range starts at {requested_start}."
        )

    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    expected = pd.MultiIndex.from_product(
        [
            range(start_year, end_year + 1),
            ["Q1", "Q2", "Q3", "Q4"],
        ],
        names=["fiscal_year", "fiscal_quarter"],
    )
    observed = calls.set_index(["fiscal_year", "fiscal_quarter"]).index
    missing = expected.difference(observed)
    if len(missing) > 0:
        print("\nMissing fiscal year/quarter combinations within observed range:")
        missing_df = missing.to_frame(index=False)
        print(missing_df.to_string(index=False))
    else:
        print("\nNo missing fiscal year/quarter combinations within observed range.")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and clean WRDS Capital IQ earnings-call transcripts."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--company-id", type=int, default=DEFAULT_COMPANY_ID)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect WRDS schema/table/columns; do not extract transcripts.",
    )
    parser.add_argument(
        "--skip-availability-check",
        action="store_true",
        help="Skip the step-by-step WRDS availability report.",
    )
    parser.add_argument(
        "--require-proofed-copy",
        action="store_true",
        help=(
            "Require Final + Proofed Copy transcripts. By default the script "
            "uses Final transcripts and prefers Proofed Copy when available."
        ),
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    ticker = args.ticker.upper()

    db = connect_wrds()
    try:
        table_refs, _ = inspect_ciq_schema(db)
        print_schema_report(table_refs)
        if args.inspect_only:
            return
        if not args.skip_availability_check:
            run_availability_checks(
                db=db,
                table_refs=table_refs,
                company_id=args.company_id,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        components_raw = query_components(
            db=db,
            table_refs=table_refs,
            company_id=args.company_id,
            start_date=args.start_date,
            end_date=args.end_date,
            require_proofed_copy=args.require_proofed_copy,
        )
    finally:
        db.close()

    components = clean_components(components_raw, ticker=ticker)
    calls = build_call_level(components)
    paths = write_outputs(components=components, calls=calls, ticker=ticker)

    validate_and_report(
        calls=calls,
        components=components,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print("\nOutput files:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    print("LLM-ready JSONL generated: yes")


if __name__ == "__main__":
    main()
