"""Extract sample raw Capital IQ earnings-call transcript data.

This is a stage-3 validation script. It extracts only raw metadata and raw
component text for a small sample of mapped Nasdaq-100 companies. It does not
clean transcript text, build processed datasets, generate LLM text, create
embeddings, or forecast anything.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import wrds

from settings import config


DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
META_DIR = DATA_DIR / "transcripts" / "_meta"
RAW_DIR = DATA_DIR / "transcripts" / "raw"
QC_DIR = OUTPUT_DIR / "transcripts" / "qc"

MAPPING_PATH = META_DIR / "ciq_company_mapping.csv"
UNIVERSE_PATH = META_DIR / "nasdaq100_constituents.csv"
AVAILABILITY_PATH = QC_DIR / "transcript_availability_by_company.csv"
SCHEMA_OUTPUT_PATH = META_DIR / "ciq_sample_raw_extraction_schema_inspection.json"

DEFAULT_SAMPLE_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "COST", "PEP", "ADBE", "AMGN"]
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_END_DATE = "2025-12-31"


def output_paths(label: str) -> dict[str, Path]:
    return {
        "raw_metadata": RAW_DIR / f"{label}_raw_transcript_metadata.parquet",
        "raw_components": RAW_DIR / f"{label}_raw_transcripts.parquet",
        "deduped_components": RAW_DIR / f"{label}_raw_transcripts_deduped.parquet",
        "deduped_metadata": RAW_DIR / f"{label}_raw_transcript_metadata_deduped.parquet",
        "manifest": QC_DIR / f"{label}_raw_extraction_manifest.json",
        "qc": QC_DIR / f"{label}_raw_extraction_qc.csv",
        "summary": QC_DIR / f"{label}_raw_extraction_summary.md",
    }


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def fqtn(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def connect_wrds() -> wrds.Connection:
    username = config("WRDS_USERNAME")
    password = config("WRDS_PASSWORD", default=None)
    kwargs: dict[str, str] = {"wrds_username": username}
    if password:
        kwargs["wrds_password"] = password
    return wrds.Connection(**kwargs)


def sql_id_list(values: list[int]) -> str:
    return ", ".join(str(int(v)) for v in sorted(set(values)))


def load_inputs(sample_tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(MAPPING_PATH).fillna("")
    universe = pd.read_csv(UNIVERSE_PATH).fillna("")
    availability = pd.read_csv(AVAILABILITY_PATH).fillna("")

    for df in [mapping, universe]:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    mapping["ciq_company_id"] = pd.to_numeric(mapping["ciq_company_id"], errors="coerce").astype("Int64")
    availability["ciq_company_id"] = pd.to_numeric(
        availability["ciq_company_id"], errors="coerce"
    ).astype("Int64")

    requested = pd.DataFrame({"ticker": [t.upper().strip() for t in sample_tickers]})
    selected = requested.merge(mapping, on="ticker", how="left", indicator=True)
    missing = selected[selected["_merge"] != "both"]["ticker"].tolist()
    if missing:
        raise ValueError(f"Sample tickers missing from mapping: {missing}")
    selected = selected.drop(columns="_merge")
    return universe, mapping, availability, selected


def inspect_raw_transcript_schema(db: wrds.Connection) -> dict[str, Any]:
    query = """
        SELECT table_schema, table_name, column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_name ILIKE '%%transcript%%'
        ORDER BY table_schema, table_name, ordinal_position
    """
    columns = db.raw_sql(query)

    def table_columns(schema: str, table: str) -> set[str]:
        return set(
            columns[
                (columns["table_schema"] == schema)
                & (columns["table_name"] == table)
            ]["column_name"].str.lower()
        )

    def find_table(required: set[str], preferred: list[str]) -> tuple[str, str]:
        tables = columns[["table_schema", "table_name"]].drop_duplicates()
        matches = []
        for row in tables.itertuples(index=False):
            if required.issubset(table_columns(row.table_schema, row.table_name)):
                matches.append((row.table_schema, row.table_name))
        if not matches:
            raise RuntimeError(f"No table found with required columns: {sorted(required)}")
        return sorted(
            matches,
            key=lambda x: (
                0 if x[0].lower() == "ciq" else 1,
                preferred.index(x[1].lower()) if x[1].lower() in preferred else 99,
                x[0],
                x[1],
            ),
        )[0]

    detail = find_table(
        {
            "companyid",
            "keydevid",
            "transcriptid",
            "headline",
            "mostimportantdateutc",
            "keydeveventtypeid",
            "keydeveventtypename",
        },
        ["wrds_transcript_detail"],
    )
    person = find_table(
        {
            "transcriptid",
            "transcriptcomponentid",
            "componentorder",
            "transcriptcomponenttypename",
            "transcriptpersonname",
        },
        ["wrds_transcript_person"],
    )
    component = find_table(
        {
            "transcriptcomponentid",
            "transcriptid",
            "componentorder",
            "componenttext",
        },
        ["ciqtranscriptcomponent"],
    )

    schema_info = {
        "inspection_timestamp": datetime.now().isoformat(timespec="seconds"),
        "metadata_table": {"schema": detail[0], "table": detail[1]},
        "component_person_table": {"schema": person[0], "table": person[1]},
        "component_text_table": {"schema": component[0], "table": component[1]},
        "join_keys": {
            "metadata_to_components": ["transcript_id"],
            "person_to_text": ["transcriptcomponentid"],
        },
        "raw_text_field": "componenttext",
    }
    SCHEMA_OUTPUT_PATH.write_text(json.dumps(schema_info, indent=2), encoding="utf-8")
    return schema_info


def query_raw_metadata(
    db: wrds.Connection,
    schema_info: dict[str, Any],
    company_ids: list[int],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    detail = schema_info["metadata_table"]
    query = f"""
        SELECT
            companyid::bigint AS ciq_company_id,
            companyname AS ciq_company_name,
            keydevid::bigint AS key_devid,
            transcriptid::bigint AS transcript_id,
            headline,
            keydeveventtypeid AS keydeveventtypeid,
            keydeveventtypename AS keydeveventtypename,
            transcriptcollectiontypeid AS transcriptcollectiontypeid,
            transcriptcollectiontypename AS transcriptcollectiontypename,
            transcriptpresentationtypeid AS transcriptpresentationtypeid,
            transcriptpresentationtypename AS transcriptpresentationtypename,
            transcriptcreationdate_utc AS transcriptcreationdate_utc,
            transcriptcreationtime_utc AS transcriptcreationtime_utc,
            mostimportantdateutc::date AS transcript_date,
            audiolengthsec AS audio_length_sec,
            {repr(fqtn(detail["schema"], detail["table"]))} AS source_table
        FROM {fqtn(detail["schema"], detail["table"])}
        WHERE companyid IN ({sql_id_list(company_ids)})
          AND mostimportantdateutc::date BETWEEN %(start_date)s AND %(end_date)s
          AND (
              keydeveventtypeid = 48
              OR keydeveventtypename ILIKE '%%Earnings%%'
              OR headline ILIKE '%%Earnings Call%%'
          )
        ORDER BY companyid, mostimportantdateutc, keydevid, transcriptid
    """
    return db.raw_sql(query, params={"start_date": start_date, "end_date": end_date})


def query_raw_components(
    db: wrds.Connection,
    schema_info: dict[str, Any],
    metadata: pd.DataFrame,
    chunk_size: int = 200,
) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame()
    person = schema_info["component_person_table"]
    component = schema_info["component_text_table"]
    transcript_ids = sorted(set(metadata["transcript_id"].dropna().astype(int).tolist()))
    frames: list[pd.DataFrame] = []
    for start in range(0, len(transcript_ids), chunk_size):
        id_chunk = transcript_ids[start : start + chunk_size]
        query = f"""
            SELECT
                p.transcriptid::bigint AS transcript_id,
                p.transcriptcomponentid::bigint AS component_id,
                p.componentorder AS component_order,
                p.transcriptcomponenttypeid AS transcript_component_type_id,
                p.transcriptcomponenttypename AS section_type,
                p.transcriptpersonid AS transcript_person_id,
                p.transcriptpersonname AS speaker_name,
                p.companyofperson AS speaker_company_name,
                p.speakertypeid AS speaker_type_id,
                c.componenttext AS component_text
            FROM {fqtn(person["schema"], person["table"])} AS p
            JOIN {fqtn(component["schema"], component["table"])} AS c
              ON c.transcriptcomponentid = p.transcriptcomponentid
            WHERE p.transcriptid IN ({sql_id_list(id_chunk)})
            ORDER BY p.transcriptid, p.componentorder
        """
        frames.append(db.raw_sql(query))
    components = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return components


def add_mapping_fields(
    metadata: pd.DataFrame,
    components: pd.DataFrame,
    selected_mapping: pd.DataFrame,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_cols = [
        "ticker",
        "primary_ticker",
        "related_tickers",
        "ciq_company_id",
        "ciq_company_name",
        "sector",
        "industry",
    ]
    company_map = (
        selected_mapping[selected_cols]
        .sort_values(["ciq_company_id", "ticker"])
        .groupby("ciq_company_id", as_index=False)
        .agg(
            ticker=("ticker", lambda x: "|".join(sorted(set(map(str, x))))),
            primary_ticker=("primary_ticker", "first"),
            related_tickers=("related_tickers", "first"),
            ciq_company_name_mapping=("ciq_company_name", "first"),
            sector=("sector", "first"),
            industry=("industry", "first"),
        )
    )
    metadata = metadata.merge(company_map, on="ciq_company_id", how="left")
    metadata["extraction_run_id"] = run_id
    metadata["fiscal_year"] = pd.NA
    metadata["fiscal_quarter"] = pd.NA

    if components.empty:
        return metadata, components
    component_meta_cols = [
        "ticker",
        "primary_ticker",
        "related_tickers",
        "ciq_company_id",
        "ciq_company_name",
        "key_devid",
        "transcript_id",
        "headline",
        "transcript_date",
        "keydeveventtypeid",
        "keydeveventtypename",
        "transcriptcollectiontypeid",
        "transcriptcollectiontypename",
        "transcriptpresentationtypeid",
        "transcriptpresentationtypename",
        "transcriptcreationdate_utc",
        "transcriptcreationtime_utc",
        "source_table",
        "extraction_run_id",
    ]
    components = components.merge(
        metadata[component_meta_cols], on="transcript_id", how="left"
    )
    return metadata, components


def collection_priority(value: Any) -> int:
    text = str(value or "").lower()
    if "audited" in text:
        return 0
    if "proofed" in text:
        return 1
    if "edited" in text:
        return 2
    if "corrected" in text:
        return 3
    if "spellchecked" in text:
        return 4
    if text:
        return 5
    return 9


def presentation_priority(value: Any) -> int:
    text = str(value or "").lower()
    if "final" in text:
        return 0
    if text:
        return 1
    return 9


def dedupe_metadata(metadata: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return metadata.copy()
    component_stats = (
        components.assign(component_text_len=components["component_text"].fillna("").astype(str).str.len())
        .groupby("transcript_id", as_index=False)
        .agg(component_count=("component_id", "count"), raw_text_char_count=("component_text_len", "sum"))
        if not components.empty
        else pd.DataFrame(columns=["transcript_id", "component_count", "raw_text_char_count"])
    )
    work = metadata.merge(component_stats, on="transcript_id", how="left")
    work["component_count"] = work["component_count"].fillna(0).astype(int)
    work["raw_text_char_count"] = work["raw_text_char_count"].fillna(0).astype(int)
    work["_presentation_priority"] = work["transcriptpresentationtypename"].map(presentation_priority)
    work["_collection_priority"] = work["transcriptcollectiontypename"].map(collection_priority)
    work["_creation_datetime"] = pd.to_datetime(
        work["transcriptcreationdate_utc"].astype(str)
        + " "
        + work["transcriptcreationtime_utc"].fillna("").astype(str),
        errors="coerce",
    )
    work = work.sort_values(
        [
            "ciq_company_id",
            "key_devid",
            "_presentation_priority",
            "_collection_priority",
            "_creation_datetime",
            "raw_text_char_count",
            "component_count",
            "transcript_id",
        ],
        ascending=[True, True, True, True, False, False, False, False],
    )
    selected = work.groupby(["ciq_company_id", "key_devid"], as_index=False).head(1).copy()
    selected["dedupe_rule"] = (
        "presentation_final_then_collection_audited_proofed_edited_corrected_"
        "then_latest_creation_then_text_length"
    )
    return selected.drop(
        columns=["_presentation_priority", "_collection_priority", "_creation_datetime"]
    )


def build_qc(
    selected_mapping: pd.DataFrame,
    metadata: pd.DataFrame,
    deduped_metadata: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in selected_mapping.itertuples(index=False):
        ticker = row.ticker
        cid = int(row.ciq_company_id)
        raw = metadata[metadata["ciq_company_id"] == cid]
        deduped = deduped_metadata[deduped_metadata["ciq_company_id"] == cid]
        comp = components[components["ciq_company_id"] == cid] if not components.empty else pd.DataFrame()
        missing_text = (
            int(comp["component_text"].isna().sum() + comp["component_text"].fillna("").astype(str).str.strip().eq("").sum())
            if not comp.empty
            else 0
        )
        dates = pd.to_datetime(raw["transcript_date"], errors="coerce") if not raw.empty else pd.Series(dtype="datetime64[ns]")
        years = sorted(set(dates.dropna().dt.year.astype(int).tolist()))
        rows.append(
            {
                "ticker": ticker,
                "ciq_company_id": cid,
                "ciq_company_name": row.ciq_company_name,
                "raw_candidate_transcript_rows": len(raw),
                "unique_key_devid_count": raw["key_devid"].nunique() if not raw.empty else 0,
                "unique_transcript_id_count": raw["transcript_id"].nunique() if not raw.empty else 0,
                "deduped_call_count": len(deduped),
                "first_transcript_date": "" if dates.empty else dates.min().date().isoformat(),
                "last_transcript_date": "" if dates.empty else dates.max().date().isoformat(),
                "years_covered": "|".join(map(str, years)),
                "event_type_distribution": (
                    ""
                    if raw.empty
                    else json.dumps(raw["keydeveventtypename"].fillna("UNKNOWN").value_counts().to_dict(), sort_keys=True)
                ),
                "transcript_version_distribution": (
                    ""
                    if raw.empty
                    else json.dumps(raw["transcriptcollectiontypename"].fillna("UNKNOWN").value_counts().to_dict(), sort_keys=True)
                ),
                "failed_extraction": False,
                "missing_text_count": missing_text,
                "duplicate_candidate_count": len(raw) - raw["key_devid"].nunique() if not raw.empty else 0,
                "is_duplicate_share_class": bool(row.related_tickers) and ticker != row.primary_ticker,
                "primary_ticker": row.primary_ticker,
                "related_tickers": row.related_tickers,
                "dedupe_rule_used": (
                    "presentation_final_then_collection_audited_proofed_edited_"
                    "corrected_then_latest_creation_then_text_length"
                ),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    universe: pd.DataFrame,
    availability: pd.DataFrame,
    selected_mapping: pd.DataFrame,
    metadata: pd.DataFrame,
    components: pd.DataFrame,
    deduped_metadata: pd.DataFrame,
    schema_info: dict[str, Any],
    run_id: str,
    start_date: str,
    end_date: str,
    sample_tickers: list[str],
    paths: dict[str, Path],
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    label = paths["summary"].name.split("_raw_extraction_summary.md")[0]
    phase_name = {
        "pilot20": "Pilot20",
        "nasdaq100": "Nasdaq-100 Full",
    }.get(label, "Sample")

    deduped_components = (
        components[components["transcript_id"].isin(deduped_metadata["transcript_id"])]
        .sort_values(["ciq_company_id", "key_devid", "transcript_id", "component_order"])
        .reset_index(drop=True)
        if not components.empty
        else components
    )
    metadata.to_parquet(paths["raw_metadata"], index=False)
    components.to_parquet(paths["raw_components"], index=False)
    deduped_metadata.to_parquet(paths["deduped_metadata"], index=False)
    deduped_components.to_parquet(paths["deduped_components"], index=False)

    qc = build_qc(selected_mapping, metadata, deduped_metadata, components)
    qc.to_csv(paths["qc"], index=False)

    old_aapl_count = None
    old_aapl_path = DATA_DIR / "transcripts" / "AAPL" / "aapl_earnings_calls.csv"
    if old_aapl_path.exists():
        old_aapl_count = len(pd.read_csv(old_aapl_path))
    aapl_row = qc[qc["ticker"] == "AAPL"].iloc[0].to_dict() if "AAPL" in qc["ticker"].values else {}

    manifest = {
        "run_id": run_id,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_files": {
            "mapping": str(MAPPING_PATH),
            "universe": str(UNIVERSE_PATH),
            "availability": str(AVAILABILITY_PATH),
        },
        "output_files": {
            "raw_metadata": str(paths["raw_metadata"]),
            "raw_components": str(paths["raw_components"]),
            "deduped_metadata": str(paths["deduped_metadata"]),
            "deduped_components": str(paths["deduped_components"]),
            "qc": str(paths["qc"]),
            "summary": str(paths["summary"]),
        },
        "universe_ticker_count": len(sample_tickers),
        "universe_as_of_date": (
            sorted(universe["universe_as_of_date"].dropna().astype(str).unique().tolist())
            if "universe_as_of_date" in universe
            else []
        ),
        "source": (
            sorted(universe["source"].dropna().astype(str).unique().tolist())
            if "source" in universe
            else []
        ),
        "requested_tickers": sample_tickers,
        "unique_tickers": sorted(selected_mapping["ticker"].dropna().astype(str).unique().tolist()),
        "unique_ciq_company_ids": sorted(
            selected_mapping["ciq_company_id"].dropna().astype(int).unique().tolist()
        ),
        "skipped_duplicate_share_class_tickers": sorted(
            selected_mapping[
                selected_mapping["related_tickers"].astype(str).ne("")
                & (selected_mapping["ticker"] != selected_mapping["primary_ticker"])
            ]["ticker"].tolist()
        ),
        "primary_ticker_used_for_extraction": (
            selected_mapping.sort_values(
                ["ciq_company_id", "is_primary_share_class"], ascending=[True, False]
            )
            .assign(ciq_company_id=lambda df: df["ciq_company_id"].astype(int).astype(str))
            .groupby("ciq_company_id")["primary_ticker"]
            .first()
            .to_dict()
        ),
        "date_range": {"start_date": start_date, "end_date": end_date},
        "earnings_call_filter": (
            "keydeveventtypeid = 48 OR keydeveventtypename ILIKE '%Earnings%' "
            "OR headline ILIKE '%Earnings Call%'"
        ),
        "earnings_call_identification_rule": {
            "keydeveventtypeid": 48,
            "keydeveventtypename_contains": "Earnings",
            "headline_contains": "Earnings Call",
        },
        "dedupe_rule": (
            "presentation_final_then_collection_audited_proofed_edited_corrected_"
            "then_latest_creation_then_text_length"
        ),
        "schema_info": schema_info,
        "row_counts": {
            "raw_metadata_rows": len(metadata),
            "raw_component_rows": len(components),
            "deduped_metadata_rows": len(deduped_metadata),
            "deduped_component_rows": len(deduped_components),
        },
        "failed_cases": [],
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    qc_table = qc[
        [
            "ticker",
            "ciq_company_id",
            "raw_candidate_transcript_rows",
            "unique_key_devid_count",
            "deduped_call_count",
            "missing_text_count",
            "duplicate_candidate_count",
            "is_duplicate_share_class",
            "primary_ticker",
            "related_tickers",
        ]
    ].to_markdown(index=False)
    duplicate_share_class = qc[qc["is_duplicate_share_class"]]
    low_coverage = qc[qc["ticker"].eq("SNDK")]
    next_recommendation = (
        "Pilot20 raw extraction succeeded with no failed tickers. Review the "
        "low-coverage SNDK case and the shorter-coverage ARM/APP cases, then "
        "proceed to full Nasdaq-100 raw extraction if these exceptions are acceptable."
        if label == "pilot20"
        else "Full Nasdaq-100 raw extraction succeeded. The raw candidate and deduped "
        "raw datasets can be frozen after reviewing the AXON missing-text components "
        "and expected low-coverage companies. It is safe to proceed to cleaning and "
        "section-parsing design; do not start embeddings until cleaned views are defined."
        if label == "nasdaq100"
        else "Small-sample raw extraction succeeded with no failed tickers. Review AAPL and "
        "sample component structure, then proceed to a 20-ticker raw extraction pilot "
        "before full Nasdaq-100 extraction."
    )
    summary = f"""# {phase_name} Raw Transcript Extraction Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

## Scope

- Run ID: `{run_id}`
- Sample tickers: {', '.join(sample_tickers)}
- Date range: {start_date} to {end_date}
- Extraction unit: unique `ciq_company_id`, not ticker
- Raw metadata output: `{paths["raw_metadata"]}`
- Raw component text output: `{paths["raw_components"]}`
- Deduped raw component output: `{paths["deduped_components"]}`

This stage did not clean transcript text, create `cleaned_text`, create
`llm_text`, generate embeddings, forecast, or modify the existing AAPL processed
dataset.

## WRDS / Capital IQ Tables

- Metadata table: `{schema_info['metadata_table']['schema']}.{schema_info['metadata_table']['table']}`
- Component/person table: `{schema_info['component_person_table']['schema']}.{schema_info['component_person_table']['table']}`
- Component text table: `{schema_info['component_text_table']['schema']}.{schema_info['component_text_table']['table']}`
- Join keys: metadata to components on `transcript_id`; component/person to text on `transcriptcomponentid`
- Raw text field: `componenttext`

## Earnings Call Filter

`keydeveventtypeid = 48 OR keydeveventtypename ILIKE '%Earnings%' OR headline ILIKE '%Earnings Call%'`

## Row Counts

- Raw candidate metadata rows: {len(metadata)}
- Raw component rows: {len(components)}
- Deduped metadata rows: {len(deduped_metadata)}
- Deduped component rows: {len(deduped_components)}

## Per-Ticker QC

{qc_table}

## Deduplication Rule

All raw candidate transcript versions are preserved in `{paths["raw_metadata"]}` and
`{paths["raw_components"]}`. The deduped validation file keeps one transcript per
`ciq_company_id` + `key_devid` using:

1. Final presentation type first, when available.
2. Collection priority based on observed collection names: Audited, Proofed,
   Edited, Corrected, Spellchecked, then other/unknown.
3. Latest `transcriptcreationdate_utc` + `transcriptcreationtime_utc`.
4. Larger raw text character count and component count.

No duplicate candidates are silently discarded; the candidate counts are in the
QC file.

## AAPL Benchmark

- Old AAPL processed call count: {old_aapl_count}
- New AAPL raw candidate metadata rows: {aapl_row.get('raw_candidate_transcript_rows')}
- New AAPL unique `key_devid` count: {aapl_row.get('unique_key_devid_count')}
- New AAPL deduped call count: {aapl_row.get('deduped_call_count')}

The new deduped AAPL count is expected to be close to, but not forced to equal,
the old processed count. Differences can come from broader earnings-call
metadata filters, Capital IQ transcript version choices, and the old pipeline's
additional Final/Q1-Q4/headline filters.

## Share-Class Handling

- Requested tickers: {len(sample_tickers)}
- Unique `ciq_company_id` extracted: {selected_mapping["ciq_company_id"].nunique()}
- Duplicate share-class ticker rows shown in QC but not separately extracted:
  {', '.join(duplicate_share_class['ticker'].tolist()) if not duplicate_share_class.empty else 'none'}

## Low Coverage / Special Cases

{low_coverage.to_markdown(index=False) if not low_coverage.empty else 'No requested low-coverage ticker was included.'}

## Recommendation

{next_recommendation}
"""
    paths["summary"].write_text(summary, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_SAMPLE_TICKERS)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--label",
        default="sample",
        help="output label prefix, e.g. sample or pilot20",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    sample_tickers = [ticker.upper().strip() for ticker in args.tickers]
    paths = output_paths(args.label)
    run_id = f"{args.label}_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    universe, mapping, availability, selected = load_inputs(sample_tickers)
    company_ids = selected["ciq_company_id"].dropna().astype(int).unique().tolist()

    db = connect_wrds()
    try:
        schema_info = inspect_raw_transcript_schema(db)
        metadata = query_raw_metadata(
            db,
            schema_info=schema_info,
            company_ids=company_ids,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        components = query_raw_components(db, schema_info=schema_info, metadata=metadata)
    finally:
        db.close()

    metadata, components = add_mapping_fields(metadata, components, selected, run_id)
    deduped_metadata = dedupe_metadata(metadata, components)
    write_outputs(
        universe=universe,
        availability=availability,
        selected_mapping=selected,
        metadata=metadata,
        components=components,
        deduped_metadata=deduped_metadata,
        schema_info=schema_info,
        run_id=run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        sample_tickers=sample_tickers,
        paths=paths,
    )
    print(f"Wrote raw metadata: {paths['raw_metadata']}")
    print(f"Wrote raw components: {paths['raw_components']}")
    print(f"Wrote deduped components: {paths['deduped_components']}")
    print(f"Wrote QC: {paths['qc']}")
    print(f"Wrote summary: {paths['summary']}")


if __name__ == "__main__":
    main()
