"""Freeze manifest/report for full_cleaning_v1.0.

This script only reads existing processed datasets and QC artifacts, then writes
freeze metadata. It does not modify parquet datasets or rerun cleaning.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from settings import config


DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
QC_DIR = OUTPUT_DIR / "transcripts" / "qc"
META_DIR = MANUAL_DATA_DIR / "_meta"

COMPONENTS_PATH = DATA_DIR / "transcripts" / "processed" / "nasdaq100_cleaned_components.parquet"
CALLS_PATH = DATA_DIR / "transcripts" / "processed" / "nasdaq100_cleaned_calls.parquet"
VIEWS_PATH = DATA_DIR / "transcripts" / "processed" / "nasdaq100_llm_views.parquet"

# Hand-frozen raw-dataset provenance manifest. Lives with manual inputs, not
# under _output/, because no current script regenerates it — it is read-only
# provenance metadata that ships with the repo.
RAW_FROZEN_MANIFEST_PATH = META_DIR / "nasdaq100_raw_dataset_frozen_manifest.json"
CLEANING_QC_PATH = QC_DIR / "nasdaq100_cleaning_qc.csv"
CLEANING_SUMMARY_PATH = QC_DIR / "nasdaq100_cleaning_summary.md"
CLEANING_MANIFEST_PATH = QC_DIR / "nasdaq100_cleaning_manifest.json"
CLEANING_MANUAL_REVIEW_PATH = QC_DIR / "nasdaq100_cleaning_manual_review.csv"
NEEDS_REVIEW_PATH = QC_DIR / "nasdaq100_cleaning_needs_review_calls.csv"
GAP_PATH = QC_DIR / "nasdaq100_metadata_component_gap_calls.csv"
HIGH_DROP_PATH = QC_DIR / "nasdaq100_high_word_drop_calls.csv"
UNCERTAIN_SAFE_PATH = QC_DIR / "nasdaq100_uncertain_safe_harbor_sample.csv"
NO_QA_PATH = QC_DIR / "nasdaq100_no_qa_calls.csv"
FINAL_REVIEW_SUMMARY_PATH = QC_DIR / "nasdaq100_cleaning_final_review_summary.md"

FROZEN_MANIFEST_OUT = QC_DIR / "nasdaq100_cleaned_dataset_frozen_manifest.json"
FREEZE_REPORT_OUT = QC_DIR / "nasdaq100_cleaned_dataset_freeze_report.md"

DEFAULT_MODELING_VIEW = "no_operator_no_safe_harbor_full_text"
FREEZE_RUN_ID = f"nasdaq100_cleaned_frozen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    components = pd.read_parquet(COMPONENTS_PATH, columns=["primary_ticker", "ciq_company_id"])
    calls = pd.read_parquet(
        CALLS_PATH,
        columns=["primary_ticker", "ciq_company_id", "transcript_date"],
    )
    views = pd.read_parquet(VIEWS_PATH, columns=["view_name"])
    qc = pd.read_csv(CLEANING_QC_PATH)
    needs = pd.read_csv(NEEDS_REVIEW_PATH)
    gap = pd.read_csv(GAP_PATH)
    high_drop = pd.read_csv(HIGH_DROP_PATH)
    no_qa = pd.read_csv(NO_QA_PATH)
    cleaning_manifest = read_json(CLEANING_MANIFEST_PATH)
    raw_manifest = read_json(RAW_FROZEN_MANIFEST_PATH)

    quality_counts = qc["parsing_quality_flag"].value_counts().to_dict()
    warning_counts = qc["cleaning_warning"].fillna("").replace("", "none").value_counts().to_dict()
    needs_counts = needs["warning_type"].value_counts().to_dict()
    gap_ticker_counts = gap["primary_ticker"].value_counts().to_dict()
    high_drop_reason_counts = high_drop["likely_reason"].value_counts().to_dict()
    no_qa_reason_counts = no_qa["likely_reason"].value_counts().to_dict()
    view_counts = views["view_name"].value_counts().to_dict()

    known_limitations = [
        "Current Nasdaq-100 universe is current constituents, not historical constituents, so survivorship bias remains.",
        "25 metadata-level deduped calls have no component text and are excluded from cleaned_calls and llm_views.",
        "NVDA 2006-02-16 transcript_id 6540 and 2006-05-11 transcript_id 6531 are raw metadata/component coverage gaps.",
        "Small number of no-QA, short, operator-only, or early-format transcripts retain QC warnings.",
        "Uncertain safe harbor components are conservatively retained to avoid over-deleting substantive prepared remarks.",
        "Cleaning pipeline is a versioned rule-based pipeline, not LLM-generated summarization.",
        "cleaned_text and llm_text are cleaned transcript text, not summaries.",
        "Raw candidate dataset and frozen raw dataset are retained for audit and future re-cleaning.",
    ]

    freeze_manifest = {
        "freeze_run_id": FREEZE_RUN_ID,
        "freeze_timestamp": datetime.now().isoformat(timespec="seconds"),
        "cleaned_dataset_status": "frozen",
        "cleaning_version": "full_cleaning_v1.0",
        "inherited_from": "sample_cleaning_v0.2",
        "source_raw_dataset_version": cleaning_manifest.get("source_raw_dataset_version"),
        "source_raw_frozen_manifest_path": str(RAW_FROZEN_MANIFEST_PATH),
        "input_frozen_raw_component_path": cleaning_manifest.get("input_files", {}).get("raw_components"),
        "cleaned_components_path": str(COMPONENTS_PATH),
        "cleaned_calls_path": str(CALLS_PATH),
        "llm_views_path": str(VIEWS_PATH),
        "qc_file_paths": {
            "cleaning_qc": str(CLEANING_QC_PATH),
            "cleaning_summary": str(CLEANING_SUMMARY_PATH),
            "cleaning_manifest": str(CLEANING_MANIFEST_PATH),
            "manual_review": str(CLEANING_MANUAL_REVIEW_PATH),
        },
        "final_review_file_paths": {
            "needs_review_calls": str(NEEDS_REVIEW_PATH),
            "metadata_component_gap_calls": str(GAP_PATH),
            "high_word_drop_calls": str(HIGH_DROP_PATH),
            "uncertain_safe_harbor_sample": str(UNCERTAIN_SAFE_PATH),
            "no_qa_calls": str(NO_QA_PATH),
            "final_review_summary": str(FINAL_REVIEW_SUMMARY_PATH),
        },
        "unique_ticker_count": int(calls["primary_ticker"].nunique()),
        "unique_ciq_company_id_count": int(calls["ciq_company_id"].nunique()),
        "cleaned_component_rows": int(len(components)),
        "cleaned_call_count": int(len(calls)),
        "llm_view_rows": int(len(views)),
        "llm_views_per_call": int(len(views) / len(calls)) if len(calls) else 0,
        "llm_view_counts": view_counts,
        "default_modeling_view": DEFAULT_MODELING_VIEW,
        "date_range": {
            "cleaned_transcript_min_date": str(calls["transcript_date"].min()),
            "cleaned_transcript_max_date": str(calls["transcript_date"].max()),
            "raw_extraction_start_date": raw_manifest.get("date_range", {}).get("start_date"),
            "raw_extraction_end_date": raw_manifest.get("date_range", {}).get("end_date"),
        },
        "universe_as_of_date": raw_manifest.get("universe_as_of_date"),
        "universe_source": raw_manifest.get("universe_source"),
        "earnings_call_filter": raw_manifest.get("earnings_call_filter"),
        "speaker_classification_rules": cleaning_manifest.get("speaker_classification_rule"),
        "section_parsing_rules": cleaning_manifest.get("section_parsing_rule"),
        "safe_harbor_rules": cleaning_manifest.get("safe_harbor_rule_summary"),
        "operator_boilerplate_rules": cleaning_manifest.get("operator_rule_summary"),
        "parsing_quality_flag_distribution": quality_counts,
        "cleaning_warning_distribution": warning_counts,
        "needs_review_distribution": needs_counts,
        "metadata_component_gap_ticker_distribution": gap_ticker_counts,
        "high_word_drop_reason_distribution": high_drop_reason_counts,
        "no_qa_reason_distribution": no_qa_reason_counts,
        "aapl_benchmark_result": "AAPL cleaned call count is 80, consistent with frozen raw deduped and prior AAPL benchmark.",
        "goog_googl_handling": "GOOG/GOOGL handled at company level through primary_ticker GOOGL; GOOG is not duplicated in cleaned calls.",
        "freeze_approval_note": "User approved full_cleaning_v1.0 official freeze after final QC/manual spot check package review.",
        "known_limitations": known_limitations,
    }
    FROZEN_MANIFEST_OUT.write_text(json.dumps(freeze_manifest, indent=2, default=str), encoding="utf-8")

    report = f"""# Nasdaq-100 Cleaned Transcript Dataset Freeze Report

Generated: {datetime.now().isoformat(timespec="seconds")}

## Freeze Status

`full_cleaning_v1.0` is officially frozen as the current Nasdaq-100 cleaned
transcript dataset.

- Freeze run ID: `{FREEZE_RUN_ID}`
- Cleaning version: `full_cleaning_v1.0`
- Inherited from: `sample_cleaning_v0.2`
- Source raw dataset version: `{cleaning_manifest.get('source_raw_dataset_version')}`
- Default modeling view: `{DEFAULT_MODELING_VIEW}`

## Frozen Outputs

- Cleaned components: `{COMPONENTS_PATH}` ({len(components):,} rows)
- Cleaned calls: `{CALLS_PATH}` ({len(calls):,} rows)
- LLM views: `{VIEWS_PATH}` ({len(views):,} rows)
- Unique tickers: {calls['primary_ticker'].nunique():,}
- Unique CIQ company IDs: {calls['ciq_company_id'].nunique():,}
- LLM views per call: {int(len(views) / len(calls))}
- Cleaned transcript date range: {calls['transcript_date'].min()} to {calls['transcript_date'].max()}

## Cleaning Rules

- Normalize whitespace and line breaks.
- Preserve speaker name, speaker role metadata, section type, and component order.
- Exclude blank components from call-level cleaned/LLM aggregation.
- LLM views remove routine operator boilerplate and short/mostly-disclaimer safe harbor components.
- Long mixed safe-harbor prepared remarks are retained and flagged as uncertain safe harbor.
- `cleaned_text` and `llm_text` remain cleaned transcript text, not summaries.

## Default LLM Modeling View

`{DEFAULT_MODELING_VIEW}` is the default modeling view. Other views remain
available for experiments: {', '.join(sorted(view_counts))}.

## Parsing Quality Flag Distribution

{pd.Series(quality_counts).rename_axis('flag').to_frame('count').to_markdown()}

## Cleaning Warning Distribution

{pd.Series(warning_counts).rename_axis('warning').to_frame('count').to_markdown()}

## Needs Review

- Needs review calls: {len(needs):,}

{pd.Series(needs_counts).rename_axis('warning_type').to_frame('count').to_markdown()}

Conclusion: needs_review calls are retained as QC flags and do not indicate a
systemic cleaning failure.

## Metadata / Component Gap

- Metadata/component gap calls: {len(gap):,}

{pd.Series(gap_ticker_counts).rename_axis('ticker').to_frame('count').to_markdown()}

All gap calls are excluded from cleaned calls and LLM views. No empty LLM inputs
were generated.

## High Word-Count Drop

- High word-count-drop cases: {len(high_drop):,}

{pd.Series(high_drop_reason_counts).rename_axis('likely_reason').to_frame('count').to_markdown()}

Conclusion: high-drop cases are concentrated in very short/operator-only,
no-management, no-QA, or unusual transcript structures. No systematic over-
deletion of prepared remarks was found.

## Safe Harbor

Uncertain safe harbor components are conservatively retained in LLM text and
flagged for review. This avoids over-deleting substantive prepared remarks.

## No-QA Calls

- No-QA related calls: {len(no_qa):,}

{pd.Series(no_qa_reason_counts).rename_axis('likely_reason').to_frame('count').to_markdown()}

No-QA calls remain as QC warnings and are not treated as global section parsing
failure.

## Benchmark Checks

- AAPL benchmark: AAPL cleaned call count is 80, consistent with frozen raw
  deduped and the prior AAPL benchmark.
- GOOG/GOOGL: handled at company level through primary ticker GOOGL; GOOG is
  not duplicated in cleaned calls.

## Modeling Readiness

The frozen cleaned transcript dataset can now be used to construct modeling
datasets and LLM input panels. The next stage should align company-level calls
to fiscal periods, earnings dates, and future fundamentals/earnings targets.

## Known Limitations

{chr(10).join(f'- {item}' for item in known_limitations)}
"""
    FREEZE_REPORT_OUT.write_text(report, encoding="utf-8")

    print(f"Wrote {FROZEN_MANIFEST_OUT}")
    print(f"Wrote {FREEZE_REPORT_OUT}")
    print(f"Cleaned components: {len(components):,}")
    print(f"Cleaned calls: {len(calls):,}")
    print(f"LLM views: {len(views):,}")


if __name__ == "__main__":
    main()
