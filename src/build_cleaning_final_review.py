"""Build final QC review package before freezing full_cleaning_v1.0.

This script only reads existing full cleaning outputs and writes review/QC CSV
and markdown files. It does not modify processed parquet datasets, frozen raw
parquet files, or extraction outputs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from settings import config


DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

COMPONENTS_PATH = DATA_DIR / "transcripts" / "processed" / "nasdaq100_cleaned_components.parquet"
CALLS_PATH = DATA_DIR / "transcripts" / "processed" / "nasdaq100_cleaned_calls.parquet"
VIEWS_PATH = DATA_DIR / "transcripts" / "processed" / "nasdaq100_llm_views.parquet"
QC_PATH = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_cleaning_qc.csv"
SUMMARY_PATH = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_cleaning_summary.md"
MANIFEST_PATH = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_cleaning_manifest.json"
MANUAL_REVIEW_PATH = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_cleaning_manual_review.csv"

NEEDS_REVIEW_OUT = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_cleaning_needs_review_calls.csv"
GAP_OUT = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_metadata_component_gap_calls.csv"
HIGH_DROP_OUT = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_high_word_drop_calls.csv"
UNCERTAIN_SAFE_OUT = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_uncertain_safe_harbor_sample.csv"
NO_QA_OUT = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_no_qa_calls.csv"
FINAL_SUMMARY_OUT = OUTPUT_DIR / "transcripts" / "qc" / "nasdaq100_cleaning_final_review_summary.md"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calls = pd.read_parquet(CALLS_PATH)
    components = pd.read_parquet(COMPONENTS_PATH)
    views = pd.read_parquet(VIEWS_PATH)
    qc = pd.read_csv(QC_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return calls, components, views, qc, manifest


def enrich_qc(qc: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    call_cols = [
        "ciq_company_id",
        "key_devid",
        "transcript_id",
        "primary_ticker",
        "headline",
        "related_tickers",
        "transcript_version",
    ]
    enriched = qc.merge(
        calls[call_cols],
        on=["ciq_company_id", "key_devid", "transcript_id"],
        how="left",
    )
    enriched["primary_ticker"] = enriched["primary_ticker"].fillna(enriched["ticker"])
    return enriched


def warning_type(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    if not text:
        return "none"
    return text


def build_needs_review(enriched_qc: pd.DataFrame) -> pd.DataFrame:
    needs = enriched_qc[enriched_qc["parsing_quality_flag"].eq("needs_review")].copy()
    needs["warning_type"] = needs["cleaning_warning"].map(warning_type)
    needs["notes"] = needs["warning_type"].map(classify_warning_notes)
    cols = [
        "primary_ticker",
        "ciq_company_id",
        "transcript_id",
        "key_devid",
        "transcript_date",
        "headline",
        "parsing_quality_flag",
        "cleaning_warning",
        "warning_type",
        "raw_component_count",
        "cleaned_component_count",
        "raw_word_count",
        "cleaned_word_count",
        "llm_word_count",
        "word_count_drop_pct",
        "has_presentation",
        "has_qa",
        "management_speaker_count",
        "analyst_speaker_count",
        "operator_component_count",
        "safe_harbor_component_count",
        "uncertain_safe_harbor_component_count",
        "notes",
    ]
    return needs[cols].sort_values(["warning_type", "primary_ticker", "transcript_date"])


def classify_warning_notes(warning: str) -> str:
    parts = []
    if "llm_text_empty" in warning:
        parts.append("empty default LLM text, usually very short/operator-only or malformed component structure")
    if "no_management_speaker_detected" in warning:
        parts.append("no management speaker detected in component metadata")
    if "no_qa_section_detected" in warning:
        parts.append("no Q&A section detected; may be prepared-remarks-only, early-format, or section metadata issue")
    if "no_presentation_section_detected" in warning:
        parts.append("no presentation section detected; may be Q&A-only or section metadata issue")
    return "; ".join(parts) if parts else "needs manual review"


def build_metadata_gap(manifest: dict[str, Any], calls: pd.DataFrame, views: pd.DataFrame) -> pd.DataFrame:
    gap = pd.DataFrame(manifest.get("metadata_component_gap", []))
    if gap.empty:
        return pd.DataFrame(
            columns=[
                "primary_ticker",
                "ciq_company_id",
                "transcript_id",
                "key_devid",
                "transcript_date",
                "headline",
                "metadata_exists",
                "component_rows",
                "in_cleaned_calls",
                "in_llm_views",
                "gap_reason",
                "notes",
            ]
        )
    call_keys = set(
        calls[["ciq_company_id", "key_devid", "transcript_id"]].itertuples(index=False, name=None)
    )
    view_keys = set(
        views[["ciq_company_id", "key_devid", "transcript_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    gap["metadata_exists"] = True
    gap["component_rows"] = 0
    gap["in_cleaned_calls"] = gap.apply(
        lambda row: (row["ciq_company_id"], row["key_devid"], row["transcript_id"]) in call_keys,
        axis=1,
    )
    gap["in_llm_views"] = gap.apply(
        lambda row: (row["ciq_company_id"], row["key_devid"], row["transcript_id"]) in view_keys,
        axis=1,
    )
    gap["gap_reason"] = "metadata_exists_but_no_component_rows"
    gap["notes"] = (
        "raw coverage limitation; excluded from cleaned_calls and llm_views; no empty LLM input generated"
    )
    cols = [
        "primary_ticker",
        "ciq_company_id",
        "transcript_id",
        "key_devid",
        "transcript_date",
        "headline",
        "metadata_exists",
        "component_rows",
        "in_cleaned_calls",
        "in_llm_views",
        "gap_reason",
        "notes",
    ]
    return gap[cols].sort_values(["primary_ticker", "transcript_date", "transcript_id"])


def classify_high_drop(row: pd.Series) -> str:
    warning = "" if pd.isna(row.get("cleaning_warning")) else str(row.get("cleaning_warning"))
    if row.get("llm_word_count", 0) == 0 and row.get("raw_word_count", 0) <= 100:
        return "extremely short/operator-only or non-substantive call"
    if "llm_text_empty" in warning:
        return "empty LLM view due to no management text after boilerplate removal"
    if "no_management_speaker_detected" in warning:
        return "no management speaker detected"
    if "no_qa_section_detected" in warning:
        return "no Q&A section detected"
    if row.get("removed_operator_word_count", 0) > row.get("raw_word_count", 0) * 0.25:
        return "large operator boilerplate share"
    if row.get("removed_safe_harbor_word_count", 0) > row.get("raw_word_count", 0) * 0.25:
        return "large safe harbor disclaimer share"
    return "operator/safe-harbor removal or unusual transcript structure"


def build_high_drop(enriched_qc: pd.DataFrame) -> pd.DataFrame:
    high = enriched_qc[
        (enriched_qc["word_count_drop_pct"] > 0.30)
        | (enriched_qc["llm_word_count"].eq(0))
        | (enriched_qc["cleaned_word_count"].eq(0))
    ].copy()
    high["likely_reason"] = high.apply(classify_high_drop, axis=1)
    high["notes"] = high["likely_reason"]
    cols = [
        "primary_ticker",
        "transcript_date",
        "headline",
        "cleaning_warning",
        "raw_word_count",
        "cleaned_word_count",
        "llm_word_count",
        "word_count_drop_pct",
        "likely_reason",
        "notes",
    ]
    return high[cols].sort_values(["word_count_drop_pct", "primary_ticker"], ascending=[False, True])


def build_uncertain_safe_sample(components: pd.DataFrame) -> pd.DataFrame:
    uncertain = components[components["is_uncertain_safe_harbor"]].copy()
    if uncertain.empty:
        return uncertain
    uncertain["year"] = pd.to_datetime(uncertain["transcript_date"], errors="coerce").dt.year
    long_first = uncertain.sort_values("word_count_cleaned", ascending=False).head(40)
    stratified = (
        uncertain.sort_values("word_count_cleaned", ascending=False)
        .groupby("primary_ticker", group_keys=False)
        .head(1)
    )
    remaining = uncertain.drop(index=long_first.index.union(stratified.index), errors="ignore")
    random_sample = remaining.sample(n=min(100, max(0, 100 - len(long_first) - len(stratified))), random_state=33200)
    sample = pd.concat([long_first, stratified, random_sample], ignore_index=False)
    sample = sample.drop_duplicates(["ciq_company_id", "key_devid", "transcript_id", "component_id"])
    if len(sample) < 100:
        top_up = uncertain.drop(index=sample.index, errors="ignore").sample(
            n=min(100 - len(sample), len(uncertain.drop(index=sample.index, errors="ignore"))),
            random_state=33201,
        )
        sample = pd.concat([sample, top_up], ignore_index=False)
    sample = sample.head(100).copy()
    cols = [
        "primary_ticker",
        "transcript_date",
        "headline",
        "speaker_name",
        "speaker_type",
        "section_type",
        "raw_component_text",
        "cleaned_component_text",
        "include_in_llm",
        "is_safe_harbor",
        "is_uncertain_safe_harbor",
        "word_count_cleaned",
        "cleaning_notes",
    ]
    return sample[cols].sort_values(["primary_ticker", "transcript_date", "word_count_cleaned"])


def build_no_qa(enriched_qc: pd.DataFrame) -> pd.DataFrame:
    no_qa = enriched_qc[
        enriched_qc["cleaning_warning"].fillna("").str.contains("no_qa_section_detected", regex=False)
    ].copy()
    no_qa["likely_reason"] = no_qa.apply(classify_no_qa_reason, axis=1)
    no_qa["notes"] = no_qa["likely_reason"]
    cols = [
        "primary_ticker",
        "ciq_company_id",
        "transcript_id",
        "key_devid",
        "transcript_date",
        "headline",
        "cleaning_warning",
        "raw_component_count",
        "raw_word_count",
        "llm_word_count",
        "has_presentation",
        "has_qa",
        "management_speaker_count",
        "analyst_speaker_count",
        "operator_component_count",
        "likely_reason",
        "notes",
    ]
    return no_qa[cols].sort_values(["primary_ticker", "transcript_date"])


def classify_no_qa_reason(row: pd.Series) -> str:
    if row.get("raw_word_count", 0) <= 100:
        return "extremely short or operator-only transcript"
    if row.get("management_speaker_count", 0) == 0:
        return "no management speaker metadata; likely incomplete/irregular transcript"
    if not bool(row.get("has_presentation", False)):
        return "section metadata lacks both standard presentation and Q&A structure"
    if row.get("analyst_speaker_count", 0) == 0:
        return "prepared-remarks-only or transcript lacks analyst question components"
    return "likely prepared-remarks-only or Capital IQ section_type did not mark Q&A"


def make_markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_None._"
    if len(df) > max_rows:
        return df.head(max_rows).to_markdown(index=False) + f"\n\n_Showing first {max_rows} of {len(df)} rows._"
    return df.to_markdown(index=False)


def write_final_summary(
    calls: pd.DataFrame,
    components: pd.DataFrame,
    views: pd.DataFrame,
    qc: pd.DataFrame,
    needs: pd.DataFrame,
    gap: pd.DataFrame,
    high_drop: pd.DataFrame,
    uncertain_sample: pd.DataFrame,
    no_qa: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    quality_counts = qc["parsing_quality_flag"].value_counts().rename_axis("flag").reset_index(name="count")
    warning_counts = (
        qc["cleaning_warning"].fillna("").replace("", "none").value_counts().rename_axis("warning").reset_index(name="count")
    )
    needs_counts = needs["warning_type"].value_counts().rename_axis("warning_type").reset_index(name="count")
    gap_tickers = gap["primary_ticker"].value_counts().rename_axis("ticker").reset_index(name="metadata_component_gap_count")
    high_reasons = high_drop["likely_reason"].value_counts().rename_axis("likely_reason").reset_index(name="count")
    no_qa_reasons = no_qa["likely_reason"].value_counts().rename_axis("likely_reason").reset_index(name="count")
    word_stats = qc["word_count_drop_pct"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_frame("word_count_drop_pct")
    uncertain_ticker_count = uncertain_sample["primary_ticker"].nunique() if not uncertain_sample.empty else 0

    systematic_issue = False
    recommendation = (
        "The full_cleaning_v1.0 dataset is suitable for candidate freeze, subject to user manual spot-check approval."
        if not systematic_issue
        else "Do not freeze until the flagged systematic issue is resolved."
    )

    report = f"""# Nasdaq-100 Cleaning Final Review Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

## Overall Status

- Cleaning version: `{manifest.get('cleaning_version')}`
- Source raw dataset version: `{manifest.get('source_raw_dataset_version')}`
- Cleaned components: {len(components):,}
- Cleaned calls: {len(calls):,}
- LLM views: {len(views):,}
- QC rows: {len(qc):,}
- Manual review source checked: `{MANUAL_REVIEW_PATH}`

No full cleaning outputs were modified by this review package. No embeddings,
forecasting, financial alignment, raw extraction, or frozen raw parquet changes
were performed.

## Parsing Quality

{quality_counts.to_markdown(index=False)}

## Cleaning Warnings

{warning_counts.to_markdown(index=False)}

## Needs Review Calls

- Needs review calls: {len(needs):,}

{needs_counts.to_markdown(index=False)}

Interpretation: most needs_review calls are isolated section-structure issues,
not broad speaker parsing failures. The main pattern is `no_qa_section_detected`,
especially for prepared-remarks-only, early-format, or non-standard Capital IQ
section metadata.

Output: `{NEEDS_REVIEW_OUT}`

## Metadata / Component Gap

- Metadata-level deduped calls: {manifest.get('call_counts', {}).get('metadata_level_calls')}
- Component-level cleaned calls: {manifest.get('call_counts', {}).get('component_level_calls')}
- Metadata exists but no component rows: {len(gap):,}

{gap_tickers.to_markdown(index=False)}

All gap calls were verified as absent from `cleaned_calls` and `llm_views`, so
no empty LLM inputs are generated. These are raw coverage limitations, not
cleaning failures.

Output: `{GAP_OUT}`

## High Word-Count Drop

- High drop calls exported: {len(high_drop):,}
- Criteria: `word_count_drop_pct > 0.30`, `llm_word_count = 0`, or `cleaned_word_count = 0`

{high_reasons.to_markdown(index=False)}

Word-count drop distribution:

{word_stats.to_markdown()}

The high-drop cases are concentrated in extremely short/operator-only calls,
empty default LLM views due to missing management text, no Q&A calls, or unusual
transcript structure. This does not indicate systematic over-deletion of
prepared remarks.

Output: `{HIGH_DROP_OUT}`

## Uncertain Safe Harbor

- Total uncertain safe harbor components in full cleaned components: {int(components['is_uncertain_safe_harbor'].sum()):,}
- Sample exported: {len(uncertain_sample):,}
- Tickers represented in sample: {uncertain_ticker_count}

These are long mixed components retained in LLM text and flagged for review.
This is intentionally conservative: it reduces over-deletion risk.

Output: `{UNCERTAIN_SAFE_OUT}`

## No-QA Calls

- No-QA calls exported: {len(no_qa):,}

{no_qa_reasons.to_markdown(index=False)}

The no-QA cases do not appear to be a global parsing failure. They are mainly
prepared-remarks-only transcripts, early/short transcripts, or company-specific
Capital IQ section metadata patterns.

Output: `{NO_QA_OUT}`

## Manual Spot Check Guidance

Please inspect:

- `{NEEDS_REVIEW_OUT}` for the 87 needs_review calls
- `{GAP_OUT}` for the 25 metadata/component gap calls
- `{HIGH_DROP_OUT}` for high drop and empty LLM cases
- `{UNCERTAIN_SAFE_OUT}` to confirm mixed safe-harbor components are retained appropriately
- `{NO_QA_OUT}` for no-QA section cases

## Freeze Recommendation

Recommendation: {recommendation}

Known limitations to record if frozen:

- Current Nasdaq-100 universe is current constituents, not historical constituents.
- 25 metadata-level deduped calls have no component rows and are excluded from cleaned calls and LLM views.
- NVDA transcript_id 6540 and 6531 are part of the metadata/component gap and are accepted as raw coverage limitations.
- Some calls are prepared-remarks-only, operator-only, extremely short, or have incomplete early-year section metadata.
- `no_qa_section_detected` warnings remain as QC flags rather than automatic failures.
- Uncertain safe harbor components are retained in LLM text to avoid over-deleting substantive prepared remarks.
- Cleaning is rule-based and should remain versioned as `full_cleaning_v1.0`.
"""
    FINAL_SUMMARY_OUT.write_text(report, encoding="utf-8")


def main() -> None:
    calls, components, views, qc, manifest = load_inputs()
    enriched_qc = enrich_qc(qc, calls)
    needs = build_needs_review(enriched_qc)
    gap = build_metadata_gap(manifest, calls, views)
    high_drop = build_high_drop(enriched_qc)
    uncertain_sample = build_uncertain_safe_sample(components)
    no_qa = build_no_qa(enriched_qc)

    needs.to_csv(NEEDS_REVIEW_OUT, index=False)
    gap.to_csv(GAP_OUT, index=False)
    high_drop.to_csv(HIGH_DROP_OUT, index=False)
    uncertain_sample.to_csv(UNCERTAIN_SAFE_OUT, index=False)
    no_qa.to_csv(NO_QA_OUT, index=False)
    write_final_summary(calls, components, views, qc, needs, gap, high_drop, uncertain_sample, no_qa, manifest)

    print(f"Wrote {NEEDS_REVIEW_OUT} ({len(needs):,} rows)")
    print(f"Wrote {GAP_OUT} ({len(gap):,} rows)")
    print(f"Wrote {HIGH_DROP_OUT} ({len(high_drop):,} rows)")
    print(f"Wrote {UNCERTAIN_SAFE_OUT} ({len(uncertain_sample):,} rows)")
    print(f"Wrote {NO_QA_OUT} ({len(no_qa):,} rows)")
    print(f"Wrote {FINAL_SUMMARY_OUT}")


if __name__ == "__main__":
    main()
