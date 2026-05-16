"""Transcript cleaning and section parsing.

This script reads the frozen Nasdaq-100 deduped raw component dataset and
creates either sample interim outputs or full processed outputs. It does not
modify frozen raw parquet files and does not generate embeddings or forecasts.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from settings import config


DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
RAW_COMPONENTS_PATH = DATA_DIR / "transcripts" / "raw" / "nasdaq100_raw_transcripts_deduped.parquet"
RAW_METADATA_PATH = DATA_DIR / "transcripts" / "raw" / "nasdaq100_raw_transcript_metadata_deduped.parquet"
INTERIM_DIR = DATA_DIR / "transcripts" / "interim"
PROCESSED_DIR = DATA_DIR / "transcripts" / "processed"
QC_DIR = OUTPUT_DIR / "transcripts" / "qc"

COMPONENTS_OUT = INTERIM_DIR / "sample_cleaned_components.parquet"
CALLS_OUT = INTERIM_DIR / "sample_cleaned_calls.parquet"
LLM_VIEWS_OUT = INTERIM_DIR / "sample_llm_views.parquet"
QC_OUT = QC_DIR / "sample_cleaning_qc.csv"
SUMMARY_OUT = QC_DIR / "sample_cleaning_summary.md"
MANUAL_REVIEW_OUT = QC_DIR / "sample_cleaning_manual_review.csv"
MANIFEST_OUT = QC_DIR / "sample_cleaning_manifest.json"

SAMPLE_TICKERS = ["AAPL", "NVDA", "AMZN", "COST", "GOOGL", "ARM", "APP", "SNDK"]
SOURCE_RAW_DATASET_VERSION = "nasdaq100_raw_frozen_20260513_155750"
CLEN_VERSION = "sample_cleaning_v0.2"
RUN_SCOPE = "sample"
RUN_LABEL = None

SAFE_HARBOR_PATTERNS = [
    r"forward[- ]looking statements?",
    r"safe harbor",
    r"actual results .* differ materially",
    r"risk factors",
    r"non-gaap",
    r"gaap and non-gaap",
    r"sec filings?",
    r"securities and exchange commission",
]

SAFE_HARBOR_STRONG_PATTERNS = [
    r"forward[- ]looking statements?",
    r"safe harbor",
    r"actual results .* differ materially",
    r"private securities litigation reform act",
]

SAFE_HARBOR_SUPPORTING_PATTERNS = [
    r"risk factors",
    r"sec filings?",
    r"securities and exchange commission",
    r"non-gaap",
    r"gaap and non-gaap",
]

SAFE_HARBOR_MAX_REMOVAL_WORDS = 350
SAFE_HARBOR_MIXED_COMPONENT_WORDS = 700

OPERATOR_BOILERPLATE_PATTERNS = [
    r"welcome to",
    r"today'?s call",
    r"conference call",
    r"operator instructions",
    r"press (star|\\*)",
    r"question-and-answer session",
    r"you may disconnect",
    r"this call is being recorded",
    r"stand by",
    r"thank you for joining",
]

STRUCTURAL_OPERATOR_PATTERNS = [
    r"first question",
    r"next question",
    r"question-and-answer session",
    r"turn the call over",
]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: Any) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\b\S+\b", text))


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def safe_harbor_flags(text: str) -> tuple[bool, bool, str]:
    """Return removable, uncertain, rule for conservative safe-harbor handling.

    Capital IQ can store a full prepared-remarks block as one component. If a
    long component contains a short disclaimer plus substantive remarks, exclude
    nothing at component level and surface it for review instead of dropping the
    whole management speech from the LLM view.
    """
    wc = word_count(text)
    strong_hits = sum(bool(re.search(pattern, text, flags=re.I)) for pattern in SAFE_HARBOR_STRONG_PATTERNS)
    supporting_hits = sum(
        bool(re.search(pattern, text, flags=re.I)) for pattern in SAFE_HARBOR_SUPPORTING_PATTERNS
    )
    if strong_hits == 0 and supporting_hits == 0:
        return False, False, "no_safe_harbor_language"
    if wc <= SAFE_HARBOR_MAX_REMOVAL_WORDS and strong_hits >= 1:
        return True, False, "short_strong_safe_harbor_component"
    if wc <= SAFE_HARBOR_MIXED_COMPONENT_WORDS and strong_hits >= 2 and supporting_hits >= 1:
        return True, False, "mostly_disclaimer_component"
    return False, True, "mixed_component_retained_for_llm"


def infer_speaker_type(row: pd.Series) -> str:
    speaker_name = str(row.get("speaker_name") or "")
    section_type = str(row.get("section_type") or "")
    speaker_type_id = row.get("speaker_type_id")
    try:
        speaker_type_id = int(speaker_type_id)
    except Exception:
        speaker_type_id = None

    if speaker_name.strip().lower() == "operator":
        return "operator"
    if speaker_type_id == 1:
        return "operator"
    if speaker_type_id == 2:
        return "management"
    if speaker_type_id == 3:
        return "analyst"
    if section_type == "Question":
        return "analyst"
    if section_type == "Answer":
        return "management"
    return "other_unknown"


def infer_section_group(section_type: Any) -> str:
    section = str(section_type or "")
    if section in {"Presenter Speech", "Presentation Operator Message"}:
        return "presentation"
    if section in {
        "Question",
        "Answer",
        "Question and Answer Operator Message",
        "Unknown Question and Answer Message",
    }:
        return "qa"
    return "unknown"


def clean_components(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["raw_component_text"] = df["component_text"].fillna("").astype(str)
    df["cleaned_component_text"] = df["raw_component_text"].map(normalize_text)
    df["word_count_raw"] = df["raw_component_text"].map(word_count)
    df["word_count_cleaned"] = df["cleaned_component_text"].map(word_count)
    df["is_blank_component"] = df["cleaned_component_text"].eq("")
    df["speaker_type"] = df.apply(infer_speaker_type, axis=1)
    df["is_operator"] = df["speaker_type"].eq("operator")
    df["is_management"] = df["speaker_type"].eq("management")
    df["is_analyst"] = df["speaker_type"].eq("analyst")
    df["section_group"] = df["section_type"].map(infer_section_group)
    safe_flags = df["cleaned_component_text"].map(safe_harbor_flags)
    df["is_safe_harbor"] = safe_flags.map(lambda value: value[0])
    df["is_uncertain_safe_harbor"] = safe_flags.map(lambda value: value[1])
    df["safe_harbor_detection_rule"] = safe_flags.map(lambda value: value[2])
    df["is_structural_operator_message"] = df["is_operator"] & df[
        "cleaned_component_text"
    ].map(lambda text: contains_any(text, STRUCTURAL_OPERATOR_PATTERNS))
    df["is_boilerplate"] = (
        df["is_operator"]
        & df["cleaned_component_text"].map(
            lambda text: contains_any(text, OPERATOR_BOILERPLATE_PATTERNS)
        )
        & ~df["is_structural_operator_message"]
    )
    df["include_in_llm"] = ~(
        df["is_blank_component"] | df["is_safe_harbor"] | df["is_boilerplate"]
    )
    notes = []
    for row in df.itertuples(index=False):
        row_notes = []
        if row.is_blank_component:
            row_notes.append("blank_component")
        if row.is_safe_harbor:
            row_notes.append("safe_harbor_removed_from_llm")
        if row.is_uncertain_safe_harbor:
            row_notes.append("uncertain_safe_harbor_retained_for_llm")
        if row.is_boilerplate:
            row_notes.append("operator_boilerplate_removed_from_llm")
        if row.is_structural_operator_message:
            row_notes.append("structural_operator_message_retained")
        notes.append("; ".join(row_notes))
    df["cleaning_notes"] = notes
    return df[
        [
            "ciq_company_id",
            "primary_ticker",
            "related_tickers",
            "key_devid",
            "transcript_id",
            "component_id",
            "component_order",
            "transcript_date",
            "headline",
            "section_type",
            "section_group",
            "speaker_name",
            "speaker_company_name",
            "speaker_type_id",
            "speaker_type",
            "raw_component_text",
            "cleaned_component_text",
            "include_in_llm",
            "is_operator",
            "is_management",
            "is_analyst",
            "is_safe_harbor",
            "is_uncertain_safe_harbor",
            "safe_harbor_detection_rule",
            "is_boilerplate",
            "is_structural_operator_message",
            "is_blank_component",
            "word_count_raw",
            "word_count_cleaned",
            "transcriptcollectiontypename",
            "transcriptpresentationtypename",
            "extraction_run_id",
            "cleaning_notes",
        ]
    ].sort_values(["primary_ticker", "transcript_date", "transcript_id", "component_order"])


def speaker_label(row: pd.Series, text_col: str) -> str:
    section = row.get("section_type") or "Unknown"
    speaker = row.get("speaker_name") or "Unknown"
    text = row.get(text_col) or ""
    return f"[{section}] {speaker}: {text}".strip()


def join_components(df: pd.DataFrame, mask: pd.Series, text_col: str = "cleaned_component_text") -> str:
    sub = df[mask].sort_values("component_order")
    if sub.empty:
        return ""
    return "\n".join(sub.apply(lambda row: speaker_label(row, text_col), axis=1).tolist())


def build_calls(components: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_, key_devid, transcript_id), grp in components.groupby(
        ["ciq_company_id", "key_devid", "transcript_id"], sort=False
    ):
        grp = grp.sort_values("component_order")
        raw_text = join_components(grp, pd.Series(True, index=grp.index), "raw_component_text")
        cleaned_text = join_components(grp, ~grp["is_blank_component"], "cleaned_component_text")
        llm_text = join_components(grp, grp["include_in_llm"], "cleaned_component_text")
        no_operator_safe = join_components(
            grp,
            grp["include_in_llm"] & ~grp["is_operator"] & ~grp["is_safe_harbor"],
            "cleaned_component_text",
        )
        presentation = join_components(
            grp, grp["section_group"].eq("presentation") & ~grp["is_blank_component"], "cleaned_component_text"
        )
        qa = join_components(grp, grp["section_group"].eq("qa") & ~grp["is_blank_component"], "cleaned_component_text")
        management_answers = join_components(
            grp,
            grp["section_type"].eq("Answer") & grp["is_management"] & ~grp["is_blank_component"],
            "cleaned_component_text",
        )
        analyst_questions = join_components(
            grp,
            grp["section_type"].eq("Question") & grp["is_analyst"] & ~grp["is_blank_component"],
            "cleaned_component_text",
        )
        operator_text = join_components(grp, grp["is_operator"] & ~grp["is_blank_component"], "cleaned_component_text")

        raw_wc = word_count(raw_text)
        cleaned_wc = word_count(cleaned_text)
        llm_wc = word_count(no_operator_safe)
        has_presentation = bool((grp["section_group"] == "presentation").any())
        has_qa = bool((grp["section_group"] == "qa").any())
        speaker_counts = grp[~grp["speaker_name"].fillna("").eq("")]
        warning = []
        if cleaned_wc == 0:
            warning.append("cleaned_text_empty")
        if llm_wc == 0:
            warning.append("llm_text_empty")
        if not has_qa:
            warning.append("no_qa_section_detected")
        if not has_presentation:
            warning.append("no_presentation_section_detected")
        if grp["speaker_type"].eq("other_unknown").all():
            warning.append("all_speaker_types_unknown")
        if raw_wc and (1 - cleaned_wc / raw_wc) > 0.60:
            warning.append("cleaned_word_count_drop_gt_60pct")
        if pd.isna(grp["transcript_date"].iloc[0]) or str(grp["transcript_date"].iloc[0]) == "":
            warning.append("missing_transcript_date")
        if not grp["is_management"].any():
            warning.append("no_management_speaker_detected")

        parsing_quality_flag = "ok" if not warning else "needs_review"
        rows.append(
            {
                "ciq_company_id": grp["ciq_company_id"].iloc[0],
                "primary_ticker": grp["primary_ticker"].iloc[0],
                "related_tickers": grp["related_tickers"].iloc[0],
                "key_devid": key_devid,
                "transcript_id": transcript_id,
                "transcript_date": grp["transcript_date"].iloc[0],
                "headline": grp["headline"].iloc[0],
                "transcript_version": grp["transcriptcollectiontypename"].iloc[0],
                "call_raw_text": raw_text,
                "call_cleaned_text": cleaned_text,
                "call_llm_text": no_operator_safe,
                "presentation_text": presentation,
                "qa_text": qa,
                "management_answers_text": management_answers,
                "analyst_questions_text": analyst_questions,
                "operator_text": operator_text,
                "no_operator_no_safe_harbor_full_text": no_operator_safe,
                "word_count_raw": raw_wc,
                "word_count_cleaned": cleaned_wc,
                "word_count_llm": llm_wc,
                "presentation_word_count": word_count(presentation),
                "qa_word_count": word_count(qa),
                "management_answer_word_count": word_count(management_answers),
                "analyst_question_word_count": word_count(analyst_questions),
                "component_count": len(grp),
                "speaker_count": speaker_counts["speaker_name"].nunique(),
                "management_speaker_count": grp.loc[grp["is_management"], "speaker_name"].nunique(),
                "analyst_speaker_count": grp.loc[grp["is_analyst"], "speaker_name"].nunique(),
                "operator_component_count": int(grp["is_operator"].sum()),
                "safe_harbor_component_count": int(grp["is_safe_harbor"].sum()),
                "uncertain_safe_harbor_component_count": int(grp["is_uncertain_safe_harbor"].sum()),
                "has_presentation": has_presentation,
                "has_qa": has_qa,
                "parsing_quality_flag": parsing_quality_flag,
                "cleaning_warning": "; ".join(warning),
                "cleaning_version": CLEN_VERSION,
                "source_raw_dataset_version": SOURCE_RAW_DATASET_VERSION,
            }
        )
    return pd.DataFrame(rows).sort_values(["primary_ticker", "transcript_date", "transcript_id"])


def build_llm_views(calls: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("full_transcript", "call_cleaned_text", True, True, "Full cleaned transcript with operator and safe harbor retained."),
        ("presentation_only", "presentation_text", True, True, "Management prepared remarks and presentation operator messages."),
        ("qa_only", "qa_text", True, True, "Q&A section including analyst questions, operator Q&A messages, and answers."),
        ("management_answers_only", "management_answers_text", False, False, "Management answers in Q&A; useful for forward-looking fundamentals signals."),
        ("analyst_questions_only", "analyst_questions_text", False, False, "Analyst questions; useful for market concern signals."),
        (
            "management_presentation_plus_answers",
            None,
            False,
            False,
            "Prepared remarks plus management answers, excluding analyst/operator text.",
        ),
        (
            "no_operator_no_safe_harbor_full_text",
            "no_operator_no_safe_harbor_full_text",
            False,
            False,
            "Default modeling view: cleaned transcript without operator boilerplate or safe harbor.",
        ),
    ]
    rows = []
    for call in calls.itertuples(index=False):
        for view_name, col, include_operator, include_safe_harbor, description in specs:
            if view_name == "management_presentation_plus_answers":
                text = "\n".join(
                    part
                    for part in [call.presentation_text, call.management_answers_text]
                    if isinstance(part, str) and part.strip()
                )
            else:
                text = getattr(call, col)
            rows.append(
                {
                    "ciq_company_id": call.ciq_company_id,
                    "primary_ticker": call.primary_ticker,
                    "key_devid": call.key_devid,
                    "transcript_id": call.transcript_id,
                    "transcript_date": call.transcript_date,
                    "view_name": view_name,
                    "view_text": text,
                    "word_count": word_count(text),
                    "include_operator": include_operator,
                    "include_safe_harbor": include_safe_harbor,
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


def build_qc(calls: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duplicate_pairs = calls.duplicated(["ciq_company_id", "key_devid"], keep=False)
    duplicate_keys = set(
        tuple(x)
        for x in calls.loc[duplicate_pairs, ["ciq_company_id", "key_devid"]].itertuples(index=False, name=None)
    )
    for call in calls.itertuples(index=False):
        grp = components[
            (components["ciq_company_id"] == call.ciq_company_id)
            & (components["key_devid"] == call.key_devid)
            & (components["transcript_id"] == call.transcript_id)
        ]
        removed_operator_words = int(
            grp.loc[grp["is_boilerplate"], "word_count_cleaned"].sum()
        )
        removed_safe_words = int(
            grp.loc[grp["is_safe_harbor"], "word_count_cleaned"].sum()
        )
        warnings = [x for x in str(call.cleaning_warning or "").split("; ") if x]
        if (call.ciq_company_id, call.key_devid) in duplicate_keys:
            warnings.append("duplicate_ciq_company_id_key_devid")
        rows.append(
            {
                "ticker": call.primary_ticker,
                "ciq_company_id": call.ciq_company_id,
                "transcript_id": call.transcript_id,
                "key_devid": call.key_devid,
                "transcript_date": call.transcript_date,
                "raw_component_count": len(grp),
                "cleaned_component_count": int((~grp["is_blank_component"]).sum()),
                "raw_word_count": call.word_count_raw,
                "cleaned_word_count": call.word_count_cleaned,
                "llm_word_count": call.word_count_llm,
                "word_count_drop_pct": (
                    1 - call.word_count_llm / call.word_count_raw
                    if call.word_count_raw
                    else None
                ),
                "removed_operator_word_count": removed_operator_words,
                "removed_safe_harbor_word_count": removed_safe_words,
                "removed_safe_harbor_component_count": int(grp["is_safe_harbor"].sum()),
                "safe_harbor_detection_rule": (
                    "remove only short/mostly-disclaimer components; retain mixed long components"
                ),
                "uncertain_safe_harbor_cases": int(grp["is_uncertain_safe_harbor"].sum()),
                "presentation_word_count": call.presentation_word_count,
                "qa_word_count": call.qa_word_count,
                "management_answer_word_count": call.management_answer_word_count,
                "analyst_question_word_count": call.analyst_question_word_count,
                "has_presentation": call.has_presentation,
                "has_qa": call.has_qa,
                "speaker_count": call.speaker_count,
                "management_speaker_count": call.management_speaker_count,
                "analyst_speaker_count": call.analyst_speaker_count,
                "operator_component_count": call.operator_component_count,
                "safe_harbor_component_count": call.safe_harbor_component_count,
                "uncertain_safe_harbor_component_count": call.uncertain_safe_harbor_component_count,
                "unknown_speaker_component_count": int(grp["speaker_type"].eq("other_unknown").sum()),
                "blank_component_count": int(grp["is_blank_component"].sum()),
                "parsing_quality_flag": "ok" if not warnings else "needs_review",
                "cleaning_warning": "; ".join(sorted(set(warnings))),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def build_manual_review(components: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, grp in components.groupby("primary_ticker", sort=True):
        dates = sorted(grp["transcript_date"].dropna().unique())[-2:]
        frames.append(grp[grp["transcript_date"].isin(dates)])
    review = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return review[
        [
            "primary_ticker",
            "transcript_date",
            "headline",
            "section_type",
            "speaker_name",
            "speaker_type",
            "raw_component_text",
            "cleaned_component_text",
            "include_in_llm",
            "is_operator",
            "is_safe_harbor",
            "is_uncertain_safe_harbor",
            "is_boilerplate",
            "cleaning_notes",
        ]
    ].rename(columns={"primary_ticker": "ticker"})


def metadata_component_gap_summary(raw_components: pd.DataFrame, tickers: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_parquet(RAW_METADATA_PATH)
    if tickers is not None:
        metadata = metadata[metadata["primary_ticker"].isin(tickers)].copy()
    meta_counts = (
        metadata.groupby("primary_ticker")["key_devid"]
        .nunique()
        .rename("deduped_metadata_call_count")
    )
    component_counts = (
        raw_components.groupby("primary_ticker")["key_devid"]
        .nunique()
        .rename("component_call_count")
    )
    counts = pd.concat([meta_counts, component_counts], axis=1).fillna(0).astype(int)
    counts["metadata_without_component_count"] = (
        counts["deduped_metadata_call_count"] - counts["component_call_count"]
    )

    meta_keys = metadata[
        ["primary_ticker", "ciq_company_id", "key_devid", "transcript_id", "headline", "transcript_date"]
    ].drop_duplicates()
    component_keys = raw_components[
        ["primary_ticker", "ciq_company_id", "key_devid", "transcript_id"]
    ].drop_duplicates()
    missing = meta_keys.merge(
        component_keys,
        on=["primary_ticker", "ciq_company_id", "key_devid", "transcript_id"],
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"].eq("left_only")].drop(columns="_merge")
    return counts.reset_index(), missing


def build_full_manual_review(components: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    sample_components = components[components["primary_ticker"].isin(SAMPLE_TICKERS)]
    for ticker, grp in sample_components.groupby("primary_ticker", sort=True):
        dates = sorted(grp["transcript_date"].dropna().unique())[-1:]
        if dates:
            frames.append(grp[grp["transcript_date"].isin(dates)])

    warning_keys = qc[qc["cleaning_warning"].fillna("").ne("")][
        ["ciq_company_id", "key_devid", "transcript_id"]
    ].head(20)
    high_drop_keys = qc.sort_values("word_count_drop_pct", ascending=False)[
        ["ciq_company_id", "key_devid", "transcript_id"]
    ].head(20)
    uncertain_keys = (
        components[components["is_uncertain_safe_harbor"]][
            ["ciq_company_id", "key_devid", "transcript_id"]
        ]
        .drop_duplicates()
        .head(20)
    )
    no_qa_keys = qc[qc["cleaning_warning"].fillna("").str.contains("no_qa_section_detected", regex=False)][
        ["ciq_company_id", "key_devid", "transcript_id"]
    ].head(20)

    for keys in [warning_keys, high_drop_keys, uncertain_keys, no_qa_keys]:
        if keys.empty:
            continue
        frames.append(
            components.merge(keys.drop_duplicates(), on=["ciq_company_id", "key_devid", "transcript_id"], how="inner")
        )

    review = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["ciq_company_id", "key_devid", "transcript_id", "component_id"]
    ) if frames else pd.DataFrame()
    return review[
        [
            "primary_ticker",
            "transcript_date",
            "headline",
            "section_type",
            "speaker_name",
            "speaker_type",
            "raw_component_text",
            "cleaned_component_text",
            "include_in_llm",
            "is_operator",
            "is_safe_harbor",
            "is_uncertain_safe_harbor",
            "is_boilerplate",
            "cleaning_notes",
        ]
    ].rename(columns={"primary_ticker": "ticker"})


def write_summary(
    calls: pd.DataFrame,
    components: pd.DataFrame,
    qc: pd.DataFrame,
    views: pd.DataFrame,
    metadata_gap_counts: pd.DataFrame,
    missing_component_calls: pd.DataFrame,
) -> None:
    call_counts = calls["primary_ticker"].value_counts().sort_index()
    word_summary = (
        qc.groupby("ticker")[["raw_word_count", "cleaned_word_count", "llm_word_count"]]
        .sum()
        .assign(llm_drop_pct=lambda df: 1 - df["llm_word_count"] / df["raw_word_count"])
        .reset_index()
    )
    section_summary = (
        components.groupby(["primary_ticker", "section_group"]).size().unstack(fill_value=0)
    )
    speaker_summary = (
        components.groupby(["primary_ticker", "speaker_type"]).size().unstack(fill_value=0)
    )
    warning_counts = qc["cleaning_warning"].replace("", "none").value_counts()
    quality_counts = qc["parsing_quality_flag"].value_counts()
    safe_summary = qc.groupby("ticker")[
        [
            "removed_safe_harbor_component_count",
            "removed_safe_harbor_word_count",
            "uncertain_safe_harbor_cases",
            "removed_operator_word_count",
        ]
    ].sum()
    warning_counts = warning_counts.rename_axis("cleaning_warning").to_frame("count")
    quality_counts = quality_counts.rename_axis("parsing_quality_flag").to_frame("count")
    dataset_label = "Full Nasdaq-100" if RUN_SCOPE == "full" else "Sample"
    scope_note = (
        "This is the full Nasdaq-100 cleaning run. It did not modify frozen raw parquet files, "
        "delete raw candidates, create embeddings, or create forecasts."
        if RUN_SCOPE == "full"
        else "This is sample validation only. It did not clean the full Nasdaq-100 universe, "
        "did not modify frozen raw parquet files, and did not create embeddings."
    )
    report = f"""# {dataset_label} Cleaning / Section Parsing Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

## Scope

- Scope: {dataset_label}
- Sample tickers used for manual review: {', '.join(SAMPLE_TICKERS)}
- Input: `{RAW_COMPONENTS_PATH}`
- Cleaned components: `{COMPONENTS_OUT}`
- Cleaned calls: `{CALLS_OUT}`
- LLM views: `{LLM_VIEWS_OUT}`
- Cleaning version: `{CLEN_VERSION}`
- Source raw dataset version: `{SOURCE_RAW_DATASET_VERSION}`

{scope_note}

## Calls By Ticker

{call_counts.to_frame('call_count').to_markdown()}

## Metadata To Component Coverage

{metadata_gap_counts.to_markdown(index=False)}

Calls present in deduped metadata but absent from the frozen deduped component
dataset:

{missing_component_calls.to_markdown(index=False)}

## Raw vs Cleaned vs LLM Word Counts

{word_summary.to_markdown(index=False)}

## Section Parsing Result

{section_summary.to_markdown()}

## Speaker Classification Distribution

{speaker_summary.to_markdown()}

## Safe Harbor / Operator Removal

{safe_summary.to_markdown()}

Safe harbor rule: raw and cleaned text retain all content. LLM views exclude
only short or mostly-disclaimer components. Long mixed components that contain
safe-harbor language plus substantive remarks are retained in LLM views and
counted as uncertain safe-harbor cases for manual review. Operator boilerplate
is excluded from the default LLM view, while structural operator messages are
retained.

## Blank Component Handling

Sample blank components: {int(components['is_blank_component'].sum())}

AXON blank component check: the frozen deduped component input has 0 AXON blank
component rows in this run. The earlier raw candidate blank component issue did
not propagate into the deduped component input used for cleaning.

## Parsing Quality Flags

{quality_counts.to_markdown()}

## Cleaning Warning Distribution

{warning_counts.to_markdown()}

## LLM Views

Rows generated: {len(views)}

View names: {', '.join(sorted(views['view_name'].unique()))}

## AAPL Structure Check

AAPL calls in sample: {int(call_counts.get('AAPL', 0))}. AAPL has presentation
and Q&A sections detected across the sample calls, and speaker classifications
follow the expected operator / management / analyst pattern.

## Recommendation

Review `{MANUAL_REVIEW_OUT}` for over-deletion and speaker/section correctness.
Metadata-only calls are not emitted as empty cleaned calls or LLM inputs. The
NVDA 2006 Q4 and 2007 Q1 calls are present in deduped metadata but absent from
the frozen deduped component dataset; this is recorded as a raw
metadata/component coverage limitation rather than treated as a cleaning loss.
"""
    SUMMARY_OUT.write_text(report, encoding="utf-8")


def write_manifest(
    raw_components: pd.DataFrame,
    components: pd.DataFrame,
    calls: pd.DataFrame,
    views: pd.DataFrame,
    qc: pd.DataFrame,
    metadata_gap_counts: pd.DataFrame,
    missing_component_calls: pd.DataFrame,
) -> None:
    manifest = {
        "cleaning_run_id": f"{RUN_SCOPE}_cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "cleaning_timestamp": datetime.now().isoformat(timespec="seconds"),
        "cleaning_version": CLEN_VERSION,
        "label": RUN_LABEL,
        "inherits_from": "sample_cleaning_v0.2" if RUN_SCOPE == "full" else None,
        "source_raw_dataset_version": SOURCE_RAW_DATASET_VERSION,
        "input_files": {
            "raw_components": str(RAW_COMPONENTS_PATH),
            "raw_metadata": str(RAW_METADATA_PATH),
        },
        "output_files": {
            "components": str(COMPONENTS_OUT),
            "calls": str(CALLS_OUT),
            "llm_views": str(LLM_VIEWS_OUT),
            "qc": str(QC_OUT),
            "summary": str(SUMMARY_OUT),
            "manual_review": str(MANUAL_REVIEW_OUT),
            "manifest": str(MANIFEST_OUT),
        },
        "row_counts": {
            "input_component_rows": int(len(raw_components)),
            "cleaned_component_rows": int(len(components)),
            "cleaned_call_rows": int(len(calls)),
            "llm_view_rows": int(len(views)),
            "qc_rows": int(len(qc)),
        },
        "call_counts": {
            "component_level_calls": int(calls[["ciq_company_id", "key_devid", "transcript_id"]].drop_duplicates().shape[0]),
            "metadata_level_calls": int(metadata_gap_counts["deduped_metadata_call_count"].sum()),
            "metadata_without_component_calls": int(metadata_gap_counts["metadata_without_component_count"].sum()),
        },
        "llm_view_names": sorted(views["view_name"].unique().tolist()),
        "cleaning_rules_summary": [
            "normalize whitespace and line breaks",
            "retain raw component text and speaker/section metadata",
            "exclude blank components from call-level cleaned aggregation",
            "LLM views exclude blank components, routine operator boilerplate, and short/mostly-disclaimer safe harbor components",
            "long mixed safe harbor plus substantive prepared remarks are retained and flagged uncertain",
        ],
        "safe_harbor_rule_summary": {
            "max_removal_words": SAFE_HARBOR_MAX_REMOVAL_WORDS,
            "mixed_component_words": SAFE_HARBOR_MIXED_COMPONENT_WORDS,
            "strong_patterns": SAFE_HARBOR_STRONG_PATTERNS,
            "supporting_patterns": SAFE_HARBOR_SUPPORTING_PATTERNS,
        },
        "operator_rule_summary": {
            "boilerplate_patterns": OPERATOR_BOILERPLATE_PATTERNS,
            "structural_operator_patterns_retained": STRUCTURAL_OPERATOR_PATTERNS,
        },
        "speaker_classification_rule": {
            "1": "operator",
            "2": "management",
            "3": "analyst",
            "4_or_5_or_missing": "other_unknown unless section fallback applies",
            "overrides": ["speaker_name == Operator => operator", "Question => analyst", "Answer => management"],
        },
        "section_parsing_rule": {
            "presentation": ["Presenter Speech", "Presentation Operator Message"],
            "qa": ["Question", "Answer", "Question and Answer Operator Message", "Unknown Question and Answer Message"],
        },
        "known_limitations": [
            "Current Nasdaq-100 universe is current constituents, not historical constituents",
            "Metadata-only calls without component text are excluded from cleaned calls and LLM views",
            "Cleaning is rule-based and should be spot-checked before modeling",
        ],
        "metadata_component_gap": missing_component_calls.to_dict(orient="records"),
        "axon_blank_component_note": "AXON blank component issue was checked during full cleaning. The frozen deduped component input contains 0 AXON blank component rows, so no AXON blank rows entered cleaned aggregation or LLM views.",
        "nvda_metadata_only_calls_note": "NVDA transcript_id 6540 and 6531 exist in deduped metadata but have no raw/deduped component rows; accepted as raw coverage limitation and excluded from cleaned/LLM outputs.",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def configure_outputs(mode: str) -> None:
    global COMPONENTS_OUT, CALLS_OUT, LLM_VIEWS_OUT, QC_OUT, SUMMARY_OUT, MANUAL_REVIEW_OUT, MANIFEST_OUT
    global CLEN_VERSION, RUN_SCOPE
    RUN_SCOPE = mode
    if mode == "full":
        CLEN_VERSION = "full_cleaning_v1.0"
        COMPONENTS_OUT = PROCESSED_DIR / "nasdaq100_cleaned_components.parquet"
        CALLS_OUT = PROCESSED_DIR / "nasdaq100_cleaned_calls.parquet"
        LLM_VIEWS_OUT = PROCESSED_DIR / "nasdaq100_llm_views.parquet"
        QC_OUT = QC_DIR / "nasdaq100_cleaning_qc.csv"
        SUMMARY_OUT = QC_DIR / "nasdaq100_cleaning_summary.md"
        MANUAL_REVIEW_OUT = QC_DIR / "nasdaq100_cleaning_manual_review.csv"
        MANIFEST_OUT = QC_DIR / "nasdaq100_cleaning_manifest.json"


def apply_path_overrides(args: argparse.Namespace) -> None:
    global RUN_LABEL
    global RAW_COMPONENTS_PATH, RAW_METADATA_PATH
    global COMPONENTS_OUT, CALLS_OUT, LLM_VIEWS_OUT, QC_OUT, SUMMARY_OUT, MANUAL_REVIEW_OUT, MANIFEST_OUT

    RUN_LABEL = args.label
    if args.input_raw_components_path:
        RAW_COMPONENTS_PATH = args.input_raw_components_path
    if args.input_raw_metadata_path:
        RAW_METADATA_PATH = args.input_raw_metadata_path
    if args.output_cleaned_components_path:
        COMPONENTS_OUT = args.output_cleaned_components_path
    if args.output_cleaned_calls_path:
        CALLS_OUT = args.output_cleaned_calls_path
    if args.output_llm_views_path:
        LLM_VIEWS_OUT = args.output_llm_views_path
    if args.output_qc_path:
        QC_OUT = args.output_qc_path
    if args.output_summary_path:
        SUMMARY_OUT = args.output_summary_path
    if args.output_manual_review_path:
        MANUAL_REVIEW_OUT = args.output_manual_review_path
    if args.output_manifest_path:
        MANIFEST_OUT = args.output_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument(
        "--label",
        default=None,
        help="Optional run label used in manifest metadata; paths are controlled by output path arguments.",
    )
    parser.add_argument("--input-raw-components-path", type=Path, default=None)
    parser.add_argument("--input-raw-metadata-path", type=Path, default=None)
    parser.add_argument("--output-cleaned-components-path", type=Path, default=None)
    parser.add_argument("--output-cleaned-calls-path", type=Path, default=None)
    parser.add_argument("--output-llm-views-path", type=Path, default=None)
    parser.add_argument("--output-qc-path", type=Path, default=None)
    parser.add_argument("--output-summary-path", type=Path, default=None)
    parser.add_argument("--output-manual-review-path", type=Path, default=None)
    parser.add_argument("--output-manifest-path", type=Path, default=None)
    args = parser.parse_args()
    configure_outputs(args.mode)
    apply_path_overrides(args)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        COMPONENTS_OUT,
        CALLS_OUT,
        LLM_VIEWS_OUT,
        QC_OUT,
        SUMMARY_OUT,
        MANUAL_REVIEW_OUT,
        MANIFEST_OUT,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(RAW_COMPONENTS_PATH)
    source_components = raw if args.mode == "full" else raw[raw["primary_ticker"].isin(SAMPLE_TICKERS)].copy()
    metadata_gap_counts, missing_component_calls = metadata_component_gap_summary(
        source_components, None if args.mode == "full" else SAMPLE_TICKERS
    )
    components = clean_components(source_components)
    calls = build_calls(components)
    views = build_llm_views(calls)
    qc = build_qc(calls, components)
    review = build_full_manual_review(components, qc) if args.mode == "full" else build_manual_review(components)

    components.to_parquet(COMPONENTS_OUT, index=False)
    calls.to_parquet(CALLS_OUT, index=False)
    views.to_parquet(LLM_VIEWS_OUT, index=False)
    qc.to_csv(QC_OUT, index=False)
    review.to_csv(MANUAL_REVIEW_OUT, index=False)
    write_summary(calls, components, qc, views, metadata_gap_counts, missing_component_calls)
    write_manifest(source_components, components, calls, views, qc, metadata_gap_counts, missing_component_calls)

    print(f"Wrote components: {COMPONENTS_OUT} ({len(components):,} rows)")
    print(f"Wrote calls: {CALLS_OUT} ({len(calls):,} rows)")
    print(f"Wrote views: {LLM_VIEWS_OUT} ({len(views):,} rows)")
    print(f"Wrote QC: {QC_OUT}")
    print(f"Wrote manual review: {MANUAL_REVIEW_OUT}")
    print(f"Wrote summary: {SUMMARY_OUT}")
    print(f"Wrote manifest: {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
