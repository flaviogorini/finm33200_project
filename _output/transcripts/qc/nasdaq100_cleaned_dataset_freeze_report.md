# Nasdaq-100 Cleaned Transcript Dataset Freeze Report

Generated: 2026-05-13T16:33:56

## Freeze Status

`full_cleaning_v1.0` is officially frozen as the current Nasdaq-100 cleaned
transcript dataset.

- Freeze run ID: `nasdaq100_cleaned_frozen_20260513_163356`
- Cleaning version: `full_cleaning_v1.0`
- Inherited from: `sample_cleaning_v0.2`
- Source raw dataset version: `nasdaq100_raw_frozen_20260513_155750`
- Default modeling view: `no_operator_no_safe_harbor_full_text`

## Frozen Outputs

- Cleaned components: `/Users/yujiadi/Desktop/finm33200_project/_data/transcripts/processed/nasdaq100_cleaned_components.parquet` (452,655 rows)
- Cleaned calls: `/Users/yujiadi/Desktop/finm33200_project/_data/transcripts/processed/nasdaq100_cleaned_calls.parquet` (6,510 rows)
- LLM views: `/Users/yujiadi/Desktop/finm33200_project/_data/transcripts/processed/nasdaq100_llm_views.parquet` (45,570 rows)
- Unique tickers: 100
- Unique CIQ company IDs: 100
- LLM views per call: 7
- Cleaned transcript date range: 2005-11-01 to 2025-12-19

## Cleaning Rules

- Normalize whitespace and line breaks.
- Preserve speaker name, speaker role metadata, section type, and component order.
- Exclude blank components from call-level cleaned/LLM aggregation.
- LLM views remove routine operator boilerplate and short/mostly-disclaimer safe harbor components.
- Long mixed safe-harbor prepared remarks are retained and flagged as uncertain safe harbor.
- `cleaned_text` and `llm_text` remain cleaned transcript text, not summaries.

## Default LLM Modeling View

`no_operator_no_safe_harbor_full_text` is the default modeling view. Other views remain
available for experiments: analyst_questions_only, full_transcript, management_answers_only, management_presentation_plus_answers, no_operator_no_safe_harbor_full_text, presentation_only, qa_only.

## Parsing Quality Flag Distribution

| flag         |   count |
|:-------------|--------:|
| ok           |    6423 |
| needs_review |      87 |

## Cleaning Warning Distribution

| warning                                                                |   count |
|:-----------------------------------------------------------------------|--------:|
| none                                                                   |    6423 |
| no_qa_section_detected                                                 |      73 |
| llm_text_empty; no_management_speaker_detected; no_qa_section_detected |       7 |
| no_presentation_section_detected                                       |       6 |
| llm_text_empty; no_management_speaker_detected                         |       1 |

## Needs Review

- Needs review calls: 87

| warning_type                                                           |   count |
|:-----------------------------------------------------------------------|--------:|
| no_qa_section_detected                                                 |      73 |
| llm_text_empty; no_management_speaker_detected; no_qa_section_detected |       7 |
| no_presentation_section_detected                                       |       6 |
| llm_text_empty; no_management_speaker_detected                         |       1 |

Conclusion: needs_review calls are retained as QC flags and do not indicate a
systemic cleaning failure.

## Metadata / Component Gap

- Metadata/component gap calls: 25

| ticker   |   count |
|:---------|--------:|
| WMT      |       9 |
| KLAC     |       2 |
| NVDA     |       2 |
| STX      |       2 |
| TTWO     |       2 |
| XEL      |       2 |
| CTSH     |       1 |
| MSFT     |       1 |
| QCOM     |       1 |
| SBUX     |       1 |
| TXN      |       1 |
| VRTX     |       1 |

All gap calls are excluded from cleaned calls and LLM views. No empty LLM inputs
were generated.

## High Word-Count Drop

- High word-count-drop cases: 13

| likely_reason                                                      |   count |
|:-------------------------------------------------------------------|--------:|
| extremely short/operator-only or non-substantive call              |       7 |
| operator/safe-harbor removal or unusual transcript structure       |       2 |
| no Q&A section detected                                            |       2 |
| empty LLM view due to no management text after boilerplate removal |       1 |
| large operator boilerplate share                                   |       1 |

Conclusion: high-drop cases are concentrated in very short/operator-only,
no-management, no-QA, or unusual transcript structures. No systematic over-
deletion of prepared remarks was found.

## Safe Harbor

Uncertain safe harbor components are conservatively retained in LLM text and
flagged for review. This avoids over-deleting substantive prepared remarks.

## No-QA Calls

- No-QA related calls: 80

| likely_reason                                                         |   count |
|:----------------------------------------------------------------------|--------:|
| prepared-remarks-only or transcript lacks analyst question components |      73 |
| extremely short or operator-only transcript                           |       7 |

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

- Current Nasdaq-100 universe is current constituents, not historical constituents, so survivorship bias remains.
- 25 metadata-level deduped calls have no component text and are excluded from cleaned_calls and llm_views.
- NVDA 2006-02-16 transcript_id 6540 and 2006-05-11 transcript_id 6531 are raw metadata/component coverage gaps.
- Small number of no-QA, short, operator-only, or early-format transcripts retain QC warnings.
- Uncertain safe harbor components are conservatively retained to avoid over-deleting substantive prepared remarks.
- Cleaning pipeline is a versioned rule-based pipeline, not LLM-generated summarization.
- cleaned_text and llm_text are cleaned transcript text, not summaries.
- Raw candidate dataset and frozen raw dataset are retained for audit and future re-cleaning.
