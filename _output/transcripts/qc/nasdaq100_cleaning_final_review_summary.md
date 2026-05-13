# Nasdaq-100 Cleaning Final Review Summary

Generated: 2026-05-13T16:27:13

## Overall Status

- Cleaning version: `full_cleaning_v1.0`
- Source raw dataset version: `nasdaq100_raw_frozen_20260513_155750`
- Cleaned components: 452,655
- Cleaned calls: 6,510
- LLM views: 45,570
- QC rows: 6,510
- Manual review source checked: `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_cleaning_manual_review.csv`

No full cleaning outputs were modified by this review package. No embeddings,
forecasting, financial alignment, raw extraction, or frozen raw parquet changes
were performed.

## Parsing Quality

| flag         |   count |
|:-------------|--------:|
| ok           |    6423 |
| needs_review |      87 |

## Cleaning Warnings

| warning                                                                |   count |
|:-----------------------------------------------------------------------|--------:|
| none                                                                   |    6423 |
| no_qa_section_detected                                                 |      73 |
| llm_text_empty; no_management_speaker_detected; no_qa_section_detected |       7 |
| no_presentation_section_detected                                       |       6 |
| llm_text_empty; no_management_speaker_detected                         |       1 |

## Needs Review Calls

- Needs review calls: 87

| warning_type                                                           |   count |
|:-----------------------------------------------------------------------|--------:|
| no_qa_section_detected                                                 |      73 |
| llm_text_empty; no_management_speaker_detected; no_qa_section_detected |       7 |
| no_presentation_section_detected                                       |       6 |
| llm_text_empty; no_management_speaker_detected                         |       1 |

Interpretation: most needs_review calls are isolated section-structure issues,
not broad speaker parsing failures. The main pattern is `no_qa_section_detected`,
especially for prepared-remarks-only, early-format, or non-standard Capital IQ
section metadata.

Output: `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_cleaning_needs_review_calls.csv`

## Metadata / Component Gap

- Metadata-level deduped calls: 6535
- Component-level cleaned calls: 6510
- Metadata exists but no component rows: 25

| ticker   |   metadata_component_gap_count |
|:---------|-------------------------------:|
| WMT      |                              9 |
| KLAC     |                              2 |
| NVDA     |                              2 |
| STX      |                              2 |
| TTWO     |                              2 |
| XEL      |                              2 |
| CTSH     |                              1 |
| MSFT     |                              1 |
| QCOM     |                              1 |
| SBUX     |                              1 |
| TXN      |                              1 |
| VRTX     |                              1 |

All gap calls were verified as absent from `cleaned_calls` and `llm_views`, so
no empty LLM inputs are generated. These are raw coverage limitations, not
cleaning failures.

Output: `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_metadata_component_gap_calls.csv`

## High Word-Count Drop

- High drop calls exported: 13
- Criteria: `word_count_drop_pct > 0.30`, `llm_word_count = 0`, or `cleaned_word_count = 0`

| likely_reason                                                      |   count |
|:-------------------------------------------------------------------|--------:|
| extremely short/operator-only or non-substantive call              |       7 |
| operator/safe-harbor removal or unusual transcript structure       |       2 |
| no Q&A section detected                                            |       2 |
| empty LLM view due to no management text after boilerplate removal |       1 |
| large operator boilerplate share                                   |       1 |

Word-count drop distribution:

|       |   word_count_drop_pct |
|:------|----------------------:|
| count |          6510         |
| mean  |             0.0573375 |
| std   |             0.0415099 |
| min   |             0         |
| 50%   |             0.0547136 |
| 75%   |             0.0676203 |
| 90%   |             0.0811796 |
| 95%   |             0.0910023 |
| 99%   |             0.116917  |
| max   |             1         |

The high-drop cases are concentrated in extremely short/operator-only calls,
empty default LLM views due to missing management text, no Q&A calls, or unusual
transcript structure. This does not indicate systematic over-deletion of
prepared remarks.

Output: `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_high_word_drop_calls.csv`

## Uncertain Safe Harbor

- Total uncertain safe harbor components in full cleaned components: 5,717
- Sample exported: 100
- Tickers represented in sample: 65

These are long mixed components retained in LLM text and flagged for review.
This is intentionally conservative: it reduces over-deletion risk.

Output: `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_uncertain_safe_harbor_sample.csv`

## No-QA Calls

- No-QA calls exported: 80

| likely_reason                                                         |   count |
|:----------------------------------------------------------------------|--------:|
| prepared-remarks-only or transcript lacks analyst question components |      73 |
| extremely short or operator-only transcript                           |       7 |

The no-QA cases do not appear to be a global parsing failure. They are mainly
prepared-remarks-only transcripts, early/short transcripts, or company-specific
Capital IQ section metadata patterns.

Output: `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_no_qa_calls.csv`

## Manual Spot Check Guidance

Please inspect:

- `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_cleaning_needs_review_calls.csv` for the 87 needs_review calls
- `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_metadata_component_gap_calls.csv` for the 25 metadata/component gap calls
- `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_high_word_drop_calls.csv` for high drop and empty LLM cases
- `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_uncertain_safe_harbor_sample.csv` to confirm mixed safe-harbor components are retained appropriately
- `/Users/yujiadi/Desktop/finm33200_project/_output/transcripts/qc/nasdaq100_no_qa_calls.csv` for no-QA section cases

## Freeze Recommendation

Recommendation: The full_cleaning_v1.0 dataset is suitable for candidate freeze, subject to user manual spot-check approval.

Known limitations to record if frozen:

- Current Nasdaq-100 universe is current constituents, not historical constituents.
- 25 metadata-level deduped calls have no component rows and are excluded from cleaned calls and LLM views.
- NVDA transcript_id 6540 and 6531 are part of the metadata/component gap and are accepted as raw coverage limitations.
- Some calls are prepared-remarks-only, operator-only, extremely short, or have incomplete early-year section metadata.
- `no_qa_section_detected` warnings remain as QC flags rather than automatic failures.
- Uncertain safe harbor components are retained in LLM text to avoid over-deleting substantive prepared remarks.
- Cleaning is rule-based and should remain versioned as `full_cleaning_v1.0`.
