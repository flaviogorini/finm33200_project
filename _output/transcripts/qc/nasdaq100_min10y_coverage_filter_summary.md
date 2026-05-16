# Nasdaq-100 Min-10Y Coverage Filter Summary

Generated: 2026-05-15T21:01:26

## Rule

Keep a company-level transcript record if either:

1. `distinct_transcript_years >= 10`, calculated by `ciq_company_id` from cleaned call `transcript_date` calendar years; or
2. `primary_ticker` is one of the manual exceptions: BKR, PDD, ZS.

Manual exception reason: call count is near the usable threshold despite fewer than 10 distinct transcript years.

## Counts

- Full unique `ciq_company_id`: 100
- Automatic keep count: 87
- Manual exception keep count: 3
- Final kept unique `ciq_company_id`: 90
- Dropped unique `ciq_company_id`: 10
- Filtered universe ticker rows: 91
- Filtered mapping ticker rows: 91
- Filtered cleaned calls rows: 6,341
- Filtered LLM views rows: 44,387
- LLM view counts: {"analyst_questions_only": 6341, "full_transcript": 6341, "management_answers_only": 6341, "management_presentation_plus_answers": 6341, "no_operator_no_safe_harbor_full_text": 6341, "presentation_only": 6341, "qa_only": 6341}

## Manual Exceptions

| primary_ticker   |   ciq_company_id | ciq_company_name     | first_transcript_date   | last_transcript_date   |   distinct_transcript_years |   deduped_call_count | coverage_status       | coverage_filter_reason                |
|:-----------------|-----------------:|:---------------------|:------------------------|:-----------------------|----------------------------:|---------------------:|:----------------------|:--------------------------------------|
| BKR              |        425005479 | Baker Hughes Company | 2017-10-20              | 2025-10-24             |                           9 |                   33 | keep_manual_exception | call_count_near_threshold_manual_keep |
| PDD              |        572577901 | PDD Holdings Inc.    | 2018-08-30              | 2025-11-18             |                           8 |                   30 | keep_manual_exception | call_count_near_threshold_manual_keep |
| ZS               |         58838228 | Zscaler, Inc.        | 2018-06-06              | 2025-11-25             |                           8 |                   31 | keep_manual_exception | call_count_near_threshold_manual_keep |

## Dropped Companies

| primary_ticker   |   ciq_company_id | ciq_company_name                 | first_transcript_date   | last_transcript_date   |   distinct_transcript_years |   deduped_call_count | coverage_filter_reason          |
|:-----------------|-----------------:|:---------------------------------|:------------------------|:-----------------------|----------------------------:|---------------------:|:--------------------------------|
| ABNB             |        115705393 | Airbnb, Inc.                     | 2021-02-25              | 2025-11-06             |                           5 |                   20 | distinct_transcript_years_lt_10 |
| APP              |        231651802 | AppLovin Corporation             | 2021-05-12              | 2025-11-05             |                           5 |                   19 | distinct_transcript_years_lt_10 |
| ARM              |        667281196 | Arm Holdings plc                 | 2023-11-08              | 2025-11-05             |                           3 |                    9 | distinct_transcript_years_lt_10 |
| CEG              |          3136719 | Constellation Energy Corporation | 2022-05-12              | 2025-11-07             |                           4 |                   13 | distinct_transcript_years_lt_10 |
| CRWD             |        420347413 | CrowdStrike Holdings, Inc.       | 2019-07-18              | 2025-12-02             |                           7 |                   27 | distinct_transcript_years_lt_10 |
| DASH             |        243735719 | DoorDash, Inc.                   | 2021-02-25              | 2025-11-05             |                           5 |                   20 | distinct_transcript_years_lt_10 |
| DDOG             |        134521275 | Datadog, Inc.                    | 2019-11-12              | 2025-11-06             |                           7 |                   25 | distinct_transcript_years_lt_10 |
| GEHC             |       1804385326 | GE HealthCare Technologies Inc.  | 2023-01-30              | 2025-10-29             |                           3 |                   12 | distinct_transcript_years_lt_10 |
| PLTR             |         43580005 | Palantir Technologies Inc.       | 2020-11-12              | 2025-11-03             |                           6 |                   21 | distinct_transcript_years_lt_10 |
| SNDK             |       1860586153 | Sandisk Corporation              | 2025-05-07              | 2025-11-06             |                           1 |                    3 | distinct_transcript_years_lt_10 |

## GOOG / GOOGL Handling

Coverage is computed by `ciq_company_id`, not ticker. GOOG / GOOGL are represented once at company level:

| primary_ticker   | related_tickers   |   ciq_company_id | ciq_company_name   |   distinct_transcript_years |   deduped_call_count | coverage_status   |
|:-----------------|:------------------|-----------------:|:-------------------|----------------------------:|---------------------:|:------------------|
| GOOGL            | GOOG|GOOGL        |            29096 | Alphabet Inc.      |                          20 |                   79 | keep_min10y       |

## Outputs

- Filtered universe: `_data/transcripts/_meta/nasdaq100_constituents_min10y_coverage.csv`
- Filtered CIQ mapping: `_data/transcripts/_meta/ciq_company_mapping_min10y_coverage.csv`
- Filtered cleaned calls: `_data/transcripts/processed/nasdaq100_cleaned_calls_min10y_coverage.parquet`
- Filtered LLM views: `_data/transcripts/processed/nasdaq100_llm_views_min10y_coverage.parquet`
- Coverage filter QC: `_output/transcripts/qc/nasdaq100_min10y_coverage_filter_qc.csv`
- Manifest: `_output/transcripts/qc/nasdaq100_min10y_coverage_dataset_manifest.json`

Filtered components were not generated in this step to avoid creating another large file unless needed.

## Known Limitations

- This is a coverage-based research filter, not a change to the frozen full cleaned transcript dataset.
- Short-history companies remain available in the original full cleaned parquet files.
- Manual exceptions BKR, PDD, and ZS are kept despite fewer than 10 distinct transcript years.
- The universe remains current Nasdaq-100, not historical constituents.
