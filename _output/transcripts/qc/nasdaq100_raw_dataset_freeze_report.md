# Nasdaq-100 Raw Transcript Dataset Freeze Report

Generated: 2026-05-13T15:57:50
Freeze run ID: `nasdaq100_raw_frozen_20260513_155750`

## Freeze Status

- Raw candidate dataset frozen: yes
- Deduped raw metadata dataset frozen: yes
- Deduped raw component dataset frozen: yes
- Raw candidate versions retained for audit: yes
- Old AAPL processed dataset modified: no

## Frozen Inputs

- Universe file: `/Users/yujiadi/Desktop/finm33200_project/_data/transcripts/_meta/nasdaq100_constituents.csv`
- Company mapping file: `/Users/yujiadi/Desktop/finm33200_project/_data/transcripts/_meta/ciq_company_mapping.csv`
- Universe as-of date: ['2026-05-13']
- Universe source: ['https://en.wikipedia.org/wiki/Nasdaq-100; Parsed from the Wikipedia Nasdaq-100 current components table; cross-check note: Nasdaq official companies page observed during setup was marked Last updated 05/19/2025 and may lag current 2026 changes.']

## Frozen Outputs

- Raw candidate metadata: `_data/transcripts/raw/nasdaq100_raw_transcript_metadata.parquet`
- Raw candidate components: `_data/transcripts/raw/nasdaq100_raw_transcripts.parquet`
- Deduped raw metadata: `_data/transcripts/raw/nasdaq100_raw_transcript_metadata_deduped.parquet`
- Deduped raw components: `_data/transcripts/raw/nasdaq100_raw_transcripts_deduped.parquet`

## Counts

- Unique ticker count: 101
- Unique CIQ company IDs: 100
- Raw candidate metadata rows: 20,467
- Raw component rows: 1,396,726
- Deduped call count: 6,535
- Deduped component rows: 452,655

## Extraction Logic

- Date range: 2005-01-01 to 2025-12-31
- Metadata table: `ciq.wrds_transcript_detail`
- Component/person table: `ciq.wrds_transcript_person`
- Text table: `ciq.ciqtranscriptcomponent`
- Join keys: `{"metadata_to_components": ["transcript_id"], "person_to_text": ["transcriptcomponentid"]}`
- Earnings-call filter: `keydeveventtypeid = 48 OR keydeveventtypename ILIKE '%Earnings%' OR headline ILIKE '%Earnings Call%'`
- Dedupe rule: `presentation_final_then_collection_audited_proofed_edited_corrected_then_latest_creation_then_text_length`

## QC Conclusions

- Failed extraction companies: none
- Companies with deduped call count = 0: none
- Companies with missing/blank component text: AXON
- AXON blank component rows: 2 accepted known issue. Raw remains frozen; cleaning stage should drop/flag blank cleaned components.
- AAPL benchmark: old processed count 80; new deduped count 80; aligned = True.
- GOOG/GOOGL: one company-level extraction unit; GOOG duplicate share class is not separately extracted.

## Low Coverage / Short History

| ticker   |   ciq_company_id |   deduped_call_count | years_covered            |
|:---------|-----------------:|---------------------:|:-------------------------|
| ABNB     |        115705393 |                   20 | 2021|2022|2023|2024|2025 |
| APP      |        231651802 |                   19 | 2021|2022|2023|2024|2025 |
| ARM      |        667281196 |                    9 | 2023|2024|2025           |
| CEG      |          3136719 |                   13 | 2022|2023|2024|2025      |
| DASH     |        243735719 |                   20 | 2021|2022|2023|2024|2025 |
| GEHC     |       1804385326 |                   12 | 2023|2024|2025           |
| SNDK     |       1860586153 |                    3 | 2025                     |

These are accepted as coverage limitations, not pipeline failures.

## Readiness

The raw dataset is officially frozen and can be used as the source for cleaning / section parsing design. Cleaning should use the deduped raw component dataset as the primary input and keep the raw candidate dataset for audit and future re-deduplication.
