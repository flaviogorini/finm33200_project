# Min10y Transcript Pipeline

This is the transcript pipeline to follow for final project replication.
Use this document instead of the older full Nasdaq-100 transcript instructions
in `README.md`, `dodo.py`, or `docs/transcript_pipeline.md`.

## Scope

The final project uses the min10y universe: roughly 90 unique Capital IQ
company IDs with at least 10 years of earnings-call transcript coverage, not
the full current Nasdaq-100 universe.

Core universe inputs:

- `data_manual/_meta/ciq_company_mapping_min10y_coverage.csv`
- `data_manual/_meta/nasdaq100_constituents_min10y_coverage.csv`

The min10y mapping has 91 ticker rows but 90 unique company IDs. GOOG and
GOOGL share the same Capital IQ company ID, so extraction is done at the
unique `ciq_company_id` level to avoid duplicate downloads.

## Step 1: Raw Extraction

Recommended command:

```bash
python src/extract_min10y_transcripts.py
```

Equivalent expanded command:

```bash
python src/extract_sample_raw_transcripts.py \
  --label nasdaq100_min10y \
  --mapping-path data_manual/_meta/ciq_company_mapping_min10y_coverage.csv \
  --universe-path data_manual/_meta/nasdaq100_constituents_min10y_coverage.csv \
  --schema-output-path _output/transcripts/qc/nasdaq100_min10y_schema_inspection.json \
  --start-date 2005-01-01 \
  --end-date 2025-12-31
```

Inputs:

- `data_manual/_meta/ciq_company_mapping_min10y_coverage.csv`
- `data_manual/_meta/nasdaq100_constituents_min10y_coverage.csv`
- WRDS credentials through `.env` or environment variables
- Capital IQ transcript access through WRDS

Outputs:

- `_data/transcripts/raw/nasdaq100_min10y_raw_transcript_metadata.parquet`
- `_data/transcripts/raw/nasdaq100_min10y_raw_transcript_metadata_deduped.parquet`
- `_data/transcripts/raw/nasdaq100_min10y_raw_transcripts.parquet`
- `_data/transcripts/raw/nasdaq100_min10y_raw_transcripts_deduped.parquet`
- `_output/transcripts/qc/nasdaq100_min10y_schema_inspection.json`
- `_output/transcripts/qc/nasdaq100_min10y_raw_extraction_manifest.json`
- `_output/transcripts/qc/nasdaq100_min10y_raw_extraction_qc.csv`
- `_output/transcripts/qc/nasdaq100_min10y_raw_extraction_summary.md`

If no `--tickers` argument is passed, the extractor reads all ticker rows from
the min10y mapping file and deduplicates extraction by unique `ciq_company_id`.

## Step 2: Cleaning

Recommended command:

```bash
python src/clean_min10y_transcripts.py
```

Equivalent expanded command:

```bash
python src/clean_sample_transcripts.py \
  --mode full \
  --label nasdaq100_min10y \
  --input-raw-components-path _data/transcripts/raw/nasdaq100_min10y_raw_transcripts_deduped.parquet \
  --input-raw-metadata-path _data/transcripts/raw/nasdaq100_min10y_raw_transcript_metadata_deduped.parquet \
  --output-cleaned-components-path _data/transcripts/processed/nasdaq100_cleaned_components_min10y_coverage.parquet \
  --output-cleaned-calls-path _data/transcripts/processed/nasdaq100_cleaned_calls_min10y_coverage.parquet \
  --output-llm-views-path _data/transcripts/processed/nasdaq100_llm_views_min10y_coverage.parquet \
  --output-qc-path _output/transcripts/qc/nasdaq100_min10y_cleaning_qc.csv \
  --output-summary-path _output/transcripts/qc/nasdaq100_min10y_cleaning_summary.md \
  --output-manual-review-path _output/transcripts/qc/nasdaq100_min10y_cleaning_manual_review.csv \
  --output-manifest-path _output/transcripts/qc/nasdaq100_min10y_cleaning_manifest.json
```

Inputs:

- `_data/transcripts/raw/nasdaq100_min10y_raw_transcripts_deduped.parquet`
- `_data/transcripts/raw/nasdaq100_min10y_raw_transcript_metadata_deduped.parquet`

Outputs:

- `_data/transcripts/processed/nasdaq100_cleaned_components_min10y_coverage.parquet`
- `_data/transcripts/processed/nasdaq100_cleaned_calls_min10y_coverage.parquet`
- `_data/transcripts/processed/nasdaq100_llm_views_min10y_coverage.parquet`
- `_output/transcripts/qc/nasdaq100_min10y_cleaning_qc.csv`
- `_output/transcripts/qc/nasdaq100_min10y_cleaning_summary.md`
- `_output/transcripts/qc/nasdaq100_min10y_cleaning_manual_review.csv`
- `_output/transcripts/qc/nasdaq100_min10y_cleaning_manifest.json`

For downstream modeling, start from:

```text
_data/transcripts/processed/nasdaq100_llm_views_min10y_coverage.parquet
```

The default modeling view remains:

```text
no_operator_no_safe_harbor_full_text
```

## Legacy / Archive Candidates

The files below may be useful for audit, debugging, or historical comparison,
but they are not the final transcript pipeline.

Full Nasdaq-100 outputs:

- `_data/transcripts/raw/nasdaq100_raw_*`
- `_data/transcripts/processed/nasdaq100_cleaned_components.parquet`
- `_data/transcripts/processed/nasdaq100_cleaned_calls.parquet`
- `_data/transcripts/processed/nasdaq100_llm_views.parquet`
- `_output/transcripts/qc/nasdaq100_raw_*`
- `_output/transcripts/qc/nasdaq100_cleaning_*`
- `_output/transcripts/qc/nasdaq100_cleaned_dataset_*`

AAPL-only outputs:

- `_data/transcripts/AAPL/*`

Sample outputs:

- `_data/transcripts/raw/sample_*`
- `_data/transcripts/interim/sample_*`
- `_output/transcripts/qc/sample_*`

Pilot20 outputs:

- `_data/transcripts/raw/pilot20_*`
- `_output/transcripts/qc/pilot20_*`

Test min10y smoke-test outputs:

- `_data/transcripts/raw/test_min10y_*`
- `_output/transcripts/qc/test_min10y_*`

Do not delete or move these files until the team confirms that any needed audit
copies exist in external storage. The immediate goal is to make the min10y path
clear, not to remove historical artifacts.

## Current Caveats

- Some legacy scripts and docs still default to full `nasdaq100_*` paths.
  For final project replication, follow this min10y document.
- `src/extract_sample_raw_transcripts.py` and
  `src/clean_sample_transcripts.py` have historical names. Use the min10y
  wrapper scripts unless you need to customize paths manually.
- Large parquet outputs are local data artifacts and may not be present in a
  fresh GitHub checkout. Reproduce them from WRDS or retrieve them from the
  team data store.
