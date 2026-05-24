# Transcript Pipeline Reproducibility Checklist

This checklist explains how to reproduce the frozen Nasdaq-100 earnings call
transcript datasets that are too large for ordinary GitHub commits.

## Required Inputs In The Repository

- `src/build_nasdaq100_universe.py`
- `src/build_ciq_company_mapping.py`
- `src/check_transcript_mapping_availability.py`
- `src/extract_sample_raw_transcripts.py`
- `src/clean_sample_transcripts.py`
- `src/build_cleaning_final_review.py`
- `src/freeze_cleaned_dataset.py`
- `data_manual/_meta/nasdaq100_constituents.csv`
- `data_manual/_meta/ciq_company_mapping.csv`
- `_output/transcripts/qc/nasdaq100_raw_dataset_frozen_manifest.json`
- `_output/transcripts/qc/nasdaq100_cleaned_dataset_frozen_manifest.json`
- `_output/transcripts/qc/nasdaq100_cleaned_dataset_freeze_report.md`
- `docs/transcript_pipeline.md`

The three large processed parquet files are not committed to ordinary GitHub:

- `_data/transcripts/processed/nasdaq100_cleaned_components.parquet`
- `_data/transcripts/processed/nasdaq100_cleaned_calls.parquet`
- `_data/transcripts/processed/nasdaq100_llm_views.parquet`

## Required Permissions

- WRDS account access.
- Capital IQ transcript data access through WRDS.
- Local credentials configured outside Git.

Do not commit:

- `.env`
- WRDS username or password
- API keys
- local credential files

## Environment Dependencies

The project has a `requirements.txt`. For this transcript pipeline, the relevant
packages are:

- `pandas`
- `pyarrow`
- `wrds`
- `python-decouple`
- `requests` or standard-library URL access
- `lxml` or another pandas-compatible HTML table parser

Install with:

```bash
pip install -r requirements.txt
```

If the full project requirements are too broad for a teammate's environment, a
future improvement would be to add a small `requirements-transcripts.txt` with
only the transcript pipeline dependencies.

## Credential Setup

Use local environment variables or a local `.env` file:

```text
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
DATA_DIR=_data
OUTPUT_DIR=_output
```

The pipeline reads these values through `src/settings.py`. Do not write
credentials into scripts.

## Reproduction Commands

Run from the repository root.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build or refresh current Nasdaq-100 universe
python src/build_nasdaq100_universe.py

# 3. Build / validate Capital IQ company mapping
python src/build_ciq_company_mapping.py

# 4. Check mapping QC and transcript availability
python src/check_transcript_mapping_availability.py

# 5. Full raw extraction
# The script name is historical; use --label nasdaq100 for the full run.
TICKERS=$(python - <<'PY'
import pandas as pd
mapping = pd.read_csv("data_manual/_meta/ciq_company_mapping.csv")
print(" ".join(mapping["ticker"].astype(str).str.upper().tolist()))
PY
)
python src/extract_sample_raw_transcripts.py \
  --label nasdaq100 \
  --start-date 2005-01-01 \
  --end-date 2025-12-31 \
  --tickers $TICKERS

# 6. Full cleaning and section parsing
python src/clean_sample_transcripts.py --mode full

# 7. Final cleaning review package
python src/build_cleaning_final_review.py

# 8. Cleaned dataset freeze manifest/report
python src/freeze_cleaned_dataset.py
```

## Min-10Y Coverage Reproduction Commands

Use this path when reproducing only the 90-company min-10Y research sample.
This does not require first generating the full 100-company cleaned parquet
files.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract raw transcripts directly from the min-10Y mapping/universe.
# If --tickers is omitted, all tickers in the mapping are used, deduplicated by
# unique ciq_company_id for extraction.
python src/extract_sample_raw_transcripts.py \
  --label nasdaq100_min10y \
  --mapping-path data_manual/_meta/ciq_company_mapping_min10y_coverage.csv \
  --universe-path data_manual/_meta/nasdaq100_constituents_min10y_coverage.csv \
  --schema-output-path _output/transcripts/qc/nasdaq100_min10y_schema_inspection.json \
  --start-date 2005-01-01 \
  --end-date 2025-12-31

# 3. Clean and parse the min-10Y deduped raw component output.
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

Expected min-10Y outputs:

- `_data/transcripts/processed/nasdaq100_cleaned_components_min10y_coverage.parquet`
- `_data/transcripts/processed/nasdaq100_cleaned_calls_min10y_coverage.parquet`
- `_data/transcripts/processed/nasdaq100_llm_views_min10y_coverage.parquet`

Expected min-10Y row counts:

- Unique `ciq_company_id`: 90
- Cleaned calls: 6,341
- LLM views: 44,387
- Default view `no_operator_no_safe_harbor_full_text`: 6,341 rows

Counts may drift slightly if Capital IQ updates historical transcripts.

## Expected Outputs

Raw extraction outputs:

- `_data/transcripts/raw/nasdaq100_raw_transcript_metadata.parquet`
- `_data/transcripts/raw/nasdaq100_raw_transcript_metadata_deduped.parquet`
- `_data/transcripts/raw/nasdaq100_raw_transcripts.parquet`
- `_data/transcripts/raw/nasdaq100_raw_transcripts_deduped.parquet`
- `_output/transcripts/qc/nasdaq100_raw_extraction_manifest.json`
- `_output/transcripts/qc/nasdaq100_raw_extraction_qc.csv`
- `_output/transcripts/qc/nasdaq100_raw_extraction_summary.md`

Processed outputs:

- `_data/transcripts/processed/nasdaq100_cleaned_components.parquet`
- `_data/transcripts/processed/nasdaq100_cleaned_calls.parquet`
- `_data/transcripts/processed/nasdaq100_llm_views.parquet`
- `_output/transcripts/qc/nasdaq100_cleaning_qc.csv`
- `_output/transcripts/qc/nasdaq100_cleaning_summary.md`
- `_output/transcripts/qc/nasdaq100_cleaning_manifest.json`
- `_output/transcripts/qc/nasdaq100_cleaning_final_review_summary.md`
- `_output/transcripts/qc/nasdaq100_cleaned_dataset_frozen_manifest.json`
- `_output/transcripts/qc/nasdaq100_cleaned_dataset_freeze_report.md`

Expected row counts for `full_cleaning_v1.0`:

- Cleaned components: 452,655
- Cleaned calls: 6,510
- LLM views: 45,570
- Unique tickers: 100
- Unique Capital IQ company IDs: 100
- LLM views per call: 7

Default modeling view:

```text
no_operator_no_safe_harbor_full_text
```

## Confirming Reproduction Success

Check the frozen manifest:

```bash
python - <<'PY'
import json
import pandas as pd

with open("_output/transcripts/qc/nasdaq100_cleaned_dataset_frozen_manifest.json") as f:
    manifest = json.load(f)

components = pd.read_parquet("_data/transcripts/processed/nasdaq100_cleaned_components.parquet")
calls = pd.read_parquet("_data/transcripts/processed/nasdaq100_cleaned_calls.parquet")
views = pd.read_parquet("_data/transcripts/processed/nasdaq100_llm_views.parquet")

print(manifest["cleaning_version"])
print(len(components), len(calls), len(views))
print(views["view_name"].value_counts().sort_index())
PY
```

Expected:

- `full_cleaning_v1.0`
- `452655 6510 45570`
- seven views, each with 6,510 rows

## Common Issues

### WRDS Login Failure

Check `WRDS_USERNAME` and `WRDS_PASSWORD` in your local environment or `.env`.
Do not commit these credentials.

### Missing Capital IQ Access

The WRDS login may work while Capital IQ tables are unavailable. The schema
inspection scripts will fail if the account lacks access to the relevant CIQ
tables.

### Large Files Not In GitHub

The processed parquet files exceed ordinary GitHub file-size limits. Obtain
them from external storage or regenerate them locally.

### Metadata / Component Gap

The frozen version has 25 metadata-level deduped calls without component text.
These are excluded from cleaned calls and LLM views. They are raw coverage
limitations, not cleaning failures.

### GitHub 100MB Limit

Do not commit the large processed or raw parquet files with ordinary Git.
Use external storage or Git LFS only if the team explicitly chooses that path.

### Script Naming

Two scripts have historical sample-oriented names:

- `src/extract_sample_raw_transcripts.py`
- `src/clean_sample_transcripts.py`

They support full reproduction with `--label nasdaq100` and `--mode full`.
A future cleanup could add clearer wrappers such as `src/extract_transcripts.py`
or `src/run_transcript_pipeline.py`.

## Known Reproducibility Caveat

The current repository can reproduce the three large processed parquet files
from WRDS / Capital IQ access and the committed metadata/mapping files.

One pipeline polish item remains: raw dataset freeze report generation is not
currently exposed as a standalone script. The committed raw frozen manifest and
freeze report document the frozen raw dataset. Reproducing the processed parquet
files does not require regenerating those raw freeze documents.
