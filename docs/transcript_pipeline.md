# Nasdaq-100 Earnings Call Transcript Pipeline

## Purpose

This pipeline builds a frozen Nasdaq-100 earnings call transcript dataset for
LLM-based fundamentals forecasting. The output is designed to support modeling
experiments that use earnings call text to forecast future company fundamentals,
earnings, and related outcomes.

## Universe

The first frozen version uses the current Nasdaq-100 universe as of 2026-05-13.
It does not use historical Nasdaq-100 constituents.

This choice keeps the pipeline stable and reproducible for the initial modeling
phase, but it introduces survivorship bias. Historical constituent tracking,
ticker changes, mergers, delistings, and company identifier changes should be
handled in a later universe-construction phase.

## Final Frozen Dataset

The frozen cleaned transcript dataset is versioned as `full_cleaning_v1.0`.
It inherits its cleaning rules from the validated sample run
`sample_cleaning_v0.2`.

The final dataset has three logical layers:

- Cleaned components: one row per transcript component / speaker turn.
- Cleaned calls: one row per earnings call.
- LLM views: long-format table with multiple text views per call.

The default modeling view is:

```text
no_operator_no_safe_harbor_full_text
```

Other LLM views are retained for experiments:

- `full_transcript`
- `presentation_only`
- `qa_only`
- `management_answers_only`
- `analyst_questions_only`
- `management_presentation_plus_answers`
- `no_operator_no_safe_harbor_full_text`

## Key Counts

- Cleaned components: 452,655
- Cleaned calls: 6,510
- LLM views: 45,570
- Unique tickers: 100
- Unique Capital IQ company IDs: 100
- LLM views per call: 7
- Cleaned transcript date range: 2005-11-01 to 2025-12-19

## File Descriptions

### Metadata

- `_data/transcripts/_meta/nasdaq100_constituents.csv`
  - Current Nasdaq-100 universe used by the pipeline.
  - Includes ticker, company name, exchange, sector/industry fields, related
    tickers, and universe source/as-of date.

- `_data/transcripts/_meta/ciq_company_mapping.csv`
  - Ticker to Capital IQ company ID mapping.
  - Used to run transcript extraction at the company level rather than the
    share-class level.

### Final Processed Data

These files are the main modeling inputs but are too large for ordinary GitHub
commits. They should be obtained from external storage or regenerated locally.

- `_data/transcripts/processed/nasdaq100_cleaned_components.parquet`
  - Component-level cleaned transcript dataset.
  - Preserves speaker, section, component order, raw component text, cleaned
    component text, and cleaning flags.

- `_data/transcripts/processed/nasdaq100_cleaned_calls.parquet`
  - Call-level dataset.
  - Contains call-level raw, cleaned, LLM-ready, presentation, Q&A, management
    answer, analyst question, and operator text fields.

- `_data/transcripts/processed/nasdaq100_llm_views.parquet`
  - Long-format LLM input dataset.
  - Recommended starting point for teammates.
  - Filter to `view_name == "no_operator_no_safe_harbor_full_text"` for the
    default modeling view.

### Final Manifests And Reports

- `_output/transcripts/qc/nasdaq100_raw_dataset_frozen_manifest.json`
  - Frozen raw dataset manifest.
  - Records the raw extraction source, date range, WRDS/Capital IQ tables,
    earnings call filter, deduplication rule, and known limitations.

- `_output/transcripts/qc/nasdaq100_raw_dataset_freeze_report.md`
  - Human-readable raw dataset freeze report.

- `_output/transcripts/qc/nasdaq100_cleaned_dataset_frozen_manifest.json`
  - Frozen cleaned dataset manifest.
  - Records `full_cleaning_v1.0`, output paths, row counts, cleaning rules,
    default modeling view, QC distributions, and known limitations.

- `_output/transcripts/qc/nasdaq100_cleaned_dataset_freeze_report.md`
  - Human-readable cleaned dataset freeze report.

- `_output/transcripts/qc/nasdaq100_cleaning_summary.md`
  - Full cleaning summary with call counts, metadata/component coverage, word
    counts, speaker classification, and warning distributions.

- `_output/transcripts/qc/nasdaq100_cleaning_final_review_summary.md`
  - Final pre-freeze manual review summary.
  - Documents needs-review calls, metadata/component gaps, high word-count-drop
    cases, uncertain safe harbor handling, and the freeze recommendation.

### Supporting Final QC Files

- `_output/transcripts/qc/nasdaq100_cleaning_manifest.json`
  - Manifest for the full cleaning run.

- `_output/transcripts/qc/nasdaq100_raw_extraction_manifest.json`
  - Manifest for the full raw extraction run.

- `_output/transcripts/qc/nasdaq100_raw_extraction_summary.md`
  - Summary of full raw extraction.

- `_output/transcripts/qc/nasdaq100_metadata_component_gap_calls.csv`
  - Calls present in deduped metadata but missing component text.

- `_output/transcripts/qc/nasdaq100_cleaning_needs_review_calls.csv`
  - Calls with cleaning or parsing warnings.

- `_output/transcripts/qc/nasdaq100_high_word_drop_calls.csv`
  - Calls with high raw-to-LLM word count drop or empty LLM text.

- `_output/transcripts/qc/nasdaq100_no_qa_calls.csv`
  - Calls where Q&A was not detected.

## Reproduction Steps

Reproduction requires WRDS / Capital IQ access. Do not store WRDS usernames,
passwords, API keys, or other credentials in the repository.

Install the project dependencies first. For the transcript pipeline, the
required packages are covered by `requirements.txt`; the relevant subset is
`pandas`, `pyarrow`, `wrds`, `python-decouple`, `requests`/standard-library
URL access, and an HTML table parser such as `lxml`.

Configure credentials locally. For example, use environment variables or a
local `.env` file:

```text
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
DATA_DIR=_data
OUTPUT_DIR=_output
```

Do not commit `.env` or any credential file.

High-level reproduction flow:

1. Build or verify the current Nasdaq-100 universe.
2. Build or verify the ticker to Capital IQ company ID mapping.
3. Inspect WRDS / Capital IQ schema before querying transcript tables.
4. Check transcript metadata availability by company ID.
5. Extract raw earnings call metadata and component text at the unique
   `ciq_company_id` level.
6. Preserve all raw transcript candidates and separately create metadata-level
   deduped raw transcripts.
7. Freeze the raw transcript dataset.
8. Run `full_cleaning_v1.0` cleaning and section parsing.
9. Generate final QC, manual review package, and cleaned dataset freeze
   manifest/report.

Command sequence:

```bash
# Step 1: prepare Python environment
pip install -r requirements.txt

# Step 2: build or refresh the current Nasdaq-100 universe
python src/build_nasdaq100_universe.py

# Step 3: build / validate Capital IQ company mapping
python src/build_ciq_company_mapping.py

# Step 4: run enhanced mapping QC and transcript availability check
python src/check_transcript_mapping_availability.py

# Step 5: full raw extraction at unique ciq_company_id level
# The script name is historical: it supports full extraction through --label.
TICKERS=$(python - <<'PY'
import pandas as pd
mapping = pd.read_csv("_data/transcripts/_meta/ciq_company_mapping.csv")
print(" ".join(mapping["ticker"].astype(str).str.upper().tolist()))
PY
)
python src/extract_sample_raw_transcripts.py \
  --label nasdaq100 \
  --start-date 2005-01-01 \
  --end-date 2025-12-31 \
  --tickers $TICKERS

# Step 6: run full cleaning / section parsing
python src/clean_sample_transcripts.py --mode full

# Step 7: build final cleaning review package
python src/build_cleaning_final_review.py

# Step 8: freeze cleaned dataset manifest/report
python src/freeze_cleaned_dataset.py
```

Notes:

- `src/extract_sample_raw_transcripts.py` and
  `src/clean_sample_transcripts.py` were originally developed during sample
  validation. They now support full runs using `--label nasdaq100` and
  `--mode full`, respectively.
- Raw dataset freeze artifacts are committed as reference files. The current
  repository does not yet include a dedicated standalone raw-freeze script; the
  full raw extraction script writes the raw extraction manifest, QC, summary,
  raw candidate parquet, and deduped raw parquet needed by the cleaning step.
- The processed parquet outputs are large and are not committed to ordinary
  GitHub. Reproduce them locally or obtain them from the project data store.

Core scripts:

- `src/build_nasdaq100_universe.py`
- `src/build_ciq_company_mapping.py`
- `src/check_transcript_mapping_availability.py`
- `src/extract_sample_raw_transcripts.py`
- `src/clean_sample_transcripts.py`
- `src/build_cleaning_final_review.py`
- `src/freeze_cleaned_dataset.py`

## Teammate Usage

For modeling, start from:

```text
_data/transcripts/processed/nasdaq100_llm_views.parquet
```

Use the default view:

```python
view_name == "no_operator_no_safe_harbor_full_text"
```

The processed parquet files are too large for normal GitHub commits. Teammates
should obtain them from the agreed external storage location or regenerate them
with WRDS / Capital IQ access.

## Known Limitations

- The universe is the current Nasdaq-100 universe as of 2026-05-13, not a
  historical Nasdaq-100 universe.
- This creates survivorship bias.
- 25 metadata-only calls have no component text and are excluded from cleaned
  calls and LLM views.
- NVDA transcript IDs `6540` and `6531` are early metadata/component coverage
  gaps.
- Some no-Q&A, short, operator-only, and early-format transcripts retain QC
  warnings.
- Uncertain safe harbor components are conservatively retained to avoid
  over-deleting substantive prepared remarks.
- Cleaning is a versioned rule-based pipeline, not LLM-generated summary.
- Cleaned text and LLM text are still cleaned transcript text, not summaries.
- Raw candidate and frozen raw datasets are retained outside normal GitHub
  tracking for audit and future re-cleaning.

## Next Stage

The next stage is transcript modeling dataset construction:

- Align transcript events to fiscal periods.
- Match calls to earnings announcement dates.
- Link company-level transcripts to Compustat fundamentals.
- Link to analyst forecasts where available.
- Build future 1- to 4-quarter fundamentals / earnings forecast targets.
- Decide whether modeling is company-level only or mapped back to ticker-level
  observations for multi-share-class companies.
