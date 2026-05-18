"""Run min10y transcript cleaning with the canonical input/output paths.

This wrapper keeps the final-project min10y cleaning command short while
delegating all cleaning logic to ``clean_sample_transcripts.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "src" / "clean_sample_transcripts.py"
    cmd = [
        sys.executable,
        str(script),
        "--mode",
        "full",
        "--label",
        "nasdaq100_min10y",
        "--input-raw-components-path",
        "_data/transcripts/raw/nasdaq100_min10y_raw_transcripts_deduped.parquet",
        "--input-raw-metadata-path",
        "_data/transcripts/raw/nasdaq100_min10y_raw_transcript_metadata_deduped.parquet",
        "--output-cleaned-components-path",
        "_data/transcripts/processed/nasdaq100_cleaned_components_min10y_coverage.parquet",
        "--output-cleaned-calls-path",
        "_data/transcripts/processed/nasdaq100_cleaned_calls_min10y_coverage.parquet",
        "--output-llm-views-path",
        "_data/transcripts/processed/nasdaq100_llm_views_min10y_coverage.parquet",
        "--output-qc-path",
        "_output/transcripts/qc/nasdaq100_min10y_cleaning_qc.csv",
        "--output-summary-path",
        "_output/transcripts/qc/nasdaq100_min10y_cleaning_summary.md",
        "--output-manual-review-path",
        "_output/transcripts/qc/nasdaq100_min10y_cleaning_manual_review.csv",
        "--output-manifest-path",
        "_output/transcripts/qc/nasdaq100_min10y_cleaning_manifest.json",
        *sys.argv[1:],
    ]
    return subprocess.run(cmd, cwd=repo_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
