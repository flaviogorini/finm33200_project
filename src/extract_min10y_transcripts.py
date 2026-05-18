"""Run the min10y WRDS Capital IQ transcript raw extraction.

This wrapper exists so teammates do not need to remember the long set of
min10y arguments for ``extract_sample_raw_transcripts.py``. It intentionally
does not implement extraction logic; it delegates to the existing script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "src" / "extract_sample_raw_transcripts.py"
    cmd = [
        sys.executable,
        str(script),
        "--label",
        "nasdaq100_min10y",
        "--mapping-path",
        "_data/transcripts/_meta/ciq_company_mapping_min10y_coverage.csv",
        "--universe-path",
        "_data/transcripts/_meta/nasdaq100_constituents_min10y_coverage.csv",
        "--schema-output-path",
        "_output/transcripts/qc/nasdaq100_min10y_schema_inspection.json",
        "--start-date",
        "2005-01-01",
        "--end-date",
        "2025-12-31",
        *sys.argv[1:],
    ]
    return subprocess.run(cmd, cwd=repo_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
