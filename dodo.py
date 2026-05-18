"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

After the May-2026 cleanup, the active pipeline is focused on earnings-transcript
sentiment for the Nasdaq-100 universe. Archived contributor code lives under
`src/_archive/` and is intentionally not referenced here.
"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
USER = environ.get("USER") or environ.get("USERNAME", "")  # USER on POSIX, USERNAME on Windows


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["python ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull():
    """Pull data from external sources."""
    yield {
        "name": "manual_macro",
        "doc": "Parse manual Bloomberg macro Excel into parquet",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_manual_macro.py",
        ],
        "targets": [DATA_DIR / "Macro_Data_US.parquet"],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_manual_macro.py",
            str(Path(MANUAL_DATA_DIR) / "Macro_Data_US.xlsx"),
        ],
        "clean": [],
    }
    yield {
        "name": "manual_companies",
        "doc": "Parse manual Bloomberg company prediction Excel into parquet",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_manual_companies.py",
        ],
        "targets": [
            DATA_DIR / "US_Companies_Forecast.parquet",
            DATA_DIR / "US_Companies_Hist_Data.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_manual_companies.py",
            "./src/pull_manual_macro.py",
            str(Path(MANUAL_DATA_DIR) / "US_Companies_Prediction_Data.xlsx"),
        ],
        "clean": [],
    }


def task_build_sentiment():
    """Embed transcripts, score sentiment, roll up to monthly carry-forward.

    Use SYNTHETIC=1 in env to skip OpenAI calls (random unit vectors).
    """
    synth = "--synthetic" if environ.get("SYNTHETIC") else ""
    yield {
        "name": "embed",
        "doc": "Chunk + embed transcript components",
        "actions": [f"python ./src/embed_transcripts.py {synth}".strip()],
        "file_dep": [
            "./src/embed_transcripts.py",
            str(Path(DATA_DIR) / "transcripts" / "AAPL" / "aapl_transcript_components.csv"),
        ],
        "targets": [DATA_DIR / "embeddings_transcripts.parquet"],
        "clean": True,
    }
    yield {
        "name": "score",
        "doc": "Cosine-vs-anchor sentiment scoring",
        "actions": [f"python ./src/score_transcript_sentiment.py {synth}".strip()],
        "file_dep": [
            "./src/score_transcript_sentiment.py",
            str(Path(DATA_DIR) / "embeddings_transcripts.parquet"),
        ],
        "targets": [DATA_DIR / "sentiment_transcripts.parquet"],
        "clean": True,
    }
    yield {
        "name": "monthly",
        "doc": "Carry per-call sentiment forward to monthly EOM",
        "actions": ["python ./src/build_sentiment_features.py"],
        "file_dep": [
            "./src/build_sentiment_features.py",
            str(Path(DATA_DIR) / "sentiment_transcripts.parquet"),
        ],
        "targets": [DATA_DIR / "features_sentiment_monthly.parquet"],
        "clean": True,
    }


def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR.

    Scope shrank after the May-2026 cleanup: the only test still wired into
    the active pipeline is `src/test_misc_tools.py`. Archived test files
    under `src/_archive/` are excluded via pytest's default discovery
    (leading-underscore directory).
    """
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        import subprocess

        result = subprocess.run(
            ["pytest", "src", f"--junitxml={test_output}"],
        )
        if result.returncode != 0:
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": [
            "./src/misc_tools.py",
            "./src/test_misc_tools.py",
        ],
        "clean": True,
        "verbosity": 2,
    }
