"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
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

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


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
    """Pull data from external sources.

    Note: the cookiecutter `crsp_stock` and `crsp_compustat` yields were
    removed; their parquet outputs aren't consumed by anything in the
    current pipeline. The scripts (src/pull_CRSP_*.py) remain on disk
    in case CRSP is reintroduced as a data source later.
    """
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
    yield {
        "name": "sec_10q_filings",
        "doc": "Pull SEC 10-Q filings from SEC EDGAR via HTTPS (edgartools)",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_sec_10q_filings.py",
        ],
        "targets": [DATA_DIR / "sec_10q" / "_meta" / "filing_index.csv"],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_sec_10q_filings.py",
        ],
        "clean": [],
    }


def task_process_10q():
    """Process the cached 10-Q filings into a (date, ticker) monthly panel.

    Stages: clean → score → panel. The `embed` and `analyze` stages are
    intentionally NOT dependencies of `panel`: they require OPENAI_API_KEY
    and incur API cost, so they are opt-in via `doit process_10q:embed` /
    `doit process_10q:analyze`. When the key is unset, both scripts print a
    notice and exit cleanly.

    `analyze` is the generative-AI layer (gpt-4o-mini reads each 10-Q and
    scores disclosure change vs the prior filing) that powers model variants
    V4. Re-run `process_10q:panel` and `build_panel` after `analyze` so
    the AI columns reach the unified panel.
    """
    yield {
        "name": "clean",
        "doc": "Parse SGML/HTML 10-Q filings into per-section text files",
        "actions": ["python ./src/clean_sec_10q_text.py"],
        "targets": [DATA_DIR / "sec_10q" / "_meta" / "cleaned_index.csv"],
        "file_dep": [
            "./src/settings.py",
            "./src/clean_sec_10q_text.py",
            DATA_DIR / "sec_10q" / "_meta" / "filing_index.csv",
        ],
        "clean": [],
    }
    yield {
        "name": "score",
        "doc": "Compute LM-dictionary sentiment + lexical drift features per filing",
        "actions": ["python ./src/score_sec_10q_text.py"],
        "targets": [DATA_DIR / "sec_10q" / "10q_features.parquet"],
        "file_dep": [
            "./src/settings.py",
            "./src/score_sec_10q_text.py",
            DATA_DIR / "sec_10q" / "_meta" / "cleaned_index.csv",
        ],
        "clean": [],
    }
    yield {
        "name": "panel",
        "doc": "Build point-in-time (date, ticker) monthly 10-Q panel",
        "actions": ["python ./src/build_10q_monthly_panel.py"],
        "targets": [DATA_DIR / "sec_10q_monthly_panel.parquet"],
        "file_dep": [
            "./src/settings.py",
            "./src/build_10q_monthly_panel.py",
            DATA_DIR / "sec_10q" / "10q_features.parquet",
        ],
        "clean": [],
    }
    yield {
        "name": "embed",
        "doc": (
            "Optional: OpenAI embeddings + FAISS for semantic drift. "
            "Run only when OPENAI_API_KEY is set; not a dependency of `panel`."
        ),
        "actions": ["python ./src/embed_sec_10q_text.py"],
        "targets": [],
        "file_dep": [
            "./src/settings.py",
            "./src/embed_sec_10q_text.py",
            DATA_DIR / "sec_10q" / "_meta" / "cleaned_index.csv",
        ],
        "uptodate": [False],
        "clean": [],
    }
    yield {
        "name": "analyze",
        "doc": (
            "Optional: gpt-4o-mini structured analysis of 10-Q disclosure "
            "change vs the prior filing (powers model variants V4). "
            "Needs OPENAI_API_KEY; not a dependency of `panel`. Responses "
            "are cached under _data/sec_10q/_llm_cache/ so re-runs are free."
        ),
        "actions": ["python ./src/analyze_sec_10q_llm.py"],
        "targets": [],
        "file_dep": [
            "./src/settings.py",
            "./src/analyze_sec_10q_llm.py",
            DATA_DIR / "sec_10q" / "_meta" / "cleaned_index.csv",
        ],
        "uptodate": [False],
        "clean": [],
    }


def task_build_features():
    """Build per-source monthly feature parquets from the manual Bloomberg pulls."""
    yield {
        "name": "fundamentals",
        "doc": "PIT daily Bloomberg → monthly EOM fundamentals",
        "actions": ["python ./src/build_fundamentals_features.py"],
        "file_dep": [
            "./src/build_fundamentals_features.py",
            str(Path(DATA_DIR) / "US_Companies_Hist_Data.parquet"),
        ],
        "targets": [DATA_DIR / "features_fundamentals_monthly.parquet"],
        "clean": True,
    }
    yield {
        "name": "consensus",
        "doc": "Bloomberg BEST_* consensus → monthly EOM",
        "actions": ["python ./src/build_consensus_features.py"],
        "file_dep": [
            "./src/build_consensus_features.py",
            str(Path(DATA_DIR) / "US_Companies_Forecast.parquet"),
        ],
        "targets": [DATA_DIR / "features_consensus_monthly.parquet"],
        "clean": True,
    }
    yield {
        "name": "macro",
        "doc": "Bloomberg macro → wide monthly EOM (VIX, treasuries, …)",
        "actions": ["python ./src/build_macro_features.py"],
        "file_dep": [
            "./src/build_macro_features.py",
            str(Path(DATA_DIR) / "Macro_Data_US.parquet"),
        ],
        "targets": [DATA_DIR / "features_macro_monthly.parquet"],
        "clean": True,
    }
    yield {
        "name": "returns",
        "doc": "Monthly trailing + forward return labels from PX_LAST",
        "actions": ["python ./src/build_return_labels.py"],
        "file_dep": [
            "./src/build_return_labels.py",
            str(Path(DATA_DIR) / "US_Companies_Hist_Data.parquet"),
        ],
        "targets": [DATA_DIR / "labels_returns_monthly.parquet"],
        "clean": True,
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


def task_build_panel():
    """Join all monthly feature parquets into the unified (date, ticker) panel."""
    return {
        "actions": ["python ./src/build_panel.py"],
        "file_dep": [
            "./src/build_panel.py",
            str(Path(DATA_DIR) / "features_fundamentals_monthly.parquet"),
            str(Path(DATA_DIR) / "features_consensus_monthly.parquet"),
            str(Path(DATA_DIR) / "features_macro_monthly.parquet"),
            str(Path(DATA_DIR) / "labels_returns_monthly.parquet"),
        ],
        "targets": [DATA_DIR / "panel_monthly.parquet"],
        "clean": True,
    }


def task_predict_returns():
    """CKX-style return-prediction model on the unified panel.

    Runs the nested variant ladder (V0a/V0b/V1/V2/V3 + V4) through
    walk-forward CV with Ridge + GBR. V3 auto-skips if the 10-Q lexicon
    panel hasn't been built; V4 auto-skips until `process_10q:analyze`
    has produced the generative-AI 10-Q columns.
    """
    return {
        "actions": ["python ./src/predict_returns_ckx.py"],
        "file_dep": [
            "./src/predict_returns_ckx.py",
            "./src/build_panel.py",
            str(Path(DATA_DIR) / "panel_monthly.parquet"),
        ],
        "targets": [
            OUTPUT_DIR / "ckx_predictions.parquet",
            OUTPUT_DIR / "ckx_metrics.json",
            OUTPUT_DIR / "ckx_portfolio.parquet",
        ],
        "clean": True,
        "verbosity": 2,
    }


def _clean_chronos2_outputs():
    """`doit clean forecast_chronos2` removes all chronos2_forecast_* artifacts.

    The forecast filename is parameterised by `as_of`, so we can't list a
    fixed `targets` — glob and unlink instead.
    """
    for pattern in ("chronos2_forecast_*.parquet", "chronos2_forecast_*.png"):
        for p in Path(OUTPUT_DIR).glob(pattern):
            p.unlink(missing_ok=True)


def task_forecast_chronos2():
    """Zero-shot 4Q forecast of revenue + net_income for AAPL with Chronos-2."""
    return {
        "actions": ["python ./src/forecast_chronos2.py AAPL"],
        "file_dep": [
            "./src/forecast_chronos2.py",
            "./src/build_panel.py",
            str(Path(DATA_DIR) / "panel_monthly.parquet"),
        ],
        "uptodate": [False],  # forecast as_of changes; always re-run
        "verbosity": 2,
        "clean": [_clean_chronos2_outputs],
    }


def task_forecast():
    """Return-forecasting experiments (slice A: generic stock, slice B: US-company factors)."""
    # slice A — single-stock baselines + Chronos2 (no covariates)
    yield {
        "name": "stock_baseline",
        "doc": "Baseline (mean/AR1/ARIMA/zero) 1-step return forecasts for a single ticker",
        "actions": ["python ./src/stock_baselines.py --ticker AAPL"],
        "targets": [DATA_DIR / "AAPL_Baseline_Forecasts.parquet"],
        "file_dep": [
            "./src/stock_baselines.py",
            "./src/stock_returns.py",
            "./src/forecast_utils.py",
            DATA_DIR / "US_Companies_Hist_Data.parquet",
        ],
        "task_dep": ["pull:manual_companies"],
        "clean": [],
    }
    yield {
        "name": "stock_chronos",
        "doc": "Chronos2 1-step return forecasts for a single ticker",
        "actions": ["python ./src/stock_chronos.py --ticker AAPL"],
        "targets": [DATA_DIR / "AAPL_Chronos_Forecasts.parquet"],
        "file_dep": [
            "./src/stock_chronos.py",
            "./src/stock_returns.py",
            "./src/forecast_utils.py",
            DATA_DIR / "US_Companies_Hist_Data.parquet",
        ],
        "task_dep": ["pull:manual_companies"],
        "clean": [],
    }
    # slice B — 13-ticker US-company analyst-factor experiment
    yield {
        "name": "us_panel",
        "doc": "Build US-company monthly returns + analyst-factor panel",
        "actions": ["python ./src/us_company_factors.py"],
        "targets": [DATA_DIR / "US_Company_Panel.parquet"],
        "file_dep": [
            "./src/us_company_factors.py",
            DATA_DIR / "US_Companies_Forecast.parquet",
            DATA_DIR / "US_Companies_Hist_Data.parquet",
        ],
        "task_dep": ["pull:manual_companies"],
        "clean": [],
    }
    yield {
        "name": "us_regression",
        "doc": "Per-ticker 3-factor regressions of monthly US returns",
        "actions": ["python ./src/us_company_forecasts.py --only regression"],
        "targets": [DATA_DIR / "US_Regression_Forecasts.parquet"],
        "file_dep": [
            "./src/us_company_forecasts.py",
            "./src/forecast_utils.py",
            "./src/us_company_factors.py",
            DATA_DIR / "US_Company_Panel.parquet",
        ],
        "clean": [],
    }
    yield {
        "name": "us_chronos",
        "doc": "Chronos2 monthly forecasts of US returns with factor covariates",
        "actions": ["python ./src/us_company_forecasts.py --only chronos"],
        "targets": [DATA_DIR / "US_Chronos_Forecasts.parquet"],
        "file_dep": [
            "./src/us_company_forecasts.py",
            "./src/forecast_utils.py",
            "./src/us_company_factors.py",
            DATA_DIR / "US_Company_Panel.parquet",
        ],
        "clean": [],
    }


# =============================================================================
# Generative-AI copilot tasks (Gonzalo)
# -----------------------------------------------------------------------------
# Everything below this banner extends the pipeline with the copilot layer:
# Chronos-2 fundamentals backtest, RAG-grounded decision digest, digest
# verifiers, and the Streamlit dashboard. Kept as pure additions after the
# existing tasks so this section can be merged/reverted as a unit without
# touching anyone else's work above.
# =============================================================================


def task_backtest_chronos2_fundamentals():
    """Historical Chronos-2 **fundamentals** backtest: 5 tickers x 4 quarters vs Consensus + Naive.

    Produces `_output/chronos2_backtest.parquet` + summary JSON. Local CPU,
    zero API cost. See `docs_src/results/forecasts.md` for the metric
    definitions and the FY-period caveat on the consensus comparison.
    """
    return {
        "actions": ["python ./src/backtest_chronos2_fundamentals.py"],
        "file_dep": [
            "./src/backtest_chronos2_fundamentals.py",
            "./src/forecast_chronos2.py",
            "./src/build_panel.py",
            str(Path(DATA_DIR) / "panel_monthly.parquet"),
        ],
        "targets": [
            Path(OUTPUT_DIR) / "chronos2_backtest.parquet",
            Path(OUTPUT_DIR) / "chronos2_backtest_summary.json",
        ],
        "clean": True,
        "verbosity": 2,
    }


def task_generate_digests():
    """Generate the 5x4 grid of decision digests (one OpenAI call per cell, cached).

    Each digest pre-fetches the returns view, fundamentals view, and top-K
    10-Q + transcript chunks for (ticker, as_of) and makes exactly one
    structured-output call. Cached under `_data/digest_cache/`. Re-runs are
    free; bump PROMPT_VERSION in `generate_digest.py` to force a paid re-run.

    Cost: ~$2 on first run, $0 thereafter. Exits 0 if OPENAI_API_KEY is unset.
    """
    return {
        "actions": ["python ./src/generate_digest.py --all"],
        "file_dep": [
            "./src/generate_digest.py",
            str(Path(OUTPUT_DIR) / "ckx_predictions.parquet"),
            str(Path(OUTPUT_DIR) / "chronos2_backtest.parquet"),
        ],
        # No fixed target file — cache lives at _data/digest_cache/*.json
        # with one file per (ticker, as_of, prompt_version). Use uptodate=False
        # and let the script's own per-cell cache short-circuit re-runs.
        "uptodate": [False],
        "verbosity": 2,
    }


def task_eval_digest():
    """Compute citation / numeric-grounding / direction-match verifiers on cached digests.

    Three independent verifiers, reported per-digest and aggregated. See
    `docs_src/project_overview/methodology.md` for the design (and for why we
    deliberately do NOT compute a Fuentes-style triangular composite).
    """
    return {
        "actions": ["python ./src/eval_digest.py"],
        "file_dep": [
            "./src/eval_digest.py",
            str(Path(OUTPUT_DIR) / "ckx_predictions.parquet"),
        ],
        "targets": [
            Path(OUTPUT_DIR) / "digest_eval.parquet",
            Path(OUTPUT_DIR) / "digest_eval_summary.json",
        ],
        "clean": True,
        "verbosity": 2,
    }


def task_dashboard():
    """Launch the Streamlit results dashboard (long-running server).

    Convenience wrapper for `streamlit run src/dashboard.py`. Read-only — it
    views the pipeline outputs (ckx_* artifacts + panel_monthly.parquet) and
    never recomputes. Not wired into `run_notebooks` / `build_chartbook_site`.
    """
    return {
        "actions": ["streamlit run ./src/dashboard.py"],
        "file_dep": ["./src/dashboard.py"],
        "uptodate": [False],  # it's a server, not a build artifact
        "verbosity": 2,
    }


notebook_tasks = {
    "01_example_notebook_interactive.ipynb.py": {
        "path": "./src/01_example_notebook_interactive.ipynb.py",
        "file_dep": [],
        "targets": [],
    },
    "02_aapl_panel_chronos2_demo.ipynb.py": {
        "path": "./src/02_aapl_panel_chronos2_demo.ipynb.py",
        "file_dep": [str(Path(DATA_DIR) / "panel_monthly.parquet")],
        "targets": [],
    },
    "03_ckx_return_prediction.ipynb.py": {
        "path": "./src/03_ckx_return_prediction.ipynb.py",
        "file_dep": [
            str(Path(OUTPUT_DIR) / "ckx_predictions.parquet"),
            str(Path(OUTPUT_DIR) / "ckx_metrics.json"),
        ],
        "targets": [],
    },
    "04_stock_return_evaluation.ipynb.py": {
        "path": "./src/04_stock_return_evaluation.ipynb.py",
        "file_dep": [
            DATA_DIR / "AAPL_Baseline_Forecasts.parquet",
            DATA_DIR / "AAPL_Chronos_Forecasts.parquet",
        ],
        "targets": [],
    },
    "05_us_company_evaluation.ipynb.py": {
        "path": "./src/05_us_company_evaluation.ipynb.py",
        "file_dep": [
            DATA_DIR / "US_Regression_Forecasts.parquet",
            DATA_DIR / "US_Chronos_Forecasts.parquet",
        ],
        "targets": [],
    },
}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix("")  # strips .py, leaves .ipynb
        notebook_name = notebook_path.stem  # e.g. "01_example_notebook_interactive"
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on

###############################################################
## Cookiecutter LaTeX templates removed from the workflow
##
## task_compile_latex_docs() used to live here. It compiled four
## example .tex / Beamer templates from ./reports/ via 8 sequential
## XeLaTeX invocations — slow on first run (font/package fetch) and
## not consumed by anything in the real pipeline. Templates remain
## on disk under ./reports/ for manual `latexmk` use if needed.
###############################################################

sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"])
        for notebook in notebook_tasks.keys()
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook build -f",
        ],  # Use docs as build destination
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": [
            "run_notebooks",
        ],
        "clean": True,
    }


def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR"""
    src_py_files = list(Path("./src").glob("*.py"))
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        import subprocess

        result = subprocess.run(
            ["pytest", f"--junitxml={test_output}"],
        )
        if result.returncode != 0:
            # Remove the XML so doit won't consider the target up-to-date
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": src_py_files,
        "clean": True,
        "verbosity": 2,
    }
