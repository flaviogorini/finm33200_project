"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

Pipeline (run ``doit list`` to see all tasks):

    config                Create _data/ + _output/
    build_meta:universe   Wikipedia -> Nasdaq-100 constituents CSV
    build_meta:ciq_mapping  WRDS metadata -> CIQ company-ID mapping CSV
    pull:manual_macro     Bloomberg macro Excel -> parquet
    pull:manual_companies Bloomberg per-company Excel -> parquet (PX_LAST, BEst NI, ...)
    pull:transcripts      WRDS Capital IQ Nasdaq-100 transcript bulk pull (slow)
    clean:transcripts     Clean + segment transcripts -> processed parquet
    cleaning_review       QC review of cleaned transcripts -> _output/transcripts/qc/
    freeze:transcripts    Cleaned-dataset freeze manifest
    build_sentiment:embed   Chunk + embed transcripts (OpenAI)
    build_sentiment:score   Cosine-vs-anchor scoring -> sentiment_transcripts.parquet
    build_sentiment:monthly Monthly carry-forward -> features_sentiment_monthly.parquet
    build_signals:call_vectors    Per-call n_chars-weighted mean embedding
    build_signals:delta_vectors   Δ call vector + days_since_earnings
    build_signals:lm              LM Δ net-positivity scores
    build_features:returns        Monthly fwd_ret_21d panel
    build_features:momentum       12-1 monthly momentum panel
    build_features:revisions      30-day analyst revisions panel (BEst NI)
    build_panel                   Unified signal panel
    train_ridge                   RidgeCV on Δ call vectors
    run_backtests                 5 strategies x 3 specs + IC time series
    joint_regression              Fama-MacBeth + Newey-West (HAC lag 6)
    run_notebooks                 Jupytext convert + execute + html for 99_results
    write_report                  Render reports/writeup.qmd to reports/writeup.html
    run_pytest                    Tests + calendar parity assertion

"""

import os
import shutil
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(1, "./src/")

from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
OS_TYPE = config("OS_TYPE")
USER = environ.get("USER") or environ.get("USERNAME", "")


PROCESSED_DIR = DATA_DIR / "transcripts" / "processed"
RAW_DIR = DATA_DIR / "transcripts" / "raw"
# Hand-maintained inputs (Nasdaq-100 constituents, CIQ company-ID mapping)
# live with the other manual data, NOT under _data/ which is treated as
# regenerable output.
META_DIR = MANUAL_DATA_DIR / "_meta"
QC_DIR = OUTPUT_DIR / "transcripts" / "qc"


PYTHON = sys.executable  # absolute path to the interpreter `doit` is running under


def _py(path: str) -> str:
    return f"{PYTHON} ./src/{path}"


def jupyter_execute_notebook(notebook_path: Path) -> str:
    return (
        "jupyter nbconvert --execute --to notebook "
        "--ClearMetadataPreprocessor.enabled=True --inplace "
        f"{notebook_path}"
    )


def jupyter_to_html(notebook_path: Path, output_dir: Path = OUTPUT_DIR) -> str:
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"


def _configure_quarto_env() -> None:
    """Point quarto at its sibling tool binaries for a non-activated conda env.

    The conda-forge Windows quarto package ships its runtime tools (deno,
    pandoc, deno_dom, sass, typst, ...) under the env's ``Library`` tree and
    wires the ``QUARTO_*`` env vars only through conda *activation* scripts.
    ``python -m doit`` does not activate the env, so those vars are unset and
    quarto's launcher falls back to bundled-install paths that do not exist
    (e.g. ``Library/bin/tools/x86_64/deno.exe``), failing with
    ``deno.exe is not recognized``.

    We set the vars here, computed from the interpreter location, so the
    pipeline is self-contained. We override unconditionally because the shipped
    activation script itself sets ``QUARTO_DENO_DOM`` to a non-existent
    build-machine path (``D:/bld/...``). No-op when this is not a Windows
    conda-forge quarto (e.g. an official system install, which is self-locating).
    """
    library = Path(sys.executable).parent / "Library"
    if not (library / "bin" / "quarto.cmd").exists():
        return
    tools = {
        "QUARTO_DENO": library / "bin" / "deno.exe",
        "QUARTO_DENO_DOM": library / "lib" / "deno_dom.dll",
        "QUARTO_PANDOC": library / "bin" / "pandoc.exe",
        "QUARTO_ESBUILD": library / "bin" / "esbuild.exe",
        "QUARTO_DART_SASS": library / "bin" / "sass.exe",
        "QUARTO_TYPST": library / "bin" / "typst.exe",
        "QUARTO_SHARE_PATH": library / "share" / "quarto",
        "QUARTO_CONDA_PREFIX": library,
    }
    for var, path in tools.items():
        if path.exists():
            os.environ[var] = str(path)


def _find_quarto() -> str:
    """Locate the quarto executable, quoted for use in a shell action.

    Quarto is not a Python package. Installed via conda-forge it lands in the
    env's ``Library/bin`` (Windows) or ``bin`` (Unix) -- directories that are
    only on PATH when the env is *activated*. doit spawns a plain subshell that
    does not activate the env, so a bare ``quarto`` is not found. Resolve it
    relative to the running interpreter (mirroring ``PYTHON = sys.executable``),
    then fall back to PATH for system-wide installs (official installer / Unix).
    """
    # On Windows ``sys.executable`` is ``<env>/python.exe`` so env_dir is the
    # env root; on Unix it is ``<env>/bin/python`` so env_dir is already the bin
    # dir. Cover both layouts.
    env_dir = Path(sys.executable).parent
    candidates = [
        env_dir / "Library" / "bin" / "quarto.cmd",  # Windows conda-forge
        env_dir / "Library" / "bin" / "quarto.exe",
        env_dir / "Scripts" / "quarto.cmd",
        env_dir / "quarto",  # Unix conda-forge (<env>/bin/quarto)
    ]
    for candidate in candidates:
        if candidate.exists():
            return f'"{candidate}"'
    found = shutil.which("quarto")
    return f'"{found}"' if found else "quarto"


def mv(from_path: Path, to_path: Path):
    """Return a cross-platform doit action that moves ``from_path`` into the
    directory ``to_path`` (overwriting any existing file of the same name).

    Returns a Python callable rather than a shell string so the pipeline does
    not depend on a Unix ``mv`` being on PATH (it is absent on Windows).
    """
    from_path = Path(from_path)
    to_path = Path(to_path)

    def _move():
        to_path.mkdir(parents=True, exist_ok=True)
        dest = to_path / from_path.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(from_path), str(dest))

    return _move


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": [_py("settings.py")],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_build_meta():
    """Universe and CIQ-mapping metadata (run once; both need internet / WRDS)."""
    yield {
        "name": "universe",
        "doc": "Parse the current Nasdaq-100 constituents table from Wikipedia",
        "actions": [_py("build_nasdaq100_universe.py")],
        "file_dep": ["./src/build_nasdaq100_universe.py"],
        "targets": [META_DIR / "nasdaq100_constituents.csv"],
        "clean": True,
        "verbosity": 2,
    }
    yield {
        "name": "ciq_mapping",
        "doc": (
            "Map Nasdaq-100 tickers to Capital IQ company IDs via WRDS metadata. "
            "Needs WRDS_USERNAME in .env or ~/.pgpass."
        ),
        "actions": [_py("build_ciq_company_mapping.py")],
        "file_dep": [
            "./src/build_ciq_company_mapping.py",
            str(META_DIR / "nasdaq100_constituents.csv"),
        ],
        "targets": [META_DIR / "ciq_company_mapping.csv"],
        "clean": True,
        "verbosity": 2,
    }


def task_pull():
    """Pull data from external sources."""
    yield {
        "name": "manual_macro",
        "doc": "Parse manual Bloomberg macro Excel into parquet",
        "actions": [
            _py("settings.py"),
            _py("pull_manual_macro.py"),
        ],
        "targets": [DATA_DIR / "Macro_Data_US.parquet"],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_manual_macro.py",
            str(MANUAL_DATA_DIR / "Macro_Data_US.xlsx"),
        ],
        "clean": [],
    }
    yield {
        "name": "manual_companies",
        "doc": "Parse manual Bloomberg company prediction Excel into parquet",
        "actions": [
            _py("settings.py"),
            _py("pull_manual_companies.py"),
        ],
        "targets": [
            DATA_DIR / "US_Companies_Forecast.parquet",
            DATA_DIR / "US_Companies_Hist_Data.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_manual_companies.py",
            "./src/pull_manual_macro.py",
            str(MANUAL_DATA_DIR / "US_Companies_Prediction_Data.xlsx"),
        ],
        "clean": [],
    }
    yield {
        "name": "transcripts",
        "doc": (
            "Bulk-pull Nasdaq-100 earnings-call transcripts from WRDS Capital IQ. "
            "Slow (20-60 min). Needs WRDS_USERNAME in .env or ~/.pgpass."
        ),
        "actions": [
            _py("check_transcript_mapping_availability.py"),
            _py("extract_sample_raw_transcripts.py --label nasdaq100"),
        ],
        "targets": [
            RAW_DIR / "nasdaq100_raw_transcripts_deduped.parquet",
            RAW_DIR / "nasdaq100_raw_transcript_metadata_deduped.parquet",
        ],
        "file_dep": [
            "./src/check_transcript_mapping_availability.py",
            "./src/extract_sample_raw_transcripts.py",
            str(META_DIR / "ciq_company_mapping.csv"),
            str(META_DIR / "nasdaq100_constituents.csv"),
        ],
        "clean": [],
        "verbosity": 2,
    }


def task_clean_transcripts():
    """Clean and segment the bulk-pulled transcripts into the processed parquet."""
    return {
        "actions": [_py("clean_sample_transcripts.py --mode full")],
        "targets": [
            PROCESSED_DIR / "nasdaq100_cleaned_components.parquet",
            PROCESSED_DIR / "nasdaq100_cleaned_calls.parquet",
            PROCESSED_DIR / "nasdaq100_llm_views.parquet",
        ],
        "file_dep": [
            "./src/clean_sample_transcripts.py",
            str(RAW_DIR / "nasdaq100_raw_transcripts_deduped.parquet"),
            str(RAW_DIR / "nasdaq100_raw_transcript_metadata_deduped.parquet"),
        ],
        "clean": True,
        "verbosity": 2,
    }


def task_cleaning_review():
    """QC review package over the cleaned transcripts (read-only).

    Produces the 4 needs-review CSVs that ``freeze_transcripts`` then reads.
    Declared BEFORE ``freeze_transcripts`` so doit schedules it first.
    """
    return {
        "actions": [_py("build_cleaning_final_review.py")],
        "file_dep": [
            "./src/build_cleaning_final_review.py",
            str(PROCESSED_DIR / "nasdaq100_cleaned_components.parquet"),
            str(PROCESSED_DIR / "nasdaq100_cleaned_calls.parquet"),
            str(PROCESSED_DIR / "nasdaq100_llm_views.parquet"),
        ],
        "targets": [
            QC_DIR / "nasdaq100_cleaning_final_review_summary.md",
            QC_DIR / "nasdaq100_cleaning_needs_review_calls.csv",
            QC_DIR / "nasdaq100_metadata_component_gap_calls.csv",
            QC_DIR / "nasdaq100_high_word_drop_calls.csv",
            QC_DIR / "nasdaq100_no_qa_calls.csv",
        ],
        "clean": True,
    }


def task_freeze_transcripts():
    """Write the cleaned-dataset freeze manifest.

    Reads the 4 needs-review CSVs from ``cleaning_review`` plus the hand-frozen
    raw provenance manifest from ``data_manual/_meta/``.
    """
    return {
        "actions": [_py("freeze_cleaned_dataset.py")],
        "targets": [
            QC_DIR / "nasdaq100_cleaned_dataset_frozen_manifest.json",
            QC_DIR / "nasdaq100_cleaned_dataset_freeze_report.md",
        ],
        "file_dep": [
            "./src/freeze_cleaned_dataset.py",
            str(PROCESSED_DIR / "nasdaq100_cleaned_calls.parquet"),
            str(QC_DIR / "nasdaq100_cleaning_qc.csv"),
            str(QC_DIR / "nasdaq100_cleaning_manifest.json"),
            str(QC_DIR / "nasdaq100_cleaning_needs_review_calls.csv"),
            str(QC_DIR / "nasdaq100_metadata_component_gap_calls.csv"),
            str(QC_DIR / "nasdaq100_high_word_drop_calls.csv"),
            str(QC_DIR / "nasdaq100_no_qa_calls.csv"),
            str(MANUAL_DATA_DIR / "_meta" / "nasdaq100_raw_dataset_frozen_manifest.json"),
        ],
        "clean": True,
    }


def task_build_sentiment():
    """Embed transcripts, score sentiment, roll up to monthly carry-forward.

    Set SYNTHETIC=1 to skip OpenAI calls (random unit vectors).
    """
    synth = "--synthetic" if environ.get("SYNTHETIC") else ""
    yield {
        "name": "embed",
        "doc": "Chunk + embed transcript components (OpenAI text-embedding-3-small)",
        "actions": [_py(f"embed_transcripts.py {synth}").strip()],
        "file_dep": [
            "./src/embed_transcripts.py",
            str(PROCESSED_DIR / "nasdaq100_cleaned_components.parquet"),
        ],
        "targets": [DATA_DIR / "embeddings_transcripts.parquet"],
        "clean": True,
    }
    yield {
        "name": "score",
        "doc": "Cosine-vs-anchor sentiment scoring",
        "actions": [_py(f"score_transcript_sentiment.py {synth}").strip()],
        "file_dep": [
            "./src/score_transcript_sentiment.py",
            str(DATA_DIR / "embeddings_transcripts.parquet"),
        ],
        "targets": [DATA_DIR / "sentiment_transcripts.parquet"],
        "clean": True,
    }
    yield {
        "name": "monthly",
        "doc": "Carry per-call sentiment forward to monthly EOM",
        "actions": [_py("build_sentiment_features.py")],
        "file_dep": [
            "./src/build_sentiment_features.py",
            str(DATA_DIR / "sentiment_transcripts.parquet"),
        ],
        "targets": [DATA_DIR / "features_sentiment_monthly.parquet"],
        "clean": True,
    }


def task_build_signals():
    """Per-call transcript-derived signal artifacts."""
    yield {
        "name": "call_vectors",
        "doc": "Per-call n_chars-weighted mean embedding (1536-D unit vector)",
        "actions": [_py("build_call_vectors.py")],
        "file_dep": [
            "./src/build_call_vectors.py",
            str(DATA_DIR / "embeddings_transcripts.parquet"),
        ],
        "targets": [DATA_DIR / "call_vectors.parquet"],
        "clean": True,
    }
    yield {
        "name": "delta_vectors",
        "doc": "Δ call vector + days_since_earnings per call",
        "actions": [_py("build_delta_vectors.py")],
        "file_dep": [
            "./src/build_delta_vectors.py",
            str(DATA_DIR / "call_vectors.parquet"),
        ],
        "targets": [DATA_DIR / "delta_vectors.parquet"],
        "clean": True,
    }
    yield {
        "name": "lm",
        "doc": "Loughran-McDonald Δ net-positivity per call (full_transcript view)",
        "actions": [_py("score_transcript_lm.py")],
        "file_dep": [
            "./src/score_transcript_lm.py",
            str(PROCESSED_DIR / "nasdaq100_llm_views.parquet"),
        ],
        "targets": [DATA_DIR / "lm_scores_transcripts.parquet"],
        "clean": True,
    }


def task_build_features():
    """Bloomberg-derived monthly panels."""
    yield {
        "name": "returns",
        "doc": "Monthly EOM prices + 21-bday forward return",
        "actions": [_py("build_returns_monthly.py")],
        "file_dep": [
            "./src/build_returns_monthly.py",
            "./src/calendar_utils.py",
            str(DATA_DIR / "US_Companies_Hist_Data.parquet"),
        ],
        "targets": [DATA_DIR / "returns_monthly.parquet"],
        "clean": True,
    }
    yield {
        "name": "momentum",
        "doc": "12-1 momentum monthly panel",
        "actions": [_py("build_momentum_monthly.py")],
        "file_dep": [
            "./src/build_momentum_monthly.py",
            str(DATA_DIR / "returns_monthly.parquet"),
        ],
        "targets": [DATA_DIR / "momentum_monthly.parquet"],
        "clean": True,
    }
    yield {
        "name": "revisions",
        "doc": "30-day analyst revisions (BEst NI) monthly panel",
        "actions": [_py("build_revisions_monthly.py")],
        "file_dep": [
            "./src/build_revisions_monthly.py",
            "./src/calendar_utils.py",
            str(DATA_DIR / "US_Companies_Forecast.parquet"),
        ],
        "targets": [DATA_DIR / "revisions_monthly.parquet"],
        "clean": True,
    }


def task_train_ridge():
    """RidgeCV with expanding-window refits on Δ call vectors (Strategy 2)."""
    return {
        "actions": [_py("train_ridge.py")],
        "file_dep": [
            "./src/train_ridge.py",
            "./src/calendar_utils.py",
            str(DATA_DIR / "delta_vectors.parquet"),
            str(DATA_DIR / "US_Companies_Hist_Data.parquet"),
        ],
        "targets": [DATA_DIR / "ridge_predictions.parquet"],
        "clean": True,
    }


def task_build_factor_baseline():
    """FF5 monthly factors + per-call CAR3 for the §5.6 nested α progression.

    Produces ff5_monthly.parquet (HTTP fetch from Ken French) and
    car3_per_call.parquet (announcement-window CAR computed from existing
    daily prices). Both feed the v2 writeup: car3_per_call gets carried
    forward into the signal panel as sig_car3; ff5_monthly is the RHS
    factor set for factor_regression.
    """
    return {
        "actions": [_py("build_factor_baseline.py")],
        "file_dep": [
            "./src/build_factor_baseline.py",
            "./src/calendar_utils.py",
            "./src/build_returns_monthly.py",
            str(DATA_DIR / "US_Companies_Hist_Data.parquet"),
            str(DATA_DIR / "lm_scores_transcripts.parquet"),
        ],
        "targets": [
            DATA_DIR / "ff5_monthly.parquet",
            DATA_DIR / "car3_per_call.parquet",
        ],
        "clean": True,
    }


def task_build_panel():
    """Unified monthly signal panel (date × ticker → 6 sigs + fwd_ret_21d).

    Declares file_dep on every per-signal input so doit forces all signal
    producers (build_sentiment, build_signals:lm, train_ridge,
    build_features:returns/momentum/revisions, build_factor_baseline) to
    complete before the panel is assembled. Without these, a fresh `doit`
    run could schedule build_panel before train_ridge and produce a panel
    with `sig_ridge` silently all-NaN.
    """
    return {
        "actions": [_py("build_signal_panel.py")],
        "file_dep": [
            "./src/build_signal_panel.py",
            str(DATA_DIR / "returns_monthly.parquet"),
            str(DATA_DIR / "momentum_monthly.parquet"),
            str(DATA_DIR / "revisions_monthly.parquet"),
            str(DATA_DIR / "features_sentiment_monthly.parquet"),
            str(DATA_DIR / "lm_scores_transcripts.parquet"),
            str(DATA_DIR / "ridge_predictions.parquet"),
            str(DATA_DIR / "car3_per_call.parquet"),
        ],
        "targets": [DATA_DIR / "signal_panel_monthly.parquet"],
        "clean": True,
    }


def task_run_backtests():
    """5 strategies × 3 specs (main, stale-excl, post-2018) + IC time series."""
    return {
        "actions": [_py("run_backtests.py")],
        "file_dep": [
            "./src/run_backtests.py",
            "./src/backtest.py",
            str(DATA_DIR / "signal_panel_monthly.parquet"),
        ],
        "targets": [
            DATA_DIR / "results_main.parquet",
            DATA_DIR / "results_stale_excl.parquet",
            DATA_DIR / "results_post2018.parquet",
            DATA_DIR / "metrics_main.json",
            DATA_DIR / "metrics_stale_excl.json",
            DATA_DIR / "metrics_post2018.json",
            DATA_DIR / "ic_timeseries.parquet",
            DATA_DIR / "ic_summary.json",
        ],
        "clean": True,
    }


def task_joint_regression():
    """Fama-MacBeth joint regression with Newey-West HAC standard errors."""
    return {
        "actions": [_py("joint_regression.py")],
        "file_dep": [
            "./src/joint_regression.py",
            str(DATA_DIR / "signal_panel_monthly.parquet"),
        ],
        "targets": [DATA_DIR / "fm_results.json"],
        "clean": True,
    }


def task_factor_regression():
    """Nested time-series α progression on FF5 + Mom + CAR3 + Rev for §5.6.

    For each text strategy in {ridge, anchor, lm}, runs 4 nested OLS regressions
    of the strategy's monthly LS return on a growing factor set. HAC SE
    (Newey-West, lag 6). Reports own-history AND post-2018 sample windows.
    """
    return {
        "actions": [_py("factor_regression.py")],
        "file_dep": [
            "./src/factor_regression.py",
            str(DATA_DIR / "ff5_monthly.parquet"),
            str(DATA_DIR / "results_main.parquet"),
        ],
        "targets": [
            DATA_DIR / "strategy_factor_returns_monthly.parquet",
            DATA_DIR / "factor_alpha.json",
        ],
        "clean": True,
    }


notebook_tasks = {
    "99_results.ipynb.py": {
        "path": Path("./src/99_results.ipynb.py"),
        "file_dep": [
            DATA_DIR / "signal_panel_monthly.parquet",
            DATA_DIR / "metrics_main.json",
            DATA_DIR / "metrics_stale_excl.json",
            DATA_DIR / "metrics_post2018.json",
            DATA_DIR / "results_main.parquet",
            DATA_DIR / "results_stale_excl.parquet",
            DATA_DIR / "results_post2018.parquet",
            DATA_DIR / "ic_timeseries.parquet",
            DATA_DIR / "ic_summary.json",
            DATA_DIR / "fm_results.json",
        ],
        "targets": [
            OUTPUT_DIR / "99_hit_rates_main.png",
            OUTPUT_DIR / "99_rolling_ic.png",
            OUTPUT_DIR / "99_cum_returns_period1_2008.png",
            OUTPUT_DIR / "99_drawdown_period1_2008.png",
            OUTPUT_DIR / "99_cum_returns_period2_ridge.png",
            OUTPUT_DIR / "99_drawdown_period2_ridge.png",
        ],
    },
    "99_results_v2.ipynb.py": {
        "path": Path("./src/99_results_v2.ipynb.py"),
        "file_dep": [
            DATA_DIR / "signal_panel_monthly.parquet",
            DATA_DIR / "metrics_main.json",
            DATA_DIR / "metrics_stale_excl.json",
            DATA_DIR / "metrics_post2018.json",
            DATA_DIR / "results_main.parquet",
            DATA_DIR / "results_stale_excl.parquet",
            DATA_DIR / "results_post2018.parquet",
            DATA_DIR / "ic_timeseries.parquet",
            DATA_DIR / "ic_summary.json",
            DATA_DIR / "fm_results.json",
            DATA_DIR / "factor_alpha.json",
            DATA_DIR / "strategy_factor_returns_monthly.parquet",
        ],
        "targets": [
            OUTPUT_DIR / "99_v2_hit_rates_main.png",
            OUTPUT_DIR / "99_v2_rolling_ic.png",
            OUTPUT_DIR / "99_v2_cum_returns_period1_2008.png",
            OUTPUT_DIR / "99_v2_drawdown_period1_2008.png",
            OUTPUT_DIR / "99_v2_cum_returns_period2_ridge.png",
            OUTPUT_DIR / "99_v2_drawdown_period2_ridge.png",
            DATA_DIR / "signal_corr_stock.json",
            DATA_DIR / "signal_corr_portfolio.json",
        ],
    },
}


def task_run_notebooks():
    """Convert + execute jupytext notebooks and copy outputs to _output/."""
    for notebook, spec in notebook_tasks.items():
        pyfile_path = Path(spec["path"])
        notebook_path = pyfile_path.with_suffix("")  # strips .py -> .ipynb
        notebook_name = notebook_path.stem
        yield {
            "name": notebook,
            "actions": [
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
            ],
            "file_dep": [
                str(pyfile_path),
                *[str(p) for p in spec["file_dep"]],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                OUTPUT_DIR / f"{notebook_name}.ipynb",
                *[str(p) for p in spec["targets"]],
            ],
            "clean": True,
            "verbosity": 2,
        }


def task_write_report():
    """Render reports/writeup.qmd to reports/writeup.html via Quarto."""
    qmd = Path("./reports/writeup.qmd")
    html = Path("./reports/writeup.html")
    _configure_quarto_env()
    quarto = _find_quarto()
    return {
        "actions": [f"{quarto} render {qmd} --to html"],
        "file_dep": [
            str(qmd),
            str(DATA_DIR / "metrics_main.json"),
            str(DATA_DIR / "metrics_stale_excl.json"),
            str(DATA_DIR / "metrics_post2018.json"),
            str(DATA_DIR / "ic_summary.json"),
            str(DATA_DIR / "fm_results.json"),
            str(OUTPUT_DIR / "99_hit_rates_main.png"),
            str(OUTPUT_DIR / "99_rolling_ic.png"),
            str(OUTPUT_DIR / "99_cum_returns_period1_2008.png"),
            str(OUTPUT_DIR / "99_cum_returns_period2_ridge.png"),
            str(OUTPUT_DIR / "99_drawdown_period2_ridge.png"),
        ],
        "targets": [str(html)],
        "clean": True,
        "verbosity": 2,
    }


def task_write_report_v2():
    """Render reports/writeup_v2.qmd to reports/writeup_v2.html via Quarto.

    v2 adds CAR3 as a 6th first-class signal, §5.4 signal correlations,
    and §5.6 nested time-series α progression. The v1 writeup at
    reports/writeup.html stays untouched as the 5-signal reference.
    """
    qmd = Path("./reports/writeup_v2.qmd")
    html = Path("./reports/writeup_v2.html")
    _configure_quarto_env()
    quarto = _find_quarto()
    return {
        "actions": [f"{quarto} render {qmd} --to html"],
        "file_dep": [
            str(qmd),
            str(DATA_DIR / "metrics_main.json"),
            str(DATA_DIR / "metrics_stale_excl.json"),
            str(DATA_DIR / "metrics_post2018.json"),
            str(DATA_DIR / "ic_summary.json"),
            str(DATA_DIR / "fm_results.json"),
            str(DATA_DIR / "factor_alpha.json"),
            str(DATA_DIR / "signal_corr_stock.json"),
            str(DATA_DIR / "signal_corr_portfolio.json"),
            str(OUTPUT_DIR / "99_v2_hit_rates_main.png"),
            str(OUTPUT_DIR / "99_v2_rolling_ic.png"),
            str(OUTPUT_DIR / "99_v2_cum_returns_period1_2008.png"),
            str(OUTPUT_DIR / "99_v2_cum_returns_period2_ridge.png"),
            str(OUTPUT_DIR / "99_v2_drawdown_period2_ridge.png"),
        ],
        "targets": [str(html)],
        "clean": True,
        "verbosity": 2,
    }


def task_run_pytest():
    """Run pytest on the src/ test files."""
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
            "./src/calendar_utils.py",
            "./src/backtest.py",
            "./src/test_backtest_engine.py",
        ],
        "clean": True,
        "verbosity": 2,
    }
