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
    """Pull data from external sources"""
    yield {
        "name": "crsp_stock",
        "doc": "Pull CRSP stock data from WRDS",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_CRSP_stock.py",
        ],
        "targets": [DATA_DIR / "CRSP_monthly_stock.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_stock.py"],
        "clean": [],
    }
    yield {
        "name": "crsp_compustat",
        "doc": "Pull CRSP Compustat data from WRDS",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_CRSP_Compustat.py",
        ],
        "targets": [DATA_DIR / "CRSP_Compustat.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_compustat.py"],
        "clean": [],
    }
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
        "doc": "Pull SEC 10-Q filings (metadata + raw/clean text) from WRDS via SFTP",
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

    Stages: clean → score → panel. The `embed` stage is intentionally NOT a
    dependency of `panel`: it requires OPENAI_API_KEY and incurs API cost,
    so it is opt-in via `doit process_10q:embed`. When the key is unset,
    the embed script prints a notice and exits cleanly.
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


def task_summary_stats():
    """Generate summary statistics tables"""
    file_dep = ["./src/example_table.py"]
    file_output = [
        "example_table.tex",
        "pandas_to_latex_simple_table1.tex",
    ]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "python ./src/example_table.py",
            "python ./src/pandas_to_latex_demo.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


notebook_tasks = {
    "01_example_notebook_interactive.ipynb.py": {
        "path": "./src/01_example_notebook_interactive.ipynb.py",
        "file_dep": [],
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
## Task below is for LaTeX compilation
###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/report_example.tex",
        "./reports/my_article_header.sty",
        "./reports/slides_example.tex",
        "./reports/my_beamer_header.sty",
        "./reports/my_common_header.sty",
        "./reports/report_simple_example.tex",
        "./reports/slides_simple_example.tex",
        "./src/example_plot.py",
        "./src/example_table.py",
    ]
    targets = [
        "./reports/report_example.pdf",
        "./reports/slides_example.pdf",
        "./reports/report_simple_example.pdf",
        "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_example.tex",  # Clean
            # Simple templates based on small adjustments to Overleaf templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_simple_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_simple_example.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }

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
