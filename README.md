finm33200_project
=================

## About this project

Generative and agentic AI Final Project

## Quick Start

The quickest way to run code in this repo is to use the following steps.

You must have TexLive (or another LaTeX distribution) installed on your computer and available in your path.
You can do this by downloading and installing it from here ([windows](https://tug.org/texlive/windows.html#install)
and [mac](https://tug.org/mactex/mactex-download.html) installers).


First, create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
Then install the dependencies:
```bash
pip install -r requirements.txt
```

Finally, run the project tasks:
```bash
doit
```
And that's it!


### Other commands

#### Unit Tests and Doc Tests

You can run the unit test, including doctests, with the following command:
```
pytest --doctest-modules
```

You can build the documentation with:
```
rm ./src/.pytest_cache/README.md
jupyter-book build -W ./
```
Use `del` instead of rm on Windows


#### Setting Environment Variables

You can [export your environment variables](https://stackoverflow.com/questions/43267413/how-to-set-environment-variables-from-env-file)
from your `.env` files like so, if you wish. This can be done easily in a Linux or Mac terminal with the following command:
```bash
set -a  # automatically export all variables
source .env
set +a
```
On Windows (PowerShell):
```powershell
Get-Content .env | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
```

### Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting Python code.

```bash
# Auto-fix linting issues (e.g., unused imports, undefined names)
ruff check . --fix

# Format code (consistent style, spacing, line length)
ruff format .

# Sort imports, then fix linting issues, then format
ruff format . && ruff check --select I --fix . && ruff check --fix .
```

- `ruff check --fix` applies safe auto-fixes for linting violations
- `ruff format` formats code similar to Black
- `--select I` targets only import sorting rules (isort-compatible)

### General Directory Structure

 - The `assets` folder is used for things like hand-drawn figures or other
   pictures that were not generated from code. These things cannot be easily
   recreated if they are deleted.

 - The `_output` folder, on the other hand, contains dataframes and figures that are
   generated from code. The entire folder should be able to be deleted, because
   the code can be run again, which would again generate all of the contents.

 - The `data_manual` is for data that cannot be easily recreated. This data
   should be version controlled. Anything in the `_data` folder or in
   the `_output` folder should be able to be recreated by running the code
   and can safely be deleted.

 - I'm using the `doit` Python module as a task runner. It works like `make` and
   the associated `Makefile`s. To rerun the code, install `doit`
   (https://pydoit.org/) and execute the command `doit` from the `src`
   directory. Note that doit is very flexible and can be used to run code
   commands from the command prompt, thus making it suitable for projects that
   use scripts written in multiple different programming languages.

 - I'm using the `.env` file as a container for absolute paths that are private
   to each collaborator in the project. You can also use it for private
   credentials, if needed. It should not be tracked in Git.

### Data and Output Storage

I'll often use a separate folder for storing data. Any data in the data folder
can be deleted and recreated by rerunning the PyDoit command (the pulls are in
the dodo.py file). Any data that cannot be automatically recreated should be
stored in the "data_manual" folder. Because of the risk of manually-created data
getting changed or lost, I prefer to keep it under version control if I can.
Thus, data in the "_data" folder is excluded from Git (see the .gitignore file),
while the "data_manual" folder is tracked by Git.

Output is stored in the "_output" directory. This includes dataframes, charts, and
rendered notebooks. When the output is small enough, I'll keep this under
version control. I like this because I can keep track of how dataframes change as my
analysis progresses, for example.

Of course, the _data directory and _output directory can be kept elsewhere on the
machine. To make this easy, I always include the ability to customize these
locations by defining the path to these directories in environment variables,
which I intend to be defined in the `.env` file, though they can also simply be
defined on the command line or elsewhere. The `settings.py` is responsible for
loading these environment variables and doing some preprocessing on them.
The `settings.py` file is the entry point for all other scripts to these
definitions. That is, all code that references these variables and others are
loaded by importing `config`.

### Naming Conventions

 - **`pull_` vs `load_`**: Files or functions that pull data from an external
 data source are prepended with "pull_", as in "pull_fred.py". Functions that
 load data that has been cached in the "_data" folder are prepended with "load_".
 For example, inside of the `pull_CRSP_Compustat.py` file there is both a
 `pull_compustat` function and a `load_compustat` function. The first pulls from
 the web, whereas the other loads cached data from the "_data" directory.


### Dependencies and Virtual Environments

#### Working with `pip` requirements

This project uses `pip` with a virtual environment. Install requirements with:
```bash
pip install -r requirements.txt
```

To update the requirements file after adding new packages:
```bash
pip freeze > requirements.txt
```

## 10-Q Signals Pipeline

A 5-stage pipeline that turns SEC 10-Q filings (pulled from WRDS) into a
point-in-time `(date, ticker)` monthly panel of text-derived features.
The panel is namespaced with `10q_*` columns so it joins cleanly with the
other team feeds (earnings transcripts, ratios, macro) on `(date, ticker)`.

### Run order

```bash
doit pull:sec_10q_filings      # WRDS SEC SFTP → _data/sec_10q/{TICKER}/wrds_*_filings/
doit process_10q:clean         # → _data/sec_10q/{TICKER}/processed_text/*.txt
doit process_10q:score         # LM-dictionary + lexical drift → 10q_features.parquet
doit process_10q:panel         # merge_asof monthly grid → sec_10q_monthly_panel.parquet
doit process_10q:embed         # OPTIONAL: OpenAI embeddings + FAISS (needs OPENAI_API_KEY)
```

The `embed` task is opt-in: it is **not** a dependency of `panel`, so the
default `doit` run does not call OpenAI or incur API cost. If
`OPENAI_API_KEY` is unset the script prints a notice and exits cleanly.

### Required environment variables

- `WRDS_USERNAME`, `WRDS_PASSWORD` — for SFTP downloads of EDGAR text via
  `/wrds/sec/warchives` and `/wrds/sec/wrds_clean_filings`.
- `OPENAI_API_KEY` — only for the optional embedding stage.
- `USE_CACHE` (default `true`) — skip re-downloading files already on disk.

### Ticker universe

Edit `TICKERS` in `src/settings.py`. The dict maps ticker → CIK (zero-padded
to 10 digits is fine either way; `cik_for()` zero-pads). Registered
defaults: `{"AAPL": "0000320193", "MSFT": "0000789019", "JPM": "0000019617"}`.
`DEFAULT_TICKERS` (the set used when callers don't pass an explicit list)
is intentionally narrow — currently `["AAPL"]` — so a default run produces
the same single-ticker artifact teammates' other pulls have already
shipped. Expand `DEFAULT_TICKERS` once the team is ready for multi-ticker
joining.

### Output panel columns

`_data/sec_10q_monthly_panel.parquet` (long format, one row per `(month-end, ticker)`):

| Column | Source |
|---|---|
| `date`, `ticker` | Primary key. |
| `filing_date` | SEC filing date — point-in-time availability. |
| `report_period` | Fiscal quarter end. |
| `accession_number`, `sec_url`, `extraction_status`, `feature_source` | Trace metadata. |
| `10q_sentiment`, `10q_positive_rate`, `10q_negative_rate` | LM dictionary (signed + raw rates). |
| `10q_uncertainty`, `10q_litigious`, `10q_constraining`, `10q_word_count` | Other LM categories + length. |
| `10q_cosine_vs_previous`, `10q_change_vs_previous` | Bag-of-words drift vs previous filing. |
| `10q_embedding_cosine_vs_previous`, `10q_embedding_change_vs_previous` | Semantic drift (present only when the embed stage has run). |

### Point-in-time guarantee

The panel builder uses
`pd.merge_asof(..., direction="backward", by="ticker")` on `filing_date`,
and `build_10q_monthly_panel.py` raises `RuntimeError` if any row would
carry a filing whose `filing_date > date`. The regression test
`src/test_10q_point_in_time.py` re-asserts the invariant on the cached
features.

