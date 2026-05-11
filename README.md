# finm33200_project

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

## Running the AAPL panel + Chronos-2 pipeline

This section walks through reproducing the unified monthly panel
(`_data/panel_monthly.parquet`) and the zero-shot Chronos-2 fundamentals
forecast (`_output/chronos2_forecast_AAPL_*.parquet`) from a clean checkout.

### Prerequisites

- Bloomberg Excels in `data_manual/` — already version-controlled, so no
  Bloomberg terminal is required for these inputs.
- A `.env` file at the repo root with `OPENAI_API_KEY=sk-...` (the project
  reads it via `python-decouple`, not `os.environ`; see the env-var
  subsection below if your key isn't being picked up).
- A virtualenv with the dependencies installed:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
  Note: `chronos-forecasting` pulls in `torch` (~2 GB). CPU is fine for
  AAPL; switch `--device mps` (Apple Silicon) or `--device cuda` for speed.

### Recommended path: `doit`

```bash
doit build_features            # parse Bloomberg Excels + 4 monthly feature parquets
doit build_sentiment           # embed AAPL transcripts + score + monthly carry-forward
doit build_panel               # join everything → _data/panel_monthly.parquet
doit forecast_chronos2         # 4Q forecast for AAPL → _output/chronos2_forecast_AAPL_*.parquet
```

`doit build_sentiment` calls OpenAI by default. To skip OpenAI and use
random unit vectors as a smoke test, prefix the command:

```bash
SYNTHETIC=1 doit build_sentiment
```

### Step-by-step (debug-friendly, no doit caching)

```bash
# 1. Parse Bloomberg Excels → long-format parquets in _data/
python src/pull_manual_companies.py
python src/pull_manual_macro.py

# 2. Build per-source monthly features
python src/build_fundamentals_features.py
python src/build_consensus_features.py
python src/build_macro_features.py
python src/build_return_labels.py

# 3. (Optional) Transcript sentiment via OpenAI
#    Add --synthetic to skip the OpenAI call for smoke tests.
python src/embed_transcripts.py AAPL
python src/score_transcript_sentiment.py
python src/build_sentiment_features.py

# 4. Assemble the unified (date, ticker) panel
python src/build_panel.py

# 5. Zero-shot Chronos-2 fundamentals forecast
python src/forecast_chronos2.py AAPL --as-of 2024-09-30
```

### Outputs

- `_data/panel_monthly.parquet` — single source of truth for downstream
  models. ~4 k rows × 40 cols, 13 tickers (AAPL, AMZN, BA, CVX, GS, HD,
  IBM, JPM, KO, MSFT, NKE, NVDA, VZ), monthly EOM, 2000 → today.
  Columns include fundamentals (`revenue`, `net_income`, `ebitda`,
  `pe_ratio`, …), Bloomberg consensus (`best_sales`, `best_net_income`, …),
  sentiment (`sentiment_diff`, `sentiment_diff_qoq`, `days_since_earnings`),
  macro (`vix`, `treas_10y`, `dxy`, …), trailing returns
  (`ret_1m/3m/6m/12m`), and forward-return labels (`fwd_ret_1m/3m/6m/12m`).
- `_output/chronos2_forecast_AAPL_<YYYYMMDD>.parquet` — 8 rows: 4 quarters
  × {`revenue`, `net_income`} with `forecast_q10`, `forecast_q50`,
  `forecast_q90`, plus Bloomberg `BEST_*` consensus side-by-side.

### Verifying no-lookahead bias

```bash
pytest src/test_panel_no_lookahead.py -v
```

The 5 invariants check: month-end dates only, no duplicate `(date, ticker)`
keys, no rows dated after today, sentiment activated only from past
earnings calls, and `fwd_*` columns are never used as features.

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
Thus, data in the "\_data" folder is excluded from Git (see the .gitignore file),
while the "data_manual" folder is tracked by Git.

Output is stored in the "\_output" directory. This includes dataframes, charts, and
rendered notebooks. When the output is small enough, I'll keep this under
version control. I like this because I can keep track of how dataframes change as my
analysis progresses, for example.

Of course, the \_data directory and \_output directory can be kept elsewhere on the
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
  data source are prepended with "pull*", as in "pull_fred.py". Functions that
  load data that has been cached in the "\_data" folder are prepended with "load*".
  For example, inside of the `pull_CRSP_Compustat.py` file there is both a
  `pull_compustat` function and a `load_compustat` function. The first pulls from
  the web, whereas the other loads cached data from the "\_data" directory.

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
