# finm33200_project — Forecast-Spine Copilot

FINM 33200 final project. A **forecast-spine copilot** for equity analysis.
Two forecasts feed one decision output:

```
Returns forecast (V0a → V4 ladder)  ──→  Decision digest  ──→  honest evaluation
                                          (one-shot, not agentic)
```

- **Returns forecast** ([src/predict_returns_ckx.py](src/predict_returns_ckx.py))
  — five nested feature variants V0a → V4, evaluated on identical OOS rows.
  Headline metrics: rank IC (Spearman) + AUC + portfolio Sharpe. R² is
  reported but demoted (monthly-return R² is noise-bounded near zero).
- **Decision digest** ([src/generate_digest.py](src/generate_digest.py))
  — one-shot LLM call that grounds both forecasts in cited 10-Q and
  transcript chunks, emitting a structured `DigestSchema` response.
- **Honest evaluation** ([src/eval_digest.py](src/eval_digest.py))
  — three independent verifiers: citation_match, numeric_grounding (Fuentes
  Extract pattern), direction_match against realized fwd_ret. Reported
  separately, NOT combined into a triangular composite — see
  [docs_src/project_overview/methodology.md](docs_src/project_overview/methodology.md)
  for the reasoning.

The full methodology, the "considered but deliberately not implemented" list
of guest-lecture techniques, and the rubric mapping live in the
[docs_src/](docs_src/) jupyter-book.

## Quick demo

After cleaning a venv and installing requirements (see
[Prerequisites](#prerequisites)):

```powershell
doit predict_returns               # if not already current
doit generate_digests              # NEW; ~$2 OpenAI cost, cached
doit eval_digest                   # NEW; local
streamlit run src/dashboard.py     # 5 tabs: forecast, ladder, AI timeline, snippets, portfolio
```

Then open the dashboard, pick a ticker on the "Ticker forecast" tab, look at
the "Variant ladder" tab (rank IC / AUC lead bars), and inspect the cached
digest JSONs under `_data/digest_cache/`.

## Known limitations

Reproduced from [_FOLLOW_UPS.md](_FOLLOW_UPS.md) so reviewers see them
up-front:

- **13-ticker cross-section is narrow.** Cross-sectional anomaly claims are
  not supported. Results read as *"does feature X add predictive content
  within this universe"* — not *"feature X works universally."*
- **2014-2025 backtest is a strong-bull-market sample.** The V0a equal-weight
  buy-and-hold benchmark is artificially tough; the long-short Sharpes for
  V0b-V4 should be read against this caveat.
- **Reported portfolio Sharpes are gross of transaction costs.** Add a flat
  5–10 bps per turnover for a trading-realistic backtest.
- **Hyperparameters are not tuned.** Values for Ridge and GBR are stated
  explicitly in [src/predict_returns_ckx.py](src/predict_returns_ckx.py) to
  forestall the "you tuned by hand" critique. An inner CV grid is the top
  stretch goal.
- **Survivorship bias.** All 13 tickers are current large-cap survivors.

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

## Running the AAPL panel pipeline

This section walks through reproducing the unified monthly panel
(`_data/panel_monthly.parquet`) from a clean checkout.

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

### Recommended path: `doit`

```bash
doit build_features            # parse Bloomberg Excels + 4 monthly feature parquets
doit build_sentiment           # embed AAPL transcripts + score + monthly carry-forward
doit pull:sec_10q_filings      # (optional) pull AAPL 10-Q filings from SEC EDGAR
doit process_10q               # (optional) clean + score + monthly 10-Q text panel
doit process_10q:analyze       # (optional) generative-AI 10-Q analysis → V4 (needs OPENAI_API_KEY)
doit build_panel               # join everything → _data/panel_monthly.parquet
doit predict_returns           # CKX-style return classifier → _output/ckx_*.parquet
doit dashboard                 # launch the Streamlit results dashboard
```

`doit process_10q:analyze` runs `gpt-4o-mini` over each 10-Q to score how
its disclosure changed versus the prior filing — this powers model variants
V4. It is opt-in (no-ops without `OPENAI_API_KEY`); re-run
`doit process_10q:panel` and `doit build_panel` afterward so the AI columns
reach the panel. Responses are cached under `_data/sec_10q/_llm_cache/`, so
re-runs are free and reproducible.

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

# 5. (Optional) 10-Q text features — adds 10q_* columns to the panel
python src/pull_sec_10q_filings.py
python src/clean_sec_10q_text.py
python src/score_sec_10q_text.py
python src/analyze_sec_10q_llm.py            # (optional) generative-AI 10-Q analysis → V4
#   pilot first:  python src/analyze_sec_10q_llm.py --tickers AAPL JPM KO
python src/build_10q_monthly_panel.py
python src/build_panel.py    # rebuild panel to include 10q_* columns

# 6. CKX-style return-prediction model (V0a..V3, plus V4 once analyze has run)
python src/predict_returns_ckx.py

# 7. (Optional) Interactive results dashboard
streamlit run src/dashboard.py
```

### Outputs

- `_data/panel_monthly.parquet` — single source of truth for downstream
  models. ~4 k rows × 40-49 cols (49 once the 10-Q pipeline has been
  run), 13 tickers (AAPL, AMZN, BA, CVX, GS, HD, IBM, JPM, KO, MSFT,
  NKE, NVDA, VZ), monthly EOM, 2000 → today.
  Columns include fundamentals (`revenue`, `net_income`, `ebitda`,
  `pe_ratio`, …), Bloomberg consensus (`best_sales`, `best_net_income`, …),
  sentiment (`sentiment_diff`, `sentiment_diff_qoq`, `days_since_earnings`),
  10-Q text features (`10q_sentiment`, `10q_uncertainty`,
  `10q_cosine_vs_previous`, …) — AAPL only currently,
  macro (`vix`, `treas_10y`, `dxy`, …), trailing returns
  (`ret_1m/3m/6m/12m`), and forward-return labels (`fwd_ret_1m/3m/6m/12m`).
- `_output/ckx_predictions.parquet`, `_output/ckx_metrics.json`,
  `_output/ckx_portfolio.parquet` — per-row OOS predictions across all
  walk-forward folds, AUC/IC headline numbers per (variant, model), and
  monthly equity curves for the long-short (V1) and timing (V2/V3)
  strategies. Generated by `python src/predict_returns_ckx.py`.

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

## 10-Q Signals Pipeline

A 5-stage pipeline that turns SEC 10-Q filings (pulled from WRDS) into a
point-in-time `(date, ticker)` monthly panel of text-derived features.
The panel is namespaced with `10q_*` columns so it joins cleanly with the
other team feeds (earnings transcripts, ratios, macro) on `(date, ticker)`.

### Run order

```bash
doit pull:sec_10q_filings      # SEC EDGAR HTTPS → _data/sec_10q/{TICKER}/wrds_*_filings/
doit process_10q:clean         # → _data/sec_10q/{TICKER}/processed_text/*.txt
doit process_10q:score         # LM-dictionary + lexical drift → 10q_features.parquet
doit process_10q:panel         # merge_asof monthly grid → sec_10q_monthly_panel.parquet
doit process_10q:embed         # OPTIONAL: OpenAI embeddings + FAISS (needs OPENAI_API_KEY)
doit process_10q:analyze       # OPTIONAL: generative-AI 10-Q analysis → V4 (needs OPENAI_API_KEY)
```

The `embed` and `analyze` tasks are opt-in: neither is a dependency of
`panel`, so the default `doit` run does not call OpenAI or incur API cost.
If `OPENAI_API_KEY` is unset both scripts print a notice and exit cleanly.

`analyze` (`src/analyze_sec_10q_llm.py`) is the **generative-AI** layer: it
has `gpt-4o-mini` read each 10-Q's MD&A / risk / market-risk / controls /
legal sections, compare them to the same ticker's previous filing, and emit
structured numeric scores (`10q_ai_tone_score`, `10q_ai_risk_score`,
`10q_ai_disclosure_change_score`, …) plus a short summary and cited evidence
quotes. These columns power model variant V4 (AI replaces the LM lexicon).
The stage only analyzes filings from
`SEC_10Q_LLM_START_YEAR` (default 2014) onward to bound API cost, and caches
every response under `_data/sec_10q/_llm_cache/` keyed on
`(accession, prev_accession, prompt_version)` — so re-runs make zero API
calls and downstream builds are reproducible. After running `analyze`,
re-run `process_10q:panel` and `build_panel` so the AI columns reach the
unified panel. For a cheap pilot:
`python src/analyze_sec_10q_llm.py --tickers AAPL JPM KO`.

### Required environment variables

- `SEC_EDGAR_USER_AGENT` — optional override of the SEC contact
  identifier sent in every EDGAR HTTP request's User-Agent header.
  SEC requires a name + email. Defaults to a sensible value in
  `settings.py`; set in `.env` if you fork the project.
- `WRDS_USERNAME` — NOT required for the 10-Q pipeline anymore. Only
  used by other workstreams in this project (transcripts, CRSP) that
  still hit WRDS PostgreSQL via `~/.pgpass`.
- `OPENAI_API_KEY` — for the optional `embed` and `analyze` stages.
- `SEC_10Q_LLM_MODEL` (default `gpt-4o-mini`) / `SEC_10Q_LLM_START_YEAR`
  (default `2014`) — model and earliest filing year for the optional
  generative-AI 10-Q `analyze` stage.
- `USE_CACHE` (default `true`) — skip re-downloading files already on disk;
  also gates reuse of the 10-Q LLM response cache.

The 10-Q pipeline pulls filings directly from SEC EDGAR over HTTPS
using [`edgartools`](https://github.com/dgunning/edgartools). No SSH
key, no SFTP, no WRDS_PASSWORD. EDGAR is the canonical source for SEC
filings; WRDS just mirrors it.

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

| Column                                                                   | Source                                                      |
| ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `date`, `ticker`                                                         | Primary key.                                                |
| `filing_date`                                                            | SEC filing date — point-in-time availability.               |
| `report_period`                                                          | Fiscal quarter end.                                         |
| `accession_number`, `sec_url`, `extraction_status`, `feature_source`     | Trace metadata.                                             |
| `10q_sentiment`, `10q_positive_rate`, `10q_negative_rate`                | LM dictionary (signed + raw rates).                         |
| `10q_uncertainty`, `10q_litigious`, `10q_constraining`, `10q_word_count` | Other LM categories + length.                               |
| `10q_cosine_vs_previous`, `10q_change_vs_previous`                       | Bag-of-words drift vs previous filing.                      |
| `10q_embedding_cosine_vs_previous`, `10q_embedding_change_vs_previous`   | Semantic drift (present only when the embed stage has run). |
| `10q_ai_tone_score`, `10q_ai_risk_score`, `10q_ai_uncertainty_score`     | Generative-AI scores (present only when the `analyze` stage has run). |
| `10q_ai_margin_pressure`, `10q_ai_liquidity_pressure`, `10q_ai_demand_outlook` | Generative-AI structured judgments per filing.        |
| `10q_ai_disclosure_change_score`, `10q_ai_material_change_flag`          | Generative-AI disclosure-change scores vs the prior filing. |
| `10q_ai_summary`, `10q_ai_evidence`                                     | Generative-AI summary + cited quotes (text — dashboard only, never model features). |

### Point-in-time guarantee

The panel builder uses
`pd.merge_asof(..., direction="backward", by="ticker")` on `filing_date`,
and `build_10q_monthly_panel.py` raises `RuntimeError` if any row would
carry a filing whose `filing_date > date`. The regression test
`src/test_10q_point_in_time.py` re-asserts the invariant on the cached
features. The generative-AI `analyze` stage is point-in-time safe by the
same construction: filings are processed in `(ticker, filing_date)` order
and each one is only ever compared against a strictly-earlier filing, so
the AI columns ride the same `merge_asof` and lookahead assert.
