# finm33200_project — LLM embeddings vs. traditional sentiment on earnings calls

FINM 33200 final project. A comparative backtest of five cross-sectional long-short equity signals on a fixed 91-ticker Nasdaq-100 universe, identical machinery across all five, isolating the effect of signal choice from backtest mechanics.

The central question: **do off-the-shelf LLM embeddings of earnings-call transcripts add cross-sectional return information beyond a Loughran-McDonald lexicon and the standard non-text factors (price momentum, analyst revisions)?**

The five signals, all run on identical machinery (monthly rebalance, top-20 / bottom-20 equal-weighted, 21-trading-day holding period, no transaction costs):

1. **Anchor cosine on Δ sentiment** — projection of the call vector onto a 10-sentence positive-minus-negative direction, quarter-over-quarter change. The "LLM, but cheap" baseline.
2. **Ridge + PCA on Δ call vectors** — RidgeCV on PCA(50) of the 1,536-D embedding delta plus `days_since_earnings`, supervised on 21-day forward returns. The central LLM-method test.
3. **Loughran-McDonald Δ net positivity** — bag-of-words match against the LM Positive / Negative dictionaries, net-positivity Δ between consecutive calls.
4. **Price momentum 12-1** — standard cross-sectional momentum factor.
5. **Analyst revisions in Δ BEst Net Income** — 21-business-day change in consensus FY1 net income.

The project is **comparative**, not absolute: universe, holding period, ranking rule, and benchmark are identical across signals. Survivorship bias, no transaction costs, small universe, short OOS — all named in the limitations section. No claims about deployable performance.

## Where the story lives

- [`reports/writeup.html`](reports/writeup.html) — the audience-facing write-up. Motivation, signal taxonomy, headline results, robustness checks, limitations. Self-contained HTML; open it directly.
- [`METHODOLOGY.md`](METHODOLOGY.md) — methodology spec. Every transformation, hyperparameter, and choice with rationale.
- [`_output/99_results.html`](_output/99_results.html) — auto-rendered results notebook with all tables and charts.
- [`AI_USAGE.md`](AI_USAGE.md) — disclosure of AI tools used in the product and in development.

## Repo layout

| Path | What's in it |
|---|---|
| [`src/`](src/) | Pipeline modules (one responsibility per file) + `99_results.ipynb.py` (jupytext notebook). |
| [`dodo.py`](dodo.py) | doit task graph orchestrating the full pipeline end-to-end. |
| [`_data/`](_data/) | Built artifacts: parquet panels, JSON metric files. Not checked in. |
| [`_output/`](_output/) | Rendered notebook, PNG charts, pytest report. |
| [`reports/`](reports/) | Quarto write-up source (`.qmd`) and rendered HTML. |
| [`docs/`](docs/) | Pipeline reproducibility notes. |
| [`data_manual/`](data_manual/) | Bloomberg Terminal exports and the LM master dictionary. Hand-maintained. |
| [`tests/`](tests/) (via `pytest src`) | Tests live alongside the modules they cover under `src/`. |

## Reproduction

The repo uses a `conda` env named `genai_project`. Replace the interpreter path with your own if you use a different env manager.

**Prerequisites**

- Python 3.12, the dependencies in `requirements.txt`.
- [Quarto](https://quarto.org/docs/get-started/) on the PATH for the write-up step (e.g. `brew install --cask quarto-cli` on macOS).
- A WRDS Capital IQ subscription to pull the raw earnings-call transcripts.
- An OpenAI API key in `.env` (`OPENAI_API_KEY=...`) to embed the transcripts via `text-embedding-3-small`.
- The Bloomberg workbook under [`data_manual/`](data_manual/) ships with the repo (prices, BEst Net Income consensus, the Loughran-McDonald dictionary). The transcript, embedding, and signal-panel parquets under `_data/` are gitignored and rebuilt by the pipeline.

**Install dependencies**

```bash
/opt/miniconda3/envs/genai_project/bin/python -m pip install -r requirements.txt
```

**Run the full pipeline**

```bash
/opt/miniconda3/envs/genai_project/bin/python -m doit
```

This rebuilds everything end-to-end: embeddings → call vectors → signal panels → backtests → Fama-MacBeth → results notebook → write-up.

**Inspect or rerun individual phases**

```bash
/opt/miniconda3/envs/genai_project/bin/python -m doit list      # all tasks
/opt/miniconda3/envs/genai_project/bin/python -m doit run_backtests
/opt/miniconda3/envs/genai_project/bin/python -m doit run_notebooks
/opt/miniconda3/envs/genai_project/bin/python -m doit write_report
```

Each task declares its `file_dep` and `targets`, so doit only re-runs what's stale.

## Outputs

After a full run:

- `_data/metrics_main.json`, `_data/metrics_stale_excl.json`, `_data/metrics_post2018.json` — per-strategy return/risk metrics under the three robustness specifications.
- `_data/ic_summary.json`, `_data/ic_timeseries.parquet` — cross-sectional Spearman IC per strategy, summary and monthly time series.
- `_data/fm_results.json` — Fama-MacBeth joint regression coefficients and Newey-West HAC standard errors.
- `_output/99_*.png` — six charts: hit rates, rolling IC, cumulative returns and drawdowns on two common windows.
- `_output/99_results.html` — auto-rendered jupyter notebook with all tables and charts.
- `reports/writeup.html` — final audience-facing write-up (self-contained, all images embedded).

## Tests

```bash
/opt/miniconda3/envs/genai_project/bin/python -m pytest src
```

Tests live alongside the modules they cover under `src/`.
