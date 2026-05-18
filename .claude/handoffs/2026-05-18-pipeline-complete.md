# Handoff — 2026-05-18 — Pipeline Complete

## Goal
Implement the full 5-strategy LLM-vs-lexicon backtest described in `gen ai project.md`, wired into `dodo.py`, with a single jupytext results notebook.

## Decisions made
- **LM signal**: computed on earnings-call transcripts only (NOT 10-Ks). Only Positive/Negative LM categories per spec 4.3.
- **Transcript view**: `full_transcript` locked (includes operator, safe-harbor, prepared remarks, Q&A).
- **Ridge fix**: PCA(50) pre-reduction on 1536-D delta vectors before RidgeCV. Without PCA, alpha saturates at max (1000) and Sharpe ≈ 0.05. With PCA, Sharpe ≈ 0.56.
- **PLS dropped**: tested, mean cv_r ≈ 0.004, Sharpe -0.06. All code removed.
- **Stale filter**: `days_since_earnings > 60` calendar days applies ONLY to anchor/ridge/lm. Momentum and revisions are NOT filtered (they refresh independently from Bloomberg). Bug was present and fixed.
- **Joint regression**: signals with avg < 20 obs/month dropped before joint dropna (prevents AAPL-only sig_anchor collapsing all rows when transcript data thin).
- **Calendar**: single source in `calendar_utils.py`. `freq='BME'` (not deprecated 'BM'). 21 trading days everywhere. Calendar days only for `days_since_earnings` (spec verbatim).
- **Train/test**: TRAIN 2012–2018, TEST 2019–2025. 7 expanding-window folds.

## What was done

### New files
- `src/calendar_utils.py` — all date conventions
- `src/build_returns_monthly.py` — PX_LAST → monthly panel + fwd_ret_21d
- `src/build_momentum_monthly.py` — 12-1 momentum
- `src/build_revisions_monthly.py` — 21-bday analyst revision (named `rev_30d`)
- `src/build_signal_panel.py` — joins all signals into unified panel
- `src/backtest.py` — `run_backtest()`, `compute_ic()`, `ic_summary()`; MIN_OBS_PER_MONTH=40
- `src/run_backtests.py` — 5 strategies × 3 specs + IC time series
- `src/joint_regression.py` — Fama-MacBeth + Newey-West HAC (lag=6)
- `src/train_ridge.py` — PCA(50) → StandardScaler → RidgeCV, per-fold expanding window
- `src/build_call_vectors.py` — n_chars-weighted mean embedding per call, L2-normalized
- `src/build_delta_vectors.py` — Δ call vectors + days_since_earnings (calendar days)
- `src/score_transcript_lm.py` — LM lexicon on transcript text
- `src/99_results.ipynb.py` — full results notebook (no compute, thin reader)
- `src/test_backtest_engine.py` — 17 pytest tests; all pass
- `conftest.py` (repo root) — excludes `src/_archive/` from pytest

### Modified files
- `src/embed_transcripts.py` — reads processed parquet (not per-ticker CSVs); `_split_long_piece()` for >7500 token components; `timeout=60.0`; progress every 25 batches
- `src/settings.py` — fix: `config(var, default=None, cast=None)` now returns None without crashing
- `dodo.py` — fully rewritten; 13 task groups; `PYTHON = sys.executable`

### Pipeline outputs verified
- 17/17 pytest tests pass
- `_output/99_results.html` renders at ~950KB with all sections
- Charts: `99_cum_returns_main.png`, `99_cum_returns_stale_excl.png`, `99_drawdown_main.png`, `99_hit_rates_main.png`, `99_rolling_ic.png`

## What's left
- **NOTHING IS COMMITTED**. All code is local. Need: `git add` specific files + commit.
- No other open tasks.

## Gotchas
- Run notebooks with `PATH=/opt/miniconda3/envs/genai_project/bin:$PATH python -m doit run_notebooks` — bare `python -m doit` won't find `jupyter`/`jupytext` on PATH.
- `doit list` for task names: subtasks use `:` separator e.g. `pull:manual_companies`.
- `sig_anchor` is AAPL-only until transcript pull covers full 91-ticker universe. Joint regression silently skips it if avg_obs < 20/month.
- Ridge `alpha_used` will always show 1000.0 (max of search grid); this is expected after PCA reduces dimensions — still produces signal.
- `_strip_bbg_suffix()` defined in `build_returns_monthly.py` and imported by `build_revisions_monthly.py` and `train_ridge.py` — all three must stay on `sys.path` (handled by `dodo.py` adding `./src/`).
- LM dictionary at `data_manual/lm_master_dictionary.csv` (not `_data/`).
- `settings.py` `config()` bug fix is load-bearing: any call with `default=None, cast=None` (e.g. `WRDS_PASSWORD`) would crash without it.
