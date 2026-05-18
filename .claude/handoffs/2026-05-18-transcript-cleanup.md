# Handoff — repo cleanup + transcript pipeline explainer

## Goal
Cleanup multi-contributor mess in `finm33200_project` to focus on the agreed-upon goal: extract LLM-based sentiment from Nasdaq-100 earnings-call transcripts and benchmark vs naive baselines. Then explain the transcript pipeline (data model, embeddings, scoring) to the user, who didn't build it.

## Decisions
- Archive other-contributors' code (don't delete) to `src/_archive/`; **delete** user's own Chronos code.
- Strip Chronos from docs in this pass; defer broader README rewrite (still describes archived 10-Q + CKX pipeline).
- Keep `data_manual/` Bloomberg pulls (`pull_manual_macro.py`, `pull_manual_companies.py`, `pull_bbg_earning_data.py`) — user said useful.
- Archive the 10-Q stack: functional and well-tested, but only 13 tickers, doesn't match Nasdaq-100 scope, different release timing than transcripts.
- No commits made — all changes staged only; user reviews `git diff` before committing.
- Approved plan archived at `/Users/flavio/.claude/plans/okay-i-m-gonna-need-inherited-metcalfe.md`.

## What was done

### Cleanup (uncommitted, staged)
- Deleted 6 Chronos `.py` files in `src/` + 4 gitignored artifacts (`_data/*Chronos*.parquet`, `_output/0[45]_*.{ipynb,html}`, stale pyc).
- `git mv`-ed 31 files from `src/` → [src/_archive/](src/_archive/) (10-Q stack, CKX regression scaffolding, digest/eval/dashboard, old forecasting, CRSP pulls, demos, notebooks 01+03). Preserves history.
- Rewrote [dodo.py](dodo.py) — only 4 active task groups remain: `config`, `pull:{manual_macro,manual_companies}`, `build_sentiment:{embed,score,monthly}`, `run_pytest`.
- Stripped Chronos from `README.md`, `AI_USAGE.md`, `docs_src/project_overview/{goals,methodology}.md`, `docs_src/results/{forecasts,digest_examples}.md`. Verification: `grep -ri chronos` over active code+docs → zero hits.
- Active `src/` is down to 17 transcript-focused `.py` files plus `_archive/__init__.py`.

### Verification done
- `python -m doit list` → expected 4 task groups only.
- `python -c "from dodo import *"` → imports clean.
- `pytest src/test_misc_tools.py` → 5/5 pass.

### Pipeline explainer delivered
- Walked user through WRDS Capital IQ schema (component-level rows, section types like `Presenter Speech` / `Question` / `Answer`), the three on-disk views per ticker, token-budget chunking (~6000-token chunks via `tiktoken`), `text-embedding-3-small` → 1536-D vectors, anchor-cosine scoring methodology, `n_chars`-weighted mean per call, monthly carry-forward.
- Explained Loughran-McDonald dictionary structure ([data_manual/lm_master_dictionary.csv](data_manual/lm_master_dictionary.csv) — 86,554 words × 7 finance-specific category flags: Negative/Positive/Uncertainty/Litigious/Strong_Modal/Weak_Modal/Constraining). LM scoring code currently only exists in archive ([src/_archive/score_sec_10q_text.py](src/_archive/score_sec_10q_text.py)) for 10-Qs, not transcripts.

## What's left

### Immediate
- **Commit** the cleanup. Nothing committed yet; user should `git diff` first.
- **Follow-up README rewrite** — current README still describes the archived 10-Q + CKX pipeline with broken file links to `src/_archive/*` paths. Plan explicitly defers this.

### Strategic next experiments (proposed; user not yet committed)
1. **LM lexicon baseline for transcripts** — write new ~50-line `score_transcript_lm.py` that ports loader from [src/_archive/score_sec_10q_text.py](src/_archive/score_sec_10q_text.py) and applies it to per-call `full_text` in [_data/transcripts/{TICKER}/{ticker}_earnings_calls.csv](_data/transcripts/AAPL/aapl_earnings_calls.csv). Output schema should match `_data/sentiment_transcripts.parquet` for clean side-by-side comparison.
2. **Q&A-only anchor scoring** — current code averages all components equally (by `n_chars`). Component-level `transcript_component_type_name` ∈ {`Presentation Operator Message`, `Presenter Speech`, `Question and Answer Operator Message`, `Question`, `Answer`, `Unknown Question and Answer Message`} lets you separate scripted vs candid; Q&A typically carries more signal.
3. **Scale embedding pipeline to Nasdaq-100** — pipeline is hard-wired to AAPL via `task_build_sentiment:embed`'s `file_dep` on `_data/transcripts/AAPL/aapl_transcript_components.csv`. Extending to 91 tickers needs (a) the cleaned per-ticker components on disk (extraction/cleaning already done — see `_data/transcripts/raw/nasdaq100_min10y_*`), (b) loosening the dodo file_dep, (c) ~$3-4 OpenAI spend.
4. **Per-chunk LLM classification** and/or **structured extraction** as the "real LLM" comparison (~$300-500 for full Nasdaq-100).

## Gotchas
- **No `pyproject.toml`/`pytest.ini`/`setup.cfg`** exist — `src/_archive/` is excluded from pytest by the leading-underscore convention only. Adding configs may break this assumption.
- **`requirements.txt` still pulls `chronos-forecasting` + `torch`** — not pruned (plan deferred to avoid breaking other contributors' envs).
- **Correct Python env is `/opt/miniconda3/envs/genai_project`** — `doit` not on default PATH. User flagged this when default `python` was used.
- **`pull_all_transcripts.py` was kept in active src/** but not exercised end-to-end this session; relationship to `extract_sample_raw_transcripts.py` not deeply audited — both fetch from WRDS, the latter is the Nasdaq-100 min10y workhorse.
- **`task_build_sentiment` is AAPL-pinned** via its `file_dep` on the AAPL components CSV. Don't assume it scales by default.
- **Embeddings are NOT explicitly L2-normalized** in `embed_transcripts.py`, but OpenAI returns near-unit-norm vectors and `score_transcript_sentiment.py` normalizes before cosine. Don't normalize twice if porting downstream code.
- **The score is `sentiment_diff = sentiment_pos - sentiment_neg`**, both cosine values in practice 0.1–0.4. Don't interpret as a probability or a z-score without standardizing.
- **`sentiment_diff_qoq`** (quarter-on-quarter delta) is computed in the monthly rollup, not in the per-call scorer. Lives in [features_sentiment_monthly.parquet](_data/features_sentiment_monthly.parquet), not in `sentiment_transcripts.parquet`.
- **The anchor phrases** ([src/score_transcript_sentiment.py:45-59](src/score_transcript_sentiment.py#L45-L59)) are doing all the semantic work and are unvalidated. Changing them changes every score — treat as a hyperparameter.
- User reads section headings naturally — keep responses prose-heavy with clear paragraph breaks, but don't over-decorate with H2/H3 unless the content is genuinely sectional.
