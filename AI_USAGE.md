# AI usage statement

Submitted in compliance with the FINM 33200 final-project rubric requirement:
> "what tools you used, where they helped, and how you checked their outputs."

## TL;DR

- **No generative LLM is in the scoring path.** The only AI model that touches the data that produces a signal is OpenAI's `text-embedding-3-small`, which converts transcript text into numeric vectors. From that point on, every signal value is computed by deterministic linear algebra or a small ridge regression with fixed hyperparameters chosen by time-series cross-validation.
- **Claude Code was used during development.** The `.claude/` directory in this repo makes that obvious. It was used for code scaffolding, refactoring, transcript-cleanup utilities, dodo task wiring, and authoring this write-up. Every change was reviewed by the project author; no AI-authored code was merged without inspection.

## 1. AI in the product (the scoring path)

| Tool | Where it's used | Inputs / outputs |
|---|---|---|
| **OpenAI `text-embedding-3-small`** (1,536-D) | Per-chunk dense embedding of earnings-call transcripts in [src/embed_transcripts.py](src/embed_transcripts.py). | Inputs: ~6,000-token chunks of cleaned transcript text. Outputs: L2-normalised 1,536-D vectors written to `_data/embeddings_transcripts.parquet`. |

What happens *after* the embeddings is fully deterministic:

| Step | What it does | No-LLM verification |
|---|---|---|
| **Per-call vector** ([src/build_call_vectors.py](src/build_call_vectors.py)) | `n_chars`-weighted mean of chunk embeddings, re-normalised. | Pure linear algebra on the parquet from the previous step. |
| **Per-call Δ vector** ([src/build_delta_vectors.py](src/build_delta_vectors.py)) | Current call vector minus the same ticker's previous call vector. | Pure subtraction; first call per ticker is dropped. |
| **Anchor cosine (Strategy 1)** ([src/score_transcript_sentiment.py](src/score_transcript_sentiment.py)) | Cosine similarity of each call vector against ten hand-written anchor sentences (5 positive, 5 negative). No LLM at scoring time — only the anchor sentences themselves were authored by a human. | Anchor sentences are listed verbatim in `score_transcript_sentiment.py` and reproduced in [METHODOLOGY.md § 4.3](METHODOLOGY.md). |
| **Ridge + PCA (Strategy 2)** ([src/train_ridge.py](src/train_ridge.py)) | PCA(50) → StandardScaler → RidgeCV on `Δ call_vector + days_since_earnings`. Hyperparameters: $\alpha$ grid log-spaced 0.01–1000, 5-fold `TimeSeriesSplit` inside the training window. | Scikit-learn estimators with fixed random seeds and a strict train/test time split (2012–2018 / 2019–today). No leakage. |
| **LM lexicon (Strategy 3)** ([src/score_transcript_lm.py](src/score_transcript_lm.py)) | Bag-of-words match against the Loughran-McDonald Positive and Negative word lists; net positivity Δ between consecutive calls. | Word-counting only. No AI. |
| **Momentum, revisions (Strategies 4, 5)** ([src/build_momentum_monthly.py](src/build_momentum_monthly.py), [src/build_revisions_monthly.py](src/build_revisions_monthly.py)) | Standard 12-1 momentum and 21-business-day change in consensus FY1 net income from Bloomberg. | No AI. |
| **Backtest, IC, joint regression** ([src/run_backtests.py](src/run_backtests.py), [src/joint_regression.py](src/joint_regression.py)) | Identical machinery for all five signals: rank, form legs, compute returns, summarise. Fama-MacBeth with Newey-West (lag 6) for the joint test. | Pure pandas + scipy + statsmodels. |

So the only place a model produces something we don't fully understand is the embedding step. Strategies 1–2 then build interpretable structures *on top of* those embeddings (cosine projections, a 50-dimensional PCA basis, a linear regression). The downstream behaviour of those structures is fully inspectable.

## 2. AI in development

[Claude Code](https://www.anthropic.com/claude-code) (Anthropic) was used throughout the development of this project. The `.claude/` directory ships with the repo and contains the conversation handoffs, plans, and settings used during development — this disclosure is therefore already implicit.

| Area | How heavily AI-assisted | Reviewed by author |
|---|---|---|
| Pipeline scaffolding ([dodo.py](dodo.py), file/directory layout, settings module) | Heavy | Every task definition |
| Transcript cleanup utilities ([src/clean_transcript_*](src/), [src/extract_min10y_transcripts.py](src/extract_min10y_transcripts.py)) | Heavy | Spot-checked on per-ticker output |
| Backtest machinery ([src/backtest.py](src/backtest.py), [src/run_backtests.py](src/run_backtests.py)) | Moderate | Each metric formula reviewed against [METHODOLOGY.md § 5](METHODOLOGY.md) |
| Ridge model design ([src/train_ridge.py](src/train_ridge.py)) | Moderate | Train/test boundary, $\alpha$ grid, PCA components chosen by the author |
| Signal definitions, anchor sentences, methodology choices | None | Author-authored |
| The five interpretation choices that drive the results (delta vs level, carry-forward, 60-day stale filter, top-20 / bottom-20 legs, no transaction costs) | None | Author-authored |
| Quarto write-up prose ([reports/writeup.qmd](reports/writeup.qmd)), [METHODOLOGY.md](METHODOLOGY.md) | Moderate (drafting), heavy (formatting) | Every numerical claim verified against the rendered notebook / JSON files |

## 3. How outputs were checked

| What | Verification |
|---|---|
| OpenAI embeddings | Returned vectors checked for L2-norm near 1.0 and dimension 1,536. Embedding cache keyed on (text hash, model name) so identical text → identical vector. |
| Per-call signal panels | Pytest suite in `src/`. Calendar-parity assertions checked that `event_date` → `month_end` mapping is consistent across all signal columns. |
| Backtest results | Reproducible from the parquet panels — running `doit run_backtests` from a clean state produces byte-identical `_data/metrics_*.json` and `_data/results_*.parquet`. |
| Look-ahead | The ridge model's expanding-window refits explicitly fit PCA, StandardScaler, and RidgeCV only on `event_date ≤ end of year Y-1`, then score year `Y`. No transform sees future data. |
| Numerical claims in the write-up | Every number quoted in `reports/writeup.qmd` is loaded at render time from `_data/metrics_*.json`, `_data/ic_summary.json`, or `_data/fm_results.json`. There are no hand-typed numbers in the prose. |
| AI-authored code | Reviewed line by line by the author before commit. `pytest src` runs as the final `doit` task; a green test suite is the gate. |

## 4. Cost

| Item | Estimate |
|---|---|
| OpenAI `text-embedding-3-small` — chunk-level transcript embedding (one-time, cached) | ~$8 |
| OpenAI generative LLM calls | $0 (no generative LLM is used at scoring time) |
| **Total OpenAI spend** | **~$8 — well under the rubric's $50 proctor-approval threshold.** |

Re-runs are free because embedding output is persisted in `_data/embeddings_transcripts.parquet`; the pipeline only re-embeds chunks whose `text_hash` is new.
