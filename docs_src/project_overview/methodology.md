# Methodology

## Modelling pipeline

The core script is [src/predict_returns_ckx.py](../../src/predict_returns_ckx.py).
It is the only consumer of `_data/panel_monthly.parquet` (modelling code
never reads raw feature parquets — only `load_panel()`). For each of the
two forecast horizons (`fwd_ret_1m`, `fwd_ret_3m`) it:

1. Builds the nested variant ladder (V0a, V0b, V1, V2, V3, and — when the
   generative-AI 10-Q stage has run — V4, V5) of nested feature sets over
   the SAME 13-ticker pooled training panel.
2. Runs **expanding-window walk-forward cross-validation**: 12 annual
   folds from 2014 onward. For each fold:
   - **Train** on all rows with `date ≤ train_end`.
   - **Test** on rows in `(train_end, train_end + 1 year]`. The one-year
     embargo separates train from test temporally; no row is used for
     both.
   - **StandardScaler** is fit on the train slice only and applied to the
     test slice — no in-sample peeking.
3. For non-V0a variants, runs two regressors: `Ridge(alpha=1.0)` and
   `GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)`.
   Hyperparameters are fixed (no inner CV) to keep the comparison clean
   and avoid the "you tuned by hand" critique.
4. **V0a "zero" model** is a hard-coded `ZeroPredictor` that always
   predicts 0.0 — the trivial-baseline answer to "is anything > 0?"
5. Calibrates a binary up-probability `p_up` per fold via z-score +
   sigmoid of the continuous prediction, used for AUC / accuracy.

## Evaluation

For every (target, variant, model) combination the script emits:

- **Pooled metrics** across all 13 tickers' test rows:
  `n`, `auc`, `accuracy`, `oos_r2`, `ic_spearman`.
- **AAPL-only metrics** for continuity with prior single-ticker analyses,
  suffixed `_aapl` in the JSON output.

The pooled metrics are the headline; the AAPL-only metrics are an
internal-consistency check.

## Portfolio backtest

A single rule is applied across V0b, V1, V2, V3:

- **Long-short tertile.** Each month-end, sort the test-set predictions
  cross-sectionally; long the top tertile of tickers (equal-weight),
  short the bottom tertile (equal-weight). The strategy return for that
  month equals `(top_avg − bottom_avg) / horizon_months`, where the
  horizon division smears the 3-month overlapping holding period into a
  monthly equivalent.
- **V0a** falls back to an **equal-weight buy-and-hold** of all 13
  tickers, since zero predictions can't rank them. This is the benchmark
  the long-short strategies need to beat in risk-adjusted terms.

Reported Sharpes are **gross of transaction costs** — see the explicit
caveat in [goals.md](goals.md).

## What this design fixes versus the pre-audit V1/V2/V3 prototype

The original script trained V1 on the 13-ticker pooled panel but V2 and
V3 on AAPL-only. The V1 vs V3 delta therefore confounded *three*
variables at once: feature richness, training-sample size (10× difference),
and cross-section vs single-stock inductive bias. The current design fixes
that by training every variant on the same 13-ticker rows; only the
feature columns differ.

Other audit fixes carried in this codebase:

- **Both 1M and 3M horizons reported.** `fwd_ret_3m` has overlapping
  labels within each CV fold (three consecutive monthly observations
  share two months of return); the embargo handles train→test separation
  but not within-train iid violations. `fwd_ret_1m` has non-overlapping
  labels and is the cleaner statistical primary; `fwd_ret_3m` is the
  smoother secondary view.
- **Baselines.** V0a (zero) and V0b (momentum-only) make the project a
  hypothesis-test ladder rather than an absolute-metric beauty contest.
- **Fundamental momentum.** V1 now includes Novy-Marx-style YoY and QoQ
  growth in revenue, net income, and EBITDA, alongside the level columns.
- **Loughran-McDonald.** The 10-Q lexicon scoring uses the official LM
  master dictionary (354 positive / 2,355 negative words) — not the
  17/19-word fallback list.
- **Generative-AI 10-Q analysis (V4/V5).** Beyond the LM *dictionary*,
  `analyze_sec_10q_llm.py` has an OpenAI model read each 10-Q's MD&A, risk,
  market-risk, controls, and legal sections, compare them to the same
  ticker's previous filing, and emit structured numeric scores (tone, risk,
  uncertainty, margin/liquidity pressure, demand outlook, disclosure-change,
  material-change flag) plus a short summary and cited evidence quotes. V4
  swaps the LM lexicon for these LLM features; V5 stacks both. The ladder
  thus tests two extra hypotheses: "does an LLM reading the filing beat the
  word-count dictionary?" (V4 vs V3) and "do the two representations
  complement each other?" (V5). The stage is **point-in-time safe** — each
  filing is only ever compared against a strictly-earlier filing, no labels
  are shown to the model, and the AI columns ride the same backward
  `merge_asof` and lookahead assert as every other 10-Q feature. Responses
  are cached on `(accession, prev_accession, prompt_version)` so re-runs are
  reproducible and incur zero API cost. The text summary/evidence columns
  flow to the panel for the dashboard but are never model features.
- **Standardised portfolio rule.** Every variant with informative
  predictions is evaluated under the same long-short tertile rule, so any
  Sharpe gap is attributable to the predictions, not to a different
  strategy.

## Limitations to flag in the writeup

- 13 tickers is a narrow cross-section. Cross-sectional anomaly claims are
  not supported.
- Hand-curated survivorship-biased universe.
- Gross-of-cost Sharpes only.
- Fixed hyperparameters (no inner CV) — values stated explicitly so
  reviewers can see they weren't tuned on the test set.

---

## Company-data forecast — Chronos-2 backtest

**Code:** [src/forecast_chronos2.py](../../src/forecast_chronos2.py),
[src/backtest_chronos2.py](../../src/backtest_chronos2.py).

We use Amazon's Chronos-2 zero-shot foundation model for time-series to
forecast quarterly fundamentals (revenue, net_income) 4 quarters ahead for
each `(ticker, as_of)` in a 5 × 4 grid. The forecast is probabilistic — we
report q10 / q50 / q90 quantiles.

**Point-in-time guard.** Inside `forecast_for_ticker`, the quarterly history
fed to Chronos is filtered to `quarter_end <= as_of`. The backtest's
`naive_yoy` baseline uses values from 4 quarters before the target quarter
(always strictly earlier than as_of). No future information leaks.

**Three forecasters compared, no composite.**

| Forecaster | What it knows | Strength |
|---|---|---|
| Chronos-2 | Past quarterly history up to as_of | Foundation model; learns patterns from outside finance |
| Bloomberg consensus | Sell-side analyst aggregation at as_of | Human-aware, ingests guidance + qualitative context |
| Naive YoY | Value 4 quarters before the target quarter | Captures seasonality, cheap, hard to beat |

The win-rates (Chronos vs naive, Chronos vs consensus) and calibration
(fraction of realized values inside Chronos's q10–q90 band) are reported per
horizon. There is no triangular composite; each comparison is read on its
own merits.

**Why this is in the project.** The FINM 33200 Chronos lecture's own
framing was *"ok against sound statistical models, pretty well against
naive ones."* This backtest is the rubric's *"Forecasting: held-out metrics,
including failure cases"* evidence type made concrete. It also gives the
decision digest a numeric grounding for the "fundamentals trajectory vs
consensus" rationale paragraph.

---

## Decision digest — one-shot LLM grounding

**Code:** [src/generate_digest.py](../../src/generate_digest.py).
**Cache:** `_data/digest_cache/{ticker}__{as_of}__{prompt_version}.json`.
**Schema:** `DigestSchema` in `generate_digest.py` (strict JSON Schema).

For each `(ticker, as_of)` in the 5 × 4 grid, the digest generator:

1. **Pre-fetches deterministically** (no agent loop):
   - Returns view: `p_up` and `y_pred` from `ckx_predictions.parquet`,
     preferring V5 GBR, falling back to V4 / V3 etc. if V5 isn't trained.
   - Fundamentals view: 4-horizon Chronos rows from
     `chronos2_backtest.parquet` for (ticker, as_of). Empty list ⇒ outside
     the 5 × 4 grid; the digest's `failure_warnings` will flag it.
   - Disclosure context: top-3 chunks from the per-ticker 10-Q FAISS index
     (`_data/sec_10q/{TICKER}/10q_chunks.faiss`), PIT-filtered to
     `filing_date <= as_of`. Fixed retrieval query: *"recent material
     developments, risks, outlook, and changes vs the prior quarter."*
   - Transcript context: top-2 chunks from
     `_data/embeddings_transcripts.parquet`, cosine-ranked, ticker- and
     event_date-filtered (PIT). Chunk text reconstructed from the
     deterministic chunker on the components CSV.

2. **Calls OpenAI exactly once** with `gpt-4o-mini` and the strict
   `DigestSchema` response_format. Temperature 0. Cached on
   (ticker, as_of, prompt_version, model).

3. **Output (DigestSchema)**: `forecast_direction` ∈ {bullish, neutral,
   bearish}, `confidence` ∈ [0, 1], three rationale paragraphs (returns,
   fundamentals, disclosure), 3–5 evidence items each tagged with
   `source_type` and `source_id`, plus a `failure_warnings` array.

**Anti-hallucination prompt rules** (from the Fuentes lecture, slide 17 —
visible reward-hacking went from "cited evidence" to "fluent but
evidence-free"):

- *"You are reasoning AT as_of_date={X}. Do not reference events dated after
  that."* (PIT guard at the prompt layer, not just at the data layer.)
- *"When you quote a number or sentence, copy verbatim from the pre-fetched
  evidence. Do not paraphrase numerics."*
- *"Cite only from the evidence block — do not cite from training data."*

---

## Verifier design — what we built and what we didn't

The digest is evaluated by three **independent** verifiers — reported
separately, NOT combined into a composite. Code:
[src/eval_digest.py](../../src/eval_digest.py).

| Verifier | What it checks | Implementation | Cost |
|---|---|---|---|
| **citation_match_rate** (Fuentes `e` — evidence leg) | Quoted strings (≥5 words) appear verbatim in the pre-fetched evidence block | Whitespace-normalized substring | $0 |
| **numeric_grounding_rate** (Fuentes Extract — MR-RLVR slide 21) | Numbers in the rationale paragraphs (regex-extracted: currency, percentages, bps) appear verbatim in the evidence block | Deterministic regex + string contains | $0 |
| **direction_match** (Fuentes `o` — outcome leg) | `forecast_direction` sign matches realized `fwd_ret_1m` (when realized) | `y_true` from `ckx_predictions.parquet` | $0 |

### Considered but deliberately not implemented

The course material from Stockton, Olson, and Fuentes describes more
elaborate evaluation patterns. We considered each and chose not to
implement:

1. **LLM-as-judge reasoning verifier (Fuentes' `r` leg).** Would have
   required a separate LLM call per digest asking *"does the rationale
   follow from the evidence?"* The fundamental problem: we have no way to
   independently calibrate the judge against human labels in 12 days. An
   uncalibrated meta-metric is worse than no metric. Reported as the
   top stretch goal in this writeup.

2. **Triangular composite `q = ∛(e · r · o)` (Fuentes Trade-R1).** Requires
   the `r` leg. Without it, the geometric mean collapses to two-axis
   and the composite is misleading. Reporting each axis independently is
   more honest.

3. **Process-verifier on every numeric claim (Fuentes MR-RLVR full
   pattern).** The full pattern would Extract → Calculate → Cite each
   numeric assertion individually, with the Calculate step recomputing
   derived metrics (QoQ deltas, YoY growth) from raw panel values. We
   implement the Extract leg (`numeric_grounding_rate`) and stop there.
   The Calculate leg requires brittle regex over free-form rationale text;
   the failure mode (regex misses a claim and silently passes it) is
   exactly the kind of silent failure we warn about elsewhere.

4. **Agentic RAG loop with typed tools (Stockton "Agentic RAG on SEC
   Filings," phases 3-4).** The classical-RAG → agentic-tools delta in
   Stockton's deck is real for **cross-filing, multi-company analyst
   queries** (his example: "find 2-3 of 10 companies where AI risk framing
   shifted 2022 → 2024"). Our digest's question is narrower: *"summarize
   this one ticker as of this one date."* For a narrow query with a clean
   join key, classical pre-fetched retrieval works — and the course's RAG
   lesson itself notes "when the question carries a clean join key, the
   structured lookup is hard to beat." We adopted Stockton's typed-parameter
   discipline (ticker, fiscal_quarter, filing_date are explicit parameters
   in the pre-fetch helpers, not crammed into a query string) without the
   agentic loop overhead.

5. **Trace audit framework (Stockton phase 4 / HW4).** The digest is a
   single LLM call. There is no trace to audit. We log the cache JSON
   instead, which carries the pre-fetched evidence block plus the response —
   sufficient for the verifier passes.

6. **Multi-agent decomposition (Olson lesson — start single, add later).**
   Olson explicitly recommends starting with one agent and adding
   complexity only after understanding the base case. With 12 days to the
   deadline, the base case is the deliverable.

7. **Transcript-level structured LLM features (V6 ladder rung).**
   Considered as a stretch. Would have mirrored `analyze_sec_10q_llm.py`
   on the transcript corpus, adding `tx_ai_*` columns to the panel and a
   V6 model variant. Dropped because (a) V4 / V5 already demonstrate the
   prompting-for-return-prediction technique from HW2, and (b) the digest
   layer already uses transcripts as retrieval context.
