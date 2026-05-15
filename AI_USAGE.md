# AI usage statement

Submitted in compliance with the FINM 33200 final-project rubric requirement:
> "what tools you used, where they helped, and how you checked their outputs."

## Tools used

| Tool | Where it was used | Inputs / outputs |
|---|---|---|
| **Claude Code** (Anthropic) | Development environment. Architectural planning, code authoring, refactoring, debugging, writeup drafting. Operated by the project author throughout. | Source: the code, plans, and writeups in this repo. |
| **OpenAI `gpt-4o-mini`** | (1) Per-filing structured 10-Q analysis in [src/analyze_sec_10q_llm.py](src/analyze_sec_10q_llm.py). (2) Per-(ticker, as_of_date) decision digest in [src/generate_digest.py](src/generate_digest.py). | Inputs: pre-fetched 10-Q section text, transcript chunks, and panel features. Outputs: strict-JSON-Schema responses cached under `_data/sec_10q/_llm_cache/` and `_data/digest_cache/`. |
| **OpenAI `text-embedding-3-small`** | Dense-embedding 10-Q chunks ([src/embed_sec_10q_text.py](src/embed_sec_10q_text.py)), transcript chunks ([src/embed_transcripts.py](src/embed_transcripts.py)), and the fixed retrieval query in [src/generate_digest.py](src/generate_digest.py). | Outputs: L2-normalized 1536-dim vectors. |
| **Amazon `chronos-2`** via `chronos-forecasting` | Zero-shot probabilistic fundamentals forecasts in [src/forecast_chronos2.py](src/forecast_chronos2.py) and the historical backtest in [src/backtest_chronos2.py](src/backtest_chronos2.py). | Inputs: quarterly revenue / net_income / EBITDA time series (PIT-sliced). Outputs: q10 / q50 / q90 forecasts at four quarterly horizons. |

## Where AI helped (and where it didn't)

The point of the project is to evaluate this honestly. The relevant evidence is:

- **`_output/ckx_metrics.json`** — does each AI-derived feature layer (V4 = generative-AI 10-Q analysis alone; V5 = AI + LM dictionary combined) actually improve out-of-sample AUC / rank IC / portfolio Sharpe over V3 (lexicon only)?
- **`_output/chronos2_backtest_summary.json`** — Chronos-2 vs Bloomberg consensus vs naive YoY MAE / MAPE / win-rate / coverage. Reports whichever wins. The course's own framing of Chronos was *"ok against sound statistical models, pretty well against naive ones."* We expect and report a result in that range.
- **`_output/digest_eval_summary.json`** — citation_match_rate, numeric_grounding_rate, and direction_match for the LLM-authored digest. If the LLM digest hallucinates numbers, the numeric_grounding_rate exposes it; if it picks wrong directions, direction_match exposes it.

The writeup (`docs_src/results/*.md`) reports each result with both successes and failure cases. Per the rubric: *"If the AI component doesn't help, say so."*

## How outputs were checked

| Output | Verification mechanism | Code |
|---|---|---|
| 10-Q AI analysis (V4/V5 features) | (a) JSON-Schema strict mode enforced server-side; (b) post-hoc range validation in `analyze_sec_10q_llm.py:validate_response`; (c) point-in-time guard — each filing only compared against a strictly-earlier same-ticker filing. | [src/analyze_sec_10q_llm.py](src/analyze_sec_10q_llm.py), [src/test_panel_no_lookahead.py](src/test_panel_no_lookahead.py) |
| Decision digest | Three independent verifiers in [src/eval_digest.py](src/eval_digest.py): (1) `citation_match_rate` — fraction of quoted strings that appear verbatim in the pre-fetched evidence block; (2) `numeric_grounding_rate` — fraction of regex-extracted numbers in the rationale that appear verbatim in the evidence block; (3) `direction_match` — sign-match between forecast_direction and realized `fwd_ret_1m`. Reported independently, NOT combined into a triangular composite (see methodology.md for why). | [src/eval_digest.py](src/eval_digest.py) |
| Chronos forecasts | PIT guard in `forecast_for_ticker` (only history with `quarter_end <= as_of` is fed to the model). Backtest compares forecasts against held-out realized values, plus a deterministic naive-YoY baseline and Bloomberg consensus. | [src/backtest_chronos2.py](src/backtest_chronos2.py) |
| Embeddings | Outputs are L2-normalized vectors of fixed dimension (1536). Used only for cosine-similarity ranking, not as standalone signals. Drift features (`10q_cosine_vs_previous`, etc.) are reported but not load-bearing. | [src/embed_sec_10q_text.py](src/embed_sec_10q_text.py), [src/embed_transcripts.py](src/embed_transcripts.py) |
| Generated code | Every Claude-authored change reviewed by the project author. Existing tests (`pytest --doctest-modules`) and the lookahead test ([src/test_panel_no_lookahead.py](src/test_panel_no_lookahead.py)) run on a fresh venv as part of the day-11 repro test. | — |

## Cost

| Item | Estimate |
|---|---|
| OpenAI `gpt-4o-mini` — 10-Q analysis (existing, cached) | ~$3 one-time, $0 on re-runs |
| OpenAI `gpt-4o-mini` — 20 cached digests | ~$2 |
| OpenAI `text-embedding-3-small` — 10-Q chunks, transcript chunks, digest queries | ~$0.30 |
| Chronos-2 inference | $0 (local CPU) |
| **Total OpenAI spend** | **<$10 — well under the rubric's $50 proctor-approval threshold.** |

Re-runs are free because every LLM call is cache-keyed on (input identifier, prompt_version, model). Bumping `PROMPT_VERSION` is the only way to force a paid re-run.
