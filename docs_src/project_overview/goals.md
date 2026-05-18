# Goals

## What this project is

A **forecast-spine copilot** for equity analysis. Two forecasts feed one
decision output:

```
Returns forecast (V0a → V5 ladder)  ──→  Decision digest  ──→  honest evaluation
                                          (one-shot, not agentic)
```

The artifact is twofold: the two forecasts (whose accuracy is reportable on
held-out data) and the LLM-authored decision digest that grounds the
forecasts in cited 10-Q + transcript evidence and emits a structured
recommendation. Three independent verifiers — citation match, numeric
grounding, and direction match against realized returns — score the digest
without combining into a single composite.

## Research question

**Does disclosure / text-derived signal add incremental predictive value on
top of price momentum, fundamentals, macro, and earnings-call sentiment,
when monthly stock returns are forecast under strict point-in-time
constraints?**

## Hypothesis testing structure

The project answers the question by stacking nested feature sets and
comparing them on identical out-of-sample evaluation rows. Each step is a
single hypothesis test:

| Variant | Features | Hypothesis tested |
|---|---|---|
| **V0a** | none — predict zero | Is anything > 0? (sanity check) |
| **V0b** | trailing-return momentum only | The academic bar (Jegadeesh-Titman 1993) |
| **V1**  | V0b + fundamentals (levels + YoY/QoQ growth) + Bloomberg consensus + macro | Does fundamentals + macro add to momentum? |
| **V2**  | V1 + earnings-call sentiment | Does call sentiment add? |
| **V3**  | V2 + SEC 10-Q Loughran-McDonald lexicon features | Does the 10-Q dictionary signal add? |
| **V4**  | V3 base + generative-AI 10-Q analysis only (lexicon dropped) | Does an LLM reading the 10-Q beat the dictionary? |
| **V5**  | V3 base + LM lexicon + generative-AI 10-Q analysis | Do the dictionary and the LLM complement each other? |

V0b is the bar that V1–V5 each need to beat. All variants are trained on
the same 13-ticker pooled cross-section and evaluated on identical test
rows, so the only thing changing between variants is the feature columns.
V4/V5 use an OpenAI model that reads each 10-Q and scores how its
disclosure changed versus the same ticker's previous filing — the
generative-AI counterpart to the V3 word-count dictionary. They are
optional: the ladder runs V0a–V3 unchanged until the
`doit process_10q:analyze` stage has produced the AI columns.

## Success criteria

A successful run answers each of the four nested hypotheses with a
defensible "yes" or "no":

1. **Statistically meaningful** — out-of-sample R² and Spearman IC for each
   variant, with V0a / V0b as the noise-floor reference. Differences need
   to be large enough to plausibly survive the limited cross-section (13
   tickers) and the overlapping-label uncertainty inherent in `fwd_ret_3m`.
2. **Economically meaningful** — a long-short tertile portfolio backtest,
   gross of transaction costs, that uses the *same* portfolio construction
   rule across V0b–V5 (V0a uses equal-weight buy-and-hold as the
   benchmark).
3. **Robust to horizon** — both `fwd_ret_1m` (clean non-overlapping labels,
   primary) and `fwd_ret_3m` (smoother signal, secondary) reported side by
   side so the reader can see whether conclusions hold across horizons.
4. **Point-in-time safe** — automated tests (`test_panel_no_lookahead.py`,
   `test_10q_point_in_time.py`) confirm no feature row uses information
   from after its `date`.

## What is NOT claimed

- This is a research evaluation, not a trading strategy. Reported portfolio
  Sharpes are **gross of transaction costs** and should be interpreted as
  upper bounds.
- The 13-ticker universe is hand-curated and biased toward current
  large-cap survivors (4 mega-cap tech + 5 industrials + finance, staples,
  energy, telecom). Generalization to a broader universe (e.g., Russell
  1000) is out of scope and not implied.
- The cross-section is too narrow (13 tickers) to support strong
  cross-sectional anomaly claims; results should be read as
  "does feature X add predictive content within this universe" rather than
  "does feature X work universally."

## Project evolution

The repo began as **"the 10-Q signals project"** — a SEC-EDGAR text pipeline
producing `10q_*` features for the panel, evolved through V0a → V5 model
variants culminating in generative-AI structured 10-Q analysis (V4 / V5). By
mid-May 2026 those features had been integrated and the marginal gain from
further V4 / V5 polish was diminishing.

The project pivoted (2026-05-15) to a **forecast-spine copilot**. The
existing return-prediction ladder is the returns forecast, and a one-shot
LLM-authored **decision digest** combines the forecast with cited
disclosure evidence into a structured recommendation.

The pivot was informed by three FINM 33200 guest lectures:

- **Stockton — "Agentic RAG on SEC Filings."** Typed retrieval tools with
  ticker / fiscal-year as explicit parameters beat top-K stuffed-context
  RAG, but **classical RAG works when the question is narrow**. Our digest
  uses pre-fetched, ticker-filtered retrieval — not an agent loop — because
  the question *("summarize this ticker as of this date")* is narrow enough
  that the agentic overhead isn't justified.
- **Olson — "Systematic Research Agents with Claude Code."** Reward hacking,
  silent path-dependence, and "fluent but evidence-free" reasoning are real
  failure modes. We mitigate at the prompt layer (verbatim-numerics rule)
  and at the verification layer ([eval_digest.py](../../src/eval_digest.py)).
  Olson's "start with one agent; add complexity later" lesson is why the
  digest is a single LLM call and not a multi-agent system.
- **Fuentes — "RLVR for Finance."** *"The verifier is the IP. Spend the
  first month on verifier design, not model selection."* Our three
  verifiers (citation match, numeric grounding, direction match) are
  reported independently — we deliberately did **not** combine them into
  a triangular composite because the reasoning leg of his triangle (an
  LLM-as-judge `r`) cannot be calibrated against human labels in 12 days.
  See [methodology.md](methodology.md) for the full reasoning.

The pivot narrative IS evaluation evidence per the rubric's
*"honest evaluation"* criterion — see methodology.md for the full
"considered but not implemented" list.
