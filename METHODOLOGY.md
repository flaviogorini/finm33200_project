# LLM embeddings on earnings-call transcripts
## A methodology write-up of the FINM 33200 final project

---

## Headline summary

This project asks whether modern large-language-model (LLM) embeddings of
earnings-call transcripts add cross-sectional information beyond traditional
quantitative signals, when used to rank stocks for a monthly long-short equity
strategy on a 100-ticker Nasdaq-100 universe.

Five signals are compared on identical backtest machinery (monthly rebalance,
top-quintile / bottom-quintile equal-weighted legs, 21-trading-day holding
period, no transaction costs):

1. **Anchor cosine on Δ sentiment** — a hand-written ten-sentence anchor set,
   cosine similarity to the call vector, quarter-over-quarter change. The
   "LLM, hand-anchored" approach: no labels, no training, all interpretive
   burden carried by the choice of anchor sentences.
2. **Ridge regression with PCA pre-reduction on Δ call vectors** — the
   "LLM, learned" counterpart. PCA(50) collapses the 1,536-D embedding delta
   into 50 principal components before RidgeCV; `days_since_earnings` is
   concatenated after PCA as a freshness control. Trained on 2012-2018, so
   predictions only start in Feb 2019 and the test sample is shorter than
   the other signals' by ~13 years.
3. **Loughran–McDonald net-positivity lexicon on Δ score** — the
   classical text-NLP baseline; only Positive and Negative word lists used.
4. **Price momentum 12-1** — the standard equity factor; included as a
   non-text benchmark.
5. **Analyst revisions in BEst Net Income** — 21-business-day estimate
   revisions from Bloomberg; the second non-text benchmark.

The project is **comparative**, not absolute. Universe, holding period,
ranking rule, and benchmark are identical across signals; what differs is the
signal value used to rank stocks at each month-end. Statements made in this
write-up about a signal's performance are statements about its rank position
relative to the four others on this universe, on this window, under this
spec — not claims about a deployable trading strategy.

---

## 1. Introduction

Earnings calls are the densest periodic information event in equity markets.
A two-hour call typically contains prepared remarks from management, Q&A
with sell-side analysts, and forward guidance — material that moves prices
on the day and continues to be re-priced through the subsequent quarter as
investors digest the language.

Two long-running questions in empirical asset pricing motivate this project:

- **Does the *language* of corporate disclosure carry cross-sectional return
  predictability beyond what is in the numbers?** Tetlock (2007) and Loughran
  and McDonald (2011) showed that lexicon-based sentiment from disclosure text
  predicts returns at meaningful horizons; subsequent literature has
  consistently replicated that finding.
- **Do modern semantic representations (LLM embeddings) extract more of that
  signal than word-counting approaches?** This is the question the recent
  S&P Global Market Intelligence whitepaper (Ao and Zhao, 2025) tackles at
  scale for the Russell 3000, using a fine-tuned LLM to score transcript
  sentiment. They find LLM-based sentiment roughly doubles the long-short
  return of a Loughran–McDonald lexicon baseline (8.4 % vs 4.2 %) over
  Feb 2010–Dec 2024.

This project answers a much smaller version of the same question. We use a
fixed 100-ticker universe, off-the-shelf OpenAI embeddings (no fine-tuning),
and a single embedding pooling strategy (call-level mean). What we *can*
answer is whether, on this universe, a simple anchor-cosine sentiment score
or a learned ridge model on the raw embedding deltas adds anything over a
lexicon baseline and the standard non-text factors.

The project is explicitly a class deliverable and a methodological exercise.
Survivorship bias, the small universe, the absence of trading costs, and the
short out-of-sample period (described in Section 8) all preclude any claim
about deployable performance. They do not preclude meaningful *relative*
statements across signals run on the same data.

---

## 2. Prior work and where this project sits

The work this project most closely follows is, in order of relevance:

- **S&P Global Market Intelligence — *Familiar signal, new context: The
  evolution of earnings call sentiment analysis from lexicons to LLMs*
  (Ao and Zhao, September 2025).** Builds a fine-tuned LLM (ProntoNLP)
  that scores transcript sentiment and constructs long-short decile-spread
  strategies on Russell 3000 monthly rebalances. Headline result: 8.4 %
  annualised long-short for the LLM-based sentiment signal versus 4.2 %
  for a Loughran–McDonald baseline, Feb 2010–Dec 2024. Pairwise rank
  correlations between their LLM signals and the LM benchmark range
  0.35–0.75; the LLM signal is measuring the same underlying construct,
  but with measurable separation.
- **Ghosal (2026) — *LLM-Driven Investment Models: Evidence from Earnings
  Call Transcripts*.** Uses 2019–2023 transcripts and sentence-transformer
  embeddings to forecast one-month forward returns, builds top/bottom-decile
  long-short portfolios, and benchmarks against TF–IDF. Reports an out-of-
  sample annualised return of 34.1 % (Sharpe 1.48) for the LLM versus 7.2 %
  (Sharpe 0.42) for TF–IDF. The methodology structure of this write-up
  mirrors the section layout of that paper.
- **Loughran and McDonald (2011) — *When is a Liability not a Liability?*** The
  finance-domain word lists that the LM baseline of this project rests on.
- **Tetlock (2007).** Foundational evidence that disclosure language carries
  return-relevant information beyond what the numbers convey.
- **Glasserman and Lin (2023) — *Assessing Look-Ahead Bias in Stock Return
  Predictions Generated by GPT Sentiment Analysis.*** Cautions about the
  embedding model itself having seen test-period data during training.

This project differs from S&P's work in scope (100 tickers vs Russell 3000)
and in technique (off-the-shelf embeddings with a simple pooling rule, vs a
fine-tuned LLM). It differs from Ghosal (2026) primarily in holding a
tighter universe constant and in including non-text benchmarks (momentum,
revisions) so that LLM signals are compared not only against word counts but
against the standard cross-sectional factors that any equity practitioner
would already have access to.

---

## 3. Data

### 3.1 Universe

The investable universe is **100 Nasdaq tickers** drawn from current
Nasdaq-100 membership. The list is fixed across the entire backtest period:
there is no point-in-time index membership adjustment. The transcript-derived
signals (anchor, ridge, LM) cover the 100 names; the price- and
Bloomberg-derived signals (momentum, revisions) cover one additional name
where Bloomberg data is available.

**Survivorship bias.** Names that delisted, were acquired, or fell out of
the index do not appear. Bias should largely cancel in the *relative*
comparison across signals — they all see substantially the same universe —
but symmetry of bias across signals is an assumption, not a verified
property. This is named in the limitations section and is not corrected.

### 3.2 Transcripts

For each of the 100 tickers, full earnings-call transcripts were pulled from
WRDS Capital IQ. The raw schema is component-level (one row per speech act,
with section type ∈ {`Presentation Operator Message`, `Presenter Speech`,
`Question and Answer Operator Message`, `Question`, `Answer`,
`Unknown Question and Answer Message`}) and was rolled up to per-call
`full_text` for the analyses in this project. We use the **complete
transcript** view — operator instructions, safe-harbor disclaimers, prepared
remarks, and Q&A — without removing boilerplate. This is a deliberate choice
to keep the embedding input identical across calls and tickers; a separate
Q&A-only specification is listed as future work.

### 3.3 Embeddings

Each call transcript is chunked to fit OpenAI's embedding API context
window. Chunking uses `tiktoken` token counts and aims for ~6,000-token
chunks, with a hard split for any single component longer than 7,500 tokens.
Each chunk is sent to OpenAI's `text-embedding-3-small` (1,536-D, near
unit-norm). The chunk-level embedding is multiplied by an `n_chars` weight
and the call-level vector is the weighted average across chunks, then
explicitly L2-normalised. Saved as `_data/embeddings_transcripts.parquet`
and `_data/call_vectors.parquet`.

### 3.4 Prices and analyst estimates

Daily prices for the universe and Bloomberg `BEst Net Income` consensus
estimates were extracted from Bloomberg Terminal exports under
`data_manual/`. Prices are converted to a monthly panel (last business day
of month) and to a 21-trading-day forward return aligned to each
earnings-call `event_date` for the ridge model. Analyst revision panels are
constructed at month-end as a 21-business-day percentage change in
consensus BEst Net Income (`rev_30d` column in the unified panel).

### 3.5 Loughran–McDonald master dictionary

`data_manual/lm_master_dictionary.csv` — 86,554 words with seven
finance-specific category flags (Negative, Positive, Uncertainty, Litigious,
Strong_Modal, Weak_Modal, Constraining). Only Positive and Negative are used
for the LM baseline, following standard practice.

### 3.6 Coverage by signal

The five signals do not see identical row counts. The Ridge + PCA model
relies on the embedding pipeline being run end-to-end on all 100 tickers; the
anchor-cosine sentiment uses the same pipeline; LM lexicon needs only the
text; momentum needs only prices; revisions need only Bloomberg estimates.

The starkest asymmetry is **Ridge + PCA's shorter history**: the model is
trained on 2012-2018 calls and only starts emitting predictions in
**February 2019**. Every other signal in the panel extends back to the
mid-2000s. Headline results tables in the write-up therefore show Ridge
with ~86 months while other signals show 150–300+ months. The post-2018
subsample (`metrics_post2018.json`) restricts every strategy to the same
shorter window for apples-to-apples comparison.

The relevant per-strategy month counts are reported in the results notebook
(`_output/99_results.ipynb`, section 3a) and the "common-window" sections of
the notebook restrict to overlapping windows for the chart-level visual
comparisons.

---

## 4. Methodology

The pipeline transforms raw transcripts and prices into five comparable
monthly signal columns, then runs every signal through identical backtest
machinery. The remainder of this section walks through each transformation.

### 4.1 Text preprocessing and chunking

Transcripts are used as-is — no lowercasing, no stopword removal, no
boilerplate stripping. The motivation is that the embedding model was
trained on natural English text; aggressive cleaning would push the input
out of the model's training distribution. Chunking is the only preparation
step:

1. Tokenise the call text with `tiktoken` (the encoder for
   `text-embedding-3-small`).
2. Walk the token sequence in chunks of up to ~6,000 tokens. A new chunk
   starts when the running total exceeds the budget.
3. For any single transcript-component piece exceeding 7,500 tokens, hard-
   split it before chunking.

This guarantees every chunk fits the 8,192-token context window with margin
and preserves component-level boundaries when they fit.

### 4.2 Per-call call vectors and Δ call vectors

For every chunk we get a 1,536-D unit-norm embedding. The per-call vector
is the **`n_chars`-weighted average** of those chunk embeddings,
re-normalised to unit length. The character-weighting (rather than uniform
averaging across chunks) means longer prepared sections naturally dominate
the per-call representation in proportion to how much they say, which is
the same property a human reader would apply.

For sentiment-style signals we work in **changes**, not levels:

$$\Delta v_{i,t} = v_{i,t} - v_{i,t-1}$$

where $v_{i,t}$ is the call vector for ticker $i$ at event date $t$ and
$t-1$ is the immediately preceding earnings call for the same ticker,
regardless of calendar gap. The first call per ticker is dropped — there
is nothing to difference against. This eliminates the company-fixed-effect
direction in embedding space (some industries naturally talk more about
operations, some more about R&D) and isolates the *new* language of the
quarter.

### 4.3 Strategy 1 — Anchor cosine on Δ sentiment

The anchor approach measures the projection of the call vector onto a
hand-written "positive" and "negative" direction in embedding space.

**Anchor phrases** (in `src/score_transcript_sentiment.py`):

| Polarity | Five sentences each |
|---|---|
| Positive | "We had a record-breaking quarter with strong revenue growth." / "We are raising our full-year guidance based on robust demand." / "Our profitability and margins expanded significantly." / "We delivered exceptional results that exceeded expectations." / "Demand for our products remains very strong worldwide." |
| Negative | "We missed expectations and guidance was below consensus." / "We are lowering our outlook due to weak demand and macro headwinds." / "Margins contracted and operating expenses rose materially." / "The quarter was disappointing with revenue declines across segments." / "We are seeing significant softness and customer pullback." |

The positive and negative anchor vectors are the L2-normalised means of the
embeddings of the five sentences in each list. Let $s_{i,t}$ denote the
per-call sentiment-diff score:

$$s_{i,t} = \cos(v_{i,t}, a_{\text{pos}}) - \cos(v_{i,t}, a_{\text{neg}})$$

The signal used for monthly ranking is its quarter-over-quarter change:

$$\Delta s_{i,t} = s_{i,t} - s_{i,t-1}$$

(stored in the panel as column `sig_anchor`).

**Why this method.** Anchor cosine is the cheapest possible way to project
an embedding onto an interpretable direction. It does not require labels,
training, or tuning. Its weakness is also its simplicity: the anchor
sentences themselves carry all of the modelling burden, and there is no
data-driven validation that the chosen anchors point in the most
return-relevant direction. We use it as the "LLM baseline" rather than a
serious contender.

### 4.4 Strategy 2 — Ridge regression with PCA pre-reduction on Δ call vectors

Strategy 2 is the **learned-direction LLM test**: a counterpart to the
hand-anchored Strategy 1, not a replacement for it. Strategies 1 and 2 are
both legitimate, first-class LLM tests with different assumptions. Strategy
1 puts all interpretive burden on ten hand-written anchor sentences;
Strategy 2 lets supervised learning identify return-predictive directions
in the embedding delta. They answer different questions and the comparison
between them is itself part of the project's result.

**Features.** Per call $(i, t)$:
- $\Delta v_{i,t}$ — the 1,536-D call-vector delta (Section 4.2).
- $\tau_{i,t}$ — calendar days since the prior call for the same ticker
  (`days_since_earnings` in the panel), included as a freshness control so
  the model can learn any decay structure.

Total: 1,537 raw features per observation.

**Target.** The realised 21-trading-day forward return from $t$, computed
with `calendar_utils.fwd_ret_bd(prices, event_date, h=21, gap=1)`. The gap
of one business day mirrors the backtest execution rule (Section 4.9):
trades execute at the open of T+1.

**Why ridge.** A plain OLS regression on a 1,537-D feature vector with the
~3,000 training observations available per fold is in the high-dimensional
regime — more knobs than examples. The L2 penalty $\alpha \cdot \|\beta\|^2$
shrinks coefficients toward zero, stabilises estimates, and is the standard
tool in this regime, especially when features are correlated (which
embedding dimensions always are).

**Why PCA.** With the raw 1,537-D feature vector, the cross-validated
$\alpha$ saturates at the maximum of the search grid (1,000) in every fold
and the Sharpe ratio of the resulting long-short is essentially zero
(~0.05). The interpretation is that ridge alone cannot bring 1,537
correlated dimensions under control with this much data; it just shrinks
everything to zero. **PCA(50) pre-reduction** addresses this by
concentrating signal into the top 50 principal components of the training
deltas; ridge then operates in a 51-dimensional space (50 PCs +
`days_since_earnings`) it can actually regularise meaningfully. With PCA,
the Sharpe rises to ~0.56.

`days_since_earnings` is **concatenated *after* the PCA**, not put through
it. The reasoning: it is a single scalar dimension on a different scale
from the embedding directions; including it in the PCA input would either
waste a principal axis on it or wash it out under standardisation. Keeping
it separate preserves it as a first-class freshness feature.

**Estimation procedure.**
1. **Train period:** `event_date` in $[2012\text{-}01\text{-}01,
   2018\text{-}12\text{-}31]$.
2. **Test period:** $[2019\text{-}01\text{-}01,$ today$]$.
3. **Expanding-window refits:** At the start of each calendar year $Y$
   in the test period, refit using all calls with `event_date` $\leq$ end
   of $Y-1$. Use that frozen model to score calls in year $Y$. Seven
   expanding folds over 2019–2025.
4. **Per-fold inner pipeline:** `PCA(n_components=50)` →
   `StandardScaler` → `RidgeCV` with $\alpha \in
   \{0.01, 0.1, 1, 10, 100, 1000\}$ and `TimeSeriesSplit(n_splits=5)`
   inside the training window.
5. **All transforms fit on train only.** PCA loadings, the standard
   scaler's means and variances, and the chosen $\alpha$ are each fit on
   the training slice and applied unchanged to the test slice. There is
   no information leakage from the test period.
6. **Output.** For each call in the test period, a predicted 21-day
   forward return `y_pred`. That `y_pred` becomes `sig_ridge` for the
   month containing `event_date`.

**Alternative that was tried and dropped.** **Partial Least Squares
(PLS)** was tested as a learned-direction alternative to PCA + ridge. With
cross-validation on the training window it produced a mean CV $r^2$ of
~0.004 and an out-of-sample long-short Sharpe of −0.06. The code was
removed; PLS is documented here only because it was tried, not because
it survives in the implementation.

**A note on the saturated $\alpha$.** Even after PCA, `RidgeCV` still
selects $\alpha = 1000$ (the top of the grid) in every fold. This is
expected: PCA already projects onto the high-variance subspace of the
delta vectors, so the ridge penalty's job is primarily to keep the
coefficients on those 50 PCs small. The fact that the chosen $\alpha$
sits at the top of the grid is *not* the same problem as in the raw-1,537
case — there, the high $\alpha$ was driving predictions to zero; here it
is producing the signal that backtests at Sharpe ~0.56.

### 4.5 Strategy 3 — Loughran–McDonald lexicon on Δ net positivity

The classical text-NLP baseline. For each call $(i, t)$ we tokenise the
full transcript text (the same `full_text` view fed to the embedder),
match each token against the LM Positive and Negative lists, and compute:

Let $A_{i,t}$ and $B_{i,t}$ be the per-call counts of LM-Positive and
LM-Negative tokens. The per-call LM net-positivity score is

$$L_{i,t} = \frac{A_{i,t} - B_{i,t}}{A_{i,t} + B_{i,t}}.$$

If the denominator is zero (a call with no LM-listed words) the score is
null and the observation is dropped. As with the anchor signal, the monthly
ranking signal is the call-to-call change

$$\Delta L_{i,t} = L_{i,t} - L_{i,t-1}$$

(stored in the panel as column `sig_lm`).

Only Positive and Negative categories are used. The Uncertainty, Litigious,
Strong_Modal, Weak_Modal, and Constraining lists are not standard
ingredients of a net-positivity score and are excluded by design.

### 4.6 Strategy 4 — Price momentum 12-1

The standard cross-sectional momentum factor. At month-end $m$, for each
ticker $i$:

$$M_{i,m} = \prod_{k=2}^{12} (1 + r_{i, m-k}) - 1$$

The cumulative simple monthly total return from 12 months ago to one month
ago. The most recent month is skipped to side-step the well-known
one-month reversal effect.

Computed in `build_momentum_monthly.py` and signal column `sig_mom`.

### 4.7 Strategy 5 — Analyst revisions in Δ blended-forward net income

Let $B_{i,m}$ denote the Bloomberg consensus `BEst Net Income` for ticker
$i$ at month-end $m$. At month-end $m$ the signal is the 21-business-day
percentage change:

$$R_{i,m} = \frac{B_{i,m} - B_{i,m-21}}{\left| B_{i,m-21} \right|}$$

The 21-business-day change in Bloomberg's consensus net income estimate,
normalised by absolute value of the prior estimate (the absolute value
handles the rare case where the prior estimate is negative without
flipping the sign of the revision).

**The Bloomberg field is *blended forward*, not FY1.** The
`BEST_NET_INCOME` column pulled from the Bloomberg Terminal export is the
`1BF` ("1-blended-forward") measure, not raw FY1 net income. By Bloomberg's
definition (workbook `Info` sheet), `1BF` is a days-weighted blend of FY1
and FY2:

$$B = \frac{d}{D}\cdot\mathrm{FY1} + \left(1 - \frac{d}{D}\right)\cdot\mathrm{FY2}$$

where $d$ = number of trading days until the next fiscal year-end and $D$
= trading days in a year. Once the FY1 reporting date has passed, the
blend switches to FY2 and FY3 with the same weighting scheme. This means
that on a typical month-end the consensus we see is a smooth ~12-month
forward estimate, not a raw FY1 number that jumps at year-ends. The 21-day
revision picks up genuine analyst updates, not the mechanical FY1→FY2
roll.

Net income is used rather than P/E inversion to keep the signal driven
purely by estimate revisions, not by price moves in the denominator.

Computed in `build_revisions_monthly.py` and signal column `sig_rev`.

### 4.8 Cross-sectional signal mechanics

Three rules apply uniformly across all five signals once the per-source
columns exist.

**Delta computation.** All sentiment-style signals (anchor, ridge, LM) are
computed as the change between consecutive calls for the same ticker, with
no calendar alignment. First call per ticker is dropped. Momentum and
revisions are already changes by construction.

**Monthly carry-forward.** At each month-end $m$, every ticker is assigned
its most recent valid signal value:
- For anchor, ridge, LM: the value from the most recent earnings call
  with `event_date` $\leq m$.
- For momentum and revisions: computed directly from prices / estimates
  as of $m$.

So one Δ-sentiment value can drive up to three rebalances before the next
call refreshes it, and the same is true of Δ-LM and Δ-ridge.

**Staleness filter (robustness only).** Under the main specification the
carry-forward is unconditional. Under the stale-call exclusion
specification, any ticker with `days_since_earnings > 60` calendar days
that month is **dropped** from the cross-section *for the sentiment
strategies only* (anchor, ridge, LM). Momentum and revisions are unaffected
because they refresh independently every month — applying the filter to
them would shrink their universe for no methodological reason. Under the
stale-excl spec, the momentum and revisions rows are therefore identical
to their main-spec rows by construction; the comparison is meaningful only
for the three sentiment signals.

### 4.9 Portfolio construction and holding period

Identical machinery across all five signals (`src/backtest.py`,
`run_backtest`):

| Parameter | Value |
|---|---|
| Universe | 100 Nasdaq tickers (Section 3.1) |
| Rebalance frequency | Monthly (last business day of month) |
| Ranking | Cross-sectional rank of the signal across all tickers with non-null signal that month |
| Long leg | Top quintile (top 20 % of valid signal values that month, `round(n*0.20)` names) |
| Short leg | Bottom quintile (bottom 20 % of valid signal values that month) |
| Weighting | Equal-weight within each leg (1/n_leg per stock; n_leg ≈ 16–20 names) |
| Trade execution | Open of T+1 where T = month-end signal date |
| Holding period | 21 trading days |
| Transaction costs | None |
| Minimum observations / month | 40 valid signal values required to form a portfolio |

The monthly long-short return is

$$R_m^{LS} = \frac{1}{n^L_m}\sum_{i \in \text{Long}_m} r_{i,m} - \frac{1}{n^S_m}\sum_{j \in \text{Short}_m} r_{j,m}$$

where $r_{i,m}$ is the simple total return of stock $i$ over the 21-day
holding period and $n^L_m = n^S_m = \text{round}(n_m \cdot 0.20)$ is the
quintile-leg size at month $m$.

**Why no transaction costs.** The project is a comparative test. All five
signals rebalance on the same monthly cadence over the same universe, and
turnover is moderate by design. Adding a uniform cost to all strategies
would shift every Sharpe by approximately the same constant and not change
the ranking. We name the absence of costs as a limitation but do not
attempt to model them.

**Why `MIN_OBS_PER_MONTH = 40`.** Sorting top-quintile / bottom-quintile
out of a cross-section smaller than ~40 starts to produce overlap between
the two legs and degenerate rankings. Forty is the floor below which we
report "no portfolio formed" for that month rather than carry through a
malformed rebalance.

### 4.10 Joint regression — Fama-MacBeth with Newey-West HAC

To answer "after controlling for the other signals, does the LLM signal
contribute marginal information?" we run a cross-sectional Fama-MacBeth
regression. Let $z^k_{i,m}$ denote the monthly cross-sectional z-score of
signal $k$ for ticker $i$ at month-end $m$, where $k$ ranges over the five
signal columns $\{a, r, l, p, v\}$ — anchor, ridge, LM, price momentum,
analyst revisions. At each rebalance month $m$ we run

$$r_{i,m} = \alpha_m + \beta^m_a z^a_{i,m} + \beta^m_r z^r_{i,m} + \beta^m_l z^l_{i,m} + \beta^m_p z^p_{i,m} + \beta^m_v z^v_{i,m} + \varepsilon_{i,m}$$

z-scoring within each month makes the $\beta$ magnitudes comparable across
signals. The time-series average of each $\beta^m_k$ is reported with
**Newey-West standard errors at lag 6** to handle serial correlation in
the coefficient series. Implementation in `src/joint_regression.py`.

A signal is dropped before the joint regression if its average number of
valid cross-sectional observations per month is below 20. This is a
defensive guard: if a sentiment signal is only present for a handful of
tickers in some month (typically because the transcript pipeline lags),
including that thin column would drive a `dropna` to wipe out almost all
rows of the joint panel, collapsing the regression onto whichever tickers
*do* have the thin signal.

---

## 5. Performance metrics

Three families of metrics are reported for each strategy. None of the three
fully describes a signal on its own.

### 5.1 Return / risk metrics — from the portfolio time series

| Metric | Definition |
|---|---|
| Annualised return | Mean monthly long-short return $\times 12$ |
| Annualised volatility | Std dev of monthly long-short returns $\times \sqrt{12}$ |
| Sharpe ratio | Annualised return ÷ annualised volatility; zero risk-free rate |
| Maximum drawdown | Worst peak-to-trough on the cumulative long-short series |
| Hit rate | Share of months with positive long-short return; 50 % is coin-flip |
| Information ratio vs benchmark | $\text{mean}(r_{LS} - r_{\text{bench}}) \cdot 12 \big/ \text{std}(r_{LS} - r_{\text{bench}}) \cdot \sqrt{12}$ |

The benchmark for the information ratio is the equal-weighted return of
the universe with non-null signal that month, not the cap-weighted
Nasdaq-100 — internal consistency with how the legs are weighted.

### 5.2 Predictive-power metrics — from the cross-sectional Spearman IC

Information Coefficient (IC) at month $m$ is the **Spearman rank
correlation** between the signal value and the realised 21-day forward
return across the cross-section that month. This is a *single number per
month* describing whether the full ordering of stocks lined up with the
return ordering. Strategies are then summarised over the IC time series:

| Metric | Definition |
|---|---|
| `spearman_ic_mean` | Average monthly Spearman rank IC |
| `spearman_ic_std` | Standard deviation of the monthly IC |
| `ic_ir` | mean / std (Sharpe-analog at the ranking level) |
| `pct_months_positive` | Share of months with IC > 0 |
| `n_months` | Months with a valid IC observation |

### 5.3 Why both portfolio metrics AND IC

Portfolio return depends only on the **tails** of the ranking — the
top-quintile and bottom-quintile legs. A signal that gets one or two big
winners right can post a positive long-short return even if the middle
stocks are randomly ordered. IC measures the **full ordering**. The two
views answer different questions:

- Positive IC, near-zero portfolio return → signal orders well but the
  tails are dominated by noise specific to those names.
- Positive portfolio return, near-zero IC → signal got lucky on a small
  number of names.
- Positive on both → the signal genuinely orders stocks.

The third case is the only one that justifies a methodology claim.

---

## 6. Robustness checks

Three specifications are reported side-by-side in the results notebook:

1. **Main specification** — every rebalance uses every available
   carry-forward signal value.
2. **Stale-call exclusion** — sentiment strategies drop tickers with
   `days_since_earnings > 60` for that month. Tests whether the signal is
   event-driven (decays quickly post-call) or persists across the carry-
   forward window. Momentum and revisions are unaffected (Section 4.8).
3. **Post-2018 subsample** — restrict the entire main-spec backtest to
   `event_date >= 2018-01-01`. Tests whether signal performance is
   driven by the early part of the sample or persists into the recent
   regime, mirroring the S&P observation that the lexicon baseline has
   weakened over time.

The notebook additionally constructs two **common-window** views for the
cumulative-return charts (sections 9 and 10 of the notebook). These are
not separate robustness specifications; they are an honest visualisation
fix for the fact that strategies have different start dates. A
cumulative-return chart in which one line starts in 2000 and another in
2019 has non-comparable endpoints; restricting to common windows
(2008-onward excluding Ridge+PCA; Ridge's first month onward including
all five) puts the chart on a like-for-like footing.

---

## 7. Implementation pipeline

The repository is organised as a `doit`-driven pipeline. Every numbered
phase below corresponds to a `doit` task group; running `doit` from the
project root rebuilds the entire chain from raw inputs to the executed
results notebook.

| Phase | Module | Output |
|---|---|---|
| 0. Config | `dodo.py task_config` | Ensures `_data/`, `_output/` exist |
| 1. Pull manual data | `pull_manual_companies.py`, `pull_bbg_earning_data.py` | Bloomberg-derived prices and estimates in `data_manual/` and `_data/` |
| 2. Extract transcripts | `extract_min10y_transcripts.py`, `clean_min10y_transcripts.py`, `freeze_cleaned_dataset.py` | Per-ticker cleaned transcripts under `_data/transcripts/` |
| 3. Embed transcripts | `embed_transcripts.py` | `_data/embeddings_transcripts.parquet` |
| 4. Score anchor sentiment | `score_transcript_sentiment.py`, `build_sentiment_features.py` | `_data/sentiment_transcripts.parquet`, `_data/features_sentiment_monthly.parquet` |
| 5. Score LM lexicon | `score_transcript_lm.py` | `_data/lm_transcripts.parquet` |
| 6. Build call & delta vectors | `build_call_vectors.py`, `build_delta_vectors.py` | `_data/call_vectors.parquet`, `_data/delta_vectors.parquet` |
| 7. Build returns / momentum / revisions panels | `build_returns_monthly.py`, `build_momentum_monthly.py`, `build_revisions_monthly.py` | `_data/returns_monthly.parquet`, `_data/momentum_monthly.parquet`, `_data/revisions_monthly.parquet` |
| 8. Train ridge model | `train_ridge.py` | `_data/ridge_predictions.parquet` |
| 9. Build unified signal panel | `build_signal_panel.py` | `_data/signal_panel_monthly.parquet` |
| 10. Run backtests | `run_backtests.py` (calls `backtest.py`) | `_data/results_*.parquet`, `_data/metrics_*.json`, `_data/ic_*.{json,parquet}` |
| 11. Joint regression | `joint_regression.py` | `_data/fm_results.json` |
| 12. Render results notebook | `99_results.ipynb.py` via `jupytext` + `nbconvert` | `_output/99_results.ipynb`, `_output/99_results.html`, `_output/99_*.png` |
| 13. Pytest | `task_run_pytest` | `_output/pytest_results.xml` |

The methodology is encoded in `src/`; the notebook is a **thin reader**
that loads pre-computed artifacts and renders tables / charts.

---

## 8. Limitations

### 8.1 Universe and bias

- **Survivorship bias.** 100 tickers drawn from current Nasdaq-100
  membership. Names that delisted, were acquired, or fell out of the index
  do not appear. Bias should largely (but not perfectly) cancel in
  *relative* comparison across signals.
- **Small universe.** 100 tickers and quintile legs of ~16–20 names limit
  statistical power and concentrate exposure. A claim like "Sharpe of
  strategy A is twice strategy B's" is much weaker on 100 names than on
  the Russell 3000.
- **Bias may not be symmetric across signals.** The "bias cancels in
  comparison" argument requires that the universe-filtering removes
  comparable information from each signal. Differential effects are
  possible and untested.

### 8.2 Modelling assumptions

- **No transaction costs.** Defensible as a comparative test (Section 4.9),
  not defensible as a deployable-strategy claim.
- **Single embedding model.** Only `text-embedding-3-small` (OpenAI) is
  tested. Alternative embedders (FinBERT, larger OpenAI models, sentence-
  transformers fine-tuned on financial text) are not compared.
- **Anchor sentences unvalidated.** Different anchors yield different
  scores. No sensitivity sweep on the choice of the five-on-five anchor set.
- **Embedding-model look-ahead.** Glasserman and Lin (2023) point out that
  the embedding model itself has seen text from the test period during its
  pre-training. We do not correct for this; the embedding-as-a-feature
  formulation is more robust to it than a direct LLM-generates-forecast
  setup, but the bias is still present.

### 8.3 Sample size

- **Short out-of-sample period.** The ridge model's test period
  (2019-onward) covers ~7 years. Statistical power scales with $\sqrt{T}$.
- **Unequal n_months across strategies.** Some signals have a 20+-year
  history on this universe (momentum, revisions); others (ridge) have
  ~7 years. We address the cumulative-return chart asymmetry with the
  common-window sections of the notebook, but the headline summary table
  remains intentionally on each strategy's full history.

### 8.4 Pipeline scope

- **Full transcript only.** We do not split prepared remarks from Q&A.
- **No structured extraction.** No event tagging, no aspect / theme /
  polarity / importance four-tuples. The S&P-style methodology is the
  obvious extension and is out of scope here.

---

## 9. Open methodology questions and future work

1. **Q&A vs prepared remarks.** Component-level transcript data supports a
   Q&A-only specification; Q&A typically carries more candid information
   than scripted prepared remarks. Doubles the embedding work.
2. **Sentence-level vs call-level pooling.** Currently we pool to a single
   call vector via $n_{\text{chars}}$-weighting. Alternatives: max-pooling
   over chunk embeddings, attention-weighted pooling, or scoring at the
   sentence level and averaging.
3. **Anchor sensitivity.** A grid of alternative anchor sets would tell
   us how much of the Strategy 1 result depends on the specific phrasing
   chosen.
4. **Alternative embedding models.** FinBERT, OpenAI
   `text-embedding-3-large`, sentence-transformers fine-tuned on
   financial text.
5. **Longer test period.** Re-pull transcripts pre-2012 to extend the
   ridge training window backwards, which would increase the
   out-of-sample test period proportionally.

---

## 10. Appendix

### A. Hyperparameters and choices made

| Choice | Value | Where set | Rationale |
|---|---|---|---|
| Embedding model | `text-embedding-3-small` (OpenAI, 1,536-D) | `embed_transcripts.py` | Cheapest unit-norm embedder; cost-vs-quality trade-off |
| Chunk size (tokens) | ~6,000 | `embed_transcripts.py` | Stays under 8,192 context with margin |
| Hard split threshold | 7,500 tokens | `embed_transcripts.py` | Largest single piece allowed without sub-splitting |
| Anchor sentences | 5 positive + 5 negative | `score_transcript_sentiment.py` | Symmetry; hand-written |
| Anchor pooling | Mean of L2-normalised sentence embeddings, then renormalise | `score_transcript_sentiment.py` | Standard |
| PCA components | 50 | `train_ridge.py` (`PCA_COMPONENTS`) | Empirical: low enough for ridge to regularise meaningfully |
| Ridge $\alpha$ grid | $\{0.01, 0.1, 1, 10, 100, 1000\}$ | `train_ridge.py` (`ALPHA_GRID`) | Log-spaced; covers the regime where the kept variance can be shrunk |
| Inner CV folds | 5, `TimeSeriesSplit` | `train_ridge.py` (`CV_SPLITS`) | Standard for ridge $\alpha$ selection in a time series |
| Ridge train window | 2012-01-01 → 2018-12-31 | `train_ridge.py` | Earliest reliable transcript / Bloomberg overlap |
| Ridge test window | 2019-01-01 → today | `train_ridge.py` | Expanding-window refits per calendar year |
| Holding period | 21 trading days | `calendar_utils.py` (`HOLDING_BDAYS`) | One monthly rebalance |
| Execution gap | 1 business day after signal | `calendar_utils.py` (`EXEC_GAP_BDAYS`) | Trade at open of T+1 |
| Quintile legs | Top / bottom 20 % of valid signals (~16–20 names per leg) | `backtest.py` | Standard quintile cut |
| Min obs per month | 40 | `backtest.py` (`MIN_OBS_PER_MONTH`) | Floor below which legs overlap |
| Stale filter (robustness only) | `days_since_earnings > 60` calendar days | `run_backtests.py` | Drops cells but keeps the cross-section comparable |
| FM Newey-West lag | 6 | `joint_regression.py` | Standard for monthly cross-sectional with autocorrelated $\beta$ |
| FM signal inclusion floor | Avg 20 obs / month | `joint_regression.py` | Guard against thin signals collapsing the joint panel |

### B. Net-positivity formula (LM lexicon, Loughran–McDonald)

For each call $(i, t)$, let $A_{i,t}$ be the count of words in the LM
Positive list and $B_{i,t}$ the count in the LM Negative list, scanning
the full transcript text. The per-call score is

$$L_{i,t} = \begin{cases} \dfrac{A_{i,t} - B_{i,t}}{A_{i,t} + B_{i,t}} & \text{if } A_{i,t} + B_{i,t} > 0 \\[2pt] \text{null} & \text{if } A_{i,t} + B_{i,t} = 0 \end{cases}$$

and the ranking signal is the call-to-call change

$$\Delta L_{i,t} = L_{i,t} - L_{i,t-1}$$

where $t-1$ is the previous call for the same ticker.

### C. Output artefacts referenced in the final write-up

| Artefact | Path | Source |
|---|---|---|
| Return / risk table — full history | section 3a of the notebook | `metrics_main.json` |
| Spearman IC table | section 3b of the notebook | `ic_summary.json` |
| Hit-rate bar chart | section 4 of the notebook | `_output/99_hit_rates_main.png` |
| Rolling 12-month IC chart | section 5 of the notebook | `_output/99_rolling_ic.png` |
| Stale-call robustness table | section 6 of the notebook | `metrics_stale_excl.json` |
| Post-2018 robustness table | section 7 of the notebook | `metrics_post2018.json` |
| Fama-MacBeth joint regression | section 8 of the notebook | `fm_results.json` |
| Cumulative return — period 1 (2008→, excl. Ridge) | section 9 of the notebook | `_output/99_cum_returns_period1_2008.png` |
| Drawdown — period 1 (2008→) | section 9 of the notebook | `_output/99_drawdown_period1_2008.png` |
| Cumulative return — period 2 (Ridge start →) | section 10 of the notebook | `_output/99_cum_returns_period2_ridge.png` |
| Drawdown — period 2 (Ridge start →) | section 10 of the notebook | `_output/99_drawdown_period2_ridge.png` |

### D. Files removed from the implementation along the way

- **PLS regression on Δ call vectors** — tested, mean CV $r^2$ ~0.004,
  long-short Sharpe −0.06. Code removed before merge.
- **10-Q SEC pipeline** — removed. The original project scaffold pointed at
  10-Q text; the agreed scope is earnings-call transcripts only.
- **Chronos forecasting** — earlier exploratory code removed entirely
  (not just archived).
