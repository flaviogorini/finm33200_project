# Project Scope: LLM Embeddings vs Traditional Signals on Earnings Call Transcripts

## 1. Goal

Test whether LLM-derived sentiment signals extracted from earnings call transcripts add value over traditional (lexicon-based) sentiment and standard equity factors (price momentum, analyst revisions) when used to rank stocks cross-sectionally for a long-short equity strategy.

**The goal is NOT to build a deployable trading strategy.** It is to provide a clean comparative test of methods, all run through identical backtest machinery, so that performance differences are attributable to signal quality rather than backtest mechanics.

## 2. Universe

- **Composition:** 100 Nasdaq tickers, hand-selected with a minimum-10-year history filter.
- **Fixed list:** The same 100 tickers are used throughout the entire backtest period. No point-in-time index membership.
- **Acknowledged bias:** This universe has survivorship bias by construction (tickers that delisted, were acquired, or fell out of major indices are excluded). Because all five strategies are evaluated on the same biased universe, the bias largely (though not perfectly) cancels in the *comparison* between strategies. This is explicitly stated as a limitation; we are not making absolute return claims, only relative ones.

## 3. Data Inventory

**Already on disk** (no further pulls required for core pipeline):
- `_data/transcripts/raw/nasdaq100_min10y_*` — full transcript text for all 100 tickers, in component-level (one row per speech act), call-level, and JSONL formats.
- `_data/embeddings_transcripts.parquet` — one row per chunk: `transcript_id, ticker, event_date, chunk_idx, n_chars, embedding` (1,536-D vector from OpenAI `text-embedding-3-small`).
- `_data/sentiment_transcripts.parquet` — one row per call: `transcript_id, ticker, event_date, sentiment_pos, sentiment_neg, sentiment_diff` (anchor-cosine scores).
- `_data/features_sentiment_monthly.parquet` — monthly panel with carried-forward sentiment, `sentiment_diff_qoq`, `days_since_earnings`.
- `data_manual/lm_master_dictionary.csv` — Loughran-McDonald financial sentiment word lists.

**Bloomberg manual extractions available:**
- Analyst forecast data: Best P/E, Best Net Sales, Best Net Income, Best EBITDA — historical revisions.
- (Macro factors VIX, rates, CDS — extracted but **out of scope** for this project; see Section 13.)

**Still to construct:**
- Per-call **call vector** = `n_chars`-weighted average of chunk embeddings (already used in anchor-cosine pipeline; needs to be saved as a per-call artifact for the learned regression).
- Per-call **delta vector** = current call vector minus same ticker's previous call vector.
- Per-call **LM lexicon score**.
- Monthly **price momentum 12-1** values.
- Monthly **analyst revision** values.

## 4. The Five Strategies

All five strategies feed into identical backtest machinery (Section 6). What differs is only the signal value used to rank stocks at each month-end.

### 4.1 Strategy 1 — Anchor Cosine on Δ Sentiment (existing)

This is the signal already coded in `score_transcript_sentiment.py`.

- For each call $i$ at date $t$: `sentiment_diff_{i,t}` = cosine(call_vector, pos_anchor) − cosine(call_vector, neg_anchor).
- **Signal value used in ranking:** `Δsentiment_{i,t} = sentiment_diff_{i,t} − sentiment_diff_{i,t-1}` where $t-1$ is the previous call for the same ticker, regardless of calendar gap.
- This is the existing `sentiment_diff_qoq` column.
- **First call per ticker:** dropped (no prior to difference against).

### 4.2 Strategy 2 — Learned Ridge Regression on Δ Call Vector

**Why this method:** Anchor cosine measures projection onto a hand-written direction (10 anchor sentences). Learned regression instead lets the *data* tell us which directions in embedding space predict returns. This is the central LLM-method test of the project.

**Why ridge specifically:** A standard linear regression on a 1,536-dimensional feature vector with ~5,000–8,000 training observations would be statistically rich enough to overfit — the model has more knobs than it has examples. Ridge regression adds an L2 penalty $\alpha \cdot \|\beta\|^2$ to the loss function, which shrinks coefficients toward zero. This stabilizes estimates, reduces variance, and is the standard tool when features outnumber observations or when features are correlated (which embeddings always are). The penalty strength $\alpha$ is the one hyperparameter; we tune it within the training window only.

**Features:**
- `delta_vector_{i,t}` (1,536-D): current call vector minus previous call vector for same ticker.
- `days_since_earnings_{i,t}` (scalar): days since previous call for same ticker. Included as a freshness control so the model can learn any decay structure.

Total: 1,537 features per observation.

**Target:** 21-trading-day forward return from call date $t$.

**Estimation procedure:**
1. **Train/test split:** Time-based. Train period: 2010 → (cutoff date, TBD; see Section 12). Test period: cutoff → most recent data.
2. **Expanding-window refits:** At the start of each calendar year $Y$ in the test period, refit the model using all calls with event_date ≤ end of year $Y-1$. Use that frozen model to predict returns for all calls in year $Y$.
3. **Hyperparameter selection:** Within each training window, select $\alpha$ via 5-fold time-series cross-validation. Search grid: $\alpha \in \{0.01, 0.1, 1, 10, 100, 1000\}$ (log-spaced).
4. **Standardization:** Standardize features within each training window. Apply the same scaler to the test period — do not refit the scaler on test data.
5. **Output:** For each call in the test period, a predicted 21-day forward return.

**Signal value used in ranking:** the predicted return from the regression. Higher predicted return → higher rank.

### 4.3 Strategy 3 — Loughran-McDonald Lexicon on Δ Net Positivity

**Per-call LM score:**
- Take the full transcript text for call $(i, t)$ as one bag of words.
- $\text{pos}_{i,t}$ = count of words matching the LM **Positive** word list.
- $\text{neg}_{i,t}$ = count of words matching the LM **Negative** word list.
- $\text{LM}_{i,t} = \dfrac{\text{pos}_{i,t} - \text{neg}_{i,t}}{\text{pos}_{i,t} + \text{neg}_{i,t}}$.
- If $\text{pos}_{i,t} + \text{neg}_{i,t} = 0$, the score is null (drop observation).

**Signal value used in ranking:** $\Delta\text{LM}_{i,t} = \text{LM}_{i,t} - \text{LM}_{i,t-1}$, same ticker, previous call.

**Other LM word lists** (Uncertainty, Litigious, StrongModal, WeakModal) are **not** used. Standard sentiment construction uses only Positive and Negative.

### 4.4 Strategy 4 — Price Momentum (12-1)

**Definition:** At month-end $m$, for each ticker:

$$\text{Mom}_{i,m} = \prod_{k=2}^{12} (1 + r_{i, m-k}) - 1$$

where $r_{i, m-k}$ is the simple monthly total return for ticker $i$ in month $m-k$.

This is the standard 12-month-minus-1-month momentum factor: cumulative return from 12 months ago to 1 month ago, skipping the most recent month to avoid short-term reversal effects.

**Signal value used in ranking:** $\text{Mom}_{i,m}$.

### 4.5 Strategy 5 — Analyst Revisions (Δ Best Net Income)

**Definition:** At month-end $m$, for each ticker:

$$\text{Rev}_{i,m} = \frac{\text{BEst\_NI}_{i,m} - \text{BEst\_NI}_{i,m-30}}{|\text{BEst\_NI}_{i,m-30}|}$$

30-day change in consensus FY1 net income estimate, normalized by absolute value of the prior estimate. The absolute value in the denominator handles the rare case of a negative prior estimate without flipping the sign of the revision.

**Signal value used in ranking:** $\text{Rev}_{i,m}$.

Net income is used (not P/E inversion) because (a) it is directly available in the Bloomberg extraction, and (b) it isolates pure estimate revisions from price-driven P/E movement.

## 5. Signal Construction Details (Apply to All Strategies)

### 5.1 Delta computation

For all sentiment-based signals (Strategies 1, 2, 3):
$$\Delta x_{i,t} = x_{i,t} - x_{i,t-1}$$
where $t-1$ is the *immediately preceding* earnings call for the same ticker, regardless of calendar gap. No alignment to fiscal-quarter cadence. First call per ticker is dropped.

### 5.2 Monthly carry-forward

At each month-end date $m$, every ticker has a signal value:
- For Strategies 1, 2, 3: the signal from the most recent earnings call where event_date ≤ $m$.
- For Strategies 4, 5: computed directly from price/estimate data as of $m$.

The carry-forward logic in `build_sentiment_features.py` already handles this for Strategy 1. The same logic must be applied to Strategies 2 and 3.

### 5.3 Staleness handling

**Headline approach:** None. Flat carry-forward (current behavior). For Strategy 2, `days_since_earnings` enters as a regression feature, so the model can learn any decay relationship if present.

**Robustness check (Section 8):** Re-run all backtests dropping observations where days_since_earnings > 60.

## 6. Backtest Mechanics (Identical Across All Strategies)

| Parameter                                | Value                                                                                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Universe                                 | 100 Nasdaq tickers (Section 2)                                                                                                                   |
| Rebalance frequency                      | Monthly (last trading day of month)                                                                                                              |
| Ranking                                  | Cross-sectional rank of signal value across all 100 tickers at month-end                                                                         |
| Long leg                                 | Top quintile = 20 tickers (highest signal)                                                                                                       |
| Short leg                                | Bottom quintile = 20 tickers (lowest signal)                                                                                                     |
| Weighting within each leg                | Equal-weight (5% per stock)                                                                                                                      |
| Trade execution                          | Enter at open of T+1, where T = month-end signal date                                                                                            |
| Holding period                           | 21 trading days (until next monthly rebalance)                                                                                                   |
| Transaction costs                        | **None.** Strategy is comparative, turnover is moderate (100 stocks, monthly rebalance), and including costs would not change relative rankings. |
| Tickers with missing signal at month-end | Excluded from that month's ranking                                                                                                               |

**Portfolio return at month $m$:**
$$R_m^{LS} = \frac{1}{20}\sum_{i \in \text{Long}_m} r_{i,m} - \frac{1}{20}\sum_{j \in \text{Short}_m} r_{j,m}$$

where $r_{i,m}$ is the simple total return of stock $i$ over the 21-day holding period.

## 7. Reporting Metrics

For each of the five strategies, report:

**Return / risk metrics:**
- Annualized return (mean monthly return × 12)
- Annualized volatility (std monthly returns × √12)
- Sharpe ratio (assuming zero risk-free rate; this is a long-short strategy)
- Maximum drawdown
- Information ratio vs equal-weighted Nasdaq-100 benchmark
- Hit rate (% of months with positive long-short return)

**Predictive-power metrics (computed separately, not from portfolio returns):**
- **Mean Information Coefficient (IC):** Average of monthly cross-sectional Spearman rank correlations between signal value and realized 21-day forward return.
- **IC standard deviation**
- **IC Information Ratio = mean IC / std IC** (analog of Sharpe at the cross-sectional ranking level)
- **% of months with positive IC**

**Why both:** Portfolio return depends only on the tails of the ranking (top 20 / bottom 20). IC measures the *full ordering*. A signal can have positive portfolio return from getting one winner right while having near-zero IC if the middle 60 are randomly ordered. High IC + positive return = the signal genuinely orders stocks. This dual metric is standard in cross-sectional asset pricing.

**Robustness alternative:** Report Pearson rank correlation as a footnote alongside Spearman. Spearman is the headline because returns are heavy-tailed; Pearson is reported as a sanity check that outliers aren't driving results.

## 8. Robustness Checks

Run all five backtests and the joint regression under the following alternative specifications and report side-by-side with the main results:

1. **Stale-call exclusion:** Drop monthly observations where days_since_earnings > 60 (for the three sentiment strategies). Tests whether the signal is event-driven and decays quickly.
2. **Time-period slice:** Report results separately for full sample vs post-2018 sample. Mirrors S&P's "recent performance" claim — the lexicon signal has weakened over time as algorithmic listening has become commoditized; does our LLM signal show similar or different behavior?

## 9. Joint Regression (Cross-Sectional Multi-Factor Test)

In addition to the five univariate long-short strategies, run **one** monthly Fama-MacBeth-style cross-sectional regression to test whether the LLM signals add information *beyond* the other factors.

**Specification at each month $m$:**
$$r_{i,m} = \alpha_m + \beta_1^m \cdot \Delta\text{LLM}_i + \beta_2^m \cdot \Delta\text{LLM-reg}_i + \beta_3^m \cdot \Delta\text{LM}_i + \beta_4^m \cdot \text{Mom}_i + \beta_5^m \cdot \text{Rev}_i + \varepsilon_{i,m}$$

All right-hand-side variables standardized (z-score) within each month so coefficients are comparable in magnitude.

**Reporting:** Time-series average of each $\beta^m$ across months, Newey-West standard errors, t-statistics. This answers: "After controlling for the other signals, does the LLM-regression signal contribute marginal information?"

## 10. Success Criteria (Pre-Committed)

These are committed *before* seeing results to prevent goalpost-shifting.

**Primary claim of the project:** The learned-regression LLM signal (Strategy 2) produces a positive long-short return that is statistically distinguishable from zero (t-stat > 2 on monthly returns) AND from the lexicon baseline (paired t-test on monthly returns, p < 0.05).

**Secondary claim:** The anchor-cosine LLM signal (Strategy 1) also beats the lexicon baseline, supporting the broader "embedding representations add value over word counts" hypothesis.

**Tertiary claim:** At least one LLM signal retains a significant coefficient in the joint regression after controlling for momentum and analyst revisions, indicating it carries information beyond established factors.

**Null finding is acceptable and informative.** If neither LLM signal beats lexicon, the project's finding is: "LLM embeddings do not add cross-sectional information beyond word counts in the Nasdaq-100 mega-cap universe." This is a legitimate and publishable result for a class project; the write-up framing must allow for this outcome.

## 11. What the Final Deliverable Contains

(Deliverable shape — paper / presentation / notebook — to be decided later. These are the analytical artifacts that must exist regardless.)

1. **Main results table:** 5 strategies × all return/risk + IC metrics from Section 7.
2. **Robustness tables:** Same metrics for (a) stale-exclusion specification, (b) post-2018 subsample.
3. **Joint regression table:** Coefficients, Newey-West t-stats, R² from Section 9.
4. **Cumulative return chart:** All 5 strategies' equity curves on one plot.
5. **IC time-series chart:** Rolling 12-month mean IC for each of the 5 strategies on one plot. Shows signal decay over time, important for the post-2018 narrative.

## 12. Open Questions (To Be Resolved Before Implementation)

1. **Train/test cutoff date.** Depends on data availability across all 100 tickers in the database. Suggested approach: identify the earliest date by which ≥90% of tickers have at least one earnings call on record, use that as the start of training. Then split training/test roughly 60/40 by time. To be locked in once the database is inspected.
2. **Exact monthly rebalance day.** Standard is last trading day of month. Confirm this matches the date convention used in `features_sentiment_monthly.parquet`.

## 13. Explicitly Out of Scope

These were considered and explicitly excluded to keep the project bounded:

- **Q&A vs prepared remarks split.** Component-level transcript data supports it, but doubles the embedding/scoring work. Listed as future work; not implemented in this project.
- **LLM structured extraction (S&P-style 4-tuple tagging).** Too expensive in time and money for the full transcript corpus.
- **Macro factors (VIX, rates, CDS) as backtest signals.** These are time-series, not cross-sectional, and cannot drive a stock ranking. Possible robustness use as regime conditioners is also out of scope.
- **Transaction costs.** Comparative study; turnover is moderate; relative rankings would not change.
- **Optimization beyond ridge regression.** No random forests, gradient boosting, neural nets. Ridge is the headline; the experiment is about signal content, not model complexity.
- **Multiple forecast horizons.** Only 21-day forward return is tested.
- **Level-version learned regression.** Only the change/delta version is run. Justification: with 100 tickers, level regression is exposed to company-specific dominance of the learned directions.
- **Bias correction beyond acknowledgment.** No Heckman correction, no inverse-probability weighting. Survivorship bias is named in limitations and assumed comparable across strategies.

## 14. Limitations (To Be Stated in Write-Up)

- **Survivorship bias** in the universe construction.
- **Small universe** (100 tickers, 20-name quintile legs) limits statistical power and concentration; results may not generalize to broader universes.
- **Bias may not be symmetric across strategies.** The "bias cancels in comparison" argument requires that all signals lose comparable information from the universe filtering. Differential effects are possible and not tested.
- **No live-data simulation.** Embeddings are computed with current OpenAI model on historical text. The embedding model itself was trained on data including some of the test period — potential look-ahead/distraction bias as discussed in Glasserman & Lin (2023). Not corrected.
- **Single embedding model.** Only `text-embedding-3-small` is tested; alternative embedding models (FinBERT, OpenAI `text-embedding-3-large`, etc.) are not compared.
- **Anchor sentences for Strategy 1 are not validated.** Different anchors → different scores. No sensitivity analysis on anchor choice.
- **Out-of-sample period length.** Depends on cutoff (Section 12). Statistical power scales with √T.

## 15. Implementation Checklist

(For coordination across team. Each item is independently verifiable.)

- [ ] Save per-call call vectors to a parquet file (one row per call, 1,536-D vector).
- [ ] Compute and save per-call delta vectors.
- [ ] Compute and save per-call LM scores and Δ LM scores.
- [ ] Pull monthly total returns for all 100 tickers from start of data to present.
- [ ] Construct monthly momentum 12-1 panel.
- [ ] Construct monthly analyst revision panel from Bloomberg Best NI data.
- [ ] Build unified monthly signal panel: one row per (date, ticker), one column per strategy.
- [ ] Implement backtest engine (Section 6) — single function, takes a signal column name, returns portfolio time series and metrics.
- [ ] Implement ridge regression training loop with expanding-window refits and within-fold α CV.
- [ ] Implement IC computation function.
- [ ] Implement Fama-MacBeth joint regression with Newey-West standard errors.
- [ ] Lock train/test cutoff (Section 12, Question 1).
- [ ] Run all 5 backtests, main specification.
- [ ] Run all 5 backtests, stale-exclusion robustness.
- [ ] Run all 5 backtests, post-2018 robustness.
- [ ] Run joint regression.
- [ ] Generate main results table, robustness tables, joint regression table.
- [ ] Generate cumulative return chart and rolling IC chart.
- [ ] Write up limitations section per Section 14.
