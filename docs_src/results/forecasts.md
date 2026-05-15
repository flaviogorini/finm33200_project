# Forecast results

Two forecasts feed the decision digest. Both are evaluated honestly here — no
target threshold is hardcoded; we report what the data shows.

## Returns forecast — V0a → V5 ladder

**Code:** [src/predict_returns_ckx.py](../../src/predict_returns_ckx.py)
**Outputs:** `_output/ckx_metrics.json`, `_output/ckx_predictions.parquet`, `_output/ckx_portfolio.parquet`

The variant ladder tests four nested hypotheses (see [goals.md](../project_overview/goals.md)):

1. **V0b vs V0a** — does anything beat zero?
2. **V1 vs V0b** — does fundamentals + macro add to momentum?
3. **V2/V3 vs V1** — does text (call sentiment, then 10-Q lexicon) add?
4. **V4/V5 vs V3** — does the LLM beat the dictionary, and do they complement each other?

### Headline metrics — rank IC + AUC + portfolio Sharpe

Rank IC (Spearman) and AUC are the primary metrics. Monthly-return R² is
noise-bounded near zero — we report it but do not lead with it. The
return-prediction lesson in FINM 33200 quotes the literature directly:
*"even the right decision still loses money about 45% of the time"* (Fuentes
2026, RLVR for Finance, slide 9). Low headline accuracy is expected; rank
ordering is what survives the noise.

> _The actual numbers will be filled in by `_output/ckx_metrics.json` after
> running `doit predict_returns`. Read them with the variant ladder tab in the
> dashboard or by opening the JSON directly._

For each (target, variant, model) the metrics JSON reports:

- `n` — number of OOS rows
- `auc` — ROC AUC for predicting `fwd_ret > 0`
- `accuracy` — at decision threshold p_up > 0.5
- `oos_r2` — out-of-sample R²; *demoted from headline*
- `ic_spearman` — rank IC

### Portfolio backtest

Long-short tertile, equal-weight, monthly rebalance. V0a falls back to
equal-weight buy-and-hold. **Gross of transaction costs** — see the explicit
caveat in [goals.md](../project_overview/goals.md). The output parquet
`_output/ckx_portfolio.parquet` carries per-variant cumulative returns; the
dashboard's "Portfolio" tab plots them.

## Fundamentals forecast — Chronos-2 vs Consensus vs Naive

**Code:** [src/backtest_chronos2.py](../../src/backtest_chronos2.py)
**Outputs:** `_output/chronos2_backtest.parquet`, `_output/chronos2_backtest_summary.json`

### Setup

- **Universe:** 5 tickers spanning sectors — AAPL, MSFT (mega-cap tech),
  JPM (financials), KO (consumer staples), NVDA (semis).
- **As-of grid:** quarter-ends in 2024 — `2024-03-31`, `2024-06-30`,
  `2024-09-30`, `2024-12-31`. All four are old enough that the 4-quarter
  horizon ends inside 2025-Q4, which has reported by mid-2026.
- **Targets:** revenue and net_income.
- **Horizons:** 1, 2, 3, 4 quarters ahead.
- **Total cells:** 5 × 4 × 2 × 4 = 160.

### Three forecasters under one roof

| Forecaster | Information at as_of |
|---|---|
| **Chronos-2** | Quarterly history up to `as_of` only. PIT guard inside [`forecast_for_ticker`](../../src/forecast_chronos2.py). |
| **Bloomberg consensus** | The panel's `best_*` column at as_of. This is sell-side consensus for the next fiscal year — a single number per as_of, not per-horizon. The per-horizon attachment is therefore a rough apples-to-oranges; we flag it explicitly in the summary. |
| **Naive YoY** | Realized value 4 quarters before the target quarter. Always strictly earlier than as_of, so PIT-safe. The honest "what does last year predict?" baseline. |

### Metrics reported per (target, horizon_q)

| Metric | Meaning |
|---|---|
| `chronos_mae`, `naive_mae`, `consensus_mae` | Mean absolute error in the metric's units (dollars). |
| `chronos_mape`, `naive_mape`, `consensus_mape` | Mean absolute percentage error. |
| `chronos_beats_naive_win_rate` | Fraction of cells where Chronos was closer to realized than naive YoY. |
| `chronos_beats_consensus_win_rate` | Same vs consensus, with the FY-period caveat. |
| `chronos_q10q90_coverage_rate` | Calibration: fraction of realized values inside Chronos's 80% prediction interval. A well-calibrated model lands near 80%. |

### Honest expectation

The course's framing of Chronos was *"ok job against sound statistical
models, pretty well against naive ones."* The win-rate we report is the
operationalization of that statement.

- If Chronos's `chronos_beats_naive_win_rate` is materially above 0.5,
  Chronos contributes signal vs the simplest baseline.
- If Chronos's `chronos_beats_consensus_win_rate` is near 0.5, the
  foundation model and the consensus of sell-side analysts are roughly
  even — the honest framing.
- Calibration rate near 0.8 means the prediction intervals are usable;
  far from 0.8 means we should report the q50 forecast but not the band.

The summary JSON reports each leg independently. The writeup's failure-case
section ([digest_examples.md](digest_examples.md)) walks through the most
illustrative case where Chronos was confidently wrong, and the most
illustrative case where Chronos identified a divergence from consensus that
later materialized.
