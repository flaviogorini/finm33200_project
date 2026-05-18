# Forecast results

Two forecasts feed the decision digest. Both are evaluated honestly here — no
target threshold is hardcoded; we report what the data shows.

## Returns forecast — V0a → V4 ladder

**Code:** [src/predict_returns_ckx.py](../../src/predict_returns_ckx.py)
**Outputs:** `_output/ckx_metrics.json`, `_output/ckx_predictions.parquet`, `_output/ckx_portfolio.parquet`

The variant ladder tests four nested hypotheses (see [goals.md](../project_overview/goals.md)):

1. **V0b vs V0a** — does anything beat zero?
2. **V1 vs V0b** — does fundamentals + macro add to momentum?
3. **V2/V3 vs V1** — does text (call sentiment, then 10-Q lexicon) add?
4. **V4 vs V3** — does the LLM reading the filing beat the word-count dictionary?

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

