# Methodology

## Modelling pipeline

The core script is [src/predict_returns_ckx.py](../../src/predict_returns_ckx.py).
It is the only consumer of `_data/panel_monthly.parquet` (modelling code
never reads raw feature parquets — only `load_panel()`). For each of the
two forecast horizons (`fwd_ret_1m`, `fwd_ret_3m`) it:

1. Builds five variants (V0a, V0b, V1, V2, V3) of nested feature sets
   over the SAME 13-ticker pooled training panel.
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
