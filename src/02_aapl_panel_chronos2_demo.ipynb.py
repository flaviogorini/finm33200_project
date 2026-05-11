# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # AAPL Monthly Panel + Chronos-2 Forecast Demo
#
# Single-ticker proof-of-concept end-to-end:
#
# 1. Load the unified `(date, ticker)` monthly panel.
# 2. Inspect carry-forward semantics for sentiment / fundamentals.
# 3. Run a Chronos-2 zero-shot 4-quarter forecast for revenue + net income.
# 4. Compare against Bloomberg `BEST_*` analyst consensus at the same as-of date.
#
# Prerequisites (run from the repo root):
#
# ```bash
# # Parse Bloomberg Excel exports → parquets
# python src/pull_manual_companies.py
# python src/pull_manual_macro.py
#
# # Build per-source monthly features
# python src/build_fundamentals_features.py
# python src/build_consensus_features.py
# python src/build_macro_features.py
# python src/build_return_labels.py
#
# # Optional: transcript sentiment (skip with SYNTHETIC=1 if no OPENAI_API_KEY)
# SYNTHETIC=1 python src/embed_transcripts.py AAPL --synthetic
# python src/score_transcript_sentiment.py --synthetic
# python src/build_sentiment_features.py
#
# # Assemble the panel
# python src/build_panel.py
# ```
#
# Or just `doit build_panel` once dependencies are installed.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_panel import load_panel
from forecast_chronos2 import (
    DEFAULT_HORIZON_Q,
    DEFAULT_TARGETS,
    _load_chronos_pipeline,
    forecast_for_ticker,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKER = "AAPL"

# %% [markdown]
# ## 1. Load the panel

# %%
panel = load_panel(DATA_DIR, tickers=[TICKER])
print(f"{len(panel)} rows × {len(panel.columns)} cols")
print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
panel.head()

# %% [markdown]
# ## 2. Carry-forward sanity checks
#
# Sentiment should be constant between earnings calls; revenue / net income
# should be constant between quarterly reports.

# %%
if "sentiment_diff" in panel.columns:
    sent = panel[["date", "sentiment_diff", "last_event_date", "days_since_earnings"]].dropna()
    print(f"Sentiment carry-forward sample (one row per month):")
    print(sent.tail(12).to_string(index=False))

# %%
fund = panel[["date", "revenue", "net_income"]].dropna(how="all").tail(24)
print("Fundamentals carry-forward sample:")
print(fund.to_string(index=False))

# %% [markdown]
# ## 3. Chronos-2 zero-shot forecast

# %%
as_of = panel["date"].max() - pd.DateOffset(months=12)  # leave a year of out-of-sample
as_of = pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)
print(f"As-of date: {as_of.date()}")

# %%
pipeline = _load_chronos_pipeline(device="cpu")  # change to "mps" or "cuda" if available
forecasts = forecast_for_ticker(
    panel, pipeline, TICKER, DEFAULT_TARGETS, as_of, DEFAULT_HORIZON_Q
)
forecasts

# %%
out_path = OUTPUT_DIR / f"chronos2_forecast_{TICKER}_{as_of.strftime('%Y%m%d')}.parquet"
forecasts.to_parquet(out_path, index=False)
print(f"Saved → {out_path}")

# %% [markdown]
# ## 4. Plot: history + forecast vs consensus

# %%
fig, axes = plt.subplots(len(DEFAULT_TARGETS), 1, figsize=(10, 4 * len(DEFAULT_TARGETS)), sharex=False)
if len(DEFAULT_TARGETS) == 1:
    axes = [axes]

for ax, target in zip(axes, DEFAULT_TARGETS):
    history = (
        panel[panel["date"] <= as_of]
        .set_index("date")[target]
        .resample("QE").last().dropna()
    )
    ax.plot(history.index, history.values, label=f"{target} (actual)", color="C0")

    f = forecasts[forecasts["target"] == target].copy()
    if not f.empty:
        # Future quarter-end dates
        future_idx = pd.date_range(
            start=as_of + pd.offsets.QuarterEnd(0), periods=DEFAULT_HORIZON_Q + 1, freq="QE"
        )[1:]
        ax.plot(future_idx, f["forecast_q50"], "o-", label="Chronos-2 median", color="C1")
        ax.fill_between(
            future_idx, f["forecast_q10"], f["forecast_q90"],
            color="C1", alpha=0.2, label="Chronos-2 q10–q90",
        )
        consensus = f["consensus"].iloc[0]
        if pd.notna(consensus):
            ax.axhline(consensus, color="C2", linestyle="--", label=f"BEST_* consensus = {consensus:,.0f}")

    ax.set_title(f"{TICKER} {target} — history & 4Q forecast (as-of {as_of.date()})")
    ax.set_ylabel(target)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig_path = OUTPUT_DIR / f"chronos2_forecast_{TICKER}_{as_of.strftime('%Y%m%d')}.png"
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"Saved → {fig_path}")

# %% [markdown]
# ## 5. (Optional) CKX-style direct return-prediction smoke test
#
# Stand-in: regress 3-month forward return on contemporaneous sentiment_diff.
# Real CKX needs a wider feature panel and proper time-series CV.

# %%
from sklearn.linear_model import LogisticRegression

if "sentiment_diff" in panel.columns and "fwd_ret_3m" in panel.columns:
    X = panel[["sentiment_diff"]].copy()
    y = (panel["fwd_ret_3m"] > 0).astype(int)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    if len(X) >= 30:
        model = LogisticRegression().fit(X, y)
        in_sample_acc = model.score(X, y)
        print(f"In-sample accuracy: {in_sample_acc:.3f}  (n = {len(X)})")
        print(f"Coefficient: {model.coef_[0][0]:.3f}")
    else:
        print(f"Too few rows ({len(X)}) for a regression smoke test.")
else:
    print("Sentiment or forward-return columns missing — skipping CKX smoke test.")
