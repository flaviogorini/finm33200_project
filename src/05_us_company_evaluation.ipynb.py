# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03. US-Company Forecast Evaluation
#
# 1-step monthly forecasts of returns for 13 US large-caps. Each ticker is
# forecasted **independently**:
#
# * **Factor regression** — OLS of `r_{t+1}` on three lagged changes in
#   Bloomberg consensus forecasts: P/E growth, profit-margin growth
#   (NI / Sales), and sales growth. Fit per ticker under both schemes.
# * **Chronos2 with covariates** — the same three factor-changes are fed
#   as **covariates** alongside the return history. `cross_learning=False`
#   so each ticker only sees its own data.
# * **Mean baseline** — per-ticker mean of training returns.
#
# Two estimation schemes: expanding history, and 5-year rolling window.
# Test window: 2010-01-31 → today.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from us_company_forecasts import (
    load_us_chronos_forecasts,
    load_us_regression_forecasts,
)

# %% [markdown]
# ## Load forecasts

# %%
reg_df = load_us_regression_forecasts()
chr_df = load_us_chronos_forecasts().assign(model="chronos")
all_df = pd.concat(
    [
        reg_df[["ticker", "date", "scheme", "model", "forecast", "actual"]],
        chr_df[["ticker", "date", "scheme", "model", "forecast", "actual"]],
    ],
    ignore_index=True,
)
print(f"regression rows: {len(reg_df):,}")
print(f"chronos rows   : {len(chr_df):,}")
print(f"combined rows  : {len(all_df):,}")
print(f"tickers        : {sorted(all_df['ticker'].unique())}")
print(f"date range     : {all_df['date'].min().date()} → {all_df['date'].max().date()}")

# %% [markdown]
# ## Hit ratio (pooled across tickers)

# %%
hit = all_df.assign(hit=np.sign(all_df["forecast"]) == np.sign(all_df["actual"]))
hit_pooled = (
    hit.groupby(["scheme", "model"])["hit"]
    .mean()
    .unstack("model")
    .round(3)
)
hit_pooled

# %% [markdown]
# ## Hit ratio by ticker

# %%
hit_by_ticker = (
    hit.groupby(["ticker", "scheme", "model"])["hit"]
    .mean()
    .unstack("model")
    .round(3)
)
hit_by_ticker

# %% [markdown]
# ## Out-of-sample R²
#
# `R²_oos = 1 − SSE(model) / SSE(benchmark)`. Positive ⇒ model beats benchmark.
# Computed pooled across all (ticker × date) pairs.

# %%
def _sse(group):
    return float(((group["forecast"] - group["actual"]) ** 2).sum())


def oos_r2_table(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    rows = []
    for scheme, sg in df.groupby("scheme"):
        bench = sg[sg["model"] == benchmark]
        bench_sse = _sse(bench)
        for model, mg in sg.groupby("model"):
            rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    f"r2_vs_{benchmark}": 1.0 - _sse(mg) / bench_sse,
                }
            )
    return (
        pd.DataFrame(rows)
        .pivot(index="scheme", columns="model", values=f"r2_vs_{benchmark}")
        .round(4)
    )


# %%
oos_r2_table(all_df, benchmark="mean")

# %%
oos_r2_table(all_df, benchmark="factor")

# %% [markdown]
# ## Information coefficient (rank correlation)
#
# Rank correlation between forecast and realized return strips out the
# bull-market drift artifact that inflates the hit ratio of any
# always-positive forecaster. Two flavours:
#
# * **Cross-sectional IC** — for each month, compute Spearman correlation
#   between the 13 tickers' forecasts and their realized returns, then
#   average over months. Tests *"does the model rank stocks correctly
#   within a given month?"*. This is the standard quant-equity
#   information coefficient.
# * **Time-series IC** — for each ticker, compute Spearman correlation
#   between the model's forecasts and the realized returns over all
#   months, then average across tickers. Tests *"for a given ticker, are
#   higher forecasts associated with higher realized returns over time?"*.
#
# A useful magnitude reference: cross-sectional ICs of 0.03–0.05 are
# considered modest but real in quant equity research; ≥0.10 is strong.
# Values near 0 mean no rank-ordering content.

# %%
from scipy.stats import spearmanr


def _safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return spearmanr(a[mask], b[mask]).statistic


# Cross-sectional IC: one IC per (scheme, model, date), then average.
cs_rows = []
for (scheme, model, date), grp in all_df.groupby(["scheme", "model", "date"]):
    cs_rows.append(
        {
            "scheme": scheme,
            "model": model,
            "date": date,
            "ic": _safe_spearman(grp["forecast"], grp["actual"]),
        }
    )
cs_ic = pd.DataFrame(cs_rows)
cs_ic_summary = (
    cs_ic.groupby(["scheme", "model"])["ic"]
    .agg(mean_ic="mean", share_positive=lambda s: (s > 0).mean())
    .round(3)
)
print("Cross-sectional IC (mean across months) and share of months with IC > 0:")
cs_ic_summary

# %%
# Time-series IC: one IC per (scheme, model, ticker), then average across tickers.
ts_rows = []
for (scheme, model, ticker), grp in all_df.groupby(["scheme", "model", "ticker"]):
    ts_rows.append(
        {
            "scheme": scheme,
            "model": model,
            "ticker": ticker,
            "ic": _safe_spearman(grp["forecast"], grp["actual"]),
        }
    )
ts_ic = pd.DataFrame(ts_rows)
ts_ic_summary = (
    ts_ic.groupby(["scheme", "model"])["ic"]
    .agg(mean_ic="mean", share_positive=lambda s: (s > 0).mean())
    .round(3)
)
print("Time-series IC (mean across tickers) and share of tickers with IC > 0:")
ts_ic_summary

# %% [markdown]
# ## MAE and RMSE (pooled)

# %%
err = all_df.assign(
    abs_err=(all_df["forecast"] - all_df["actual"]).abs(),
    sq_err=(all_df["forecast"] - all_df["actual"]) ** 2,
)
mae = err.groupby(["scheme", "model"])["abs_err"].mean().unstack("model").round(5)
mae

# %%
rmse = (
    err.groupby(["scheme", "model"])["sq_err"]
    .mean()
    .pow(0.5)
    .unstack("model")
    .round(5)
)
rmse

# %% [markdown]
# ## Forecast vs realized — per-ticker grid (rolling scheme)
#
# Black = realized monthly return, blue = factor regression, violet =
# Chronos median, light purple band = Chronos 10–90% prediction interval.

# %%
def plot_per_ticker(df: pd.DataFrame, scheme: str) -> plt.Figure:
    tickers = sorted(df["ticker"].unique())
    n = len(tickers)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.6 * nrows), sharex=False)
    axes = axes.flatten()
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        sub = df[(df["ticker"] == ticker) & (df["scheme"] == scheme)]
        if sub.empty:
            ax.set_visible(False)
            continue
        actual = sub.drop_duplicates("date").sort_values("date")
        ax.plot(actual["date"], actual["actual"], color="black", lw=0.6, label="actual")

        chr_sub = sub[sub["model"] == "chronos"].sort_values("date")
        fact_sub = sub[sub["model"] == "factor"].sort_values("date")
        if not chr_sub.empty:
            chr_full = chr_df[
                (chr_df["ticker"] == ticker) & (chr_df["scheme"] == scheme)
            ].sort_values("date")
            ax.fill_between(
                chr_full["date"], chr_full["q10"], chr_full["q90"],
                color="xkcd:light lavender", alpha=0.5,
            )
            ax.plot(chr_sub["date"], chr_sub["forecast"], color="xkcd:violet",
                    lw=0.9, label="Chronos")
        if not fact_sub.empty:
            ax.plot(fact_sub["date"], fact_sub["forecast"], color="xkcd:blue",
                    lw=0.7, label="factor", ls="--")
        ax.set_title(ticker.replace(" US Equity", ""), fontsize=10)
        ax.axhline(0, color="grey", lw=0.4)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)
    for j in range(len(tickers), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"US monthly return forecasts ({scheme} scheme)")
    fig.tight_layout()
    return fig


# %%
fig_roll = plot_per_ticker(all_df, scheme="rolling")
fig_roll

# %%
fig_exp = plot_per_ticker(all_df, scheme="expanding")
fig_exp
