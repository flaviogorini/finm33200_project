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
# # 04. Stock Return Forecast Evaluation: Chronos vs Baselines
#
# Out-of-sample evaluation of 1-step return forecasts for a single US stock
# at weekly and monthly frequency. Compares the **Chronos2** zero-shot
# forecaster against four baselines (`mean`, `ar1`, `arima(1,0,1)`, `zero`)
# under two conditioning schemes:
#
# * **expanding** — train = all returns strictly before target date
# * **rolling**   — train = last 5 years of returns
#
# Default trial ticker: **AAPL**. Any ticker present in
# `_data/US_Companies_Hist_Data.parquet` (with `PX_LAST`) can be selected by
# changing the `TICKER` constant below and re-running the upstream pulls:
#
# ```bash
# python src/stock_baselines.py --ticker <TICKER>
# python src/stock_chronos.py   --ticker <TICKER>
# ```
#
# Test period: 2010-01-01 → today.
#
# Metrics:
# 1. **Hit ratio** — share of forecasts whose sign matches the realized return.
# 2. **Out-of-sample R²** — `1 − SSE(model) / SSE(benchmark)`, with the `mean`
#    and `ar1` baselines as benchmarks. Positive ⇒ model beats the benchmark.
# 3. **MAE / RMSE** — point-error magnitudes.
# 4. **Forecast vs realized plots** — visual sanity check.

# %%
TICKER = "AAPL"

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_baselines import load_baseline_forecasts
from stock_chronos import load_chronos_forecasts

# %% [markdown]
# ## Load forecasts

# %%
baseline_df = load_baseline_forecasts(TICKER)
chronos_df = load_chronos_forecasts(TICKER).assign(model="chronos")
# Concat keeping all columns; q10/q90 are NaN for baseline rows, which is fine.
all_df = pd.concat([baseline_df, chronos_df], ignore_index=True)
all_df.head()

# %%
print(f"ticker        : {TICKER}")
print(f"baseline rows : {len(baseline_df):,}")
print(f"chronos rows  : {len(chronos_df):,}")
print(f"combined      : {len(all_df):,}")
print(f"date range    : {all_df['date'].min().date()} → {all_df['date'].max().date()}")

# %% [markdown]
# ## Hit ratio
#
# Share of forecasts whose sign matches the realized return. The `zero`
# baseline always reads as 0 (sign mismatch on every non-zero realization).

# %%
hit = all_df.assign(hit=np.sign(all_df["forecast"]) == np.sign(all_df["actual"]))
hit_ratio = (
    hit.groupby(["freq", "scheme", "model"])["hit"]
    .mean()
    .unstack("model")
    .round(3)
)
hit_ratio

# %% [markdown]
# ## Out-of-sample R²
#
# For each model series, `R²_oos = 1 − SSE(model) / SSE(benchmark)`. Positive
# values mean the model beats the benchmark on squared error. We benchmark
# Chronos against both `mean` and `ar1`.

# %%
def _sse(forecasts, actual):
    return float(np.sum((np.asarray(forecasts) - np.asarray(actual)) ** 2))


def oos_r2_table(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    rows = []
    for (freq, scheme), grp in df.groupby(["freq", "scheme"]):
        bench = grp[grp["model"] == benchmark]
        if bench.empty:
            continue
        bench_sse = _sse(bench["forecast"], bench["actual"])
        for model, mgrp in grp.groupby("model"):
            mgrp = mgrp.merge(
                bench[["date", "actual"]], on="date", suffixes=("", "_b")
            )
            sse = _sse(mgrp["forecast"], mgrp["actual"])
            rows.append(
                {
                    "freq": freq,
                    "scheme": scheme,
                    "model": model,
                    "r2_vs_" + benchmark: 1.0 - sse / bench_sse,
                }
            )
    return (
        pd.DataFrame(rows)
        .pivot(index=["freq", "scheme"], columns="model", values="r2_vs_" + benchmark)
        .round(4)
    )


# %%
oos_r2_table(all_df, benchmark="mean")

# %%
oos_r2_table(all_df, benchmark="ar1")

# %% [markdown]
# ## MAE and RMSE

# %%
err = all_df.assign(
    abs_err=(all_df["forecast"] - all_df["actual"]).abs(),
    sq_err=(all_df["forecast"] - all_df["actual"]) ** 2,
)
mae = (
    err.groupby(["freq", "scheme", "model"])["abs_err"]
    .mean()
    .unstack("model")
    .round(5)
)
mae

# %%
rmse = (
    err.groupby(["freq", "scheme", "model"])["sq_err"]
    .mean()
    .pow(0.5)
    .unstack("model")
    .round(5)
)
rmse

# %% [markdown]
# ## Forecast vs realized
#
# One panel per (frequency, scheme). Black = realized return, purple =
# Chronos median, light band = Chronos 10–90% prediction interval, orange
# dashed = AR(1) baseline.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=False)
for i, freq in enumerate(("M", "W")):
    for j, scheme in enumerate(("expanding", "rolling")):
        ax = axes[i, j]
        sub = all_df[(all_df["freq"] == freq) & (all_df["scheme"] == scheme)]
        if sub.empty:
            ax.set_visible(False)
            continue
        actual = sub.drop_duplicates("date").sort_values("date")
        ax.plot(actual["date"], actual["actual"], color="black", lw=0.8, label="actual")

        chronos = sub[sub["model"] == "chronos"].sort_values("date")
        if not chronos.empty:
            ax.fill_between(
                chronos["date"], chronos["q10"], chronos["q90"],
                color="xkcd:light lavender", alpha=0.6, label="Chronos 10–90%",
            )
            ax.plot(
                chronos["date"], chronos["forecast"],
                color="xkcd:violet", lw=1.0, label="Chronos median",
            )

        ar1 = sub[sub["model"] == "ar1"].sort_values("date")
        if not ar1.empty:
            ax.plot(
                ar1["date"], ar1["forecast"],
                color="xkcd:orange", lw=0.8, ls="--", label="AR(1)",
            )

        ax.set_title(f"{freq} / {scheme}")
        ax.axhline(0, color="grey", lw=0.5)
        if i == 0 and j == 0:
            ax.legend(loc="upper left", fontsize=8)
fig.suptitle(f"{TICKER} return forecasts: Chronos vs AR(1) vs realized")
fig.tight_layout()
fig
