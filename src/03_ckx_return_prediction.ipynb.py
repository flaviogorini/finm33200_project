# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # CKX-style return prediction: does sentiment + 10-Q text add signal?
#
# Walk-forward CV on the unified panel. Three feature variants:
#
# - **V1**: pooled across 13 tickers, no text features (baseline)
# - **V2**: AAPL-only, + earnings-call sentiment
# - **V3**: AAPL-only, + earnings-call sentiment + 10-Q text  *(skipped if 10-Q parquet absent)*
#
# Two regressors per variant: `Ridge`, `GradientBoostingRegressor`.
# Headline metric: AUC of `P(fwd_ret_3m > 0)` on AAPL test rows.
#
# Pipeline (run from repo root):
#
# ```bash
# python src/predict_returns_ckx.py
# # → _output/ckx_predictions.parquet, ckx_metrics.json, ckx_portfolio.parquet
# ```

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

predictions = pd.read_parquet(OUTPUT_DIR / "ckx_predictions.parquet")
metrics = json.loads((OUTPUT_DIR / "ckx_metrics.json").read_text())
portfolios = pd.read_parquet(OUTPUT_DIR / "ckx_portfolio.parquet")

predictions["date"] = pd.to_datetime(predictions["date"])
portfolios["date"] = pd.to_datetime(portfolios["date"])

print(
    f"Predictions: {len(predictions):,} rows  |  "
    f"variants: {sorted(predictions['variant'].unique())}  |  "
    f"models: {sorted(predictions['model'].unique())}"
)

# %% [markdown]
# ## 1. Headline metric table

# %%
rows = []
for v_name, v_block in metrics.items():
    for m_name, m in v_block.get("models", {}).items():
        rows.append(
            {
                "variant": v_name,
                "model": m_name,
                "n_aapl_oos": m.get("n", 0),
                "AUC": m.get("auc", float("nan")),
                "accuracy": m.get("accuracy", float("nan")),
                "OOS_R2": m.get("oos_r2", float("nan")),
                "IC_spearman": m.get("ic_spearman", float("nan")),
            }
        )
metrics_df = pd.DataFrame(rows)
metrics_df.style.format({"AUC": "{:.3f}", "accuracy": "{:.3f}",
                        "OOS_R2": "{:+.3f}", "IC_spearman": "{:+.3f}"}) \
    if hasattr(metrics_df, "style") else metrics_df

# %% [markdown]
# ## 2. AUC bar chart per variant × model

# %%
fig, ax = plt.subplots(figsize=(7, 4))
pivot = metrics_df.pivot(index="variant", columns="model", values="AUC")
pivot.plot(kind="bar", ax=ax, color=["C0", "C1"])
ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="random")
ax.set_ylabel("AUC (AAPL OOS)")
ax.set_ylim(0.40, 0.75)
ax.set_title("AUC of P(fwd_ret_3m > 0) on AAPL test rows")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ckx_auc_bars.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Cumulative-return chart per strategy
#
# - **V1**: long-short cross-sectional tertile sort across 13 tickers (GBR rank).
# - **V2 / V3**: AAPL timing — long when `P(up) > 0.55`, else cash.
#
# Returns smeared to monthly equivalent (`fwd_ret_3m / 3`) so the chart isn't
# triple-counting the 3-month overlap.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
for variant in sorted(portfolios["variant"].unique()):
    sub = portfolios[portfolios["variant"] == variant].sort_values("date")
    ax.plot(sub["date"], sub["cum_ret"], label=f"{variant} ({sub['strategy'].iloc[0]})")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_ylabel("Cumulative return (monthly compounded)")
ax.set_title("OOS portfolio backtest — variants vs benchmark")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ckx_cum_returns.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Per-fold AUC stability
#
# Fold = year of OOS test slice. A model that's stable across folds is more
# trustworthy than one that wins on average via a single lucky year.

# %%
fold_metrics = []
for (variant, model, fold), grp in predictions.groupby(["variant", "model", "fold"]):
    aapl = grp[grp["ticker"] == "AAPL"]
    if aapl.empty:
        continue
    y_bin = (aapl["y_true"] > 0).astype(int)
    if y_bin.nunique() < 2:
        continue
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_bin, aapl["p_up"])
    train_end = grp["train_end"].iloc[0]
    fold_metrics.append({"variant": variant, "model": model, "train_end": train_end, "AUC": auc})

fold_df = pd.DataFrame(fold_metrics)
if not fold_df.empty:
    fig, ax = plt.subplots(figsize=(11, 4))
    for (variant, model), grp in fold_df.groupby(["variant", "model"]):
        ax.plot(grp["train_end"], grp["AUC"], "o-", label=f"{variant}/{model}")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("train_end (fold cutoff)")
    ax.set_ylabel("AAPL test AUC")
    ax.set_title("Per-fold stability of AUC")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ckx_fold_auc.png", dpi=140, bbox_inches="tight")
    plt.show()
else:
    print("Per-fold AUC unavailable (test slices too sparse for at least one class).")

# %% [markdown]
# ## 5. Takeaways
#
# Read these off the AUC bar chart:
#
# - **V2 > V1?** If yes, OpenAI-embedded earnings-call sentiment adds signal
#   beyond fundamentals + macro alone.
# - **V3 > V2?** (Only if 10-Q parquet has been built.) If yes, 10-Q narrative
#   adds incremental signal beyond the call.
# - **Ridge vs GBR**: with only ~250 AAPL training rows per fold, GBR easily
#   overfits — Ridge is the safer baseline. If GBR materially beats Ridge,
#   suspect leakage; if it's much worse, regularisation is doing its job.
