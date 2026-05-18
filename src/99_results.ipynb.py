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
# # LLM Embeddings vs Traditional Signals on Earnings Call Transcripts
#
# This notebook is the **final deliverable** for the project. All compute lives
# in `src/*.py` modules wired into `dodo.py`; this notebook only loads
# artifacts and renders tables/charts.
#
# ## What's compared
#
# Five cross-sectional long-short signals on the ~91-ticker Nasdaq-100 min10y
# universe, identical 21-trading-day holding period, top-20 / bottom-20
# equal-weighted legs, monthly rebalance:
#
# 1. **Anchor cosine on Δ sentiment** (`sig_anchor`) — current code in
#    `score_transcript_sentiment.py`.
# 2. **Learned ridge regression on Δ call vector** (`sig_ridge`) — 1,536-D
#    OpenAI embedding deltas + days_since_earnings → 21-bday forward return.
# 3. **Loughran-McDonald Δ net positivity** (`sig_lm`) — bag-of-words baseline.
# 4. **Price momentum 12-1** (`sig_mom`).
# 5. **Analyst revisions Δ BEst Net Income** (`sig_rev`).
#
# Each strategy reports return/risk metrics + Information Coefficient (IC)
# metrics. Robustness checks: drop stale calls (>60 calendar days since
# earnings) and post-2018 subsample. A Fama-MacBeth joint regression tests
# whether LLM signals add information beyond momentum and revisions.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_LABELS = {
    "anchor": "1. Anchor cosine (LLM)",
    "ridge": "2. Ridge + PCA (LLM)",
    "lm": "3. LM lexicon",
    "momentum": "4. Momentum 12-1",
    "revisions": "5. Analyst revisions",
}
STRATEGY_ORDER = list(STRATEGY_LABELS)
COLOURS = {
    "anchor": "#1f77b4",
    "ridge": "#ff7f0e",
    "lm": "#2ca02c",
    "momentum": "#9467bd",
    "revisions": "#8c564b",
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# %% [markdown]
# ## 1. Load artifacts

# %%
panel = _load_parquet(DATA_DIR / "signal_panel_monthly.parquet")
panel["date"] = pd.to_datetime(panel["date"]) if not panel.empty else panel.get("date")

metrics_main = _load_json(DATA_DIR / "metrics_main.json")
metrics_stale = _load_json(DATA_DIR / "metrics_stale_excl.json")
metrics_post = _load_json(DATA_DIR / "metrics_post2018.json")

results_main = _load_parquet(DATA_DIR / "results_main.parquet")
results_stale = _load_parquet(DATA_DIR / "results_stale_excl.parquet")
results_post = _load_parquet(DATA_DIR / "results_post2018.parquet")

ic_ts = _load_parquet(DATA_DIR / "ic_timeseries.parquet")
ic_summary = _load_json(DATA_DIR / "ic_summary.json")

fm = _load_json(DATA_DIR / "fm_results.json")

print(f"panel rows: {len(panel):,}")
print(f"main results months: {0 if results_main.empty else results_main['date'].nunique():,}")
print(f"strategies in metrics_main: {sorted(metrics_main.keys())}")


# %% [markdown]
# ## 2. Universe & data summary

# %%
if not panel.empty:
    n_tickers = panel["ticker"].nunique()
    date_min = panel["date"].min().date()
    date_max = panel["date"].max().date()
    summary_rows = [{"ticker_count": n_tickers, "date_min": str(date_min), "date_max": str(date_max)}]
    for col in ["sig_anchor", "sig_ridge", "sig_lm", "sig_mom", "sig_rev"]:
        if col in panel.columns:
            summary_rows[0][f"{col}_non_null"] = int(panel[col].notna().sum())
    summary_df = pd.DataFrame(summary_rows)
else:
    summary_df = pd.DataFrame()
summary_df


# %% [markdown]
# ## 3a. Return / risk metrics — main specification
#
# Annualised return, annualised vol, Sharpe (zero risk-free), maximum
# drawdown, hit rate (% of months with positive long-short return), and
# information ratio against the equal-weighted benchmark.

# %%
RETURN_RISK_COLS = ["ann_ret", "ann_vol", "sharpe", "max_dd", "hit_rate", "ir_vs_bench", "n_months"]
IC_COLS = ["ic_mean", "ic_std", "ic_ir", "ic_pos_frac", "ic_n_months"]


def _return_risk_frame(metrics_block: dict) -> pd.DataFrame:
    rows = []
    for key in STRATEGY_ORDER:
        m = metrics_block.get(key)
        if not m:
            continue
        rows.append({
            "strategy": STRATEGY_LABELS[key],
            "ann_ret": m.get("ann_ret", np.nan),
            "ann_vol": m.get("ann_vol", np.nan),
            "sharpe": m.get("sharpe", np.nan),
            "max_dd": m.get("max_dd", np.nan),
            "hit_rate": m.get("hit_rate", np.nan),
            "ir_vs_bench": m.get("ir_vs_eqweight_bench", np.nan),
            "n_months": m.get("n_months", np.nan),
        })
    return pd.DataFrame(rows)


def _ic_frame(ic_block: dict) -> pd.DataFrame:
    rows = []
    for key in STRATEGY_ORDER:
        i = ic_block.get(key)
        if not i:
            continue
        rows.append({
            "strategy": STRATEGY_LABELS[key],
            "ic_mean": i.get("ic_mean", np.nan),
            "ic_std": i.get("ic_std", np.nan),
            "ic_ir": i.get("ic_ir", np.nan),
            "ic_pos_frac": i.get("ic_pos_frac", np.nan),
            "ic_n_months": i.get("ic_n_months", np.nan),
        })
    return pd.DataFrame(rows)


return_risk_main = _return_risk_frame(metrics_main)
return_risk_main


# %% [markdown]
# ## 3b. Predictive-power (IC) metrics — main specification
#
# Cross-sectional Spearman rank correlation between signal and realised
# 21-day forward return, computed at each rebalance date.
#
# - **IC mean**: average monthly IC. Reflects whether the *full ordering*
#   of stocks lines up with returns (vs. the long/short tails only,
#   which is what the table above measures).
# - **IC IR**: mean IC / std IC — analogous to a Sharpe at the
#   cross-sectional ranking level.

# %%
ic_main = _ic_frame(ic_summary)
ic_main


# %% [markdown]
# ## 4. Cumulative-return chart — main specification

# %%
def _plot_cum_returns(monthly: pd.DataFrame, title: str, save_as: Path) -> None:
    if monthly.empty:
        print("(no data)")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, grp in monthly.groupby("strategy"):
        grp = grp.sort_values("date")
        cum = (1.0 + grp["ret_ls"]).cumprod()
        ax.plot(
            grp["date"], cum, label=STRATEGY_LABELS.get(label, label),
            color=COLOURS.get(label),
        )
    ax.axhline(1.0, color="grey", linewidth=0.5)
    ax.set_ylabel("Cumulative long-short return (×, monthly compounded)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_as, dpi=140, bbox_inches="tight")
    plt.show()


_plot_cum_returns(
    results_main,
    "Main specification — cumulative long-short return",
    OUTPUT_DIR / "99_cum_returns_main.png",
)


# %% [markdown]
# ## 5. Cumulative-return chart — stale-call exclusion (≤60d)
#
# **How the staleness filter works.** Earnings calls happen quarterly, but
# we rebalance the portfolio every month. Between calls the signal value
# doesn't change — we carry the most recent call's signal forward
# unchanged through each month-end until the next call lands. So a single
# Δ sentiment value can drive **three** rebalances before getting
# refreshed:
#
# ```
# AAPL call on Jan 15                          → fresh signal
#   Jan-end rebalance: days_since_earnings=16  → uses Jan 15's Δ
#   Feb-end rebalance: days_since_earnings=44  → still uses Jan 15's Δ
#   Mar-end rebalance: days_since_earnings=75  → still uses Jan 15's Δ
# AAPL call on Apr 15                          → fresh signal
#   Apr-end rebalance: days_since_earnings=15  → uses Apr 15's Δ
# ```
#
# The **main specification** uses all four of those months. The
# **stale-call exclusion specification** drops any ticker whose
# ``days_since_earnings > 60`` from that month's cross-sectional
# ranking — so in the example above, AAPL is *excluded from the
# Mar-end portfolio entirely* (not long, not short — just absent). The
# long-short rule itself is unchanged; the only thing that changes is
# the population being ranked. Some months therefore have legs that
# shrink to fewer than 20 names per side.
#
# **Purpose.** Tests whether the signal is event-driven: if stale-excl
# Sharpe > main Sharpe, the carried-forward stale signal was adding
# noise and the signal is concentrated near the earnings event.
#
# **Scope of the filter.** Per spec section 8, the 60d filter applies
# only to the three sentiment strategies (anchor, ridge, LM). Price
# momentum and analyst revisions refresh independently every month from
# Bloomberg, so applying ``days_since_earnings > 60`` to them would
# shrink their universe for no methodological reason. The orchestrator
# routes the filter accordingly — so under the stale-excl spec the
# momentum and revisions rows in section 9a are identical to their
# main-spec rows. Any difference between main and stale-excl for those
# two strategies in this notebook would indicate a bug.

# %%
_plot_cum_returns(
    results_stale,
    "Stale-call exclusion (≤60d) — cumulative long-short return",
    OUTPUT_DIR / "99_cum_returns_stale_excl.png",
)


# %% [markdown]
# ## 6. Drawdown — running underwater curve
#
# For each strategy, ``drawdown_t = cum_t / running_peak_t - 1`` (always
# ≤ 0). Shows the depth and persistence of losses over time. Flat = at a
# new high; deep negative = sustained underperformance.

# %%
def _plot_drawdowns(monthly: pd.DataFrame, save_as: Path) -> None:
    if monthly.empty:
        print("(no data)")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, grp in monthly.groupby("strategy"):
        grp = grp.sort_values("date")
        cum = (1.0 + grp["ret_ls"]).cumprod()
        peak = cum.cummax()
        dd = cum / peak - 1.0
        ax.plot(
            grp["date"], dd.values, label=STRATEGY_LABELS.get(label, label),
            color=COLOURS.get(label),
        )
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_ylabel("Drawdown (× from peak)")
    ax.set_title("Running drawdown by strategy — main specification")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_as, dpi=140, bbox_inches="tight")
    plt.show()


_plot_drawdowns(results_main, OUTPUT_DIR / "99_drawdown_main.png")


# %% [markdown]
# ## 7. Hit rate by strategy
#
# Fraction of months with positive long-short return. 50% is the
# coin-flip baseline. A hit rate near 0.5 with a positive Sharpe means
# the *magnitude* of wins drives the return; a higher hit rate means the
# *frequency* of wins does.

# %%
def _plot_hit_rates(metrics_block: dict, save_as: Path) -> None:
    rows = [
        (STRATEGY_LABELS[k], metrics_block.get(k, {}).get("hit_rate", np.nan))
        for k in STRATEGY_ORDER if k in metrics_block
    ]
    if not rows:
        print("(no data)")
        return
    labels, vals = zip(*rows)
    colours = [COLOURS[k] for k in STRATEGY_ORDER if k in metrics_block]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(labels)), vals, color=colours)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="coin-flip (0.5)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Hit rate (fraction of months > 0)")
    ax.set_title("Hit rate by strategy — main specification")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_as, dpi=140, bbox_inches="tight")
    plt.show()


_plot_hit_rates(metrics_main, OUTPUT_DIR / "99_hit_rates_main.png")


# %% [markdown]
# ## 8. Rolling 12-month IC chart

# %%
def _plot_rolling_ic(ic_ts: pd.DataFrame, save_as: Path) -> None:
    if ic_ts.empty:
        print("(no IC data)")
        return
    ic = ic_ts.copy()
    ic["date"] = pd.to_datetime(ic["date"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, grp in ic.groupby("strategy"):
        grp = grp.sort_values("date").set_index("date")
        rolling = grp["ic"].rolling(12, min_periods=6).mean()
        ax.plot(
            rolling.index, rolling.values, label=STRATEGY_LABELS.get(label, label),
            color=COLOURS.get(label),
        )
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_ylabel("Rolling 12-month mean IC (Spearman)")
    ax.set_title("Cross-sectional IC — rolling 12-month average")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_as, dpi=140, bbox_inches="tight")
    plt.show()


_plot_rolling_ic(ic_ts, OUTPUT_DIR / "99_rolling_ic.png")


# %% [markdown]
# ## 9a. Robustness — main vs stale-call exclusion (≤60d)
#
# Tests whether the sentiment signals decay quickly (event-driven) or
# survive when the most recent call is "stale" (>60d old). Stronger
# stale-excl Sharpe = signal is concentrated near the earnings event.

# %%
return_risk_main_v_stale = (
    return_risk_main[["strategy", "ann_ret", "sharpe", "max_dd", "hit_rate", "n_months"]]
    .rename(columns={c: f"main_{c}" for c in ["ann_ret", "sharpe", "max_dd", "hit_rate", "n_months"]})
    .merge(
        _return_risk_frame(metrics_stale)[["strategy", "ann_ret", "sharpe", "max_dd", "hit_rate", "n_months"]]
        .rename(columns={c: f"stale_{c}" for c in ["ann_ret", "sharpe", "max_dd", "hit_rate", "n_months"]}),
        on="strategy", how="left",
    )
)
return_risk_main_v_stale


# %% [markdown]
# ## 9b. Robustness — main vs post-2018 subsample

# %%
return_risk_main_v_post = (
    return_risk_main[["strategy", "ann_ret", "sharpe", "max_dd", "hit_rate", "n_months"]]
    .rename(columns={c: f"main_{c}" for c in ["ann_ret", "sharpe", "max_dd", "hit_rate", "n_months"]})
    .merge(
        _return_risk_frame(metrics_post)[["strategy", "ann_ret", "sharpe", "max_dd", "hit_rate", "n_months"]]
        .rename(columns={c: f"post_{c}" for c in ["ann_ret", "sharpe", "max_dd", "hit_rate", "n_months"]}),
        on="strategy", how="left",
    )
)
return_risk_main_v_post


# %% [markdown]
# ## 10. Joint regression — Fama-MacBeth with Newey-West (lag 6)
#
# At each rebalance date m, signals are z-scored across the cross-section
# and regressed against `fwd_ret_21d`. Coefficient series are then averaged
# over time with Newey-West HAC standard errors.

# %%
if fm:
    fm_rows = []
    for s in fm.get("signals", []):
        fm_rows.append(
            {
                "signal": s,
                "beta": fm["beta"].get(s, np.nan),
                "nw_se": fm["nw_se"].get(s, np.nan),
                "nw_tstat": fm["nw_tstat"].get(s, np.nan),
            }
        )
    fm_table = pd.DataFrame(fm_rows)
    print(
        f"FM months = {fm['n_months']:,}   mean R^2 = {fm['mean_r2']:.4f}"
        f"   alpha = {fm['alpha']['beta']:+.4f} (t={fm['alpha']['t']:+.3f})"
    )
else:
    fm_table = pd.DataFrame()
fm_table


# %% [markdown]
# ## 11. Limitations
#
# - **Survivorship bias.** The 91-ticker universe is current Nasdaq-100 members
#   that have ≥10 years of transcript history; tickers that delisted or fell
#   out of the index are excluded by construction. Bias should largely cancel
#   in the *relative* comparison across strategies but may not cancel
#   symmetrically.
# - **Small universe.** 100 tickers and 20-name quintile legs limit statistical
#   power and concentrate exposure.
# - **No live-data simulation.** Embeddings are computed today on historical
#   text. The embedding model's training data includes some of the test
#   period — potential look-ahead noted but not corrected.
# - **Single embedding model.** Only `text-embedding-3-small` is tested;
#   alternatives (FinBERT, larger OpenAI embedders) are out of scope.
# - **Anchor sentences for Strategy 1 not validated.** Different anchor
#   wording gives different scores; no sensitivity sweep.
# - **Out-of-sample period.** ~7 years (2019-2025) once transcripts are
#   re-pulled. Statistical power scales with √T.

# %%
