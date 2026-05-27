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
# # LLM Embeddings vs Traditional Signals on Earnings Call Transcripts — v2
#
# **Extensions over the original 5-signal notebook**:
#
# - Adds **CAR3** (announcement-window cumulative return, [-1, +1]) as a 6th
#   first-class signal (`sig_car3`).
# - Adds **§5.4 — signal correlation matrices** (stock-level Spearman + portfolio-level Pearson)
#   to diagnose multicollinearity in the joint FM and α regressions.
# - Adds **§5.6 — nested time-series α progression** for Ridge / Anchor / LM on a
#   FF5 + Mom + CAR3 + Rev factor baseline. Reads `factor_alpha.json`.
# - Renumbers the original §5.4 (Fama-MacBeth) to §5.5, now with 6 signals.
# - PNGs written under `_output/99_v2_*.png` to leave the v1 outputs untouched.
#
# This notebook is the **v2 deliverable** for the project. All compute lives
# in `src/*.py` modules wired into `dodo.py`; this notebook only loads
# artifacts and renders tables/charts.
#
# ## What's compared
#
# Six cross-sectional long-short signals on the ~91-ticker Nasdaq-100 min10y
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
# 6. **CAR3 — earnings-window CAR** (`sig_car3`) — 3-trading-day cumulative
#    return [-1, +1] around `event_date`, carried forward 60 days.
#
# Each strategy reports return/risk metrics + Information Coefficient (IC)
# metrics. Robustness checks: drop stale calls (>60 calendar days since
# earnings) and post-2018 subsample. A Fama-MacBeth joint regression tests
# whether LLM signals add information beyond the other five at the stock
# level. A nested time-series α progression (§5.6) tests the same question
# at the portfolio level against FF5 + Mom + CAR3 + Rev.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import _summary_metrics
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
})

STRATEGY_LABELS = {
    "anchor": "1. Anchor cosine (LLM)",
    "ridge": "2. Ridge + PCA (LLM)",
    "lm": "3. LM lexicon",
    "momentum": "4. Momentum 12-1",
    "revisions": "5. Analyst revisions",
    "car3": "6. CAR3 (earnings-window)",
}
STRATEGY_ORDER = list(STRATEGY_LABELS)
# Okabe-Ito palette: colorblind-friendly. LLM strategies (anchor, ridge) share
# a blue family; the four traditional signals get distinct hues.
COLOURS = {
    "anchor": "#0072B2",     # deep blue    — LLM family
    "ridge": "#56B4E9",      # sky blue     — LLM family
    "lm": "#D55E00",         # vermillion   — traditional / lexicon
    "momentum": "#009E73",   # bluish green
    "revisions": "#CC79A7",  # reddish purple
    "car3": "#F0E442",       # yellow       — earnings-window
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

# v2 new artefacts
factor_alpha = _load_json(DATA_DIR / "factor_alpha.json")
factor_returns = _load_parquet(DATA_DIR / "strategy_factor_returns_monthly.parquet")

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
    for col in ["sig_anchor", "sig_ridge", "sig_lm", "sig_mom", "sig_rev", "sig_car3"]:
        if col in panel.columns:
            summary_rows[0][f"{col}_non_null"] = int(panel[col].notna().sum())
    summary_df = pd.DataFrame(summary_rows)
else:
    summary_df = pd.DataFrame()
summary_df


# %% [markdown]
# ## 3a. Return / risk metrics — all available history per strategy
#
# Annualised return, annualised vol, Sharpe (zero risk-free), maximum
# drawdown, hit rate (% of months with positive long-short return), and
# information ratio against the equal-weighted benchmark. Each strategy
# uses its own full history (so `n_months` differs across rows). For
# apples-to-apples cumulative-return charts on a common window, see
# sections 9 and 10.

# %%
RETURN_RISK_COLS = ["ann_ret", "ann_vol", "sharpe", "max_dd", "hit_rate", "ir_vs_bench", "n_months"]


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
            "spearman_ic_mean": i.get("ic_mean", np.nan),
            "spearman_ic_std": i.get("ic_std", np.nan),
            "ic_ir": i.get("ic_ir", np.nan),
            "pct_months_positive": i.get("ic_pos_frac", np.nan),
            "n_months": i.get("ic_n_months", np.nan),
        })
    return pd.DataFrame(rows)


return_risk_main = _return_risk_frame(metrics_main)
return_risk_main


# %% [markdown]
# ## 3b. Predictive-power — cross-sectional Spearman rank IC
#
# At each monthly rebalance date we compute the **Spearman rank
# correlation** between the strategy's signal and the realised 21-day
# forward return, taken across the cross-section of tickers available
# that month. That single number per month is the Information
# Coefficient (IC). The columns below summarise that monthly IC series:
#
# - **spearman_ic_mean** — average monthly Spearman rank IC. Reflects
#   whether the *full ordering* of stocks lines up with returns (the
#   3a table only measures the top-20 vs bottom-20 tails).
# - **spearman_ic_std** — standard deviation of the monthly IC.
# - **ic_ir** — mean / std (a Sharpe-like ratio at the ranking level).
# - **pct_months_positive** — share of months where IC > 0; 0.50 is the
#   coin-flip baseline.
# - **n_months** — number of months with a valid IC observation.

# %%
ic_main = _ic_frame(ic_summary)
ic_main


# %% [markdown]
# ## 4. Hit rate by strategy
#
# Fraction of months with positive long-short return. 50% is the
# coin-flip baseline. A hit rate near 0.5 with a positive Sharpe means
# the *magnitude* of wins drives the return; a higher hit rate means the
# *frequency* of wins does. Each bar uses each strategy's full history.

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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), vals, color=colours)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="coin-flip (0.5)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Hit rate (fraction of months > 0)")
    ax.set_title("Hit rate by strategy — main specification")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(save_as)
    plt.show()


_plot_hit_rates(metrics_main, OUTPUT_DIR / "99_v2_hit_rates_main.png")


# %% [markdown]
# ## 5. Rolling 12-month IC chart
#
# 12-month trailing mean of the monthly Spearman IC (the same per-month
# IC summarised in section 3b). Each strategy's line begins when it
# accumulates enough months of IC observations.

# %%
def _plot_rolling_ic(ic_ts: pd.DataFrame, save_as: Path) -> None:
    if ic_ts.empty:
        print("(no IC data)")
        return
    ic = ic_ts.copy()
    ic["date"] = pd.to_datetime(ic["date"])
    fig, ax = plt.subplots(figsize=(12, 5))
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
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_as)
    plt.show()


_plot_rolling_ic(ic_ts, OUTPUT_DIR / "99_v2_rolling_ic.png")


# %% [markdown]
# ## 5b. Signal correlation matrices (writeup §5.4)
#
# Two views of how the six signals relate:
#
# - **(a) Stock-level Spearman rank correlations.** Computed on the z-scored
#   signal panel across all (ticker, month) observations. This is the
#   diagnostic for §5.5 (joint Fama-MacBeth) multicollinearity: if two
#   signals are r ≈ 0.9 at the stock level, their individual β t-stats in
#   the FM regression will be unreliable even if the joint loading is real.
# - **(b) Portfolio-level Pearson correlations.** Computed on each strategy's
#   monthly long-short return series from the backtest. This is the diagnostic
#   for §5.6 (nested time-series α) multicollinearity: correlated factor
#   portfolios on the RHS produce noisy individual coefficients.
#
# A pair with |r| > 0.5 is flagged inline.

# %%
SIG_COLS = ["sig_anchor", "sig_ridge", "sig_lm", "sig_mom", "sig_rev", "sig_car3"]


def _stock_level_corr(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    cols = [c for c in SIG_COLS if c in panel.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    # Z-score per month before pooling so the rank correlation is on a
    # like-for-like cross-section. Spearman is invariant under monotonic
    # transforms but z-scoring matches the §5.5 regression preprocessing.
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        for c in cols:
            x = out[c].astype(float)
            mu = x.mean()
            sd = x.std(ddof=1)
            out[c] = (x - mu) / sd if sd and sd > 0 else 0.0
        return out
    zpanel = panel.groupby("date", group_keys=False).apply(_z)
    return zpanel[cols].corr(method="spearman")


def _portfolio_level_corr(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    wide = results.pivot_table(
        index="date", columns="strategy", values="ret_ls", aggfunc="first"
    )
    # Order columns to mirror STRATEGY_ORDER where present.
    cols = [s for s in STRATEGY_ORDER if s in wide.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return wide[cols].corr(method="pearson")


corr_stock = _stock_level_corr(panel)
corr_portfolio = _portfolio_level_corr(results_main)

# Persist for reproducibility and downstream consumers.
if not corr_stock.empty:
    (DATA_DIR / "signal_corr_stock.json").write_text(
        json.dumps(corr_stock.round(4).to_dict(), indent=2)
    )
if not corr_portfolio.empty:
    (DATA_DIR / "signal_corr_portfolio.json").write_text(
        json.dumps(corr_portfolio.round(4).to_dict(), indent=2)
    )

print("Stock-level Spearman rank correlations (z-scored signal panel):")
corr_stock.round(3) if not corr_stock.empty else "(no data)"

# %%
print("Portfolio-level Pearson correlations (monthly LS returns):")
corr_portfolio.round(3) if not corr_portfolio.empty else "(no data)"

# %%
def _flag_high_pairs(corr: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    if corr.empty:
        return []
    flags = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > threshold:
                flags.append(f"  |corr({a}, {b})| = {r:+.3f}")
    return flags


print("Stock-level pairs with |r| > 0.5:")
for line in _flag_high_pairs(corr_stock) or ["  (none)"]:
    print(line)
print("\nPortfolio-level pairs with |r| > 0.5:")
for line in _flag_high_pairs(corr_portfolio) or ["  (none)"]:
    print(line)


# %% [markdown]
# ## 6. Robustness — main vs stale-call exclusion (≤60d)
#
# **How the staleness filter works.** Earnings calls happen quarterly,
# but we rebalance the portfolio every month. Between calls the signal
# value doesn't change — we carry the most recent call's signal forward
# unchanged through each month-end until the next call lands. So a
# single Δ sentiment value can drive **three** rebalances before
# getting refreshed:
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
# momentum and revisions rows below are identical to their main-spec
# rows. Any difference between main and stale-excl for those two
# strategies would indicate a bug.

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
# ## 7. Robustness — main vs post-2018 subsample

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
# ## 8. Joint regression — Fama-MacBeth with Newey-West (lag 6)
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
# ## 8b. Nested time-series α progression (writeup §5.6)
#
# For each text strategy in {Ridge, Anchor, LM}, regress its monthly LS
# return on a growing factor baseline:
#
# - **α₀**: FF5 alone
# - **α₁**: FF5 + Mom-factor (the project's own momentum strategy LS)
# - **α₂**: FF5 + Mom + CAR3-factor (earnings-window control)
# - **α₃**: FF5 + Mom + CAR3 + Rev-factor (analyst revisions control)
#
# Read horizontally: how does α decay as we add controls? A signal whose α
# drops to zero after CAR3 is just earnings news; a signal that retains α
# through column 3 is contributing unique information.
#
# Read vertically: Ridge vs Anchor vs LM head-to-head at any column — this
# is the project's central LLM-vs-lexicon comparison after factor adjustment.
#
# HAC standard errors (Newey-West, lag 6). Reported on both each strategy's
# own backtest history AND the post-2018 common window.

# %%
SPEC_KEYS = ["alpha_0", "alpha_1", "alpha_2", "alpha_3"]
SPEC_HEADERS = ["α₀ (FF5)", "α₁ (+Mom)", "α₂ (+CAR3)", "α₃ (+Rev)"]
ALPHA_STRATEGIES = ["ridge", "anchor", "lm"]


def _alpha_table(sample_block: dict, value: str = "alpha") -> pd.DataFrame:
    rows = []
    for strat in ALPHA_STRATEGIES:
        cells = sample_block.get(strat, {}) or {}
        row = {"strategy": STRATEGY_LABELS[strat]}
        for k, header in zip(SPEC_KEYS, SPEC_HEADERS):
            c = cells.get(k)
            if c is None:
                row[header] = np.nan
            else:
                row[header] = c.get(value, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


print("α (mean monthly residual return, NW lag 6) — own-history sample:")
_alpha_table(factor_alpha.get("own_history", {}), "alpha").round(4)

# %%
print("t(α) — own-history sample:")
_alpha_table(factor_alpha.get("own_history", {}), "t").round(2)

# %%
print("α — post-2018 common-window sample:")
_alpha_table(factor_alpha.get("post_2018", {}), "alpha").round(4)

# %%
print("t(α) — post-2018 common-window sample:")
_alpha_table(factor_alpha.get("post_2018", {}), "t").round(2)


# %% [markdown]
# ## Common-window cumulative-return analysis
#
# The whole-history tables above let each strategy use every month it
# has data for. That's fine for *summary statistics* but produces
# misleading *cumulative-return charts*: a strategy starting in 2000
# compounds 19 more years of returns than one starting in 2019, so the
# endpoints aren't comparable.
#
# Sections 9 and 10 fix this by restricting to common windows. Metrics
# in these sections are **recomputed** from the monthly returns over
# the restricted window — so Sharpe / max_dd / ann_ret match what the
# chart actually shows.

# %%
def _plot_cum_returns(monthly: pd.DataFrame, title: str, save_as: Path) -> None:
    if monthly.empty:
        print("(no data)")
        return
    fig, ax = plt.subplots(figsize=(12, 5.5))
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
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_as)
    plt.show()


def _plot_drawdowns(monthly: pd.DataFrame, title: str, save_as: Path) -> None:
    if monthly.empty:
        print("(no data)")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
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
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_as)
    plt.show()


def _period_metrics_table(
    results: pd.DataFrame,
    strategies: list[str],
    start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for s in strategies:
        sub = results[(results["strategy"] == s) & (results["date"] >= start)].copy()
        m = _summary_metrics(sub)
        if not m:
            continue
        rows.append({
            "strategy": STRATEGY_LABELS[s],
            "ann_ret": m["ann_ret"],
            "ann_vol": m["ann_vol"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
            "hit_rate": m["hit_rate"],
            "ir_vs_bench": m["ir_vs_eqweight_bench"],
            "n_months": m["n_months"],
        })
    return pd.DataFrame(rows)


# %% [markdown]
# ## 9. Common window 1 — 2008 onwards, excluding Ridge + PCA
#
# Ridge + PCA only has predictions from 2019 onward (it needs a long
# training window). Excluding it lets us look at the four traditional /
# lexicon / anchor signals on a longer common window starting
# 2008-01-01.

# %%
PERIOD1_START = pd.Timestamp("2008-01-01")
PERIOD1_STRATS = ["anchor", "lm", "momentum", "revisions"]

period1_table = _period_metrics_table(results_main, PERIOD1_STRATS, PERIOD1_START)
period1_table

# %%
period1_panel = results_main[
    results_main["strategy"].isin(PERIOD1_STRATS)
    & (results_main["date"] >= PERIOD1_START)
]
_plot_cum_returns(
    period1_panel,
    f"Cumulative long-short return — {PERIOD1_START.date()} onwards (excl. Ridge+PCA)",
    OUTPUT_DIR / "99_v2_cum_returns_period1_2008.png",
)

# %%
_plot_drawdowns(
    period1_panel,
    f"Drawdown — {PERIOD1_START.date()} onwards (excl. Ridge+PCA)",
    OUTPUT_DIR / "99_v2_drawdown_period1_2008.png",
)


# %% [markdown]
# ## 10. Common window 2 — Ridge + PCA's first month onwards, all 6 strategies
#
# Start date is the first month for which the Ridge + PCA strategy has a
# realised long-short return. All six strategies are plotted on this
# common window.

# %%
ridge_rows = results_main[results_main["strategy"] == "ridge"]
PERIOD2_START = pd.to_datetime(ridge_rows["date"]).min()
PERIOD2_STRATS = list(STRATEGY_ORDER)
print(f"Period 2 starts {PERIOD2_START.date()}")

period2_table = _period_metrics_table(results_main, PERIOD2_STRATS, PERIOD2_START)
period2_table

# %%
period2_panel = results_main[
    results_main["strategy"].isin(PERIOD2_STRATS)
    & (results_main["date"] >= PERIOD2_START)
]
_plot_cum_returns(
    period2_panel,
    f"Cumulative long-short return — {PERIOD2_START.date()} onwards (all 6 strategies)",
    OUTPUT_DIR / "99_v2_cum_returns_period2_ridge.png",
)

# %%
_plot_drawdowns(
    period2_panel,
    f"Drawdown — {PERIOD2_START.date()} onwards (all 6 strategies)",
    OUTPUT_DIR / "99_v2_drawdown_period2_ridge.png",
)


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
