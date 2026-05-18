"""Cross-sectional quintile long-short backtest engine and IC computation.

All five strategies in the project (anchor-cosine, ridge, LM, momentum,
analyst revisions) feed into ``run_backtest``. The function is purely
mechanical: it takes a monthly panel with a signal column and a 21-bday
forward-return column, ranks the cross-section at each rebalance date,
forms a top-20 long / bottom-20 short equal-weight portfolio, and reports
the monthly return series plus aggregate metrics.

``compute_ic`` is the corresponding cross-sectional Spearman rank
correlation between signal and forward return, computed at each
rebalance date.

Backtest convention (matches spec section 6 and project calendar
conventions):
    - Rebalance frequency: monthly (last business day of month).
    - Long: top 20 of 100 tickers by signal value.
    - Short: bottom 20.
    - Weighting: equal within each leg (5% per stock per leg).
    - Holding period: 21 trading days — already baked into ``fwd_ret_21d``.
    - Transaction costs: zero.
    - If a ticker is missing the signal at month m, it is excluded from
      that month's ranking only.

Robustness filters supported via the ``filters`` argument:
    - ``drop_stale_gt`` (int, days): drop rows with
      ``days_since_earnings > N`` *calendar* days. Calendar days because
      the spec phrases section 8 in calendar terms.
    - ``start_date`` (str or Timestamp): keep only rebalance dates
      ``>= start_date``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LONG_QUANTILE = 0.80
SHORT_QUANTILE = 0.20
MIN_OBS_PER_MONTH = 40
RETURN_COL = "fwd_ret_21d"
DATE_COL = "date"
TICKER_COL = "ticker"
ANNUALISATION = 12


@dataclass
class BacktestResult:
    monthly: pd.DataFrame  # date, n_long, n_short, ret_long, ret_short, ret_ls, ret_bench
    metrics: dict[str, Any] = field(default_factory=dict)


def _apply_filters(panel: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if not filters:
        return panel
    out = panel
    if "drop_stale_gt" in filters and filters["drop_stale_gt"] is not None:
        thresh = int(filters["drop_stale_gt"])
        if "days_since_earnings" in out.columns:
            mask = out["days_since_earnings"].isna() | (out["days_since_earnings"] <= thresh)
            out = out[mask]
    if "start_date" in filters and filters["start_date"] is not None:
        out = out[out[DATE_COL] >= pd.Timestamp(filters["start_date"])]
    return out


def _portfolio_row(group: pd.DataFrame, sig_col: str) -> dict | None:
    """One rebalance-date row: long-short portfolio return + benchmark."""
    sub = group.dropna(subset=[sig_col, RETURN_COL])
    if len(sub) < MIN_OBS_PER_MONTH:
        return None

    n = len(sub)
    n_legs = max(int(round(n * 0.20)), 1)

    ranked = sub.sort_values(sig_col)
    short_leg = ranked.head(n_legs)
    long_leg = ranked.tail(n_legs)

    ret_long = float(long_leg[RETURN_COL].mean())
    ret_short = float(short_leg[RETURN_COL].mean())
    ret_bench = float(sub[RETURN_COL].mean())

    return {
        DATE_COL: group[DATE_COL].iloc[0],
        "n_obs": n,
        "n_long": n_legs,
        "n_short": n_legs,
        "ret_long": ret_long,
        "ret_short": ret_short,
        "ret_ls": ret_long - ret_short,
        "ret_bench": ret_bench,
    }


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    cum = (1.0 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())


def _summary_metrics(monthly: pd.DataFrame) -> dict[str, float]:
    if monthly.empty:
        return {}
    ls = monthly["ret_ls"].dropna()
    bench = monthly["ret_bench"].dropna()

    ann_ret = float(ls.mean() * ANNUALISATION)
    ann_vol = float(ls.std(ddof=1) * np.sqrt(ANNUALISATION))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    hit_rate = float((ls > 0).mean())
    max_dd = _max_drawdown(ls)

    aligned = monthly.dropna(subset=["ret_ls", "ret_bench"])
    excess = aligned["ret_ls"] - aligned["ret_bench"]
    ir_vs_bench = float(
        excess.mean() * ANNUALISATION / (excess.std(ddof=1) * np.sqrt(ANNUALISATION))
    ) if excess.std(ddof=1) > 0 else float("nan")

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
        "ir_vs_eqweight_bench": ir_vs_bench,
        "n_months": int(len(ls)),
    }


def run_backtest(
    panel: pd.DataFrame,
    sig_col: str,
    filters: dict | None = None,
) -> BacktestResult:
    """Run a quintile long-short backtest on one signal column.

    Parameters
    ----------
    panel
        Long-format monthly panel with at minimum ``date``, ``ticker``,
        ``sig_col``, and ``fwd_ret_21d`` columns.
    sig_col
        Name of the signal column to rank cross-sectionally.
    filters
        Optional dict, see module docstring for supported keys.
    """
    if sig_col not in panel.columns:
        raise KeyError(f"signal column {sig_col!r} not in panel")

    work = _apply_filters(panel, filters)
    rows: list[dict] = []
    for _, group in work.groupby(DATE_COL, sort=True):
        row = _portfolio_row(group, sig_col)
        if row is not None:
            rows.append(row)

    monthly = pd.DataFrame(rows)
    metrics = _summary_metrics(monthly)
    return BacktestResult(monthly=monthly, metrics=metrics)


def compute_ic(
    panel: pd.DataFrame,
    sig_col: str,
    ret_col: str = RETURN_COL,
    filters: dict | None = None,
) -> pd.DataFrame:
    """Cross-sectional Spearman IC at each rebalance date."""
    if sig_col not in panel.columns or ret_col not in panel.columns:
        raise KeyError(f"need both {sig_col!r} and {ret_col!r} in panel")

    work = _apply_filters(panel, filters)
    rows: list[dict] = []
    for date, group in work.groupby(DATE_COL, sort=True):
        sub = group.dropna(subset=[sig_col, ret_col])
        if len(sub) < 10:
            continue
        if sub[sig_col].nunique() < 2 or sub[ret_col].nunique() < 2:
            continue
        ic, _ = spearmanr(sub[sig_col], sub[ret_col])
        rows.append({DATE_COL: date, "ic": float(ic), "n_obs": int(len(sub))})
    return pd.DataFrame(rows)


def ic_summary(ic_series: pd.DataFrame) -> dict[str, float]:
    if ic_series.empty:
        return {}
    ic = ic_series["ic"].dropna()
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1))
    ic_ir = mean_ic / std_ic if std_ic > 0 else float("nan")
    return {
        "ic_mean": mean_ic,
        "ic_std": std_ic,
        "ic_ir": ic_ir,
        "ic_pos_frac": float((ic > 0).mean()),
        "ic_n_months": int(len(ic)),
    }
