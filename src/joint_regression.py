"""Fama-MacBeth joint regression of forward returns on the five signals.

Per spec section 9:
    1. At each rebalance date m, z-score every signal across the
       cross-section so coefficients are comparable in magnitude.
    2. Run a monthly OLS:
           fwd_ret_21d ~ z_anchor + z_ridge + z_lm + z_mom + z_rev
       (intercept included).
    3. Time-series average each coefficient. Report Newey-West standard
       errors via statsmodels (HAC, lag = 6).
    4. Also report the time-series average R^2 across monthly fits.

Tests whether the LLM signals add information beyond momentum and
analyst revisions after the other factors are partialled out at the
cross-sectional level.

If any signal is missing from the panel (e.g. transcript-derived
signals not yet built), the regression silently omits it. The signal
list and coefficient names in the output reflect what was actually
used.

Output:
    _data/fm_results.json
        {
          "signals": ["sig_mom", "sig_rev", ...],
          "n_months": int,
          "beta": {sig: mean_coef, ...},
          "nw_se": {sig: standard error, ...},
          "nw_tstat": {sig: t statistic, ...},
          "mean_r2": float,
          "alpha": {beta, se, t}
        }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
PANEL_FILENAME = "signal_panel_monthly.parquet"
OUTPUT_FILENAME = "fm_results.json"
NW_LAGS = 6
RETURN_COL = "fwd_ret_21d"
DATE_COL = "date"
ALL_SIGNALS: list[str] = ["sig_anchor", "sig_ridge", "sig_lm", "sig_mom", "sig_rev", "sig_car3"]


def _z_score(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = group.copy()
    for c in cols:
        x = out[c].astype(float)
        mu = x.mean()
        sd = x.std(ddof=1)
        out[c] = (x - mu) / sd if sd and sd > 0 else 0.0
    return out


def _fit_one_month(df: pd.DataFrame, signals: list[str]) -> dict | None:
    """OLS fit at one rebalance date. Returns coef dict + R^2 or None."""
    sub = df.dropna(subset=signals + [RETURN_COL])
    if len(sub) < len(signals) + 5:
        return None

    sub = _z_score(sub, signals)
    if any(sub[s].nunique() < 2 for s in signals):
        return None

    X = sm.add_constant(sub[signals].to_numpy(), has_constant="add")
    y = sub[RETURN_COL].to_numpy()
    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    coefs = {"const": float(model.params[0])}
    coefs.update({s: float(model.params[i + 1]) for i, s in enumerate(signals)})
    return {"coefs": coefs, "r2": float(model.rsquared), "n": int(len(sub))}


def run_fm(panel: pd.DataFrame, min_avg_obs: int = 20) -> dict:
    """Fama-MacBeth on whichever signals are present *with cross-sectional
    coverage* in the panel.

    A signal is only included if its average number of non-null rows per
    rebalance date is at least ``min_avg_obs``. This prevents an AAPL-only
    or otherwise sparse signal from collapsing the joint dropna step to
    zero usable rows per month.
    """
    candidates = [s for s in ALL_SIGNALS if s in panel.columns and panel[s].notna().any()]
    avg_obs = (
        panel.groupby("date")[candidates].apply(lambda g: g.notna().sum())
        if candidates else pd.DataFrame()
    )
    signals = [s for s in candidates if not avg_obs.empty and avg_obs[s].mean() >= min_avg_obs]
    dropped = sorted(set(candidates) - set(signals))
    if dropped:
        print(f"  [drop] sparse signals (avg < {min_avg_obs} obs/month): {dropped}")
    if not signals:
        raise ValueError("no signals meet the cross-sectional coverage threshold")

    monthly_coefs: list[dict] = []
    monthly_r2: list[float] = []

    for date, group in panel.groupby(DATE_COL, sort=True):
        fit = _fit_one_month(group, signals)
        if fit is None:
            continue
        row = {"date": date, **fit["coefs"]}
        monthly_coefs.append(row)
        monthly_r2.append(fit["r2"])

    if not monthly_coefs:
        raise RuntimeError("Fama-MacBeth produced zero valid monthly fits")

    fm = pd.DataFrame(monthly_coefs).set_index("date").sort_index()

    out = {
        "signals": signals,
        "n_months": int(len(fm)),
        "mean_r2": float(np.mean(monthly_r2)),
        "beta": {},
        "nw_se": {},
        "nw_tstat": {},
        "alpha": {},
    }

    for col in fm.columns:
        series = fm[col].astype(float).dropna()
        if series.empty:
            continue
        mean = float(series.mean())
        X_const = np.ones(len(series))
        nw = sm.OLS(series.values, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
        se = float(nw.bse[0])
        t = mean / se if se > 0 else float("nan")
        if col == "const":
            out["alpha"] = {"beta": mean, "se": se, "t": t}
        else:
            out["beta"][col] = mean
            out["nw_se"][col] = se
            out["nw_tstat"][col] = t

    return out


def main(data_dir: Path = DATA_DIR) -> None:
    panel_path = data_dir / PANEL_FILENAME
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found. Run `python src/build_signal_panel.py` first."
        )
    panel = pd.read_parquet(panel_path)
    result = run_fm(panel)
    out_path = data_dir / OUTPUT_FILENAME
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {out_path}")
    print(f"  signals  : {result['signals']}")
    print(f"  n_months : {result['n_months']}")
    print(f"  mean R^2 : {result['mean_r2']:.4f}")
    print(f"  alpha    : beta={result['alpha']['beta']:+.4f}  t={result['alpha']['t']:+.3f}")
    for s in result["signals"]:
        print(
            f"  {s:>10s} : beta={result['beta'][s]:+.4f}"
            f"  NW se={result['nw_se'][s]:.4f}"
            f"  t={result['nw_tstat'][s]:+.3f}"
        )


if __name__ == "__main__":
    main()
