"""Nested α progression for the §5.6 time-series factor regression.

For each text strategy in {Ridge, Anchor, LM}, run 4 nested OLS regressions
of the strategy's monthly long-short return on a growing set of factor
controls:

    α₀: LS_t = α + β·FF5_t                              + ε
    α₁: LS_t = α + β·FF5_t + γ·Mom_t                    + ε
    α₂: LS_t = α + β·FF5_t + γ·Mom_t + δ·CAR3_t         + ε
    α₃: LS_t = α + β·FF5_t + γ·Mom_t + δ·CAR3_t + ζ·Rev_t + ε

where Mom, CAR3, Rev are the corresponding strategies' monthly LS returns
used as factor-mimicking portfolios. FF5 = (Mkt-RF, SMB, HML, RMW, CMA)
from Ken French.

Conventions:
- HAC SE (Newey-West, lag 6) — consistent with §5.5 cross-sectional FM.
- No RF subtraction on LHS: LS portfolios are zero-cost.
- Timestamp alignment: strategy LS at signal-date T joined with FF5 row
  for the next month (the period the LS actually earned in). Implemented
  by shifting FF5 dates back one business-month-end. Residual ~1-2 BD
  boundary mismatch is documented in the writeup, not corrected.

Sample windows:
- own_history: each strategy's own backtest start.
- post_2018:   restrict to date >= 2018-12-31 (mirrors existing post2018 spec).

Outputs:
    _data/strategy_factor_returns_monthly.parquet
        The aligned monthly frame: date, FF5 cols, and each strategy's LS.
    _data/factor_alpha.json
        Nested α tables. Schema:
        {
          "own_history": {
            "ridge":  {"alpha_0": {alpha, t, r2, n}, "alpha_1": ..., ...},
            "anchor": ...,
            "lm":     ...
          },
          "post_2018": {... same shape ...}
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

FF5_FILENAME = "ff5_monthly.parquet"
RESULTS_FILENAME = "results_main.parquet"

ASSEMBLED_FILENAME = "strategy_factor_returns_monthly.parquet"
ALPHA_OUTPUT_FILENAME = "factor_alpha.json"

NW_LAGS = 6

# Strategies appearing as LHS in §5.6 — the three "text" candidates.
TEXT_STRATEGIES: list[str] = ["ridge", "anchor", "lm"]

# Non-text strategies whose LS series enter as RHS factors.
FACTOR_STRATEGIES: list[str] = ["momentum", "car3", "revisions"]

FF5_COLS: list[str] = ["mkt_rf", "smb", "hml", "rmw", "cma"]

# Nested factor specs. Each entry: (key, list_of_factor_strategies_to_add).
# FF5 is always included; this list defines what's ADDED on top.
SPECS: list[tuple[str, list[str]]] = [
    ("alpha_0", []),
    ("alpha_1", ["momentum"]),
    ("alpha_2", ["momentum", "car3"]),
    ("alpha_3", ["momentum", "car3", "revisions"]),
]

# Post-2018 cutoff — matches run_backtests.SPECIFICATIONS["post2018"]["start_date"].
POST_2018_CUTOFF = "2018-12-31"


def load_ff5_shifted(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load FF5 and shift dates back one business-month-end so that the
    FF5 row originally for month M (e.g. 2024-02-29) is keyed at the
    PREVIOUS month-end (2024-01-31), aligning with strategy LS at
    signal-date T = 2024-01-31 (which earned during Feb)."""
    ff5 = pd.read_parquet(data_dir / FF5_FILENAME)
    ff5["date"] = pd.to_datetime(ff5["date"])
    ff5 = ff5.sort_values("date").reset_index(drop=True)
    ff5["date"] = ff5["date"] - pd.offsets.BMonthEnd(1)
    return ff5[["date", *FF5_COLS]]


def load_strategy_ls(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Pivot results_main.parquet to wide form: date × strategy → ret_ls.

    Returns a DataFrame with one row per signal date and one column per
    strategy holding that strategy's monthly long-short return.
    """
    results = pd.read_parquet(data_dir / RESULTS_FILENAME)
    results["date"] = pd.to_datetime(results["date"])
    wide = results.pivot_table(
        index="date", columns="strategy", values="ret_ls", aggfunc="first"
    ).reset_index()
    return wide


def assemble_factor_returns(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Join FF5 (shifted) with each strategy's LS return at the signal date.

    Output columns: date, mkt_rf, smb, hml, rmw, cma, <strategy>_strat ...
    """
    ff5 = load_ff5_shifted(data_dir)
    ls = load_strategy_ls(data_dir)

    rename = {s: f"{s}_strat" for s in ls.columns if s != "date"}
    ls = ls.rename(columns=rename)

    merged = ls.merge(ff5, on="date", how="left")
    return merged.sort_values("date").reset_index(drop=True)


def _fit_one(
    df: pd.DataFrame,
    lhs_col: str,
    rhs_cols: list[str],
) -> dict | None:
    """OLS with HAC (NW lag 6). Returns alpha, t-stat, R², n, plus the full
    factor β / SE / t-stat dict so the writeup can render the entire
    regression — not just α."""
    sub = df.dropna(subset=[lhs_col, *rhs_cols])
    if len(sub) < len(rhs_cols) + 5:
        return None

    y = sub[lhs_col].to_numpy()
    X = sm.add_constant(sub[rhs_cols].to_numpy(), has_constant="add")
    try:
        model = sm.OLS(y, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": NW_LAGS}
        )
    except Exception:
        return None

    alpha = float(model.params[0])
    se_alpha = float(model.bse[0])
    t_alpha = alpha / se_alpha if se_alpha > 0 else float("nan")

    betas: dict[str, dict[str, float]] = {}
    for i, col in enumerate(rhs_cols):
        b = float(model.params[i + 1])
        s = float(model.bse[i + 1])
        t = b / s if s > 0 else float("nan")
        betas[col] = {"beta": b, "se": s, "t": t}

    return {
        "alpha": alpha,
        "se_alpha": se_alpha,
        "t": t_alpha,
        "r2": float(model.rsquared),
        "n": int(len(sub)),
        "betas": betas,
    }


def run_nested_for_strategy(
    frame: pd.DataFrame,
    strategy: str,
) -> dict[str, dict | None]:
    """All 4 nested α specs for one LHS strategy."""
    lhs = f"{strategy}_strat"
    if lhs not in frame.columns:
        return {key: None for key, _ in SPECS}
    out: dict[str, dict | None] = {}
    for key, add_factors in SPECS:
        rhs = list(FF5_COLS) + [f"{s}_strat" for s in add_factors]
        # Drop self-as-factor (defensive; not expected to occur for the
        # three text strategies which are not in FACTOR_STRATEGIES).
        rhs = [c for c in rhs if c != lhs]
        out[key] = _fit_one(frame, lhs, rhs)
    return out


def run_all(frame: pd.DataFrame) -> dict[str, dict[str, dict | None]]:
    """All 3 LHS strategies × 4 nested specs."""
    return {s: run_nested_for_strategy(frame, s) for s in TEXT_STRATEGIES}


def run_all_with_samples(frame: pd.DataFrame) -> dict[str, dict]:
    """Run nested α on both own-history and post-2018 sample windows."""
    own = run_all(frame)
    post = run_all(frame[frame["date"] >= pd.Timestamp(POST_2018_CUTOFF)])
    return {"own_history": own, "post_2018": post}


def main(data_dir: Path = DATA_DIR) -> None:
    ff5_path = data_dir / FF5_FILENAME
    res_path = data_dir / RESULTS_FILENAME
    if not ff5_path.exists():
        raise FileNotFoundError(
            f"{ff5_path} not found. Run `python src/build_factor_baseline.py` first."
        )
    if not res_path.exists():
        raise FileNotFoundError(
            f"{res_path} not found. Run `python src/run_backtests.py` first."
        )

    frame = assemble_factor_returns(data_dir)
    out_parquet = data_dir / ASSEMBLED_FILENAME
    frame.to_parquet(out_parquet, index=False)
    print(f"Wrote {len(frame):,} aligned monthly rows -> {out_parquet}")
    print(f"  columns: {list(frame.columns)}")
    print(
        f"  date range: {frame['date'].min().date()} -> "
        f"{frame['date'].max().date()}"
    )

    results = run_all_with_samples(frame)
    out_json = data_dir / ALPHA_OUTPUT_FILENAME
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote nested α results -> {out_json}")

    for sample, table in results.items():
        print(f"\n[{sample}]")
        for strategy in TEXT_STRATEGIES:
            row = table[strategy]
            cells = []
            for spec_key, _ in SPECS:
                cell = row[spec_key]
                if cell is None:
                    cells.append(f"{spec_key}: n/a")
                else:
                    cells.append(
                        f"{spec_key}: α={cell['alpha']:+.4f} "
                        f"t={cell['t']:+.2f} n={cell['n']}"
                    )
            print(f"  {strategy:>7s}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
