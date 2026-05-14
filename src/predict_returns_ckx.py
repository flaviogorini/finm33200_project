"""CKX-style return-prediction model on the unified monthly panel.

Predicts forward 1-month and forward 3-month returns for each (date, ticker)
using expanding-window walk-forward CV with no in-sample data leak. Both
horizons are reported side by side so the reader can see which conclusions
are robust to the choice of target.

Five variants, ALL trained on the same 13-ticker pooled cross-section and
evaluated on identical test rows so that each step measures a clean
incremental hypothesis:

    V0a — Zero forecast (predict 0). The "is anything > 0?" sanity check.
    V0b — Momentum-only Ridge / GBR on trailing-return columns alone.
          The academic bar (Jegadeesh-Titman 1993 momentum).
    V1  — V0b features + fundamentals (incl. YoY/QoQ growth) + Bloomberg
          consensus + macro. "Does fundamentals+macro add to momentum?"
    V2  — V1 + earnings-call sentiment. "Does call sentiment add?"
    V3  — V2 + SEC 10-Q text features. "Does 10-Q disclosure text add?"

Each non-V0a variant runs both Ridge and Gradient Boosting. Two heads per
regressor: classification ``P(y > 0)`` (AUC, accuracy) + regression (OOS R²,
Spearman IC). Long-short tertile portfolio backtest applies the SAME rule
across V0b/V1/V2/V3; V0a falls back to equal-weight buy-and-hold because
zero predictions can't rank tickers.

Outputs (under ``_output/``):
    ckx_predictions.parquet
        [date, ticker, variant, model, target, fold, y_true, y_pred, p_up]
    ckx_metrics.json
        {target: {variant: {model: {auc, accuracy, oos_r2, ic_spearman, n,
                                     auc_aapl, oos_r2_aapl, ...}}}}
    ckx_portfolio.parquet
        [date, variant, model, target, strategy, ret, cum_ret]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from build_panel import LABEL_COLS, load_panel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Both horizons are reported side by side. fwd_ret_1m is the headline
# (non-overlapping labels → cleaner statistics). fwd_ret_3m is retained
# for continuity with the teammate's original design and to show whether
# conclusions are horizon-robust.
TARGET_COLS: tuple[str, ...] = ("fwd_ret_1m", "fwd_ret_3m")

# Feature column groups. Variant-builders compose subsets of these.
TRAILING_RET_COLS = ("ret_1m", "ret_3m", "ret_6m", "ret_12m")
FUNDAMENTAL_LEVEL_COLS = (
    "px_last", "pe_ratio", "revenue", "net_income", "net_debt", "ebitda",
)
# Novy-Marx-style fundamental momentum (added in build_fundamentals_features.py).
FUNDAMENTAL_GROWTH_COLS = (
    "revenue_yoy", "revenue_qoq",
    "net_income_yoy", "net_income_qoq",
    "ebitda_yoy", "ebitda_qoq",
)
CONSENSUS_COLS = (
    "best_pe_ratio", "best_sales", "best_net_income", "best_net_debt", "best_ebitda",
)
MACRO_COLS = (
    "vix", "treas_10y", "treas_2y", "treas_30y", "treas_5y",
    "yield_curve_2s10s", "yield_curve_2s30s", "breakeven_10y",
    "dxy", "eurusd_iv_3m", "wti_front",
)
SENTIMENT_COLS = ("sentiment_diff", "sentiment_diff_qoq", "days_since_earnings")
# 10-Q lexicon + bag-of-words similarity. Always present after running
# `doit process_10q` (no OpenAI key needed).
SEC_10Q_CORE_COLS = (
    "10q_sentiment", "10q_positive_rate", "10q_negative_rate",
    "10q_uncertainty", "10q_litigious", "10q_constraining",
    "10q_word_count", "10q_cosine_vs_previous", "10q_change_vs_previous",
)
# Embedding-derived similarity. Optional; activated when the embed stage
# has been run (needs OPENAI_API_KEY).
SEC_10Q_EMBED_COLS = (
    "10q_embedding_cosine_vs_previous", "10q_embedding_change_vs_previous",
)


# ---- Zero predictor (V0a) -----------------------------------------------


class ZeroPredictor:
    """Predicts zero for every row. Used as the V0a sanity-check baseline."""

    def fit(self, X, y):  # noqa: D401
        return self

    def predict(self, X):
        return np.zeros(len(X))


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    features: tuple[str, ...]
    models: tuple[str, ...]  # subset of model_factories() keys


def variants_for_panel(panel_columns: list[str]) -> list[Variant]:
    """Define all five variants. Auto-skips V2/V3 if their cols aren't in panel."""
    momentum = TRAILING_RET_COLS
    fundamentals_macro = (
        FUNDAMENTAL_LEVEL_COLS
        + tuple(c for c in FUNDAMENTAL_GROWTH_COLS if c in panel_columns)
        + CONSENSUS_COLS
        + MACRO_COLS
    )
    v1_features = momentum + fundamentals_macro

    out = [
        Variant(
            name="v0a",
            description="Zero forecast (sanity check).",
            features=momentum,  # for row-filtering parity with V0b
            models=("zero",),
        ),
        Variant(
            name="v0b",
            description="Momentum-only Ridge/GBR (Jegadeesh-Titman bar).",
            features=momentum,
            models=("ridge", "gbr"),
        ),
        Variant(
            name="v1",
            description="V0b + fundamentals(+growth) + consensus + macro.",
            features=v1_features,
            models=("ridge", "gbr"),
        ),
    ]

    if all(c in panel_columns for c in SENTIMENT_COLS):
        out.append(
            Variant(
                name="v2",
                description="V1 + earnings-call sentiment.",
                features=v1_features + SENTIMENT_COLS,
                models=("ridge", "gbr"),
            )
        )
    else:
        print("  note: V2 skipped — sentiment cols missing from panel.")

    if all(c in panel_columns for c in SEC_10Q_CORE_COLS):
        extra = tuple(c for c in SEC_10Q_EMBED_COLS if c in panel_columns)
        sent_cols = SENTIMENT_COLS if all(c in panel_columns for c in SENTIMENT_COLS) else ()
        out.append(
            Variant(
                name="v3",
                description=(
                    "V2 + 10-Q text"
                    + (" (incl. embeddings)" if extra else " (lexicon only)")
                    + "."
                ),
                features=v1_features + sent_cols + SEC_10Q_CORE_COLS + extra,
                models=("ridge", "gbr"),
            )
        )
    else:
        print(
            "  note: V3 skipped — 10-Q core columns not in panel. "
            "Run `doit pull:sec_10q_filings && doit process_10q` first."
        )
    return out


# ---- Feature matrix construction ----------------------------------------


def build_feature_matrix(
    panel: pd.DataFrame, variant: Variant, target_col: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Slice the panel by variant; drop rows with any-NaN feature/target.

    Returns (X, y, meta) where meta carries [date, ticker] aligned with X.
    Hard-asserts that no fwd_* column leaks into X.
    """
    cols = list(variant.features)
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise ValueError(f"variant {variant.name}: features missing from panel: {missing}")

    keep = ["date", "ticker", target_col] + cols
    df = panel[keep].dropna()

    leak = [c for c in cols if c.startswith("fwd_") or c in LABEL_COLS]
    if leak:
        raise AssertionError(f"variant {variant.name}: forward-label leak in features: {leak}")

    X = df[cols].copy()
    y = df[target_col].copy()
    meta = df[["date", "ticker"]].copy()
    return X, y, meta


# ---- Walk-forward CV -----------------------------------------------------


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    train_end_dates: list[pd.Timestamp],
    model_factory: Callable,
) -> pd.DataFrame:
    """Expanding-window walk-forward CV.

    For each cutoff ``train_end``:
        train = rows with date <= train_end
        test  = rows with train_end < date <= train_end + 1 year
    Z-score fits on train slice only; applied to test slice.
    """
    out_rows: list[pd.DataFrame] = []
    dates = meta["date"]
    for fold_idx, train_end in enumerate(train_end_dates):
        train_mask = dates <= train_end
        test_end = train_end + pd.DateOffset(years=1)
        test_mask = (dates > train_end) & (dates <= test_end)
        if train_mask.sum() < 30 or test_mask.sum() == 0:
            continue

        scaler = StandardScaler().fit(X.loc[train_mask].values)
        X_train = scaler.transform(X.loc[train_mask].values)
        X_test = scaler.transform(X.loc[test_mask].values)

        model = model_factory()
        model.fit(X_train, y.loc[train_mask].values)
        y_pred = model.predict(X_test)

        out_rows.append(
            pd.DataFrame(
                {
                    "date": dates.loc[test_mask].values,
                    "ticker": meta["ticker"].loc[test_mask].values,
                    "fold": fold_idx,
                    "train_end": train_end,
                    "y_true": y.loc[test_mask].values,
                    "y_pred": y_pred,
                }
            )
        )

    if not out_rows:
        return pd.DataFrame(
            columns=["date", "ticker", "fold", "train_end", "y_true", "y_pred"]
        )
    return pd.concat(out_rows, ignore_index=True)


def add_p_up(predictions: pd.DataFrame) -> pd.DataFrame:
    """Convert continuous y_pred → calibrated up-probability via per-fold
    z-score then sigmoid. For ZeroPredictor (all preds == 0), p_up = 0.5."""
    out_parts: list[pd.DataFrame] = []
    for fold, grp in predictions.groupby("fold", sort=False):
        sd = grp["y_pred"].std()
        if sd == 0 or not np.isfinite(sd):
            g = grp.copy()
            g["p_up"] = 0.5
            out_parts.append(g)
            continue
        mu = grp["y_pred"].mean()
        z = (grp["y_pred"] - mu) / sd
        p = 1.0 / (1.0 + np.exp(-z))
        g = grp.copy()
        g["p_up"] = p.values
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


# ---- Evaluation ---------------------------------------------------------


def _eval_block(predictions: pd.DataFrame, suffix: str = "") -> dict:
    if predictions.empty:
        return {f"n{suffix}": 0}
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    p_up = predictions["p_up"].to_numpy()
    y_bin = (y_true > 0).astype(int)

    auc = float(roc_auc_score(y_bin, p_up)) if len(np.unique(y_bin)) > 1 else float("nan")
    acc = float(accuracy_score(y_bin, (p_up > 0.5).astype(int)))
    # r2_score is undefined when all y_pred are identical; guard.
    if np.allclose(y_pred, y_pred[0]):
        r2 = 0.0  # constant predictor → variance-explained = 0
    else:
        r2 = float(r2_score(y_true, y_pred))
    rho_arr = spearmanr(y_true, y_pred)
    rho = float(rho_arr[0]) if rho_arr[0] == rho_arr[0] else float("nan")
    return {
        f"n{suffix}": int(len(predictions)),
        f"auc{suffix}": auc,
        f"accuracy{suffix}": acc,
        f"oos_r2{suffix}": r2,
        f"ic_spearman{suffix}": rho,
    }


def evaluate(predictions: pd.DataFrame) -> dict:
    """Pooled metrics (all tickers) + AAPL-only continuity metrics."""
    out = _eval_block(predictions, "")
    aapl = predictions[predictions["ticker"] == "AAPL"]
    out.update(_eval_block(aapl, "_aapl"))
    return out


# ---- Portfolio backtests ------------------------------------------------


def _target_horizon_months(target_col: str) -> int:
    """Return the forward horizon in months for the /horizon smearing."""
    if target_col == "fwd_ret_1m":
        return 1
    if target_col == "fwd_ret_3m":
        return 3
    raise ValueError(f"Unknown target horizon: {target_col}")


def portfolio_long_short_tertile(
    predictions: pd.DataFrame, *, target_col: str, top_frac: float = 1 / 3
) -> pd.DataFrame:
    """Cross-sectional long-short. Each month, long top tertile, short
    bottom tertile (equal-weight within each side). Strategy return =
    (long_avg - short_avg) divided by horizon-in-months for smearing of
    overlapping holding periods (3M target → 1/3 each month)."""
    horizon = _target_horizon_months(target_col)
    rows: list[dict] = []
    for date, grp in predictions.groupby("date"):
        if grp["ticker"].nunique() < 4:
            continue
        ranked = grp.sort_values("y_pred")
        n = len(ranked)
        k = max(1, int(n * top_frac))
        bottom = ranked.head(k)["y_true"].mean()
        top = ranked.tail(k)["y_true"].mean()
        rows.append({"date": date, "ret": (top - bottom) / horizon})
    if not rows:
        return pd.DataFrame(columns=["date", "strategy", "ret", "cum_ret"])
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["strategy"] = "long_short_tertile"
    df["cum_ret"] = (1.0 + df["ret"]).cumprod() - 1.0
    return df


def portfolio_buy_and_hold(
    predictions: pd.DataFrame, *, target_col: str
) -> pd.DataFrame:
    """Equal-weight buy-and-hold across all tickers each month — used for
    V0a where zero predictions can't rank. This is the benchmark the long-
    short strategy needs to beat (in risk-adjusted terms)."""
    horizon = _target_horizon_months(target_col)
    rows: list[dict] = []
    for date, grp in predictions.groupby("date"):
        if grp["ticker"].nunique() < 1:
            continue
        rows.append({"date": date, "ret": float(grp["y_true"].mean()) / horizon})
    if not rows:
        return pd.DataFrame(columns=["date", "strategy", "ret", "cum_ret"])
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["strategy"] = "equal_weight_buy_hold"
    df["cum_ret"] = (1.0 + df["ret"]).cumprod() - 1.0
    return df


# ---- Main orchestrator --------------------------------------------------


def model_factories() -> dict[str, Callable]:
    return {
        "zero": lambda: ZeroPredictor(),
        "ridge": lambda: Ridge(alpha=1.0),
        "gbr": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
        ),
    }


def default_train_end_dates(panel: pd.DataFrame, target_col: str) -> list[pd.Timestamp]:
    start_year = 2014
    # Need at least one full year of forward returns past the cutoff.
    last_year = (panel["date"].max() - pd.DateOffset(years=1)).year
    return [pd.Timestamp(f"{y}-12-31") for y in range(start_year, last_year + 1)]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    panel_cols = list(panel.columns)
    tickers = sorted(panel["ticker"].unique())
    print(f"Panel: {len(panel)} rows, {len(tickers)} tickers, cols={len(panel_cols)}")

    variants = variants_for_panel(panel_cols)
    factories = model_factories()

    all_preds: list[pd.DataFrame] = []
    all_portfolios: list[pd.DataFrame] = []
    metrics_out: dict = {}

    for target_col in TARGET_COLS:
        if target_col not in panel.columns:
            print(f"\n[target {target_col}] missing from panel — skipping")
            continue
        train_ends = default_train_end_dates(panel, target_col)
        print(f"\n=== Target: {target_col} ({len(train_ends)} CV folds) ===")
        metrics_out[target_col] = {}

        for v in variants:
            X, y, meta = build_feature_matrix(panel, v, target_col)
            print(f"\n  variant {v.name} — {v.description}")
            print(f"    rows after dropna: {len(X)}, features: {len(v.features)}")

            metrics_out[target_col][v.name] = {
                "description": v.description,
                "models": {},
            }
            for model_name in v.models:
                preds = walk_forward_cv(X, y, meta, train_ends, factories[model_name])
                preds = add_p_up(preds)
                preds["variant"] = v.name
                preds["model"] = model_name
                preds["target"] = target_col

                metrics = evaluate(preds)
                metrics_out[target_col][v.name]["models"][model_name] = metrics
                print(
                    f"    {model_name:6s}  n={metrics.get('n',0):4d}  "
                    f"AUC={metrics.get('auc',float('nan')):.3f}  "
                    f"acc={metrics.get('accuracy',float('nan')):.3f}  "
                    f"R2={metrics.get('oos_r2',float('nan')):+.3f}  "
                    f"IC={metrics.get('ic_spearman',float('nan')):+.3f}  "
                    f"| AAPL: AUC={metrics.get('auc_aapl',float('nan')):.3f}  "
                    f"R2={metrics.get('oos_r2_aapl',float('nan')):+.3f}"
                )
                all_preds.append(preds)

                # Portfolio backtest: SAME long-short rule for every variant
                # with informative predictions; V0a uses buy-and-hold.
                if v.name == "v0a":
                    pf = portfolio_buy_and_hold(preds, target_col=target_col)
                else:
                    pf = portfolio_long_short_tertile(preds, target_col=target_col)
                if not pf.empty:
                    pf["variant"] = v.name
                    pf["model"] = model_name
                    pf["target"] = target_col
                    all_portfolios.append(pf)

    # Persist outputs.
    pred_path = OUTPUT_DIR / "ckx_predictions.parquet"
    pd.concat(all_preds, ignore_index=True).to_parquet(pred_path, index=False)

    metrics_path = OUTPUT_DIR / "ckx_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2, default=str))

    portfolio_path = OUTPUT_DIR / "ckx_portfolio.parquet"
    if all_portfolios:
        pd.concat(all_portfolios, ignore_index=True).to_parquet(portfolio_path, index=False)

    print(f"\nWrote predictions -> {pred_path}")
    print(f"Wrote metrics     -> {metrics_path}")
    if all_portfolios:
        print(f"Wrote portfolios  -> {portfolio_path}")


if __name__ == "__main__":
    main()
