"""CKX-style return-prediction model on the unified monthly panel.

Predicts forward 3-month return for each (date, ticker) using
expanding-window walk-forward CV with no in-sample data leak.

Three feature variants, evaluated head-to-head on AAPL test rows:

    V1 — Pooled, no text (baseline).
        All 13 tickers. Features: fundamentals + Bloomberg consensus +
        macro + trailing returns.
    V2 — AAPL-only, earnings-call sentiment added.
        V1 features + sentiment_diff, sentiment_diff_qoq, days_since_earnings.
    V3 — AAPL-only, earnings-call sentiment AND 10-Q text added.
        V2 features + the eleven 10q_* columns. Auto-skipped if the 10-Q
        panel hasn't been built yet (run `doit pull:sec_10q_filings &&
        doit process_10q` first; needs WRDS_PASSWORD).

Two heads from one regressor: classification ``P(fwd_ret_3m > 0)`` (AUC,
accuracy) + regression ``fwd_ret_3m`` (OOS R², Spearman IC). Long-short
portfolio backtest for V1 (cross-sectional). AAPL timing strategy for
V2 / V3 (long when ``P(up) > threshold``).

Outputs (under ``_output/``):
    ckx_predictions.parquet    [date, ticker, variant, model, y_true, y_pred, p_up, fold]
    ckx_metrics.json           {variant: {model: {auc, accuracy, oos_r2, ic_spearman, n}}}
    ckx_portfolio.parquet      [date, strategy, ret, cum_ret]
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

TARGET_COL = "fwd_ret_3m"

# Feature column groups. Variant-builders compose subsets of these.
FUNDAMENTAL_COLS = (
    "px_last", "pe_ratio", "revenue", "net_income", "net_debt", "ebitda",
)
CONSENSUS_COLS = (
    "best_pe_ratio", "best_sales", "best_net_income", "best_net_debt", "best_ebitda",
)
MACRO_COLS = (
    "vix", "treas_10y", "treas_2y", "treas_30y", "treas_5y",
    "yield_curve_2s10s", "yield_curve_2s30s", "breakeven_10y",
    "dxy", "eurusd_iv_3m", "wti_front",
)
TRAILING_RET_COLS = ("ret_1m", "ret_3m", "ret_6m", "ret_12m")
SENTIMENT_COLS = ("sentiment_diff", "sentiment_diff_qoq", "days_since_earnings")
# 10-Q lexicon + bag-of-words similarity. Always present after running
# `doit process_10q` (no OpenAI key needed).
SEC_10Q_CORE_COLS = (
    "10q_sentiment", "10q_positive_rate", "10q_negative_rate",
    "10q_uncertainty", "10q_litigious", "10q_constraining",
    "10q_word_count", "10q_cosine_vs_previous", "10q_change_vs_previous",
)
# Embedding-derived similarity. Only present when the optional embed
# stage has been run (`doit process_10q:embed`, requires OPENAI_API_KEY).
# V3 activates with just the core cols; embed cols are appended when present.
SEC_10Q_EMBED_COLS = (
    "10q_embedding_cosine_vs_previous", "10q_embedding_change_vs_previous",
)


@dataclass(frozen=True)
class Variant:
    name: str           # "v1" | "v2" | "v3"
    features: tuple[str, ...]
    tickers: tuple[str, ...] | None  # None = all tickers in panel
    description: str


def variants_for_panel(panel_columns: list[str], tickers_in_panel: list[str]) -> list[Variant]:
    """Define the three variants, auto-skipping V3 if 10-Q columns aren't present."""
    base_features = (
        FUNDAMENTAL_COLS + CONSENSUS_COLS + MACRO_COLS + TRAILING_RET_COLS
    )
    out = [
        Variant(
            name="v1",
            features=base_features,
            tickers=tuple(tickers_in_panel),
            description="Pooled, no text features (baseline).",
        )
    ]
    if all(c in panel_columns for c in SENTIMENT_COLS):
        out.append(
            Variant(
                name="v2",
                features=base_features + SENTIMENT_COLS,
                tickers=("AAPL",),
                description="AAPL-only, earnings-call sentiment added.",
            )
        )
    if all(c in panel_columns for c in SEC_10Q_CORE_COLS):
        extra = tuple(c for c in SEC_10Q_EMBED_COLS if c in panel_columns)
        out.append(
            Variant(
                name="v3",
                features=base_features + SENTIMENT_COLS + SEC_10Q_CORE_COLS + extra,
                tickers=("AAPL",),
                description=(
                    "AAPL-only, sentiment + 10-Q text"
                    + (" (incl. embeddings)" if extra else "")
                    + "."
                ),
            )
        )
    else:
        print(
            "  note: V3 skipped — 10-Q core columns not in panel. "
            "Run `doit pull:sec_10q_filings && doit process_10q` first, "
            "then rebuild the panel. "
            "(Embedding-similarity features are optional; "
            "add `doit process_10q:embed` for them.)"
        )
    return out


# ---- Feature matrix construction ----------------------------------------


def build_feature_matrix(
    panel: pd.DataFrame, variant: Variant
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Slice the panel by variant; drop rows with any-NaN feature/target.

    Returns (X, y, meta) where meta carries [date, ticker] aligned with X.
    Hard-asserts that no fwd_* column leaks into X.
    """
    df = panel.copy()
    if variant.tickers is not None:
        df = df[df["ticker"].isin(variant.tickers)]

    cols = list(variant.features)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"variant {variant.name}: features missing from panel: {missing}")

    keep = ["date", "ticker", TARGET_COL] + cols
    df = df[keep].dropna()

    leak = [c for c in cols if c.startswith("fwd_") or c in LABEL_COLS]
    if leak:
        raise AssertionError(f"variant {variant.name}: forward-label leak in features: {leak}")

    X = df[cols].copy()
    y = df[TARGET_COL].copy()
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

    For each cutoff `train_end`:
        train = rows with date <= train_end
        test  = rows with train_end < date <= train_end + 1 year
    Z-score fits on train slice only; applied to test slice.
    Returns DataFrame [date, ticker, fold, y_true, y_pred].
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
    """Convert continuous y_pred → calibrated up-probability via training-set
    z-score then sigmoid. Per fold so we don't peek across folds."""
    out_parts: list[pd.DataFrame] = []
    for fold, grp in predictions.groupby("fold", sort=False):
        mu = grp["y_pred"].mean()
        sd = grp["y_pred"].std() or 1.0
        z = (grp["y_pred"] - mu) / sd
        p = 1.0 / (1.0 + np.exp(-z))
        g = grp.copy()
        g["p_up"] = p.values
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


# ---- Evaluation ---------------------------------------------------------


def evaluate(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {"n": 0}
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    p_up = predictions["p_up"].to_numpy()
    y_bin = (y_true > 0).astype(int)

    auc = float(roc_auc_score(y_bin, p_up)) if len(np.unique(y_bin)) > 1 else float("nan")
    acc = float(accuracy_score(y_bin, (p_up > 0.5).astype(int)))
    r2 = float(r2_score(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "n": int(len(predictions)),
        "auc": auc,
        "accuracy": acc,
        "oos_r2": r2,
        "ic_spearman": float(rho) if rho == rho else float("nan"),
    }


# ---- Portfolio backtests ------------------------------------------------


def portfolio_backtest_long_short(
    predictions: pd.DataFrame, *, top_frac: float = 1 / 3
) -> pd.DataFrame:
    """Cross-sectional long-short. Each month, long top tertile, short bottom
    tertile (equal-weight within each side). Returns the strategy return for
    that month (= (long_avg - short_avg) of fwd_ret_3m, divided by 3 to get
    monthly equivalent, holding for 3 months smeared as 1/3 each month)."""
    rows: list[dict] = []
    for date, grp in predictions.groupby("date"):
        if grp["ticker"].nunique() < 4:
            continue
        ranked = grp.sort_values("y_pred")
        n = len(ranked)
        k = max(1, int(n * top_frac))
        bottom = ranked.head(k)["y_true"].mean()
        top = ranked.tail(k)["y_true"].mean()
        # 3m returns spread, smeared to monthly by /3.
        rows.append({"date": date, "ret": (top - bottom) / 3.0})
    if not rows:
        return pd.DataFrame(columns=["date", "strategy", "ret", "cum_ret"])
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["strategy"] = "long_short_v1"
    df["cum_ret"] = (1.0 + df["ret"]).cumprod() - 1.0
    return df


def portfolio_backtest_timing(
    predictions: pd.DataFrame, *, threshold: float = 0.55
) -> pd.DataFrame:
    """Single-ticker timing strategy. Long when p_up > threshold else cash.
    fwd_ret_3m / 3 for monthly equivalent."""
    if predictions.empty:
        return pd.DataFrame(columns=["date", "strategy", "ret", "cum_ret"])
    df = predictions.sort_values("date").reset_index(drop=True).copy()
    long_signal = df["p_up"] > threshold
    df["ret"] = np.where(long_signal, df["y_true"] / 3.0, 0.0)
    df["strategy"] = "timing"
    df["cum_ret"] = (1.0 + df["ret"]).cumprod() - 1.0
    return df[["date", "strategy", "ret", "cum_ret"]]


# ---- Main orchestrator --------------------------------------------------


def model_factories() -> dict[str, Callable]:
    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "gbr": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
        ),
    }


def default_train_end_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    start_year = 2014
    last_year = (panel["date"].max() - pd.DateOffset(years=1)).year
    return [pd.Timestamp(f"{y}-12-31") for y in range(start_year, last_year + 1)]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    panel_cols = list(panel.columns)
    tickers = sorted(panel["ticker"].unique())
    train_ends = default_train_end_dates(panel)
    print(f"Panel: {len(panel)} rows, {len(tickers)} tickers, {len(train_ends)} CV folds")

    variants = variants_for_panel(panel_cols, tickers)
    factories = model_factories()

    all_preds: list[pd.DataFrame] = []
    metrics_out: dict = {}
    portfolios: list[pd.DataFrame] = []

    for v in variants:
        X, y, meta = build_feature_matrix(panel, v)
        print(f"\n  variant {v.name} — {v.description}")
        print(f"    rows after dropna: {len(X)}, features: {len(v.features)}")

        metrics_out[v.name] = {"description": v.description, "models": {}}
        for model_name, factory in factories.items():
            preds = walk_forward_cv(X, y, meta, train_ends, factory)
            preds = add_p_up(preds)
            preds["variant"] = v.name
            preds["model"] = model_name

            # Evaluate on AAPL rows only — keeps cross-variant comparisons honest.
            aapl_preds = preds[preds["ticker"] == "AAPL"]
            metrics = evaluate(aapl_preds)
            metrics_out[v.name]["models"][model_name] = metrics
            print(
                f"    {model_name:8s}  n={metrics.get('n',0):4d}  "
                f"AUC={metrics.get('auc',float('nan')):.3f}  "
                f"acc={metrics.get('accuracy',float('nan')):.3f}  "
                f"R2={metrics.get('oos_r2',float('nan')):+.3f}  "
                f"IC={metrics.get('ic_spearman',float('nan')):+.3f}"
            )
            all_preds.append(preds)

            # Portfolio backtest using GBR predictions as the headline model.
            if model_name == "gbr":
                if v.name == "v1":
                    pf = portfolio_backtest_long_short(preds)
                    pf["variant"] = v.name
                    portfolios.append(pf)
                else:
                    pf = portfolio_backtest_timing(aapl_preds)
                    pf["variant"] = v.name
                    portfolios.append(pf)

    # Persist outputs.
    pred_path = OUTPUT_DIR / "ckx_predictions.parquet"
    pd.concat(all_preds, ignore_index=True).to_parquet(pred_path, index=False)

    metrics_path = OUTPUT_DIR / "ckx_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2, default=str))

    portfolio_path = OUTPUT_DIR / "ckx_portfolio.parquet"
    if portfolios:
        pd.concat(portfolios, ignore_index=True).to_parquet(portfolio_path, index=False)

    print(f"\nWrote predictions → {pred_path}")
    print(f"Wrote metrics     → {metrics_path}")
    if portfolios:
        print(f"Wrote portfolios  → {portfolio_path}")


if __name__ == "__main__":
    main()
