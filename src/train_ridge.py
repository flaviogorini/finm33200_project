"""Ridge regression on Δ call vectors — Strategy 2 (spec section 4.2).

Trains a ridge model with expanding-window refits to predict the
21-trading-day forward return from earnings-call event_date.

Features (1,537 total per call):
    - delta_vector (1,536-D): current call vector minus previous-call
      vector for the same ticker. Built by ``build_delta_vectors.py``.
    - days_since_earnings (scalar, calendar days): freshness control.

Target:
    21-trading-day forward return computed via
    ``calendar_utils.fwd_ret_bd(prices, event_date, h=21, gap=1)``.
    Identical to the backtest holding-period return.

Training procedure (per spec):
    - Train period: ``TRAIN_START`` ≤ event_date ≤ ``TRAIN_END_INITIAL``.
    - Expanding-window refits at the start of each calendar year ``Y``
      in the test period. The refit window is
      ``[TRAIN_START, end-of-(Y-1)]``; predictions for that year are made
      with the frozen model.
    - Per training window: ``StandardScaler`` fit on train only,
      ``RidgeCV`` with α grid {1e-2, 1e-1, 1, 10, 100, 1000} and
      ``TimeSeriesSplit(n_splits=5)`` inside the training window.

Output:
    _data/ridge_predictions.parquet
    Schema:
        transcript_id  int
        ticker         str
        event_date     date
        y_pred         float  predicted 21-bday forward return
        y_true         float  realised 21-bday forward return (for diagnostics)
        fold_year      int    calendar year of the test slice
        alpha_used     float  α selected by RidgeCV for that fold's training window
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from build_returns_monthly import _strip_bbg_suffix
from calendar_utils import fwd_ret_event_calmonth
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

DELTA_FILENAME = "delta_vectors.parquet"
HIST_FILENAME = "US_Companies_Hist_Data.parquet"
OUTPUT_FILENAME = "ridge_predictions.parquet"

ALPHA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
CV_SPLITS = 5
TRAIN_START = pd.Timestamp("2012-01-01")
TRAIN_END_INITIAL = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")

# PCA pre-reduction. With 1,537 raw features and ~3k training observations
# per fold, ridge saturates the regularization grid (alpha=1000 every fold)
# and produces zero signal. PCA collapses the embedding to a low-dimensional
# subspace before ridge sees it. days_since_earnings is concatenated AFTER
# the PCA so its single scalar dimension isn't wasted as a principal axis.
PCA_COMPONENTS = 50


def load_delta_vectors(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / DELTA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/build_delta_vectors.py` first."
        )
    df = pd.read_parquet(path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["delta_vector"] = df["delta_vector"].map(lambda x: np.asarray(x, dtype=np.float32))
    return df.sort_values(["event_date", "ticker"]).reset_index(drop=True)


def load_daily_prices(data_dir: Path = DATA_DIR) -> dict[str, pd.Series]:
    path = data_dir / HIST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/pull_manual_companies.py` first."
        )
    df = pd.read_parquet(path)
    df = df[df["field"] == "PX_LAST"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].map(_strip_bbg_suffix).astype(str).str.upper()
    df = df.rename(columns={"value": "px_last"}).dropna(subset=["px_last"])
    out: dict[str, pd.Series] = {}
    for ticker, grp in df.groupby("ticker"):
        s = grp.set_index("date")["px_last"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[ticker] = s
    return out


def build_xy(deltas: pd.DataFrame, prices_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
    """Add y_true (calendar-month forward return from event_date close to
    BME at-or-after event_date + 1 month) and the stacked X matrix.

    v3 change vs v2: target is now ``fwd_ret_event_calmonth`` instead of
    ``fwd_ret_bd(h=21, gap=1)``. Anchored to per-call event dates with a
    calendar-month horizon, matching the monthly backtest's holding period.
    """
    y_vals: list[float] = []
    for _, row in deltas.iterrows():
        s = prices_by_ticker.get(row["ticker"])
        if s is None:
            y_vals.append(float("nan"))
            continue
        y_vals.append(fwd_ret_event_calmonth(s, row["event_date"]))
    out = deltas.copy()
    out["y_true"] = y_vals
    out = out.dropna(subset=["y_true"]).reset_index(drop=True)
    return out


def _split_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_vectors_2d, days_since_earnings_1d) for one frame."""
    deltas = np.stack(df["delta_vector"].to_numpy())
    days = df["days_since_earnings"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return deltas, days


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_components: int = PCA_COMPONENTS,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Fit PCA -> StandardScaler -> RidgeCV on train, predict on test.

    PCA is fit on the training delta vectors only. ``days_since_earnings``
    is concatenated AFTER the PCA so the freshness feature isn't competing
    for principal-component space.

    Returns ``(y_pred, alpha_chosen, diagnostics_dict)`` where diagnostics
    include the cumulative variance explained by the kept PCs and the
    actual number of components retained (capped at min(n_components,
    n_train, n_features)).
    """
    deltas_train, days_train = _split_matrix(train)
    deltas_test, days_test = _split_matrix(test)
    y_train = train["y_true"].to_numpy(dtype=np.float32)

    max_pcs = min(n_components, deltas_train.shape[0], deltas_train.shape[1])
    pca = PCA(n_components=max_pcs, svd_solver="auto", random_state=0)
    pcs_train = pca.fit_transform(deltas_train)
    pcs_test = pca.transform(deltas_test)

    X_train = np.concatenate([pcs_train, days_train], axis=1)
    X_test = np.concatenate([pcs_test, days_test], axis=1)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    splits = max(2, min(CV_SPLITS, len(train) // 30))
    cv = TimeSeriesSplit(n_splits=splits)
    model = RidgeCV(alphas=ALPHA_GRID, cv=cv)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    diag = {
        "n_components": int(max_pcs),
        "var_explained": float(pca.explained_variance_ratio_.sum()),
    }
    return y_pred, float(model.alpha_), diag


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    deltas = load_delta_vectors(data_dir)
    prices_by_ticker = load_daily_prices(data_dir)
    data = build_xy(deltas, prices_by_ticker)
    print(f"  loaded {len(data):,} delta-call rows with valid 21-bday target "
          f"across {data['ticker'].nunique()} tickers")

    if data.empty:
        return pd.DataFrame()

    initial_train = data[(data["event_date"] >= TRAIN_START) & (data["event_date"] <= TRAIN_END_INITIAL)]
    test_pool = data[data["event_date"] >= TEST_START]
    if initial_train.empty or test_pool.empty:
        print(
            f"  WARN: insufficient data. initial_train={len(initial_train)} "
            f"test_pool={len(test_pool)}. Skipping ridge fit."
        )
        return pd.DataFrame()

    test_years = sorted(set(test_pool["event_date"].dt.year))
    print(f"  expanding-window folds: years {test_years}")

    rows: list[pd.DataFrame] = []
    for year in test_years:
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        train = data[(data["event_date"] >= TRAIN_START) & (data["event_date"] <= train_end)]
        test = data[
            (data["event_date"] >= pd.Timestamp(f"{year}-01-01"))
            & (data["event_date"] <= pd.Timestamp(f"{year}-12-31"))
        ]
        if len(train) < CV_SPLITS * 2 or test.empty:
            print(f"    skip year {year}: train={len(train)} test={len(test)}")
            continue
        y_pred, alpha, diag = _fit_predict(train, test)
        block = test[["transcript_id", "ticker", "event_date"]].copy()
        block["y_pred"] = y_pred
        block["y_true"] = test["y_true"].to_numpy()
        block["fold_year"] = year
        block["alpha_used"] = alpha
        block["n_components"] = diag["n_components"]
        block["var_explained"] = diag["var_explained"]
        rows.append(block)
        print(
            f"    fold {year}: train={len(train):>4d} test={len(test):>3d}  "
            f"PCs={diag['n_components']:>2d}  var={diag['var_explained']:.2%}  "
            f"alpha={alpha:>8.3f}"
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    if panel.empty:
        print(f"Wrote empty {out} (no eligible data).")
        return
    print(f"\nWrote {len(panel):,} ridge predictions -> {out}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Date range: {panel['event_date'].min().date()} -> {panel['event_date'].max().date()}")
    print(f"Mean alpha across folds: {panel['alpha_used'].mean():.3f}")


if __name__ == "__main__":
    main()
