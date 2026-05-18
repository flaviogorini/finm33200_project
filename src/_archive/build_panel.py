"""Assemble the monthly (date × ticker) panel from all feature builders.

Inputs (must already exist in DATA_DIR):
    features_fundamentals_monthly.parquet   (date, ticker, px_last, pe_ratio, …)
    features_consensus_monthly.parquet      (date, ticker, best_*)
    features_sentiment_monthly.parquet      (date, ticker, sentiment_*)   [optional]
    features_macro_monthly.parquet          (date, vix, treas_*, …)        [global]
    labels_returns_monthly.parquet          (date, ticker, ret_*, fwd_ret_*)

Output:
    _data/panel_monthly.parquet

Join order:
    1. fundamentals (left base — defines (date, ticker) universe)
    2. left-join consensus on (date, ticker)
    3. left-join sentiment on (date, ticker) — gracefully empty if missing
    4. left-join macro on (date,)            — broadcast across tickers
    5. left-join return labels on (date, ticker)

The panel is the single source of truth downstream. Modeling code should
NEVER read raw feature parquets — only ``load_panel()``.

A no-lookahead invariant test ships in ``test_panel_no_lookahead.py``:
for every row at month-end ``t``, every feature column was activated by ``t``.
``fwd_*`` columns are labels, excluded from feature checks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

OUTPUT_FILENAME = "panel_monthly.parquet"

FUNDAMENTALS = "features_fundamentals_monthly.parquet"
CONSENSUS = "features_consensus_monthly.parquet"
SENTIMENT = "features_sentiment_monthly.parquet"
MACRO = "features_macro_monthly.parquet"
RETURNS = "labels_returns_monthly.parquet"
SEC_10Q = "sec_10q_monthly_panel.parquet"

# 10-Q panel ships metadata cols alongside features; we only want the features.
# This tuple is the NUMERIC modeling list — keep it numeric-only.
SEC_10Q_FEATURE_COLS = (
    "10q_sentiment",
    "10q_positive_rate",
    "10q_negative_rate",
    "10q_uncertainty",
    "10q_litigious",
    "10q_constraining",
    "10q_word_count",
    "10q_cosine_vs_previous",
    "10q_change_vs_previous",
    "10q_embedding_cosine_vs_previous",
    "10q_embedding_change_vs_previous",
    # Generative-AI 10-Q analysis (analyze_sec_10q_llm.py); powers V4.
    "10q_ai_tone_score",
    "10q_ai_risk_score",
    "10q_ai_uncertainty_score",
    "10q_ai_margin_pressure",
    "10q_ai_liquidity_pressure",
    "10q_ai_demand_outlook",
    "10q_ai_disclosure_change_score",
    "10q_ai_material_change_flag",
)

# AI summary + cited evidence are TEXT. They flow into the panel so the
# Streamlit dashboard can show cited filing snippets, but they are NOT model
# features — the modelling layer only ever reads numeric columns.
AI_TEXT_COLS = ("10q_ai_summary", "10q_ai_evidence")

# Columns considered LABELS, not features. Excluded from no-lookahead checks
# of "feature columns" but kept in the panel so the modelling layer can split
# X / y in one place.
LABEL_COLS = ("fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m")


def _read_required(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Required upstream feature parquet is missing."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _read_optional(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    path = data_dir / name
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    fund = _read_required(FUNDAMENTALS, data_dir)
    cons = _read_required(CONSENSUS, data_dir)
    rets = _read_required(RETURNS, data_dir)
    macro = _read_required(MACRO, data_dir)
    sent = _read_optional(SENTIMENT, data_dir)

    panel = fund.merge(cons, on=["date", "ticker"], how="left", suffixes=("", "_cons"))
    if sent is not None:
        panel = panel.merge(sent, on=["date", "ticker"], how="left", suffixes=("", "_sent"))
    else:
        print(
            f"  note: {SENTIMENT} not found — sentiment columns will be absent. "
            f"Run embed_transcripts -> score_transcript_sentiment -> build_sentiment_features."
        )

    sec10q = _read_optional(SEC_10Q, data_dir)
    if sec10q is not None:
        keep = (
            ["date", "ticker"]
            + [c for c in SEC_10Q_FEATURE_COLS if c in sec10q.columns]
            + [c for c in AI_TEXT_COLS if c in sec10q.columns]
        )
        panel = panel.merge(sec10q[keep], on=["date", "ticker"], how="left")
    else:
        print(
            f"  note: {SEC_10Q} not found — 10-Q text columns will be absent. "
            f"Run `doit pull:sec_10q_filings && doit process_10q` (needs WRDS_PASSWORD)."
        )

    panel = panel.merge(macro, on="date", how="left")
    panel = panel.merge(rets, on=["date", "ticker"], how="left")

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def load_panel(
    data_dir: Path = DATA_DIR,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Single read-side entry point for downstream models.

    Args:
        data_dir:  where panel_monthly.parquet lives.
        start:     inclusive lower bound on `date` (str or Timestamp).
        end:       inclusive upper bound on `date`.
        tickers:   filter to this ticker set; default = all.
    """
    path = data_dir / OUTPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/build_panel.py` first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    return df.reset_index(drop=True)


def integrity_checks(panel: pd.DataFrame) -> None:
    """Hard-fail integrity checks. Called from main()."""
    assert panel["date"].notna().all(), "null dates"
    assert (panel.groupby(["date", "ticker"]).size() == 1).all(), (
        "duplicate (date, ticker) rows"
    )
    # Every date should be a month-end.
    bad = panel.loc[panel["date"] != panel["date"] + pd.offsets.MonthEnd(0), "date"]
    assert bad.empty, f"non-month-end dates: {bad.unique()[:5]}"


def main() -> None:
    panel = build()
    integrity_checks(panel)
    out = write(panel)
    print(f"Wrote panel: {len(panel):,} rows x {len(panel.columns)} cols -> {out}")
    print(f"Tickers ({panel['ticker'].nunique()}):", sorted(panel["ticker"].unique()))
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Columns: {list(panel.columns)}")


if __name__ == "__main__":
    main()
