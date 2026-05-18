"""Build a (date, ticker) point-in-time monthly panel of 10-Q signals.

Long-format output with one row per (month-end, ticker). Each row carries
the most recent 10-Q whose `filing_date <= month_end`, never a future
filing. The panel is the handshake artifact for the project's combined
panel — teammates' feature columns concatenate by (date, ticker).
"""

from pathlib import Path

import pandas as pd

from settings import (
    DEFAULT_TICKERS,
    SEC_10Q_DIR,
    SEC_10Q_END_DATE,
    SEC_10Q_START_DATE,
    config,
)


DATA_DIR = config("DATA_DIR")


FEATURE_COLUMNS = [
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
    # Generative-AI 10-Q analysis (analyze_sec_10q_llm.py). Numeric scores
    # only; present after the optional `doit process_10q:analyze` stage.
    "10q_ai_tone_score",
    "10q_ai_risk_score",
    "10q_ai_uncertainty_score",
    "10q_ai_margin_pressure",
    "10q_ai_liquidity_pressure",
    "10q_ai_demand_outlook",
    "10q_ai_disclosure_change_score",
    "10q_ai_material_change_flag",
]

PANEL_KEY_COLUMNS = [
    "date",
    "ticker",
    "filing_date",
    "report_period",
    "accession_number",
    "sec_url",
    "extraction_status",
    "feature_source",
    # AI summary + cited evidence are TEXT — carried for the dashboard, never
    # used as model features (kept out of FEATURE_COLUMNS deliberately).
    "10q_ai_summary",
    "10q_ai_evidence",
]

PANEL_PATH = DATA_DIR / "sec_10q_monthly_panel.parquet"
FEATURES_PATH = SEC_10Q_DIR / "10q_features.parquet"
AI_FEATURES_PATH = SEC_10Q_DIR / "10q_ai_features.parquet"

# AI columns merged in from 10q_ai_features.parquet (numeric features + text).
AI_MERGE_COLUMNS = [
    "10q_ai_tone_score",
    "10q_ai_risk_score",
    "10q_ai_uncertainty_score",
    "10q_ai_margin_pressure",
    "10q_ai_liquidity_pressure",
    "10q_ai_demand_outlook",
    "10q_ai_disclosure_change_score",
    "10q_ai_material_change_flag",
    "10q_ai_summary",
    "10q_ai_evidence",
]


def build_monthly_panel(
    features_path: Path = FEATURES_PATH,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    features = pd.read_parquet(features_path)
    features["filing_date"] = pd.to_datetime(features["filing_date"])
    if "report_period" in features.columns:
        features["report_period"] = pd.to_datetime(features["report_period"], errors="coerce")
    features = features[features["ticker"].isin(tickers)].copy()

    # Optional: merge the generative-AI 10-Q features on (ticker, accession).
    # No-op when the `process_10q:analyze` stage has not been run. Merged
    # BEFORE the merge_asof below so the AI columns ride the same
    # point-in-time machinery and the lookahead assert covers them too.
    if AI_FEATURES_PATH.exists():
        ai = pd.read_parquet(AI_FEATURES_PATH)
        ai_cols = [c for c in AI_MERGE_COLUMNS if c in ai.columns]
        features = features.merge(
            ai[["ticker", "accession_number", *ai_cols]],
            on=["ticker", "accession_number"],
            how="left",
        )
        print(f"  merged {len(ai_cols)} AI 10-Q columns from {AI_FEATURES_PATH.name}")
    else:
        print(
            f"  note: {AI_FEATURES_PATH.name} not found — AI 10-Q columns absent. "
            f"Run `doit process_10q:analyze` (needs OPENAI_API_KEY)."
        )

    features = features.sort_values(["ticker", "filing_date"])

    months = pd.date_range(SEC_10Q_START_DATE, SEC_10Q_END_DATE, freq="ME")
    monthly = (
        pd.MultiIndex.from_product([months, tickers], names=["date", "ticker"])
        .to_frame(index=False)
        .sort_values("date")  # merge_asof requires the `on` key sorted globally
        .reset_index(drop=True)
    )

    panel = pd.merge_asof(
        monthly,
        features.sort_values("filing_date").reset_index(drop=True),
        left_on="date",
        right_on="filing_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    valid = panel.dropna(subset=["filing_date"])
    if not valid.empty and (valid["filing_date"] > valid["date"]).any():
        raise RuntimeError("Lookahead bias detected: a row uses a future filing.")

    cols = [c for c in PANEL_KEY_COLUMNS + FEATURE_COLUMNS if c in panel.columns]
    panel = panel[cols]

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    print(
        f"Wrote {PANEL_PATH}  ({len(panel)} rows: "
        f"{len(months)} months x {len(tickers)} tickers)"
    )
    return panel


def load_10q_monthly_panel() -> pd.DataFrame:
    """Read the cached 10-Q monthly panel."""
    return pd.read_parquet(PANEL_PATH)


if __name__ == "__main__":
    build_monthly_panel()
