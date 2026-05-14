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
]

PANEL_PATH = DATA_DIR / "sec_10q_monthly_panel.parquet"
FEATURES_PATH = SEC_10Q_DIR / "10q_features.parquet"


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
