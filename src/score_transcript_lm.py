"""Loughran-McDonald lexicon scoring on earnings-call transcripts.

This is Strategy 3 from the project spec (section 4.3). The Loughran-McDonald
financial sentiment master dictionary is loaded from
``data_manual/lm_master_dictionary.csv`` (free download from
https://sraf.nd.edu/loughranmcdonald-master-dictionary/). The LM lexicon
loader is ported from the archived 10-Q scorer; the input here is
earnings-call transcript text, NOT 10-K text. The 10-Q stack is archived
and explicitly out of scope.

Per spec section 4.3 we use ONLY the Positive and Negative word lists.
The other LM categories (Uncertainty, Litigious, StrongModal, WeakModal,
Constraining) are not used.

Per-call score:
    LM_{i,t} = (pos - neg) / (pos + neg)        (null if denom = 0)

Then per ticker, sorted by event_date:
    lm_delta_{i,t} = LM_{i,t} - LM_{i,t-1}      (NaN for the first call)

Input source (selected per Open Question 3 in the plan):
    _data/transcripts/processed/nasdaq100_llm_views.parquet
    filtered to ``view_name == 'full_transcript'``

Output:
    _data/lm_scores_transcripts.parquet

Schema:
    transcript_id  int
    ticker         str  upper-case
    event_date     date
    lm_pos         int
    lm_neg         int
    lm_score       float  (pos - neg) / (pos + neg)
    lm_delta       float  current - previous call (same ticker)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))

LM_DICT_PATH = MANUAL_DATA_DIR / "lm_master_dictionary.csv"
LLM_VIEWS_FILENAME = "transcripts/processed/nasdaq100_llm_views.parquet"
OUTPUT_FILENAME = "lm_scores_transcripts.parquet"
SELECTED_VIEW = "full_transcript"

TOKEN_RE = re.compile(r"[a-z]+")

_FALLBACK_POSITIVE = {
    "achieve", "benefit", "efficient", "favorable", "gain", "growth", "improve",
    "improved", "improvement", "increase", "increased", "positive", "profit", "strong",
    "success", "successful", "upturn",
}
_FALLBACK_NEGATIVE = {
    "adverse", "challenge", "challenging", "decline", "decreased", "decrease", "deficit",
    "delay", "deteriorate", "difficult", "downturn", "impairment", "loss", "negative",
    "risk", "risks", "uncertain", "weak", "weakness",
}


def load_lm_lexicon(path: Path = LM_DICT_PATH) -> tuple[set[str], set[str]]:
    """Load LM Positive and Negative word lists. Falls back to a small
    in-repo wordlist if the LM CSV isn't on disk so the pipeline still runs.
    """
    if not path.exists():
        print(f"  [warn] LM dictionary not found at {path}; using fallback word list")
        return _FALLBACK_POSITIVE, _FALLBACK_NEGATIVE

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    word_col = "word" if "word" in df.columns else df.columns[0]
    df[word_col] = df[word_col].astype(str).str.lower()

    def category(col: str) -> set[str]:
        if col not in df.columns:
            return set()
        return set(df.loc[df[col].fillna(0).astype(int) != 0, word_col])

    pos = category("positive")
    neg = category("negative")
    print(f"  [lm] loaded LM dictionary: pos={len(pos):,}  neg={len(neg):,}")
    return pos, neg


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def score_call(text: str, pos_vocab: set[str], neg_vocab: set[str]) -> dict[str, float]:
    """LM positive/negative counts and net-positivity ratio for one call."""
    tokens = tokenize(text)
    pos = sum(t in pos_vocab for t in tokens)
    neg = sum(t in neg_vocab for t in tokens)
    denom = pos + neg
    score = (pos - neg) / denom if denom > 0 else float("nan")
    return {"lm_pos": pos, "lm_neg": neg, "lm_score": score}


def load_llm_views(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / LLM_VIEWS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the WRDS extract + clean pipeline first "
            "(extract_sample_raw_transcripts.py + clean_sample_transcripts.py --mode full)."
        )
    df = pd.read_parquet(path)
    df = df[df["view_name"] == SELECTED_VIEW].copy()
    if df.empty:
        raise RuntimeError(
            f"No rows with view_name == {SELECTED_VIEW!r} in {path}"
        )
    df["primary_ticker"] = df["primary_ticker"].astype(str).str.upper()
    df["transcript_date"] = pd.to_datetime(df["transcript_date"])
    return df


def build(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    views = load_llm_views(data_dir)
    print(f"  [lm] scoring {len(views):,} calls (view={SELECTED_VIEW})")

    pos_vocab, neg_vocab = load_lm_lexicon()

    rows: list[dict] = []
    for _, row in views.iterrows():
        scored = score_call(row["view_text"], pos_vocab, neg_vocab)
        rows.append(
            {
                "transcript_id": int(row["transcript_id"]),
                "ticker": row["primary_ticker"],
                "event_date": row["transcript_date"].date(),
                **scored,
            }
        )

    per_call = (
        pd.DataFrame(rows)
        .dropna(subset=["lm_score"])
        .sort_values(["ticker", "event_date"])
        .reset_index(drop=True)
    )
    per_call["lm_delta"] = per_call.groupby("ticker")["lm_score"].diff()
    return per_call


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    panel = build()
    out = write(panel)
    print(f"\nWrote {len(panel):,} per-call LM scores -> {out}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    if not panel.empty:
        print(f"Date range: {panel['event_date'].min()} -> {panel['event_date'].max()}")
        print(f"Non-null lm_delta: {panel['lm_delta'].notna().sum():,} / {len(panel):,}")
        print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
