"""Dictionary-based textual features from cleaned 10-Q text.

Reads `_data/sec_10q/_meta/cleaned_index.csv` produced by
`clean_sec_10q_text.py` and emits a long-format
`_data/sec_10q/10q_features.parquet` keyed on (ticker, accession_number).

Sentiment / tone features are computed against MD&A only (the narrative
signal). When MD&A extraction failed (`extraction_status != 'ok'`), the
filing is still scored against the full narrative as a fallback so the
panel doesn't lose the row entirely — the column `feature_source` records
which text was used.

Uses the Loughran-McDonald financial sentiment master dictionary when
available at `data_manual/lm_master_dictionary.csv`. Falls back to a small
built-in word list so the script stays runnable without the LM download.
Master dictionary (free download, requires manual fetch):
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/
"""

import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from settings import (
    DEFAULT_TICKERS,
    SEC_10Q_DIR,
    SEC_10Q_META_DIR,
    config,
)
from pull_sec_10q_filings import resolve_sec10q_path


MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
LM_DICT_PATH = MANUAL_DATA_DIR / "lm_master_dictionary.csv"
FEATURES_PATH = SEC_10Q_DIR / "10q_features.parquet"

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
_FALLBACK_UNCERTAINTY = {
    "approximate", "contingent", "depend", "depends", "fluctuate", "fluctuation",
    "may", "might", "possible", "possibly", "uncertain", "uncertainties", "uncertainty",
    "unknown", "variable", "volatility",
}
_FALLBACK_LITIGIOUS = {
    "claim", "claims", "court", "lawsuit", "lawsuits", "litigation", "plaintiff",
    "defendant", "settlement", "judgment",
}
_FALLBACK_CONSTRAINING = {
    "constrain", "constrained", "constraint", "constraints", "limit", "limited",
    "restrict", "restricted", "restriction", "obligation", "covenant",
}

TOKEN_RE = re.compile(r"[a-z]+")


def load_lm_lexicon(path: Path = LM_DICT_PATH) -> dict[str, set[str]]:
    if not path.exists():
        return {
            "positive": _FALLBACK_POSITIVE,
            "negative": _FALLBACK_NEGATIVE,
            "uncertainty": _FALLBACK_UNCERTAINTY,
            "litigious": _FALLBACK_LITIGIOUS,
            "constraining": _FALLBACK_CONSTRAINING,
        }

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    word_col = "word" if "word" in df.columns else df.columns[0]
    df[word_col] = df[word_col].astype(str).str.lower()

    def category(col: str) -> set[str]:
        if col not in df.columns:
            return set()
        return set(df.loc[df[col].fillna(0).astype(int) != 0, word_col])

    return {
        "positive": category("positive"),
        "negative": category("negative"),
        "uncertainty": category("uncertainty"),
        "litigious": category("litigious"),
        "constraining": category("constraining"),
    }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def rate(tokens: list[str], vocab: set[str]) -> float:
    if not tokens:
        return 0.0
    return sum(token in vocab for token in tokens) / len(tokens)


def score_text(text: str, lex: dict[str, set[str]]) -> dict[str, float]:
    tokens = tokenize(text)
    pos = rate(tokens, lex["positive"])
    neg = rate(tokens, lex["negative"])
    return {
        "10q_sentiment":     pos - neg,
        "10q_positive_rate": pos,
        "10q_negative_rate": neg,
        "10q_uncertainty":   rate(tokens, lex["uncertainty"]),
        "10q_litigious":     rate(tokens, lex["litigious"]),
        "10q_constraining":  rate(tokens, lex["constraining"]),
        "10q_word_count":    len(tokens),
    }


def cosine_similarity(current: str, previous: str) -> float:
    a = Counter(tokenize(current))
    b = Counter(tokenize(previous))
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pick_text_path(row: pd.Series) -> tuple[Path | None, str]:
    """Prefer MD&A; fall back to narrative if MD&A missing or weak.

    Paths in cleaned_index.csv are stored relative to SEC_10Q_DIR; resolve
    them back to absolute paths before opening.
    """
    status = str(row.get("extraction_status", ""))
    mda = row.get("mda_path")
    if status == "ok" and isinstance(mda, str):
        p = resolve_sec10q_path(mda)
        if p.exists():
            return p, "mda"
    narr = row.get("narrative_path")
    if isinstance(narr, str):
        p = resolve_sec10q_path(narr)
        if p.exists():
            return p, "narrative_fallback"
    return None, "missing"


def score_filings(
    cleaned_index_path: Path = SEC_10Q_META_DIR / "cleaned_index.csv",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    df = pd.read_csv(cleaned_index_path, parse_dates=["filing_date", "report_period"])
    df = df[df["ticker"].isin(tickers)].copy()
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    lex = load_lm_lexicon()
    using_lm = LM_DICT_PATH.exists()
    print(
        f"Using {'Loughran-McDonald' if using_lm else 'fallback small'} dictionary "
        f"(pos={len(lex['positive'])}, neg={len(lex['negative'])})"
    )

    rows = []
    previous_text_by_ticker: dict[str, str | None] = {t: None for t in tickers}
    for _, row in df.iterrows():
        ticker = row["ticker"]
        path, source = _pick_text_path(row)
        if path is None:
            print(f"  skip {ticker} {row['accession_number']}: no usable text")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        scores = score_text(text, lex)
        previous = previous_text_by_ticker.get(ticker)
        if previous is None:
            scores["10q_cosine_vs_previous"] = float("nan")
            scores["10q_change_vs_previous"] = float("nan")
        else:
            cos = cosine_similarity(text, previous)
            scores["10q_cosine_vs_previous"] = cos
            scores["10q_change_vs_previous"] = 1.0 - cos
        previous_text_by_ticker[ticker] = text
        rows.append({
            **row.to_dict(),
            "feature_source": source,
            **scores,
        })

    features = pd.DataFrame(rows)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    print(
        f"Wrote {FEATURES_PATH}  ({len(features)} rows across "
        f"{features['ticker'].nunique() if len(features) else 0} tickers)"
    )
    return features


if __name__ == "__main__":
    score_filings()
