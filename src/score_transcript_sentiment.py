"""Score earnings-call transcript sentiment from chunk embeddings.

Reads ``_data/embeddings_transcripts.parquet`` (output of
``embed_transcripts.py``) and computes a per-call sentiment score using
cosine-similarity to anchor phrases.

Anchor phrases are embedded once on demand and cached in
``_data/sentiment_anchors.parquet``. Re-embedding is cheap (10 phrases) so
they're recomputed if the cache is missing.

Output:
    _data/sentiment_transcripts.parquet

Schema:
    transcript_id   int
    ticker          str
    event_date      date
    sentiment_pos   float  cosine to mean of POSITIVE anchor embeddings
    sentiment_neg   float  cosine to mean of NEGATIVE anchor embeddings
    sentiment_diff  float  sentiment_pos - sentiment_neg  (signed score)

Per-transcript embedding is the n_chars-weighted mean of chunk embeddings.

Synthetic mode (``--synthetic``) re-uses the synthetic-embedding path from
``embed_transcripts.py`` for the anchor phrases as well, so the entire
pipeline runs without an OpenAI key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

EMBEDDINGS_FILENAME = "embeddings_transcripts.parquet"
ANCHORS_FILENAME = "sentiment_anchors.parquet"
OUTPUT_FILENAME = "sentiment_transcripts.parquet"

POSITIVE_ANCHORS = [
    "We had a record-breaking quarter with strong revenue growth.",
    "We are raising our full-year guidance based on robust demand.",
    "Our profitability and margins expanded significantly.",
    "We delivered exceptional results that exceeded expectations.",
    "Demand for our products remains very strong worldwide.",
]

NEGATIVE_ANCHORS = [
    "We missed expectations and guidance was below consensus.",
    "We are lowering our outlook due to weak demand and macro headwinds.",
    "Margins contracted and operating expenses rose materially.",
    "The quarter was disappointing with revenue declines across segments.",
    "We are seeing significant softness and customer pullback.",
]


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def load_embeddings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / EMBEDDINGS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/embed_transcripts.py [--synthetic]` first."
        )
    df = pd.read_parquet(path)
    df["embedding"] = df["embedding"].map(lambda x: np.asarray(x, dtype=np.float32))
    return df


def _embed_anchors(synthetic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_mean_unit, neg_mean_unit) anchor vectors."""
    from embed_transcripts import _embed_openai, _embed_synthetic

    embedder = _embed_synthetic if synthetic else _embed_openai
    pos = np.asarray(embedder(POSITIVE_ANCHORS), dtype=np.float32)
    neg = np.asarray(embedder(NEGATIVE_ANCHORS), dtype=np.float32)

    pos_mean = _normalise(pos.mean(axis=0))
    neg_mean = _normalise(neg.mean(axis=0))
    return pos_mean, neg_mean


def transcript_embeddings(chunks: pd.DataFrame) -> pd.DataFrame:
    """n_chars-weighted mean embedding per transcript_id."""
    rows: list[dict] = []
    for tid, grp in chunks.groupby("transcript_id", sort=False):
        weights = grp["n_chars"].to_numpy(dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        embs = np.stack(grp["embedding"].to_numpy())
        weighted = (embs * weights[:, None]).sum(axis=0)
        rows.append(
            dict(
                transcript_id=tid,
                ticker=grp["ticker"].iloc[0],
                event_date=grp["event_date"].iloc[0],
                embedding=_normalise(weighted),
            )
        )
    return pd.DataFrame(rows)


def score(transcripts: pd.DataFrame, pos: np.ndarray, neg: np.ndarray) -> pd.DataFrame:
    embs = np.stack(transcripts["embedding"].to_numpy())
    embs_unit = _normalise(embs)
    sentiment_pos = embs_unit @ pos
    sentiment_neg = embs_unit @ neg

    out = transcripts.drop(columns="embedding").copy()
    out["sentiment_pos"] = sentiment_pos
    out["sentiment_neg"] = sentiment_neg
    out["sentiment_diff"] = sentiment_pos - sentiment_neg
    return out


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="random anchor embeddings (no OpenAI call)",
    )
    args = parser.parse_args()

    chunks = load_embeddings()
    print(f"  loaded {len(chunks)} chunks across {chunks['transcript_id'].nunique()} transcripts")

    pos, neg = _embed_anchors(args.synthetic)
    transcripts = transcript_embeddings(chunks)
    scored = score(transcripts, pos, neg)
    scored = scored.sort_values(["ticker", "event_date"]).reset_index(drop=True)

    out = write(scored)
    print(f"Wrote {len(scored)} per-call sentiment rows -> {out}")
    print(scored.head().to_string(index=False))


if __name__ == "__main__":
    main()
