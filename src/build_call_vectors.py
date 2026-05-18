"""Per-call call vectors: n_chars-weighted mean of chunk embeddings.

Lifts the per-call aggregation logic out of ``score_transcript_sentiment``
into a standalone artifact so downstream code (delta vectors, ridge
training) can use it without re-running the cosine scorer.

Input:
    _data/embeddings_transcripts.parquet
        chunk-level rows: transcript_id, ticker, event_date, chunk_idx,
                          n_chars, embedding (1536-D vector)

Output:
    _data/call_vectors.parquet
        one row per call:
            transcript_id  int
            ticker         str  upper-case
            event_date     date
            embedding      list[float]  1536-D unit vector
                           (n_chars-weighted mean of the call's chunks,
                            then L2-normalised)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
INPUT_FILENAME = "embeddings_transcripts.parquet"
OUTPUT_FILENAME = "call_vectors.parquet"


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def load_chunks(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/embed_transcripts.py` first."
        )
    df = pd.read_parquet(path)
    df["embedding"] = df["embedding"].map(lambda x: np.asarray(x, dtype=np.float32))
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def build(chunks: pd.DataFrame) -> pd.DataFrame:
    """n_chars-weighted mean embedding per transcript_id, L2-normalised."""
    rows: list[dict] = []
    for tid, grp in chunks.groupby("transcript_id", sort=False):
        weights = grp["n_chars"].to_numpy(dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        embs = np.stack(grp["embedding"].to_numpy())
        weighted = (embs * weights[:, None]).sum(axis=0)
        rows.append(
            {
                "transcript_id": int(tid),
                "ticker": grp["ticker"].iloc[0],
                "event_date": grp["event_date"].iloc[0],
                "embedding": _normalise(weighted).tolist(),
            }
        )
    return pd.DataFrame(rows).sort_values(["ticker", "event_date"]).reset_index(drop=True)


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    chunks = load_chunks()
    print(f"  loaded {len(chunks):,} chunks across {chunks['transcript_id'].nunique()} calls "
          f"({chunks['ticker'].nunique()} tickers)")
    panel = build(chunks)
    out = write(panel)
    print(f"Wrote {len(panel):,} per-call vectors -> {out}")
    if not panel.empty:
        print(f"Date range: {panel['event_date'].min()} -> {panel['event_date'].max()}")


if __name__ == "__main__":
    main()
