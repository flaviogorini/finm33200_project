"""Build dense embeddings over cleaned 10-Q MD&A text.

For each filing in `_data/sec_10q/_meta/cleaned_index.csv`:
  1. Load MD&A (fall back to narrative when MD&A extraction failed).
  2. Sliding-window chunk to 2000-char pieces, 200-char overlap.
  3. Embed each chunk with OpenAI `text-embedding-3-small`, L2-normalize.
  4. Mean-pool chunks → one (1536,) vector per filing, L2-renormalized.

Outputs:
  _data/sec_10q/{ticker}/10q_chunks.parquet     -- per-chunk text + char offsets.
  _data/sec_10q/{ticker}/10q_embeddings.parquet -- per-filing pooled vectors + cosine.
  _data/sec_10q/{ticker}/10q_chunks.faiss       -- FAISS IndexFlatIP over chunk vectors.
  _data/sec_10q/10q_features.parquet (in-place) -- adds `10q_embedding_*` columns.

Cost estimate: ~$0.008 per ticker, ~$0.20 for Dow 30 at text-embedding-3-small.

When `OPENAI_API_KEY` is not set, this script prints a notice and exits 0
so the optional `doit process_10q:embed` task can be no-op'd cleanly.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from settings import (
    DEFAULT_TICKERS,
    OPENAI_API_KEY,
    SEC_10Q_DIR,
    SEC_10Q_META_DIR,
    ticker_dir,
)
from pull_sec_10q_filings import resolve_sec10q_path


EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 100

FEATURES_PATH = SEC_10Q_DIR / "10q_features.parquet"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping character windows. Returns
    list of (chunk_text, char_start, char_end)."""
    if not text:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]
        if piece.strip():
            chunks.append((piece, start, end))
        if end == len(text):
            break
    return chunks


def _embed_batch_with_retry(client, batch, model: str, max_retries: int = 8):
    """Call the embeddings endpoint, retrying on TPM rate limits."""
    from openai import RateLimitError
    for attempt in range(max_retries):
        try:
            return client.embeddings.create(model=model, input=batch)
        except RateLimitError as e:
            wait = 2.0
            m = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(e))
            if m:
                v = float(m.group(1))
                wait = v / 1000 if m.group(2) == "ms" else v
            wait = min(max(wait, 2.0) * (2 ** attempt), 60.0)
            print(f"    rate limited, sleeping {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError("embed_texts: exceeded retry budget on rate limits")


def embed_texts(client, texts: list[str]) -> np.ndarray:
    """Embed strings in batches; return L2-normalized (n, EMBED_DIM) float32."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = _embed_batch_with_retry(client, batch, EMBED_MODEL)
        for j, item in enumerate(resp.data):
            out[i + j] = np.asarray(item.embedding, dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out /= norms
    return out


def _pick_text_path(row: pd.Series) -> Path | None:
    """Resolve paths from cleaned_index.csv (relative to SEC_10Q_DIR)."""
    status = str(row.get("extraction_status", ""))
    if status == "ok" and isinstance(row.get("mda_path"), str):
        p = resolve_sec10q_path(row["mda_path"])
        if p.exists():
            return p
    if isinstance(row.get("narrative_path"), str):
        p = resolve_sec10q_path(row["narrative_path"])
        if p.exists():
            return p
    return None


def embed_one_ticker(client, ticker: str, ticker_idx: pd.DataFrame) -> pd.DataFrame:
    """Chunk + embed every filing for one ticker."""
    out_dir = ticker_dir(ticker)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_records = []
    chunk_vectors_per_filing: list[np.ndarray] = []
    filing_rows: list[dict] = []

    print(f"\n[{ticker}] embedding {len(ticker_idx)} filings...")
    for _, row in ticker_idx.iterrows():
        path = _pick_text_path(row)
        if path is None:
            print(f"  skip {row['accession_number']}: no text")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        if not chunks:
            print(f"  skip {row['accession_number']}: empty chunks")
            continue
        chunk_texts = [c[0] for c in chunks]
        vectors = embed_texts(client, chunk_texts)

        pooled = vectors.mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm

        for (piece, s, e), vec in zip(chunks, vectors):
            chunk_records.append({
                "ticker": ticker,
                "accession_number": row["accession_number"],
                "filing_date": row["filing_date"],
                "chunk_idx": len(chunk_records),
                "char_start": s,
                "char_end": e,
                "text": piece,
            })

        chunk_vectors_per_filing.append(vectors)
        filing_rows.append({
            "ticker": ticker,
            "accession_number": row["accession_number"],
            "filing_date": row["filing_date"],
            "n_chunks": len(chunks),
            "vector": pooled.astype(np.float32).tolist(),
        })
        print(f"  {row['accession_number']}: {len(chunks)} chunks -> pooled vector")

    if not filing_rows:
        return pd.DataFrame()

    filings_df = pd.DataFrame(filing_rows).sort_values("filing_date").reset_index(drop=True)

    cos_prev: list[float] = []
    prev: np.ndarray | None = None
    for vec in filings_df["vector"]:
        v = np.asarray(vec, dtype=np.float32)
        cos_prev.append(float(np.dot(v, prev)) if prev is not None else float("nan"))
        prev = v
    filings_df["10q_embedding_cosine_vs_previous"] = cos_prev
    filings_df["10q_embedding_change_vs_previous"] = 1.0 - filings_df["10q_embedding_cosine_vs_previous"]

    chunks_df = pd.DataFrame(chunk_records)
    chunks_df.to_parquet(out_dir / "10q_chunks.parquet", index=False)
    filings_df.to_parquet(out_dir / "10q_embeddings.parquet", index=False)

    try:
        import faiss
        all_chunks = np.vstack(chunk_vectors_per_filing).astype(np.float32)
        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(all_chunks)
        faiss.write_index(index, str(out_dir / "10q_chunks.faiss"))
        print(f"  wrote FAISS index: {len(all_chunks)} chunks")
    except ImportError:
        print("  faiss not installed; skipping FAISS index (parquet outputs still written)")

    return filings_df


def build_embeddings(
    cleaned_index_path: Path = SEC_10Q_META_DIR / "cleaned_index.csv",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    if not OPENAI_API_KEY:
        print(
            "Skipping embeddings: OPENAI_API_KEY is not set. "
            "Add it to .env if you want to run the optional embedding stage."
        )
        return pd.DataFrame()

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    tickers = tickers or DEFAULT_TICKERS
    idx = pd.read_csv(cleaned_index_path, parse_dates=["filing_date"])
    idx = idx[idx["ticker"].isin(tickers)].copy()

    all_filings = []
    for ticker in tickers:
        ticker_idx = idx[idx["ticker"] == ticker].sort_values("filing_date").reset_index(drop=True)
        if ticker_idx.empty:
            print(f"[{ticker}] no filings in cleaned index; skipping")
            continue
        all_filings.append(embed_one_ticker(client, ticker, ticker_idx))

    if not all_filings:
        return pd.DataFrame()
    combined = pd.concat(all_filings, ignore_index=True)

    # Merge the per-filing embedding cosine into 10q_features.parquet if present.
    if FEATURES_PATH.exists():
        features = pd.read_parquet(FEATURES_PATH)
        merge_cols = [
            "ticker", "accession_number",
            "10q_embedding_cosine_vs_previous",
            "10q_embedding_change_vs_previous",
        ]
        for c in merge_cols[2:]:
            if c in features.columns:
                features = features.drop(columns=[c])
        features = features.merge(
            combined[merge_cols],
            on=["ticker", "accession_number"],
            how="left",
        )
        features.to_parquet(FEATURES_PATH, index=False)
        print(f"\nMerged embedding features into {FEATURES_PATH}")
    else:
        print(f"\n{FEATURES_PATH} not found; per-ticker parquet outputs still written.")

    return combined


if __name__ == "__main__":
    build_embeddings()
    sys.exit(0)
