"""Embed earnings-call transcripts using OpenAI ``text-embedding-3-small``.

Reads the per-component CSV produced by ``pull_wrds_earning_transcripts.py``:

    _data/transcripts/{TICKER_LOWER}/{ticker_lower}_transcript_components.csv

and emits one parquet with one row per chunk:

    _data/embeddings_transcripts.parquet

Schema:
    transcript_id    int    Capital IQ transcript_id
    ticker           str    UPPER e.g. "AAPL"
    event_date       date   earnings call date
    chunk_idx        int    0-based chunk index within transcript
    n_chars          int    chunk character length
    embedding        list[float]  1536-dim text-embedding-3-small vector

Chunking: consecutive components are concatenated until a soft token budget
(~6000 tokens via tiktoken) is reached, then a new chunk starts. This keeps
each chunk well under the 8191-token model limit while preserving speaker flow.

Synthetic mode (``--synthetic``) skips the OpenAI call and writes random unit
vectors. Useful for end-to-end pipeline smoke tests when no API key is set.

Usage:
    python embed_transcripts.py AAPL                 # real embeddings
    python embed_transcripts.py AAPL --synthetic     # random unit vectors
    python embed_transcripts.py                      # all tickers in transcripts/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_TOKEN_BUDGET = 6000
BATCH_SIZE = 20

OUTPUT_FILENAME = "embeddings_transcripts.parquet"


def _discover_tickers() -> list[str]:
    if not TRANSCRIPTS_DIR.exists():
        return []
    return sorted([p.name.upper() for p in TRANSCRIPTS_DIR.iterdir() if p.is_dir()])


def _components_csv_path(ticker: str) -> Path:
    """Find the components CSV for a ticker (case-insensitive directory)."""
    lower = ticker.lower()
    return TRANSCRIPTS_DIR / lower / f"{lower}_transcript_components.csv"


def load_components(ticker: str) -> pd.DataFrame:
    path = _components_csv_path(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run pull_wrds_earning_transcripts.py for {ticker} first."
        )
    df = pd.read_csv(path)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["ticker"] = ticker.upper()
    df = df.sort_values(["transcript_id", "component_order"]).reset_index(drop=True)
    return df


def _count_tokens(text: str, encoder) -> int:
    return len(encoder.encode(text))


def chunk_components(
    components: pd.DataFrame, token_budget: int = CHUNK_TOKEN_BUDGET
) -> pd.DataFrame:
    """Group consecutive components into chunks under a soft token budget.

    Returns a frame with [transcript_id, ticker, event_date, chunk_idx, text, n_chars].
    """
    try:
        import tiktoken

        encoder = tiktoken.encoding_for_model(EMBED_MODEL)
    except Exception:
        encoder = None  # fall back to char/4 heuristic

    rows: list[dict] = []
    for tid, grp in components.groupby("transcript_id", sort=False):
        ticker = grp["ticker"].iloc[0]
        event_date = grp["event_date"].iloc[0]

        chunk_idx = 0
        cur_text: list[str] = []
        cur_tokens = 0
        for _, comp in grp.iterrows():
            text = comp.get("component_text_clean")
            if not isinstance(text, str) or not text.strip():
                continue
            speaker = comp.get("speaker_name", "")
            piece = f"{speaker}: {text}".strip()
            piece_tokens = (
                _count_tokens(piece, encoder) if encoder else max(1, len(piece) // 4)
            )

            if cur_tokens + piece_tokens > token_budget and cur_text:
                joined = "\n\n".join(cur_text)
                rows.append(
                    dict(
                        transcript_id=tid,
                        ticker=ticker,
                        event_date=event_date,
                        chunk_idx=chunk_idx,
                        text=joined,
                        n_chars=len(joined),
                    )
                )
                chunk_idx += 1
                cur_text, cur_tokens = [], 0

            cur_text.append(piece)
            cur_tokens += piece_tokens

        if cur_text:
            joined = "\n\n".join(cur_text)
            rows.append(
                dict(
                    transcript_id=tid,
                    ticker=ticker,
                    event_date=event_date,
                    chunk_idx=chunk_idx,
                    text=joined,
                    n_chars=len(joined),
                )
            )

    return pd.DataFrame(rows)


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API; raises if OPENAI_API_KEY is missing."""
    from openai import OpenAI

    api_key = config("OPENAI_API_KEY", default="")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env, export it, "
            "or rerun with --synthetic for a smoke-test."
        )
    client = OpenAI(api_key=api_key)

    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2**attempt
                print(f"  embed batch {i}: retry {attempt + 1} in {wait}s ({e})")
                time.sleep(wait)
    return out


def _embed_synthetic(texts: list[str], seed: int = 0) -> list[list[float]]:
    """Random unit vectors — for pipeline smoke tests only."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((len(texts), EMBED_DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.tolist()


def embed_chunks(chunks: pd.DataFrame, *, synthetic: bool) -> pd.DataFrame:
    texts = chunks["text"].tolist()
    embedder = _embed_synthetic if synthetic else _embed_openai
    print(
        f"  embedding {len(texts)} chunks "
        f"({'SYNTHETIC' if synthetic else EMBED_MODEL})..."
    )
    embs = embedder(texts)
    out = chunks.drop(columns="text").copy()
    out["embedding"] = embs
    return out


def write(panel: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    out_path = data_dir / OUTPUT_FILENAME
    panel.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*", help="tickers; default: discover all")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="random unit vectors (no OpenAI API call)",
    )
    args = parser.parse_args()

    tickers = args.tickers or _discover_tickers()
    if not tickers:
        raise SystemExit(
            f"No tickers given and no subdirectories found in {TRANSCRIPTS_DIR}."
        )

    all_chunks: list[pd.DataFrame] = []
    for t in tickers:
        try:
            comps = load_components(t)
        except FileNotFoundError as e:
            print(f"  skip {t}: {e}")
            continue
        print(f"  {t}: {len(comps)} components, {comps['transcript_id'].nunique()} calls")
        chunks = chunk_components(comps)
        print(f"  {t}: {len(chunks)} chunks after token-budget grouping")
        all_chunks.append(chunks)

    if not all_chunks:
        raise SystemExit("No transcript components found for any ticker.")

    chunks = pd.concat(all_chunks, ignore_index=True)
    embedded = embed_chunks(chunks, synthetic=args.synthetic)
    out = write(embedded)
    print(f"Wrote {len(embedded):,} chunk embeddings -> {out}")


if __name__ == "__main__":
    main()
