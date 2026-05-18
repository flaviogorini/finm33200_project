"""Embed earnings-call transcripts using OpenAI ``text-embedding-3-small``.

Two input sources are supported, preferred in this order:

  1. The processed Nasdaq-100 components parquet (full universe, ~91 tickers)
        _data/transcripts/processed/nasdaq100_cleaned_components.parquet
     Used when present. This is the canonical source for the project.

  2. Per-ticker CSVs produced by the WRDS single-ticker extractor:
        _data/transcripts/{TICKER_LOWER}/{ticker_lower}_transcript_components.csv
     Fallback for legacy / smoke-test data (AAPL-only).

The output is one parquet with one row per chunk:

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
PROCESSED_COMPONENTS_PATH = (
    TRANSCRIPTS_DIR / "processed" / "nasdaq100_cleaned_components.parquet"
)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_TOKEN_BUDGET = 6000
MAX_CHUNK_TOKENS = 7500  # well under the 8,192 hard cap of text-embedding-3-small
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


def load_components_from_processed(
    path: Path = PROCESSED_COMPONENTS_PATH,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load components for the full Nasdaq-100 from the processed parquet.

    Returns a frame with the same column names the chunker expects:
    ``[transcript_id, ticker, event_date, component_order,
       component_text_clean, speaker_name]``. Blank components are dropped.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run extract_sample_raw_transcripts.py + "
            "clean_sample_transcripts.py --mode full first."
        )
    df = pd.read_parquet(path)
    if "is_blank_component" in df.columns:
        df = df[~df["is_blank_component"]]
    rename = {
        "primary_ticker": "ticker",
        "transcript_date": "event_date",
        "cleaned_component_text": "component_text_clean",
    }
    df = df.rename(columns=rename)
    if "speaker_name" not in df.columns:
        df["speaker_name"] = ""
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df = df.sort_values(["transcript_id", "component_order"]).reset_index(drop=True)
    if tickers is not None:
        wanted = [t.upper() for t in tickers]
        df = df[df["ticker"].isin(wanted)].reset_index(drop=True)
    return df[["transcript_id", "ticker", "event_date", "component_order",
               "component_text_clean", "speaker_name"]]


def _count_tokens(text: str, encoder) -> int:
    return len(encoder.encode(text))


def _split_long_piece(
    piece: str, encoder, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[tuple[str, int]]:
    """Split a single component into sub-pieces under ``max_tokens``.

    Returns a list of ``(text, token_count)`` tuples. Uses tiktoken to
    cut at exact token boundaries when available; falls back to a
    character-budget heuristic otherwise (~4 chars per token).
    """
    if encoder is None:
        approx = max_tokens * 4
        return [(piece[i : i + approx], max(1, len(piece[i : i + approx]) // 4))
                for i in range(0, len(piece), approx)]
    ids = encoder.encode(piece)
    if len(ids) <= max_tokens:
        return [(piece, len(ids))]
    parts: list[tuple[str, int]] = []
    for i in range(0, len(ids), max_tokens):
        sub = ids[i : i + max_tokens]
        parts.append((encoder.decode(sub), len(sub)))
    return parts


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
            piece_raw = f"{speaker}: {text}".strip()

            # Split any single component longer than the model's hard cap.
            for piece, piece_tokens in _split_long_piece(piece_raw, encoder):
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


REQUEST_TIMEOUT_S = 60.0
PROGRESS_EVERY_N_BATCHES = 25


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API; raises if OPENAI_API_KEY is missing.

    Each request has a hard 60s timeout — without one the client can hang
    indefinitely if a connection drops. Retries with exponential backoff
    up to 4 attempts (1s, 2s, 4s, 8s), then re-raises.
    """
    from openai import OpenAI

    api_key = config("OPENAI_API_KEY", default="")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env, export it, "
            "or rerun with --synthetic for a smoke-test."
        )
    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_S)

    out: list[list[float]] = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(4):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 2**attempt
                print(
                    f"  embed batch {i // BATCH_SIZE + 1}/{total_batches}: "
                    f"retry {attempt + 1} in {wait}s ({type(e).__name__}: {str(e)[:120]})",
                    flush=True,
                )
                time.sleep(wait)
        batch_idx = i // BATCH_SIZE + 1
        if batch_idx % PROGRESS_EVERY_N_BATCHES == 0 or batch_idx == total_batches:
            elapsed = time.time() - t0
            rate = batch_idx / elapsed if elapsed > 0 else 0.0
            eta = (total_batches - batch_idx) / rate if rate > 0 else float("nan")
            print(
                f"  embedded {batch_idx}/{total_batches} batches "
                f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                flush=True,
            )
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
    parser.add_argument("tickers", nargs="*", help="tickers; default: all available")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="random unit vectors (no OpenAI API call)",
    )
    parser.add_argument(
        "--from-processed",
        action="store_true",
        help=(
            "Force reading from the processed Nasdaq-100 components parquet "
            "even if it exists. (Default: auto — use parquet when present, "
            "fall back to per-ticker CSVs otherwise.)"
        ),
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Force per-ticker CSV input (skip the processed parquet).",
    )
    args = parser.parse_args()

    use_processed = (
        not args.from_csv
        and (args.from_processed or PROCESSED_COMPONENTS_PATH.exists())
    )

    if use_processed:
        wanted = [t.upper() for t in args.tickers] if args.tickers else None
        comps = load_components_from_processed(tickers=wanted)
        if comps.empty:
            raise SystemExit(f"No components found in {PROCESSED_COMPONENTS_PATH}")
        print(
            f"  processed input: {comps['ticker'].nunique()} tickers, "
            f"{comps['transcript_id'].nunique()} calls, {len(comps):,} components"
        )
        chunks = chunk_components(comps)
        print(f"  -> {len(chunks):,} chunks after token-budget grouping")
    else:
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
            tcs = chunk_components(comps)
            print(f"  {t}: {len(tcs)} chunks after token-budget grouping")
            all_chunks.append(tcs)
        if not all_chunks:
            raise SystemExit("No transcript components found for any ticker.")
        chunks = pd.concat(all_chunks, ignore_index=True)

    embedded = embed_chunks(chunks, synthetic=args.synthetic)
    out = write(embedded)
    print(f"Wrote {len(embedded):,} chunk embeddings -> {out}")


if __name__ == "__main__":
    main()
