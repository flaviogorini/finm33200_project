"""Decision digest — one-shot LLM grounding of the two forecasts in cited evidence.

For (ticker, as_of_date), this:
  1. Pre-fetches deterministically (no agent loop):
        - Returns view: p_up + y_pred from the CKX V5 GBR predictions.
        - Fundamentals view: 4-horizon Chronos forecast row from the backtest.
        - Disclosure context: top-3 PIT-filtered 10-Q chunks (FAISS).
        - Transcript context: top-2 PIT-filtered transcript chunks.
  2. Calls OpenAI once with the structured `DigestSchema` (strict=True).
  3. Caches the response under
        _data/digest_cache/{ticker}__{YYYY-MM-DD}__{PROMPT_VERSION}.json
     so re-runs are free and reproducible.

Output: per-call JSON cache files (no separate parquet). Aggregating across the
20-cell grid is :mod:`eval_digest`'s job.

Usage:
    python src/generate_digest.py --ticker AAPL --as-of 2024-09-30
    python src/generate_digest.py --all       # 5 tickers x 4 as_of dates

Cost: ~$0.10 per digest with gpt-4o-mini. 20-cell warm = ~$2.
When OPENAI_API_KEY is unset, this script prints a notice and exits 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_chronos2 import BACKTEST_AS_OF_DATES, BACKTEST_TICKERS
from build_panel import load_panel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
SEC_10Q_DIR = Path(config("DATA_DIR")) / "sec_10q"

DIGEST_CACHE_DIR = DATA_DIR / "digest_cache"
PROMPT_VERSION = "v1"

# Mirrors the existing analyze_sec_10q_llm.py pattern. Hardcoded rather than
# config-driven because we don't anticipate per-environment overrides; edit
# this constant to switch models.
DIGEST_LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Fixed retrieval query — Stockton lesson: predictable, ticker-filtered.
DISCLOSURE_QUERY = "recent material developments, risks, outlook, and changes vs the prior quarter"

DIGEST_SCHEMA = {
    "name": "decision_digest",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "forecast_direction",
            "confidence",
            "rationale_returns",
            "rationale_fundamentals",
            "rationale_disclosure",
            "evidence",
            "failure_warnings",
        ],
        "properties": {
            "forecast_direction": {
                "type": "string",
                "enum": ["bullish", "neutral", "bearish"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale_returns": {"type": "string"},
            "rationale_fundamentals": {"type": "string"},
            "rationale_disclosure": {"type": "string"},
            "evidence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source_type", "source_id", "quote", "why_it_matters"],
                    "properties": {
                        "source_type": {
                            "type": "string",
                            "enum": [
                                "returns_model",
                                "chronos_forecast",
                                "10q_chunk",
                                "transcript_chunk",
                            ],
                        },
                        "source_id": {"type": "string"},
                        "quote": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                    },
                },
            },
            "failure_warnings": {"type": "array", "items": {"type": "string"}},
        },
    },
}

SYSTEM_PROMPT = (
    "You are a careful equity research analyst writing a point-in-time investment "
    "digest. You are reasoning AT a specific as_of_date — do not reference any "
    "event, price, filing, or report dated AFTER that date. When you quote a "
    "number or sentence, copy it verbatim from the pre-fetched evidence block; "
    "do not paraphrase numerics. Cite only from the evidence block — do not cite "
    "from training data. Return ONLY the structured JSON object requested."
)


# ---- pre-fetch helpers --------------------------------------------------


def get_returns_view(
    ticker: str, as_of: pd.Timestamp, predictions: pd.DataFrame
) -> dict | None:
    """Return p_up + y_pred from the CKX V5 GBR head, falling back to V3 GBR.

    Returns None when no prediction exists for (ticker, as_of) under any
    variant — caller will surface a `no_returns_view` failure warning.
    """
    for variant_pref in ("v5", "v4", "v3", "v2", "v1", "v0b"):
        sub = predictions[
            (predictions["ticker"] == ticker)
            & (predictions["date"] == as_of)
            & (predictions["variant"] == variant_pref)
            & (predictions["model"] == "gbr")
            & (predictions["target"] == "fwd_ret_1m")
        ]
        if not sub.empty:
            row = sub.iloc[0]
            return {
                "variant": variant_pref,
                "model": "gbr",
                "p_up": float(row["p_up"]),
                "y_pred": float(row["y_pred"]),
                "y_true": float(row["y_true"]) if pd.notna(row["y_true"]) else None,
            }
    return None


def get_fundamentals_view(
    ticker: str, as_of: pd.Timestamp, backtest: pd.DataFrame | None
) -> list[dict]:
    """Return the 4-horizon Chronos rows for (ticker, as_of), revenue and net_income.

    Empty list = outside the 5-ticker x 4-quarter Chronos grid; caller adds a
    `no_fundamentals_view` failure warning.
    """
    if backtest is None or backtest.empty:
        return []
    sub = backtest[(backtest["ticker"] == ticker) & (backtest["as_of_date"] == as_of)]
    if sub.empty:
        return []
    out: list[dict] = []
    for _, r in sub.sort_values(["target", "horizon_q"]).iterrows():
        out.append(
            {
                "target": str(r["target"]),
                "horizon_q": int(r["horizon_q"]),
                "target_quarter_end": pd.Timestamp(r["target_quarter_end"]).date().isoformat(),
                "chronos_q10": float(r["chronos_q10"]),
                "chronos_q50": float(r["chronos_q50"]),
                "chronos_q90": float(r["chronos_q90"]),
                "consensus": (
                    float(r["consensus"]) if pd.notna(r["consensus"]) else None
                ),
                "naive_yoy": (
                    float(r["naive_yoy"]) if pd.notna(r["naive_yoy"]) else None
                ),
            }
        )
    return out


def _embed_query(client, text: str) -> np.ndarray:
    """Embed one query string with text-embedding-3-small, L2-normalized."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.asarray(resp.data[0].embedding, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def get_disclosure_chunks(
    ticker: str, as_of: pd.Timestamp, query_vec: np.ndarray, k: int = 3
) -> list[dict]:
    """Top-k 10-Q chunks for ticker with filing_date <= as_of, FAISS-ranked.

    Returns list of {accession_number, filing_date, chunk_idx, text, score}.
    """
    ticker_dir = SEC_10Q_DIR / ticker
    chunks_path = ticker_dir / "10q_chunks.parquet"
    vecs_path = ticker_dir / "10q_embeddings.parquet"  # filing-level pooled vectors
    # We need per-chunk vectors. Re-derive scores from chunks parquet + FAISS index.
    faiss_path = ticker_dir / "10q_chunks.faiss"
    if not (chunks_path.exists() and faiss_path.exists()):
        return []
    chunks = pd.read_parquet(chunks_path)
    chunks["filing_date"] = pd.to_datetime(chunks["filing_date"])
    mask_pit = chunks["filing_date"] <= as_of
    if not mask_pit.any():
        return []

    try:
        import faiss
    except ImportError:
        return []
    index = faiss.read_index(str(faiss_path))
    # Search wider than k and filter PIT post-hoc; chunk count per filing is small.
    n_search = min(index.ntotal, max(k * 8, 30))
    scores, idxs = index.search(query_vec.reshape(1, -1).astype(np.float32), n_search)
    out: list[dict] = []
    for rank in range(idxs.shape[1]):
        row_id = int(idxs[0, rank])
        if row_id < 0 or row_id >= len(chunks):
            continue
        row = chunks.iloc[row_id]
        if row["filing_date"] > as_of:
            continue  # PIT skip
        out.append(
            {
                "accession_number": str(row["accession_number"]),
                "filing_date": pd.Timestamp(row["filing_date"]).date().isoformat(),
                "chunk_idx": int(row["chunk_idx"]),
                "text": str(row["text"]),
                "score": float(scores[0, rank]),
            }
        )
        if len(out) >= k:
            break
    return out


def get_transcript_chunks(
    ticker: str, as_of: pd.Timestamp, query_vec: np.ndarray, k: int = 2
) -> list[dict]:
    """Top-k transcript chunks for ticker with event_date <= as_of.

    Uses the existing `_data/embeddings_transcripts.parquet` directly — no
    separate FAISS index needed at this scale (a few hundred chunks per ticker).
    Reconstructs chunk text by re-running the deterministic chunker on the
    ticker's components CSV.
    """
    emb_path = DATA_DIR / "embeddings_transcripts.parquet"
    if not emb_path.exists():
        return []
    emb = pd.read_parquet(emb_path)
    emb["event_date"] = pd.to_datetime(emb["event_date"])
    sub = emb[(emb["ticker"] == ticker) & (emb["event_date"] <= as_of)]
    if sub.empty:
        return []
    # Cosine = dot product since L2-normalized at embed time.
    mat = np.vstack(sub["embedding"].to_list()).astype(np.float32)
    scores = mat @ query_vec.astype(np.float32)
    sub = sub.assign(_score=scores).sort_values("_score", ascending=False).head(k)

    # Reconstruct chunk text from components CSV.
    comp_path = DATA_DIR / "transcripts" / ticker / f"{ticker.lower()}_transcript_components.csv"
    if not comp_path.exists():
        # Without text we still return scored chunk metadata so the prompt knows it exists.
        return [
            {
                "transcript_id": int(r["transcript_id"]),
                "event_date": pd.Timestamp(r["event_date"]).date().isoformat(),
                "chunk_idx": int(r["chunk_idx"]),
                "text": "",
                "score": float(r["_score"]),
            }
            for _, r in sub.iterrows()
        ]
    components = pd.read_csv(comp_path)
    # Avoid heavy circular import: import the chunker lazily.
    sys.path.insert(0, str(Path(__file__).parent))
    from embed_transcripts import chunk_components  # noqa: E402

    chunks = chunk_components(components)
    chunks["event_date"] = pd.to_datetime(chunks["event_date"])
    out: list[dict] = []
    for _, r in sub.iterrows():
        match = chunks[
            (chunks["transcript_id"] == r["transcript_id"])
            & (chunks["chunk_idx"] == r["chunk_idx"])
        ]
        text = str(match.iloc[0]["text"]) if not match.empty else ""
        out.append(
            {
                "transcript_id": int(r["transcript_id"]),
                "event_date": pd.Timestamp(r["event_date"]).date().isoformat(),
                "chunk_idx": int(r["chunk_idx"]),
                "text": text,
                "score": float(r["_score"]),
            }
        )
    return out


# ---- prompt assembly ----------------------------------------------------


def _evidence_block(
    returns_view: dict | None,
    fundamentals: list[dict],
    disclosure: list[dict],
    transcript: list[dict],
) -> str:
    parts = ["=== EVIDENCE BLOCK ===", ""]

    parts.append("[RETURNS MODEL]")
    if returns_view is None:
        parts.append("(no CKX prediction available for this (ticker, date) — none of the trained variants produced a forecast)")
    else:
        parts.append(
            f"variant={returns_view['variant']} model={returns_view['model']} "
            f"y_pred={returns_view['y_pred']:.4f} p_up={returns_view['p_up']:.3f}"
        )
    parts.append("")

    parts.append("[CHRONOS-2 FUNDAMENTALS FORECAST (4 quarters ahead)]")
    if not fundamentals:
        parts.append(
            "(no Chronos backtest row for this (ticker, date) — outside the 5x4 grid)"
        )
    else:
        for f in fundamentals:
            cons = f"{f['consensus']:.1f}" if f["consensus"] is not None else "n/a"
            naive = f"{f['naive_yoy']:.1f}" if f["naive_yoy"] is not None else "n/a"
            parts.append(
                f"{f['target']} h={f['horizon_q']} -> "
                f"target_quarter={f['target_quarter_end']}: "
                f"chronos q50={f['chronos_q50']:.1f} "
                f"(q10={f['chronos_q10']:.1f}, q90={f['chronos_q90']:.1f}); "
                f"consensus={cons}; naive_yoy={naive}"
            )
    parts.append("")

    parts.append("[10-Q DISCLOSURE CHUNKS (most relevant, PIT-filtered)]")
    if not disclosure:
        parts.append("(no 10-Q chunks available before this date)")
    else:
        for d in disclosure:
            parts.append(
                f"--- accession={d['accession_number']} filing_date={d['filing_date']} "
                f"chunk_idx={d['chunk_idx']} score={d['score']:.3f} ---"
            )
            parts.append(d["text"][:1500])  # cap at 1500 chars per chunk to bound cost
            parts.append("")

    parts.append("[TRANSCRIPT CHUNKS (most relevant, PIT-filtered)]")
    if not transcript:
        parts.append("(no transcript chunks available before this date)")
    else:
        for t in transcript:
            parts.append(
                f"--- transcript_id={t['transcript_id']} event_date={t['event_date']} "
                f"chunk_idx={t['chunk_idx']} score={t['score']:.3f} ---"
            )
            parts.append((t["text"] or "(text unavailable)")[:1500])
            parts.append("")

    return "\n".join(parts)


def build_user_prompt(
    ticker: str,
    as_of: pd.Timestamp,
    returns_view: dict | None,
    fundamentals: list[dict],
    disclosure: list[dict],
    transcript: list[dict],
) -> str:
    evidence = _evidence_block(returns_view, fundamentals, disclosure, transcript)
    return f"""You are writing an investment digest for ticker {ticker} as_of_date={as_of.date()}.

Hard rules:
- Reason AT as_of_date. Do not reference anything dated after that.
- Cite ONLY from the evidence block. Do not invent quotes or numbers.
- Quote numbers verbatim from the evidence block.
- If a piece of evidence is missing (no fundamentals view, no transcript, etc.),
  add an entry to `failure_warnings` explaining what's missing.

Output the structured DigestSchema with five fields:
  - forecast_direction: bullish/neutral/bearish, informed by p_up and Chronos vs naive.
  - confidence: your honest confidence, 0..1. Low when evidence is thin or conflicting.
  - rationale_returns: one paragraph using p_up and y_pred from the returns model.
  - rationale_fundamentals: one paragraph using Chronos vs consensus vs naive.
  - rationale_disclosure: one paragraph citing 10-Q + transcript quotes.
  - evidence: 3-5 items, each with source_type, source_id (e.g. accession_number for
    10-Q, transcript_id for calls, "ckx_V5_gbr" for the returns model, target for chronos),
    a verbatim quote, and why_it_matters.
  - failure_warnings: any data gaps that materially weaken the call (empty list if none).

{evidence}
"""


# ---- cache + LLM call ---------------------------------------------------


def _cache_path(ticker: str, as_of: pd.Timestamp) -> Path:
    safe = re.sub(
        r"[^A-Za-z0-9_.-]", "_",
        f"{ticker}__{as_of.date().isoformat()}__{PROMPT_VERSION}",
    )
    return DIGEST_CACHE_DIR / f"{safe}.json"


def _load_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload  # full payload incl. evidence_block_text for downstream eval
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(
    path: Path,
    *,
    ticker: str,
    as_of: pd.Timestamp,
    response: dict,
    evidence_block_text: str,
    model: str,
) -> None:
    DIGEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "as_of_date": as_of.date().isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_block_text": evidence_block_text,
        "response": response,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _call_llm(client, system_prompt: str, user_prompt: str, max_retries: int = 6) -> dict:
    from openai import RateLimitError

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=DIGEST_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_schema", "json_schema": DIGEST_SCHEMA},
            )
            return json.loads(resp.choices[0].message.content)
        except RateLimitError:
            wait = min(2.0 * (2**attempt), 60.0)
            print(f"    rate limited, sleeping {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("LLM call exhausted retries")


# ---- orchestration ------------------------------------------------------


def generate_one(
    ticker: str,
    as_of: pd.Timestamp,
    *,
    predictions: pd.DataFrame,
    backtest: pd.DataFrame | None,
    client=None,
    query_vec: np.ndarray | None = None,
    force: bool = False,
) -> dict:
    """Generate (or load cached) digest for one (ticker, as_of)."""
    cache = _cache_path(ticker, as_of)
    if not force:
        cached = _load_cached(cache)
        if cached is not None:
            return cached

    if client is None or query_vec is None:
        raise RuntimeError(
            f"no cache for ({ticker}, {as_of.date()}) and no OpenAI client provided"
        )

    returns_view = get_returns_view(ticker, as_of, predictions)
    fundamentals = get_fundamentals_view(ticker, as_of, backtest)
    disclosure = get_disclosure_chunks(ticker, as_of, query_vec)
    transcript = get_transcript_chunks(ticker, as_of, query_vec)

    user_prompt = build_user_prompt(
        ticker, as_of, returns_view, fundamentals, disclosure, transcript
    )
    response = _call_llm(client, SYSTEM_PROMPT, user_prompt)

    _write_cache(
        cache,
        ticker=ticker,
        as_of=as_of,
        response=response,
        evidence_block_text=_evidence_block(
            returns_view, fundamentals, disclosure, transcript
        ),
        model=DIGEST_LLM_MODEL,
    )
    return _load_cached(cache)  # round-trip so callers always get the full payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--all", action="store_true", help="run the 5x4 grid")
    parser.add_argument("--force", action="store_true", help="ignore cache")
    args = parser.parse_args()

    if args.all:
        pairs = [(t, pd.Timestamp(d)) for t in BACKTEST_TICKERS for d in BACKTEST_AS_OF_DATES]
    elif args.ticker and args.as_of:
        pairs = [(args.ticker, pd.Timestamp(args.as_of))]
    else:
        parser.error("provide --ticker and --as-of, or --all")

    api_key = config("OPENAI_API_KEY", default="")
    needs_api = args.force or any(
        not _cache_path(t, d).exists() for t, d in pairs
    )

    if needs_api and not api_key:
        print(
            "OPENAI_API_KEY not set; cannot generate uncached digests. "
            "Set it in .env or re-run --force after setting it. Exiting 0 "
            "so doit task can no-op."
        )
        return

    predictions_path = OUTPUT_DIR / "ckx_predictions.parquet"
    if not predictions_path.exists():
        print(
            f"missing {predictions_path}. Run `python src/predict_returns_ckx.py` first."
        )
        sys.exit(1)
    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"])

    backtest_path = OUTPUT_DIR / "chronos2_backtest.parquet"
    backtest = None
    if backtest_path.exists():
        backtest = pd.read_parquet(backtest_path)
        backtest["as_of_date"] = pd.to_datetime(backtest["as_of_date"])
        backtest["target_quarter_end"] = pd.to_datetime(backtest["target_quarter_end"])

    client = None
    query_vec = None
    if needs_api:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        print(f"  embedding fixed query with {EMBED_MODEL}…")
        query_vec = _embed_query(client, DISCLOSURE_QUERY)

    for ticker, as_of in pairs:
        out_path = _cache_path(ticker, as_of)
        if not args.force and out_path.exists():
            print(f"  cache hit: {ticker} {as_of.date()}")
            continue
        print(f"  generating: {ticker} {as_of.date()}")
        try:
            generate_one(
                ticker,
                as_of,
                predictions=predictions,
                backtest=backtest,
                client=client,
                query_vec=query_vec,
                force=args.force,
            )
        except Exception as e:
            print(f"    error: {e}")
            continue


if __name__ == "__main__":
    main()
