"""Honest evaluation of the cached decision digests.

Reads every JSON under `_data/digest_cache/` produced by
:mod:`generate_digest`, and writes a row-per-digest evaluation parquet:

    _output/digest_eval.parquet
    _output/digest_eval_summary.json

Three independent verifiers — reported separately, NOT combined into a
triangular composite. The methodology section of the writeup explains the
choice: each leg here is either deterministic (citation_match,
numeric_grounding) or grounded in realized return data (direction_match), so
we trust each at face value. We deliberately did not implement an LLM-as-judge
reasoning verifier — calibrating the judge against human labels in 12 days
was not feasible, and an uncalibrated meta-metric is worse than no metric.

The three legs and their inspirations:

- ``citation_match_rate`` — fraction of ``evidence[i].quote`` strings (>= 5
  words) that appear verbatim in the pre-fetched evidence block for that run.
  This is Fuentes' ``e`` verifier (Trade-R1, slide 18): "cited data appears in
  input?".

- ``numeric_grounding_rate`` — fraction of numbers appearing in the digest's
  rationale paragraphs (matched by regex) that ALSO appear in the evidence
  block. Deterministic, zero LLM calls. This is Fuentes' MR-RLVR "Extract"
  sub-verifier (slide 21): "Are the cited figures actually in the doc?".
  This catches hallucinated numerics that ``citation_match_rate`` misses,
  because that metric only checks quoted strings; standalone numbers in
  paragraph prose are uncovered without this leg.

- ``direction_match`` — when ``fwd_ret_1m`` is realized at as_of+1m, did the
  digest's ``forecast_direction`` sign match? 1.0 if yes, 0.0 if no, 0.5 if
  "neutral". This is Fuentes' ``o`` verifier — outcome leg.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

DIGEST_CACHE_DIR = DATA_DIR / "digest_cache"

# Regexes for "noticeable" numerics in the rationale text. Anchored loosely so
# we catch the typical analyst-writeup conventions:
#   $12.4B, $1,200M, $310, 12.4%, 47 bps, 1,234.5, 26477.8
# We skip lone 1-2 digit integers ("1 quarter", "h=4") and 4-digit numbers
# that look like years (1900-2099) — both would over-match.
NUMERIC_PATTERNS = (
    r"\$\d+(?:[\.,]\d+)*\s*[BbMmKk]?",       # currency: $12.4B, $1,200M, $310
    r"\d+(?:\.\d+)?\s*%",                     # percentages: 12.4%, 47%
    r"\d+(?:\.\d+)?\s*bps\b",                 # basis points: 47 bps
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",      # thousands-comma: 104,263.0
    r"\b\d+\.\d+\b",                           # bare decimal: 26477.8, 0.627
    r"\b\d{3,}\b",                              # bare integer (3+ digits): 1234
)

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def _normalize_ws(s: str) -> str:
    """Collapse runs of whitespace so verbatim checks tolerate line wraps."""
    return re.sub(r"\s+", " ", s or "").strip()


def _norm_numeric(tok: str) -> str:
    """Canonical form for evidence-block lookup: lowercase, strip $ and commas.

    The evidence block writes raw figures (`consensus=104263.0`), so we
    normalize the LLM's `$104,263` form to bare digits before substring-
    matching against the evidence.
    """
    return tok.replace(",", "").replace("$", "").strip().lower()


def _is_year_or_short_int(tok: str) -> bool:
    """Reject 1-2 digit ints and 4-digit years — too noisy to ground."""
    bare = tok.replace(",", "")
    if re.fullmatch(r"\d+", bare):
        if len(bare) < 3:
            return True
        if _YEAR_RE.fullmatch(bare):
            return True
    return False


def citation_match_rate(response: dict, evidence_block_text: str) -> tuple[float, int, int]:
    """Fuentes `e` verifier. Returns (rate, n_matched, n_total)."""
    evidence = response.get("evidence") or []
    ev_norm = _normalize_ws(evidence_block_text).lower()
    n_total = 0
    n_matched = 0
    for item in evidence:
        quote = _normalize_ws(item.get("quote") or "")
        # Only count quotes long enough to be meaningful (>=5 words). Short
        # quotes ("Yes", "Risk") trivially appear by chance.
        if len(quote.split()) < 5:
            continue
        n_total += 1
        if quote.lower() in ev_norm:
            n_matched += 1
    if n_total == 0:
        return (float("nan"), 0, 0)
    return (n_matched / n_total, n_matched, n_total)


def _extract_numerics(text: str) -> list[str]:
    """Return every regex-matched numeric token in `text` (deduped, order-preserved).

    Spans claimed by an earlier pattern are masked out, so the bare-integer
    pattern doesn't double-count the integer part of a decimal (`26477.8`
    would otherwise also yield `26477`). Dedup key is the normalized form
    (comma-stripped, lowercased) so "104,263.0" and "104263.0" are the same.
    """
    seen: set[str] = set()
    out: list[str] = []
    claimed: list[tuple[int, int]] = []  # (start, end) spans already matched
    for pat in NUMERIC_PATTERNS:
        for m in re.finditer(pat, text):
            if any(s <= m.start() < e or s < m.end() <= e for s, e in claimed):
                continue
            tok = _normalize_ws(m.group(0))
            if _is_year_or_short_int(tok):
                continue
            claimed.append((m.start(), m.end()))
            key = _norm_numeric(tok)
            if key in seen:
                continue
            seen.add(key)
            out.append(tok)
    return out


def numeric_grounding_rate(
    response: dict, evidence_block_text: str
) -> tuple[float, int, int, list[str]]:
    """Fuentes Extract sub-verifier. Returns (rate, n_matched, n_total, unmatched_list).

    A token counts as grounded if its normalized (comma-stripped, lowercased)
    form appears in the normalized evidence block. We compare both the raw
    and comma-stripped views of evidence so "$1,234" matches "1234".
    """
    rationale_text = " ".join(
        response.get(k, "") or ""
        for k in ("rationale_returns", "rationale_fundamentals", "rationale_disclosure")
    )
    candidates = _extract_numerics(rationale_text)
    ev_raw = _normalize_ws(evidence_block_text).lower()
    ev_nocomma = ev_raw.replace(",", "")
    n_matched = 0
    unmatched: list[str] = []
    for tok in candidates:
        key = _norm_numeric(tok)
        if key in ev_raw or key in ev_nocomma:
            n_matched += 1
        else:
            unmatched.append(tok)
    n_total = len(candidates)
    if n_total == 0:
        return (float("nan"), 0, 0, [])
    return (n_matched / n_total, n_matched, n_total, unmatched)


def direction_match(response: dict, realized_fwd_ret_1m: float | None) -> float:
    """Fuentes `o` verifier. NaN if no realized return available."""
    if realized_fwd_ret_1m is None or pd.isna(realized_fwd_ret_1m):
        return float("nan")
    direction = response.get("forecast_direction")
    if direction == "neutral":
        return 0.5
    if direction == "bullish":
        return 1.0 if realized_fwd_ret_1m > 0 else 0.0
    if direction == "bearish":
        return 1.0 if realized_fwd_ret_1m < 0 else 0.0
    return float("nan")


def _iter_digests(cache_dir: Path) -> Iterable[Path]:
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("*.json"))


def evaluate_one(digest_payload: dict, predictions: pd.DataFrame) -> dict:
    """Return a dict with one row's worth of eval columns."""
    ticker = digest_payload["ticker"]
    as_of = pd.Timestamp(digest_payload["as_of_date"])
    response = digest_payload.get("response") or {}
    evidence_block_text = digest_payload.get("evidence_block_text") or ""

    cm_rate, cm_n, cm_total = citation_match_rate(response, evidence_block_text)
    ng_rate, ng_n, ng_total, ng_unmatched = numeric_grounding_rate(
        response, evidence_block_text
    )

    # Realized fwd_ret_1m comes from the CKX predictions y_true.
    realized: float | None = None
    if predictions is not None:
        sub = predictions[
            (predictions["ticker"] == ticker)
            & (predictions["date"] == as_of)
            & (predictions["target"] == "fwd_ret_1m")
        ]
        if not sub.empty and pd.notna(sub.iloc[0]["y_true"]):
            realized = float(sub.iloc[0]["y_true"])

    dm = direction_match(response, realized)

    return {
        "ticker": ticker,
        "as_of_date": as_of,
        "forecast_direction": response.get("forecast_direction"),
        "confidence": response.get("confidence"),
        "citation_match_rate": cm_rate,
        "citation_matched_n": cm_n,
        "citation_total_n": cm_total,
        "numeric_grounding_rate": ng_rate,
        "numeric_matched_n": ng_n,
        "numeric_total_n": ng_total,
        "numeric_unmatched_examples": ", ".join(ng_unmatched[:5]),
        "realized_fwd_ret_1m": realized,
        "direction_match": dm,
        "n_evidence_items": len(response.get("evidence") or []),
        "n_failure_warnings": len(response.get("failure_warnings") or []),
    }


def summarize(eval_df: pd.DataFrame) -> dict:
    realized = eval_df.dropna(subset=["direction_match"])
    return {
        "n_digests": int(len(eval_df)),
        "n_digests_with_realized": int(len(realized)),
        "mean_citation_match_rate": float(eval_df["citation_match_rate"].mean(skipna=True)),
        "mean_numeric_grounding_rate": float(
            eval_df["numeric_grounding_rate"].mean(skipna=True)
        ),
        "direction_accuracy_when_realized": (
            float(realized["direction_match"].mean()) if len(realized) else float("nan")
        ),
        "per_ticker": {
            t: {
                "n": int(len(grp)),
                "mean_citation": float(grp["citation_match_rate"].mean(skipna=True)),
                "mean_numeric_grounding": float(
                    grp["numeric_grounding_rate"].mean(skipna=True)
                ),
                "n_realized": int(grp["direction_match"].notna().sum()),
                "direction_accuracy": float(
                    grp["direction_match"].mean(skipna=True)
                ),
            }
            for t, grp in eval_df.groupby("ticker")
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    digest_paths = list(_iter_digests(DIGEST_CACHE_DIR))
    if not digest_paths:
        print(
            f"no digest cache at {DIGEST_CACHE_DIR}. "
            "Run `python src/generate_digest.py --all` first."
        )
        return

    pred_path = OUTPUT_DIR / "ckx_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"missing {pred_path}; run predict_returns_ckx.py first."
        )
    predictions = pd.read_parquet(pred_path)
    predictions["date"] = pd.to_datetime(predictions["date"])

    rows: list[dict] = []
    for p in digest_paths:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skip {p.name}: {e}")
            continue
        rows.append(evaluate_one(payload, predictions))

    if not rows:
        print("no rows; aborting.")
        return

    eval_df = pd.DataFrame(rows).sort_values(["ticker", "as_of_date"])
    out_parquet = OUTPUT_DIR / "digest_eval.parquet"
    eval_df.to_parquet(out_parquet, index=False)
    print(f"wrote {len(eval_df)} rows -> {out_parquet}")

    summary = summarize(eval_df)
    out_summary = OUTPUT_DIR / "digest_eval_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote summary -> {out_summary}")

    # Print a quick stdout snapshot.
    print(
        f"\nmean citation_match_rate = {summary['mean_citation_match_rate']:.3f}"
    )
    print(
        f"mean numeric_grounding_rate = {summary['mean_numeric_grounding_rate']:.3f}"
    )
    print(
        f"direction_accuracy (n={summary['n_digests_with_realized']}) = "
        f"{summary['direction_accuracy_when_realized']:.3f}"
    )


if __name__ == "__main__":
    main()
