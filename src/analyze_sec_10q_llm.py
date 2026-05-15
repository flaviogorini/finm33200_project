"""Generative-AI 10-Q analysis layer.

Reads `_data/sec_10q/_meta/cleaned_index.csv` produced by
`clean_sec_10q_text.py`, and for each filing from `SEC_10Q_LLM_START_YEAR`
onward, asks an OpenAI chat model to *read* the filing, compare it to the
same ticker's previous 10-Q, and emit structured numeric scores plus cited
evidence snippets. Output: `_data/sec_10q/10q_ai_features.parquet`, keyed on
(ticker, accession_number).

This is the "generative AI" counterpart to the Loughran-McDonald *dictionary*
features in `score_sec_10q_text.py` (LM = the finance dictionary, not a
language model). The lexicon stage is left untouched; this writes a separate
parquet so the two stages stay independently runnable.

Point-in-time safe: filings are processed in (ticker, filing_date) order and
each one is compared only against a strictly-earlier same-ticker filing — the
same pattern used by `score_sec_10q_text.py`. No labels are ever shown to the
model.

Caching: each analysis is cached under `_data/sec_10q/_llm_cache/` keyed by
(accession_number, prev_accession_number, prompt_version). Re-runs reuse the
cache and make zero API calls, so downstream builds are reproducible.

When `OPENAI_API_KEY` is not set, this script prints a notice and exits 0 so
the optional `doit process_10q:analyze` task can be no-op'd cleanly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from settings import (
    DEFAULT_TICKERS,
    OPENAI_API_KEY,
    SEC_10Q_DIR,
    SEC_10Q_LLM_CACHE_DIR,
    SEC_10Q_LLM_MODEL,
    SEC_10Q_LLM_START_YEAR,
    SEC_10Q_META_DIR,
    USE_CACHE,
)
from pull_sec_10q_filings import resolve_sec10q_path

FEATURES_PATH = SEC_10Q_DIR / "10q_ai_features.parquet"

# Bump when the prompt or schema changes — it is part of the cache key, so a
# bump forces a clean re-run instead of serving stale cached responses.
PROMPT_VERSION = "v1"

# Per-section char budgets fed to the model. MD&A is the primary narrative;
# the others are supporting context. Total ~28k chars/filing keeps token
# cost bounded even on long filings.
SECTION_BUDGETS = {
    "mda": 12_000,
    "risk_changes": 4_000,
    "market_risk": 4_000,
    "controls": 4_000,
    "legal": 4_000,
}
SECTION_ORDER = ("mda", "risk_changes", "market_risk", "controls", "legal")

# Numeric feature columns — these are the only AI columns that ever enter the
# model feature matrix (V4/V5). Kept in sync with SEC_10Q_AI_COLS in
# predict_returns_ckx.py.
AI_NUMERIC_COLS = (
    "10q_ai_tone_score",
    "10q_ai_risk_score",
    "10q_ai_uncertainty_score",
    "10q_ai_margin_pressure",
    "10q_ai_liquidity_pressure",
    "10q_ai_demand_outlook",
    "10q_ai_disclosure_change_score",
    "10q_ai_material_change_flag",
)
# These two are TEXT — they flow to the panel for the dashboard but must
# never be used as model features.
AI_TEXT_COLS = ("10q_ai_summary", "10q_ai_evidence")

# Scores that only make sense relative to a prior filing; set to NaN when the
# filing is the first one for its ticker.
_CHANGE_COLS = ("10q_ai_disclosure_change_score", "10q_ai_material_change_flag")

_UNIT_RANGE = (-1.0, 1.0)
_POS_RANGE = (0.0, 1.0)
_SCORE_RANGES = {
    "tone_score": _UNIT_RANGE,
    "risk_score": _POS_RANGE,
    "uncertainty_score": _POS_RANGE,
    "margin_pressure": _POS_RANGE,
    "liquidity_pressure": _POS_RANGE,
    "demand_outlook": _UNIT_RANGE,
    "disclosure_change_score": _POS_RANGE,
}

RESPONSE_SCHEMA = {
    "name": "tenq_change_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "tone_score", "risk_score", "uncertainty_score", "margin_pressure",
            "liquidity_pressure", "demand_outlook", "disclosure_change_score",
            "material_change_flag", "summary", "evidence",
        ],
        "properties": {
            "tone_score": {"type": "number", "minimum": -1, "maximum": 1},
            "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
            "uncertainty_score": {"type": "number", "minimum": 0, "maximum": 1},
            "margin_pressure": {"type": "number", "minimum": 0, "maximum": 1},
            "liquidity_pressure": {"type": "number", "minimum": 0, "maximum": 1},
            "demand_outlook": {"type": "number", "minimum": -1, "maximum": 1},
            "disclosure_change_score": {"type": "number", "minimum": 0, "maximum": 1},
            "material_change_flag": {"type": "integer", "enum": [0, 1]},
            "summary": {"type": "string"},
            "evidence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["section", "quote", "why_it_matters"],
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": list(SECTION_ORDER),
                        },
                        "quote": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                    },
                },
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You are a meticulous equity research analyst. You read SEC 10-Q filings "
    "and compare each quarter's disclosure to the same company's previous "
    "10-Q. You score tone, risk, and disclosure change strictly and "
    "conservatively. You only cite text that actually appears in the provided "
    "excerpts. The excerpts may be truncated. Return ONLY the structured JSON "
    "object requested — no prose, no markdown."
)


# ---- text loading -------------------------------------------------------


def _read_section(rel_path, budget: int) -> str:
    """Resolve a cleaned_index.csv path and read it, truncated to `budget`."""
    if not isinstance(rel_path, str) or not rel_path:
        return ""
    p = resolve_sec10q_path(rel_path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    return text[:budget]


def load_sections(row: pd.Series) -> tuple[dict[str, str], str]:
    """Load the five 10-Q sections for one filing.

    Returns (sections, source). When MD&A extraction failed, the MD&A slot
    falls back to the full narrative and `source` is "narrative_fallback".
    """
    sections = {
        "mda": _read_section(row.get("mda_path"), SECTION_BUDGETS["mda"]),
        "risk_changes": _read_section(
            row.get("risk_changes_path"), SECTION_BUDGETS["risk_changes"]
        ),
        "market_risk": _read_section(
            row.get("market_risk_path"), SECTION_BUDGETS["market_risk"]
        ),
        "controls": _read_section(
            row.get("controls_path"), SECTION_BUDGETS["controls"]
        ),
        "legal": _read_section(row.get("legal_path"), SECTION_BUDGETS["legal"]),
    }
    status = str(row.get("extraction_status", ""))
    source = "mda"
    if status != "ok" or not sections["mda"]:
        narrative = _read_section(row.get("narrative_path"), SECTION_BUDGETS["mda"])
        if narrative:
            sections["mda"] = narrative
            source = "narrative_fallback"
    return sections, source


def _has_any_text(sections: dict[str, str]) -> bool:
    return any(v for v in sections.values())


# ---- prompt -------------------------------------------------------------


def _sections_block(sections: dict[str, str]) -> str:
    parts = []
    for key in SECTION_ORDER:
        text = sections.get(key, "")
        label = key.upper().replace("_", " ")
        parts.append(f"[{label}]\n{text if text else '(not available)'}")
    return "\n\n".join(parts)


def build_user_prompt(
    ticker: str,
    cur_row: pd.Series,
    cur_sections: dict[str, str],
    prev_row: pd.Series | None,
    prev_sections: dict[str, str] | None,
) -> str:
    def _fmt_date(value) -> str:
        ts = pd.Timestamp(value)
        return ts.date().isoformat() if pd.notna(ts) else "unknown"

    if prev_row is not None:
        prev_meta = (
            f"accession {prev_row['accession_number']}, "
            f"period {_fmt_date(prev_row.get('report_period'))}, "
            f"filed {_fmt_date(prev_row.get('filing_date'))}"
        )
        prev_block = _sections_block(prev_sections or {})
    else:
        prev_meta = "NONE (this is the earliest filing for this ticker)"
        prev_block = "(no previous filing available)"

    return f"""Company ticker: {ticker}
Current filing: accession {cur_row['accession_number']}, \
period {_fmt_date(cur_row.get('report_period'))}, \
filed {_fmt_date(cur_row.get('filing_date'))}
Previous filing: {prev_meta}

Score the CURRENT filing on these dimensions (use the stated scales):
  - tone_score (-1..+1): management tone, negative..positive
  - risk_score (0..1): overall risk emphasis
  - uncertainty_score (0..1): hedging / conditionality / unpredictability
  - margin_pressure (0..1): cost inflation, pricing pressure, margin compression
  - liquidity_pressure (0..1): cash / financing / covenant / debt strain
  - demand_outlook (-1..+1): demand weakening..strengthening
  - disclosure_change_score (0..1): how materially the disclosure changed vs the
        PREVIOUS filing (0 = identical boilerplate, 1 = substantial change). If
        there is no previous filing, return 0 — the calling code overrides it.
  - material_change_flag (0 or 1): would an analyst flag a material change vs
        the prior filing? If there is no previous filing, return 0.

Then write:
  - summary: 2-3 sentences on what changed this quarter and why it matters.
  - evidence: 1-5 short verbatim quotes (each <= 60 words) drawn ONLY from the
    excerpts below, each tagged with its section and a one-line
    "why_it_matters". Prefer quotes that show CHANGE versus the prior filing.

=== CURRENT FILING EXCERPTS ===
{_sections_block(cur_sections)}

=== PREVIOUS FILING EXCERPTS ===
{prev_block}
"""


# ---- caching ------------------------------------------------------------


def _cache_path(accession: str, prev_accession: str | None) -> Path:
    prev = prev_accession or "NONE"
    raw = f"{accession}__{prev}__{PROMPT_VERSION}"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    return SEC_10Q_LLM_CACHE_DIR / f"{safe}.json"


def _load_cached(path: Path) -> dict | None:
    if not (USE_CACHE and path.exists()):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("response")
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(
    path: Path,
    response: dict,
    *,
    ticker: str,
    accession: str,
    prev_accession: str | None,
    model: str,
) -> None:
    SEC_10Q_LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "accession_number": accession,
        "prev_accession_number": prev_accession,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "response": response,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---- LLM call -----------------------------------------------------------


def _call_with_retry(client, messages, max_retries: int = 6) -> dict:
    """Call the chat completion endpoint, retrying on rate limits."""
    from openai import RateLimitError

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=SEC_10Q_LLM_MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
            )
            return json.loads(resp.choices[0].message.content)
        except RateLimitError as e:
            last_err = e
            wait = min(2.0 * (2 ** attempt), 60.0)
            print(f"    rate limited, sleeping {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"    bad JSON from model (attempt {attempt + 1}/{max_retries})")
    raise RuntimeError(f"_call_with_retry: exhausted retries ({last_err})")


def validate_response(resp: dict) -> dict:
    """Validate a model response against the expected schema/ranges.

    Raises ValueError on any violation. `strict` json_schema enforces most of
    this server-side, but we re-check defensively so a malformed response
    triggers the retry/fallback path instead of poisoning the panel.
    """
    for key, (lo, hi) in _SCORE_RANGES.items():
        if key not in resp:
            raise ValueError(f"missing field: {key}")
        val = resp[key]
        if not isinstance(val, (int, float)) or not (lo <= float(val) <= hi):
            raise ValueError(f"{key}={val!r} outside [{lo}, {hi}]")
    flag = resp.get("material_change_flag")
    if flag not in (0, 1):
        raise ValueError(f"material_change_flag={flag!r} not in {{0, 1}}")
    if not isinstance(resp.get("summary"), str) or not resp["summary"].strip():
        raise ValueError("summary missing or empty")
    evidence = resp.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence missing or empty")
    for item in evidence:
        if item.get("section") not in SECTION_ORDER:
            raise ValueError(f"evidence section invalid: {item.get('section')!r}")
    return resp


def analyze_filing(
    client,
    ticker: str,
    cur_row: pd.Series,
    cur_sections: dict[str, str],
    prev_row: pd.Series | None,
    prev_sections: dict[str, str] | None,
) -> dict:
    """Return a validated model response for one filing (cached if available)."""
    accession = str(cur_row["accession_number"])
    prev_accession = (
        str(prev_row["accession_number"]) if prev_row is not None else None
    )
    cache_path = _cache_path(accession, prev_accession)

    cached = _load_cached(cache_path)
    if cached is not None:
        try:
            return validate_response(cached)
        except ValueError:
            pass  # stale/invalid cache — fall through and re-call

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                ticker, cur_row, cur_sections, prev_row, prev_sections
            ),
        },
    ]
    last_err: Exception | None = None
    for _ in range(2):  # one call + one retry on validation failure
        resp = _call_with_retry(client, messages)
        try:
            resp = validate_response(resp)
        except ValueError as e:
            last_err = e
            continue
        _write_cache(
            cache_path, resp, ticker=ticker, accession=accession,
            prev_accession=prev_accession, model=SEC_10Q_LLM_MODEL,
        )
        return resp
    raise RuntimeError(f"analyze_filing: validation failed twice ({last_err})")


# ---- record assembly ----------------------------------------------------


def _failed_record() -> dict:
    """All-NaN feature payload for filings the model could not analyze."""
    rec = {c: float("nan") for c in AI_NUMERIC_COLS}
    rec["10q_ai_summary"] = "ANALYSIS_FAILED"
    rec["10q_ai_evidence"] = "[]"
    return rec


def response_to_record(resp: dict, *, has_previous: bool) -> dict:
    """Map a validated model response onto the AI feature columns."""
    rec = {
        "10q_ai_tone_score": float(resp["tone_score"]),
        "10q_ai_risk_score": float(resp["risk_score"]),
        "10q_ai_uncertainty_score": float(resp["uncertainty_score"]),
        "10q_ai_margin_pressure": float(resp["margin_pressure"]),
        "10q_ai_liquidity_pressure": float(resp["liquidity_pressure"]),
        "10q_ai_demand_outlook": float(resp["demand_outlook"]),
        "10q_ai_disclosure_change_score": float(resp["disclosure_change_score"]),
        "10q_ai_material_change_flag": int(resp["material_change_flag"]),
        "10q_ai_summary": str(resp["summary"]).strip(),
        "10q_ai_evidence": json.dumps(resp["evidence"], ensure_ascii=False),
    }
    # Change-vs-previous scores are undefined for the first filing of a ticker.
    if not has_previous:
        for col in _CHANGE_COLS:
            rec[col] = float("nan")
    return rec


# ---- orchestrator -------------------------------------------------------


def analyze_filings(
    cleaned_index_path: Path = SEC_10Q_META_DIR / "cleaned_index.csv",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    if not OPENAI_API_KEY:
        print(
            "Skipping 10-Q LLM analysis: OPENAI_API_KEY is not set. "
            "Add it to .env to run the optional generative-AI 10-Q stage."
        )
        return pd.DataFrame()

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    tickers = tickers or DEFAULT_TICKERS

    df = pd.read_csv(cleaned_index_path, parse_dates=["filing_date", "report_period"])
    df = df[df["ticker"].isin(tickers)].copy()
    df = df[df["filing_date"].dt.year >= SEC_10Q_LLM_START_YEAR]
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    print(
        f"Analyzing {len(df)} filings across {df['ticker'].nunique()} tickers "
        f"with {SEC_10Q_LLM_MODEL} (filings from {SEC_10Q_LLM_START_YEAR}+, "
        f"prompt {PROMPT_VERSION})"
    )

    prev_row_by_ticker: dict[str, pd.Series | None] = {t: None for t in tickers}
    prev_sections_by_ticker: dict[str, dict[str, str] | None] = {
        t: None for t in tickers
    }

    rows: list[dict] = []
    n_failed = 0
    for _, row in df.iterrows():
        ticker = row["ticker"]
        accession = str(row["accession_number"])
        cur_sections, source = load_sections(row)
        prev_row = prev_row_by_ticker.get(ticker)
        prev_sections = prev_sections_by_ticker.get(ticker)

        base = {
            "ticker": ticker,
            "accession_number": accession,
            "filing_date": row["filing_date"],
            "report_period": row.get("report_period"),
            "prev_accession_number": (
                str(prev_row["accession_number"]) if prev_row is not None else None
            ),
            "ai_feature_source": source,
            "prompt_version": PROMPT_VERSION,
        }

        if not _has_any_text(cur_sections):
            print(f"  skip {ticker} {accession}: no usable section text")
            rows.append({**base, **_failed_record()})
            n_failed += 1
            # Do not advance prev_* — a text-less filing is not a useful baseline.
            continue

        try:
            resp = analyze_filing(
                client, ticker, row, cur_sections, prev_row, prev_sections
            )
            rec = response_to_record(resp, has_previous=prev_row is not None)
        except RuntimeError as e:
            print(f"  FAIL {ticker} {accession}: {e}")
            rec = _failed_record()
            n_failed += 1
        else:
            print(
                f"  {ticker} {accession}: tone={rec['10q_ai_tone_score']:+.2f} "
                f"risk={rec['10q_ai_risk_score']:.2f} "
                f"change={rec['10q_ai_disclosure_change_score']!s:>5} "
                f"flag={rec['10q_ai_material_change_flag']!s}"
            )

        rows.append({**base, **rec})
        prev_row_by_ticker[ticker] = row
        prev_sections_by_ticker[ticker] = cur_sections

    features = pd.DataFrame(rows)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    print(
        f"\nWrote {FEATURES_PATH}  ({len(features)} rows, {n_failed} failed)"
    )
    if not features.empty:
        counts = features.groupby("ticker").size().to_dict()
        print(f"Per-ticker counts: {counts}")
    return features


def _spot_check(features: pd.DataFrame, n: int = 4) -> None:
    """Print a few random analyzed filings for a sanity read."""
    ok = features[features["10q_ai_summary"] != "ANALYSIS_FAILED"]
    if ok.empty:
        return
    sample = ok.sample(min(n, len(ok)), random_state=0)
    print("\n--- spot check ---")
    for _, r in sample.iterrows():
        print(
            f"[{r['ticker']} {r['accession_number']} "
            f"{pd.Timestamp(r['filing_date']).date()}]"
        )
        print(f"  tone={r['10q_ai_tone_score']:+.2f} "
              f"risk={r['10q_ai_risk_score']:.2f} "
              f"uncertainty={r['10q_ai_uncertainty_score']:.2f} "
              f"change={r['10q_ai_disclosure_change_score']}")
        print(f"  summary: {r['10q_ai_summary']}")
        try:
            ev = json.loads(r["10q_ai_evidence"])
            if ev:
                print(f"  evidence[0] ({ev[0]['section']}): {ev[0]['quote']}")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Generative-AI 10-Q analysis.")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Subset of tickers to analyze (default: all). Useful for a pilot.",
    )
    args = parser.parse_args()

    features = analyze_filings(tickers=args.tickers)
    if not features.empty:
        _spot_check(features)
    sys.exit(0)


if __name__ == "__main__":
    main()
