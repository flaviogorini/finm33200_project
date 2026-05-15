"""Clean SEC 10-Q filings into per-section narrative text files.

For each filing we write multiple outputs to
`_data/sec_10q/{ticker}/processed_text/`:
  - {accession}__narrative.txt : SGML-stripped, HTML-stripped, tables removed.
  - {accession}__full.txt      : same as narrative but with tables flattened
                                 to pipe-separated rows (numbers preserved).
  - {accession}__mda.txt       : Part I, Item 2 (Management's Discussion).
  - {accession}__market_risk.txt : Part I, Item 3.
  - {accession}__controls.txt    : Part I, Item 4.
  - {accession}__legal.txt       : Part II, Item 1.
  - {accession}__risk_changes.txt : Part II, Item 1A (often near-empty).

Section files are written only when extraction succeeds. The cleaned-index
records `extraction_status` per filing so downstream scoring can skip
filings where MD&A wasn't found rather than silently rescore the full doc.
"""

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from settings import (
    DEFAULT_TICKERS,
    SEC_10Q_META_DIR,
    create_sec_10q_dirs,
    processed_text_dir,
)
from pull_sec_10q_filings import _rel_to_sec10q, resolve_sec10q_path


PERIOD_OF_REPORT_RE = re.compile(
    r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", re.IGNORECASE
)

_DOC_RE = re.compile(r"<DOCUMENT>(?P<body>.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
_TYPE_RE = re.compile(r"<TYPE>([^\r\n<]+)", re.IGNORECASE)
_TEXT_RE = re.compile(r"<TEXT>(?P<body>.*?)</TEXT>", re.IGNORECASE | re.DOTALL)

_XBRL_TAIL_RE = re.compile(
    r"(?im)^\s*(?:R\d+\.htm|[\w\-]+\.x[ms][dl]|[\w\-]+\.zip|IDEA:\s*XBRL\s+DOCUMENT)\s*$"
)


def extract_period_of_report(raw_text: str) -> pd.Timestamp | None:
    match = PERIOD_OF_REPORT_RE.search(raw_text)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), format="%Y%m%d")
    except (ValueError, TypeError):
        return None


def extract_10q_body(raw_text: str) -> str:
    """Return the inside of the <DOCUMENT><TYPE>10-Q...<TEXT>...</TEXT></DOCUMENT>
    block. Falls back to the full text when SGML markers are absent."""
    for match in _DOC_RE.finditer(raw_text):
        body = match.group("body")
        type_match = _TYPE_RE.search(body)
        if type_match and type_match.group(1).strip().upper().startswith("10-Q"):
            text_match = _TEXT_RE.search(body)
            return text_match.group("body") if text_match else body

    lines = raw_text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        s = line.strip().upper()
        if s == "10-Q" or s.startswith("UNITED STATES SECURITIES"):
            start = i
            break
    return "\n".join(lines[start:])


def truncate_xbrl_tail(text: str) -> str:
    match = _XBRL_TAIL_RE.search(text)
    return text[: match.start()].rstrip() if match else text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _flatten_table(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def html_to_text(text: str, *, keep_tables: bool) -> str:
    """Strip HTML/XML tags. Either drop tables or flatten them to pipe-rows."""
    if "<" not in text or ">" not in text:
        return _normalize_whitespace(text)

    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    if keep_tables:
        for table in soup.find_all("table"):
            flat = _flatten_table(table)
            table.replace_with("\n" + flat + "\n")
    else:
        for table in soup.find_all("table"):
            table.decompose()

    return _normalize_whitespace(soup.get_text("\n"))


_ITEM_RE = re.compile(r"(?<!, )\bItem\s+(\d+[A-Z]?)\s*[\.\:\-–—]", re.IGNORECASE)
_PART_RE = re.compile(r"\bPART\s+(I{1,3}|IV)\.\s+[A-Z][A-Z]")
_PART_FLIP_BODY_MIN = 200

SECTION_NAME: dict[tuple[str, str], str] = {
    ("I",  "2"):  "mda",
    ("I",  "3"):  "market_risk",
    ("I",  "4"):  "controls",
    ("II", "1"):  "legal",
    ("II", "1a"): "risk_changes",
}


def _build_part_boundaries(narrative: str) -> list[tuple[str, int]]:
    matches = list(_PART_RE.finditer(narrative))
    if not matches:
        return []
    firsts: dict[str, int] = {}
    for m in matches:
        part = m.group(1).upper()
        firsts.setdefault(part, m.start())
    if "I" not in firsts or "II" not in firsts:
        return []
    return sorted(firsts.items(), key=lambda kv: kv[1])


def _part_for_offset(boundaries: list[tuple[str, int]], offset: int) -> str:
    current = boundaries[0][0]
    for part, start in boundaries:
        if offset >= start:
            current = part
        else:
            break
    return current


def extract_sections(narrative: str) -> tuple[dict[str, str], str]:
    item_matches = list(_ITEM_RE.finditer(narrative))
    if not item_matches:
        return {}, "none"

    boundaries = _build_part_boundaries(narrative)
    candidates: dict[tuple[str, str], list[tuple[int, int]]] = {}

    if boundaries:
        part_detection = "marker"
        for i, m in enumerate(item_matches):
            item_id = m.group(1).lower()
            start = m.end()
            end = item_matches[i + 1].start() if i + 1 < len(item_matches) else len(narrative)
            part = _part_for_offset(boundaries, m.start())
            candidates.setdefault((part, item_id), []).append((start, end))
    else:
        part_detection = "inferred"
        part = "I"
        seen_in_part: set[str] = set()
        for i, m in enumerate(item_matches):
            item_id = m.group(1).lower()
            start = m.end()
            end = item_matches[i + 1].start() if i + 1 < len(item_matches) else len(narrative)
            body_len = end - start
            if body_len >= _PART_FLIP_BODY_MIN:
                if item_id == "1" and seen_in_part & {"2", "3", "4"}:
                    part = "II"
                    seen_in_part = set()
                seen_in_part.add(item_id)
            candidates.setdefault((part, item_id), []).append((start, end))

    out: dict[str, str] = {}
    for key, name in SECTION_NAME.items():
        spans = candidates.get(key)
        if not spans:
            continue
        best_start, best_end = max(spans, key=lambda se: se[1] - se[0])
        body = narrative[best_start:best_end].strip()
        if body:
            out[name] = body
    return out, part_detection


def clean_one_file(source_path: Path) -> dict:
    raw = source_path.read_text(encoding="utf-8", errors="ignore")
    period = extract_period_of_report(raw)
    body = extract_10q_body(raw)

    narrative = truncate_xbrl_tail(html_to_text(body, keep_tables=False))
    full      = truncate_xbrl_tail(html_to_text(body, keep_tables=True))
    sections, part_detection = extract_sections(narrative)

    mda_words = len(sections.get("mda", "").split())
    if mda_words >= 200:
        status = "ok"
    elif "mda" in sections:
        status = f"weak_mda_{mda_words}w"
    elif part_detection == "none":
        status = "failed_no_items"
    else:
        status = "failed_no_mda"

    return {
        "report_period": period,
        "narrative": narrative,
        "full": full,
        "sections": sections,
        "narrative_word_count": len(narrative.split()),
        "extraction_status": status,
        "part_detection": part_detection,
    }


def clean_filings(
    metadata_path: Path = SEC_10Q_META_DIR / "filing_index.csv",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    create_sec_10q_dirs(tickers)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Filing index not found at {metadata_path}. "
            f"Run `python src/pull_sec_10q_filings.py` first."
        )
    metadata = pd.read_csv(metadata_path, parse_dates=["filing_date"])
    if "ticker" in metadata.columns:
        metadata = metadata[metadata["ticker"].isin(tickers)].reset_index(drop=True)

    rows = []
    for i, row in metadata.iterrows():
        source_path = resolve_sec10q_path(row["clean_local_path"])
        if not source_path.exists():
            print(f"  skip (missing): {source_path}")
            continue

        result = clean_one_file(source_path)
        ticker = row.get("ticker", "AAPL")
        accession = row["accession_number"]
        out_dir = processed_text_dir(ticker)
        out_dir.mkdir(parents=True, exist_ok=True)

        narrative_path = out_dir / f"{accession}__narrative.txt"
        full_path      = out_dir / f"{accession}__full.txt"
        narrative_path.write_text(result["narrative"], encoding="utf-8")
        full_path.write_text(result["full"], encoding="utf-8")

        section_paths: dict[str, str | None] = {
            f"{name}_path": None for name in SECTION_NAME.values()
        }
        section_word_counts: dict[str, int] = {}
        for name, text in result["sections"].items():
            p = out_dir / f"{accession}__{name}.txt"
            p.write_text(text, encoding="utf-8")
            section_paths[f"{name}_path"] = _rel_to_sec10q(p)
            section_word_counts[f"{name}_word_count"] = len(text.split())

        rows.append({
            **row.to_dict(),
            "report_period": result["report_period"],
            "narrative_path": _rel_to_sec10q(narrative_path),
            "full_path": _rel_to_sec10q(full_path),
            "narrative_word_count": result["narrative_word_count"],
            "extraction_status": result["extraction_status"],
            "part_detection": result["part_detection"],
            **section_paths,
            **section_word_counts,
        })

        period_str = (
            result["report_period"].date().isoformat()
            if pd.notna(result["report_period"]) else "unknown"
        )
        print(
            f"[{i + 1}/{len(metadata)}] {ticker} {accession} "
            f"period={period_str} narrative={result['narrative_word_count']:,}w "
            f"status={result['extraction_status']} "
            f"parts={result['part_detection']} "
            f"sections={sorted(result['sections'])}"
        )

    SEC_10Q_META_DIR.mkdir(parents=True, exist_ok=True)
    out = SEC_10Q_META_DIR / "cleaned_index.csv"
    cleaned_index = pd.DataFrame(rows)
    cleaned_index.to_csv(out, index=False)
    print(f"Wrote {out}")
    return cleaned_index


if __name__ == "__main__":
    clean_filings()
