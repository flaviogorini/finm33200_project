"""Quality tests for the 10-Q cleaning pipeline.

These tests run against whatever processed_text files already exist on
disk (produced by `python src/clean_sec_10q_text.py`). If no cleaned
filings are present, the tests are skipped — they're regression checks,
not full end-to-end smoke tests.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from pull_sec_10q_filings import resolve_sec10q_path
from settings import SEC_10Q_DIR, SEC_10Q_META_DIR


CLEANED_INDEX = SEC_10Q_META_DIR / "cleaned_index.csv"


def _narrative_paths() -> list[Path]:
    return sorted(SEC_10Q_DIR.glob("*/processed_text/*__narrative.txt"))


def _mda_paths() -> list[Path]:
    return sorted(SEC_10Q_DIR.glob("*/processed_text/*__mda.txt"))


def _require(paths: list[Path], label: str) -> list[Path]:
    if not paths:
        pytest.skip(f"No {label} found — run `python src/clean_sec_10q_text.py` first.")
    return paths


FORBIDDEN_HEADER_TOKENS = [
    "CONFORMED SUBMISSION TYPE",
    "ACCESSION NUMBER:",
    "BUSINESS ADDRESS:",
    "MAIL ADDRESS:",
    "FILER:",
    "STANDARD INDUSTRIAL CLASSIFICATION",
]
XBRL_LINE_RE = re.compile(r"(?im)^\s*(?:R\d+\.htm|[\w\-]+\.x[ms][dl]|[\w\-]+\.zip)\s*$")


def test_narrative_strips_sec_header():
    for p in _require(_narrative_paths(), "narrative files"):
        text = p.read_text(encoding="utf-8")
        for token in FORBIDDEN_HEADER_TOKENS:
            assert token not in text, f"{p.name}: leaked header token {token!r}"


def test_narrative_strips_xbrl_trailer():
    for p in _require(_narrative_paths(), "narrative files"):
        text = p.read_text(encoding="utf-8")
        match = XBRL_LINE_RE.search(text)
        assert match is None, (
            f"{p.name}: leaked XBRL nav line {text[match.start():match.start()+80]!r}"
        )
        assert "IDEA: XBRL DOCUMENT" not in text, f"{p.name}: leaked IDEA marker"


def test_narrative_word_count_is_reasonable():
    paths = _require(_narrative_paths(), "narrative files")
    counts = [(p, len(p.read_text(encoding="utf-8").split())) for p in paths]
    too_small = [(p.name, c) for p, c in counts if c < 1_000]
    too_large = [(p.name, c) for p, c in counts if c > 60_000]
    assert not too_small, f"narrative too small: {too_small}"
    assert not too_large, f"narrative too large (likely cleanup failed): {too_large}"


def test_mda_word_count_is_reasonable():
    paths = _require(_mda_paths(), "mda files")
    counts = [(p, len(p.read_text(encoding="utf-8").split())) for p in paths]
    too_large = [(p.name, c) for p, c in counts if c > 30_000]
    assert not too_large, f"MD&A too large (likely Part II bled in): {too_large}"

    if len(counts) >= 10:
        substantive = sum(1 for _, c in counts if c >= 1_000)
        ratio = substantive / len(counts)
        assert ratio >= 0.85, (
            f"only {substantive}/{len(counts)} MD&A files have >=1k words "
            f"(ratio {ratio:.0%}); cleaner is regressing"
        )


def test_narrative_has_low_char_per_word_ratio():
    """Heavy markup (raw filings) ~10-20 chars/word; clean prose ~5-7."""
    paths = _require(_narrative_paths(), "narrative files")
    bad = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        words = text.split()
        if not words:
            continue
        ratio = len(text) / len(words)
        if ratio > 8.5:
            bad.append((p.name, round(ratio, 2)))
    assert not bad, f"chars/word ratio too high (markup leakage?): {bad}"


def test_mda_starts_with_managements_discussion():
    """When extraction status is 'ok', the MD&A body should begin with the
    section title (modulo whitespace and apostrophes lost during HTML strip)."""
    if not CLEANED_INDEX.exists():
        pytest.skip("cleaned_index.csv missing — run clean_sec_10q_text.py first.")
    idx = pd.read_csv(CLEANED_INDEX)
    ok = idx[idx["extraction_status"] == "ok"]
    if ok.empty:
        pytest.skip("no filings with extraction_status='ok'")
    bad = []
    for _, row in ok.iterrows():
        path = resolve_sec10q_path(row["mda_path"]) if pd.notna(row.get("mda_path")) else None
        if path is None or not path.exists():
            continue
        head = path.read_text(encoding="utf-8")[:300].lower()
        if "management" not in head or "discussion" not in head:
            bad.append(path.name)
    assert not bad, f"MD&A body doesn't open with the expected heading: {bad}"
