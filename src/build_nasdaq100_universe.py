"""Build the current Nasdaq-100 universe file for transcript mapping.

This script creates only metadata used by the transcript extraction planning
stage. It does not pull transcripts, clean text, or create embeddings.
"""

from __future__ import annotations

import re
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from settings import config


DATA_DIR = Path(config("DATA_DIR"))
META_DIR = DATA_DIR / "transcripts" / "_meta"
OUTPUT_PATH = META_DIR / "nasdaq100_constituents.csv"

SOURCE_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
SOURCE_NOTE = (
    "Parsed from the Wikipedia Nasdaq-100 current components table; "
    "cross-check note: Nasdaq official companies page observed during setup "
    "was marked Last updated 05/19/2025 and may lag current 2026 changes."
)


def _normalise_company_name(value: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value))
    text = re.sub(r"\b(inc|inc\.|corp|corp\.|corporation|plc|ltd|limited|class|cl)\b", "", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().upper()


def _primary_rank(row: pd.Series) -> tuple[int, str]:
    company = str(row["company_name"]).lower()
    ticker = str(row["ticker"])
    if "class a" in company or "cl a" in company:
        return (0, ticker)
    if "class c" in company or "cl c" in company:
        return (2, ticker)
    return (1, ticker)


def load_current_components(source_url: str = SOURCE_URL) -> pd.DataFrame:
    request = urllib.request.Request(
        source_url,
        headers={
            "User-Agent": (
                "finm33200_project Nasdaq-100 universe builder "
                "(academic research; contact via project owner)"
            )
        },
    )
    with urllib.request.urlopen(request) as response:
        tables = pd.read_html(response)
    for table in tables:
        cols = [str(c).strip() for c in table.columns]
        lower = {c.lower(): c for c in cols}
        if {
            "ticker",
            "company",
            "icb industry[14]",
            "icb subsector[14]",
        }.issubset(lower):
            out = table.rename(
                columns={
                    lower["ticker"]: "ticker",
                    lower["company"]: "company_name",
                    lower["icb industry[14]"]: "sector",
                    lower["icb subsector[14]"]: "industry",
                }
            )[["ticker", "company_name", "sector", "industry"]]
            out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
            out["company_name"] = out["company_name"].astype(str).str.strip()
            out["sector"] = out["sector"].astype(str).str.strip()
            out["industry"] = out["industry"].astype(str).str.strip()
            return out.sort_values("ticker").reset_index(drop=True)
    raise RuntimeError("Could not find the Nasdaq-100 current components table.")


def add_share_class_metadata(universe: pd.DataFrame) -> pd.DataFrame:
    out = universe.copy()
    out["company_group_key"] = out["company_name"].map(_normalise_company_name)
    primary_by_group: dict[str, str] = {}
    related_by_group: dict[str, str] = {}
    for group_key, grp in out.groupby("company_group_key", sort=False):
        tickers = sorted(grp["ticker"].tolist())
        primary = (
            grp.assign(_rank=grp.apply(_primary_rank, axis=1))
            .sort_values("_rank")
            .iloc[0]["ticker"]
        )
        primary_by_group[group_key] = primary
        related_by_group[group_key] = "|".join(tickers) if len(tickers) > 1 else ""

    out["exchange"] = "Nasdaq"
    out["is_current_nasdaq100"] = True
    out["primary_ticker"] = out["company_group_key"].map(primary_by_group)
    out["related_tickers"] = out["company_group_key"].map(related_by_group)
    out["is_primary_share_class"] = out["ticker"] == out["primary_ticker"]
    out["notes"] = out["related_tickers"].map(
        lambda x: "multi_share_class_group" if x else ""
    )
    out["universe_as_of_date"] = date.today().isoformat()
    out["source"] = f"{SOURCE_URL}; {SOURCE_NOTE}"
    return out[
        [
            "ticker",
            "company_name",
            "exchange",
            "sector",
            "industry",
            "is_current_nasdaq100",
            "is_primary_share_class",
            "primary_ticker",
            "related_tickers",
            "notes",
            "universe_as_of_date",
            "source",
        ]
    ]


def main() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    universe = add_share_class_metadata(load_current_components())
    universe.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(universe):,} rows to {OUTPUT_PATH}")
    print(f"Unique company-level groups: {universe['primary_ticker'].nunique():,}")
    print("Multi-share-class groups:")
    multi = universe[universe["related_tickers"] != ""][
        ["ticker", "company_name", "primary_ticker", "related_tickers"]
    ]
    print(multi.to_string(index=False) if not multi.empty else "  none")


if __name__ == "__main__":
    main()
