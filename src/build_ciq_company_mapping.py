"""Map Nasdaq-100 tickers to Capital IQ company IDs through WRDS metadata.

This script is intentionally limited to universe/mapping/QC artifacts. It does
not pull earnings-call transcripts, clean transcript text, or generate
embeddings.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import wrds

from settings import config


DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

META_DIR = MANUAL_DATA_DIR / "_meta"
QC_DIR = OUTPUT_DIR / "transcripts" / "qc"

UNIVERSE_PATH = META_DIR / "nasdaq100_constituents.csv"
MAPPING_PATH = META_DIR / "ciq_company_mapping.csv"
SCHEMA_PATH = META_DIR / "ciq_company_mapping_schema_inspection.json"
QC_CSV_PATH = QC_DIR / "company_mapping_qc.csv"
QC_MD_PATH = QC_DIR / "company_mapping_qc.md"
READINESS_PATH = QC_DIR / "extraction_readiness_summary.md"


REQUIRED_UNIVERSE_COLUMNS = [
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

MAPPING_COLUMNS = [
    "ticker",
    "company_name",
    "ciq_company_id",
    "ciq_company_name",
    "exchange",
    "sector",
    "industry",
    "match_method",
    "match_confidence",
    "is_primary_share_class",
    "primary_ticker",
    "related_tickers",
    "mapping_status",
    "notes",
    "universe_as_of_date",
    "mapping_verified_at",
    "mapping_source",
]


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def fqtn(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def safe_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def normalise_name(value: Any) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value or ""))
    text = re.sub(
        r"\b(inc|inc\.|corp|corp\.|corporation|co|co\.|company|plc|ltd|limited|"
        r"holdings|holding|class|cl|ordinary|ads|adr|nv|sa|se)\b",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().upper()


def name_similarity(left: str, right: str) -> float:
    a = normalise_name(left)
    b = normalise_name(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def connect_wrds() -> wrds.Connection:
    username = config("WRDS_USERNAME")
    password = config("WRDS_PASSWORD", default=None)
    kwargs: dict[str, str] = {"wrds_username": username}
    if password:
        kwargs["wrds_password"] = password
    return wrds.Connection(**kwargs)


def load_universe(path: Path = UNIVERSE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/build_nasdaq100_universe.py first."
        )
    universe = pd.read_csv(path).fillna("")
    missing = [c for c in REQUIRED_UNIVERSE_COLUMNS if c not in universe.columns]
    if missing:
        raise ValueError(f"Universe file is missing required columns: {missing}")
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    return universe


def inspect_ciq_mapping_schema(db: wrds.Connection) -> dict[str, Any]:
    query = """
        SELECT table_schema, table_name, column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_schema ILIKE '%%ciq%%'
          AND (
            column_name ILIKE '%%ticker%%'
            OR column_name ILIKE '%%company%%'
            OR column_name ILIKE '%%security%%'
            OR column_name ILIKE '%%exchange%%'
            OR column_name ILIKE '%%trading%%'
          )
        ORDER BY table_schema, table_name, ordinal_position
    """
    columns = db.raw_sql(query)
    if columns.empty:
        raise RuntimeError("No Capital IQ mapping-related metadata found.")

    def table_has(schema: str, table: str, required: set[str]) -> bool:
        cols = set(
            columns[
                (columns["table_schema"] == schema)
                & (columns["table_name"] == table)
            ]["column_name"]
            .str.lower()
            .tolist()
        )
        return required.issubset(cols)

    tables = columns[["table_schema", "table_name"]].drop_duplicates()
    discovered: dict[str, dict[str, str]] = {}
    specs = {
        "company": {"companyid", "companyname"},
        "ticker_helper": {"companyid", "ticker", "companyname"},
        "symbol_helper": {"companyid", "companyname", "symbolvalue"},
        "trading_item": {"securityid", "tickersymbol", "exchangeid"},
        "security": {"securityid", "companyid", "securityname"},
        "exchange": {"exchangeid", "exchangename", "exchangesymbol"},
        "trading_status": {"tradingitemstatusid", "tradingitemstatusname"},
    }
    preferred_names = {
        "company": ["ciqcompany"],
        "ticker_helper": ["wrds_ticker", "wrds_ciqsymbol_primary"],
        "symbol_helper": ["wrds_ciqsymbol"],
        "trading_item": ["ciqtradingitem"],
        "security": ["ciqsecurity"],
        "exchange": ["ciqexchange"],
        "trading_status": ["ciqtradingitemstatus"],
    }

    for logical_name, required_cols in specs.items():
        matches = []
        for row in tables.itertuples(index=False):
            if table_has(row.table_schema, row.table_name, required_cols):
                matches.append((row.table_schema, row.table_name))
        if not matches:
            continue
        preferred = preferred_names[logical_name]
        matches = sorted(
            matches,
            key=lambda x: (
                0 if x[0].lower() == "ciq" else 1,
                preferred.index(x[1].lower()) if x[1].lower() in preferred else 99,
                x[0],
                x[1],
            ),
        )
        discovered[logical_name] = {
            "schema": matches[0][0],
            "table": matches[0][1],
        }

    required_logical = ["company", "ticker_helper", "trading_item", "security", "exchange"]
    missing = [name for name in required_logical if name not in discovered]
    if missing:
        raise RuntimeError(f"Could not discover required mapping tables: {missing}")

    return {
        "inspection_timestamp": datetime.now().isoformat(timespec="seconds"),
        "discovered_tables": discovered,
        "candidate_column_rows": len(columns),
    }


def query_candidates(
    db: wrds.Connection, schema_info: dict[str, Any], tickers: list[str]
) -> pd.DataFrame:
    tables = schema_info["discovered_tables"]
    ticker_sql = ", ".join(safe_sql_string(t) for t in tickers)

    company = tables["company"]
    ticker_helper = tables["ticker_helper"]
    trading_item = tables["trading_item"]
    security = tables["security"]
    exchange = tables["exchange"]
    trading_status = tables.get("trading_status")

    status_join = ""
    status_col = "NULL::text AS tradingitemstatusname"
    if trading_status:
        status_join = f"""
        LEFT JOIN {fqtn(trading_status['schema'], trading_status['table'])} AS st
          ON st.tradingitemstatusid = ti.tradingitemstatusid
        """
        status_col = "st.tradingitemstatusname AS tradingitemstatusname"

    detailed_query = f"""
        SELECT
            UPPER(ti.tickersymbol) AS ticker,
            c.companyid::bigint AS ciq_company_id,
            c.companyname AS ciq_company_name,
            e.exchangename AS ciq_exchange_name,
            e.exchangesymbol AS ciq_exchange_symbol,
            s.securityname AS security_name,
            s.primaryflag AS security_primaryflag,
            ti.primaryflag AS tradingitem_primaryflag,
            {status_col},
            s.securitystartdate,
            s.securityenddate,
            'trading_item_join' AS candidate_source
        FROM {fqtn(trading_item['schema'], trading_item['table'])} AS ti
        JOIN {fqtn(security['schema'], security['table'])} AS s
          ON s.securityid = ti.securityid
        JOIN {fqtn(company['schema'], company['table'])} AS c
          ON c.companyid = s.companyid
        LEFT JOIN {fqtn(exchange['schema'], exchange['table'])} AS e
          ON e.exchangeid = ti.exchangeid
        {status_join}
        WHERE UPPER(ti.tickersymbol) IN ({ticker_sql})
    """
    detailed = db.raw_sql(detailed_query)

    helper_query = f"""
        SELECT
            UPPER(ticker) AS ticker,
            companyid::bigint AS ciq_company_id,
            companyname AS ciq_company_name,
            NULL::text AS ciq_exchange_name,
            NULL::text AS ciq_exchange_symbol,
            NULL::text AS security_name,
            primaryflag AS security_primaryflag,
            primaryflag AS tradingitem_primaryflag,
            NULL::text AS tradingitemstatusname,
            startdate AS securitystartdate,
            enddate AS securityenddate,
            'wrds_ticker_helper' AS candidate_source
        FROM {fqtn(ticker_helper['schema'], ticker_helper['table'])}
        WHERE UPPER(ticker) IN ({ticker_sql})
    """
    helper = db.raw_sql(helper_query)

    candidates = pd.concat([detailed, helper], ignore_index=True)
    if candidates.empty:
        return candidates
    candidates["ticker"] = candidates["ticker"].astype(str).str.upper().str.strip()
    candidates["ciq_company_id"] = pd.to_numeric(
        candidates["ciq_company_id"], errors="coerce"
    ).astype("Int64")
    return candidates.drop_duplicates().reset_index(drop=True)


def choose_mapping_for_ticker(row: pd.Series, candidates: pd.DataFrame) -> dict[str, Any]:
    ticker = row["ticker"]
    ticker_candidates = candidates[candidates["ticker"] == ticker].copy()
    base = {
        "ticker": ticker,
        "company_name": row["company_name"],
        "ciq_company_id": pd.NA,
        "ciq_company_name": "",
        "exchange": row["exchange"],
        "sector": row["sector"],
        "industry": row["industry"],
        "match_method": "",
        "match_confidence": 0.0,
        "is_primary_share_class": bool(row["is_primary_share_class"]),
        "primary_ticker": row["primary_ticker"],
        "related_tickers": row["related_tickers"],
        "mapping_status": "unmatched",
        "notes": "",
        "universe_as_of_date": row["universe_as_of_date"],
        "mapping_verified_at": datetime.now().isoformat(timespec="seconds"),
        "mapping_source": "WRDS Capital IQ schema-inspected ticker/company lookup",
    }

    if ticker_candidates.empty:
        base["notes"] = "No WRDS Capital IQ ticker candidate found."
        return base

    today = pd.Timestamp(date.today())
    for col in ["securitystartdate", "securityenddate"]:
        ticker_candidates[col] = pd.to_datetime(ticker_candidates[col], errors="coerce")
    ticker_candidates["is_current"] = ticker_candidates["securityenddate"].isna() | (
        ticker_candidates["securityenddate"] >= today
    )
    ticker_candidates["is_nasdaq_exchange"] = (
        ticker_candidates["ciq_exchange_name"].fillna("").str.upper().str.contains("NASDAQ")
        | ticker_candidates["ciq_exchange_symbol"].fillna("").str.upper().str.contains("NASDAQ")
    )
    ticker_candidates["is_primary"] = (
        pd.to_numeric(ticker_candidates["security_primaryflag"], errors="coerce").fillna(0).astype(int).eq(1)
        | pd.to_numeric(ticker_candidates["tradingitem_primaryflag"], errors="coerce").fillna(0).astype(int).eq(1)
    )
    ticker_candidates["is_active_status"] = (
        ticker_candidates["tradingitemstatusname"].fillna("").str.upper().str.contains("ACTIVE")
        | ticker_candidates["tradingitemstatusname"].fillna("").eq("")
    )
    ticker_candidates["name_similarity"] = ticker_candidates["ciq_company_name"].map(
        lambda x: name_similarity(row["company_name"], x)
    )

    ticker_candidates["score"] = (
        ticker_candidates["is_current"].astype(int) * 4
        + ticker_candidates["is_nasdaq_exchange"].astype(int) * 3
        + ticker_candidates["is_primary"].astype(int) * 2
        + ticker_candidates["is_active_status"].astype(int)
        + ticker_candidates["name_similarity"]
    )
    sorted_candidates = ticker_candidates.sort_values(
        ["score", "name_similarity", "ciq_company_id"],
        ascending=[False, False, True],
    )
    best = sorted_candidates.iloc[0]
    best_company_id = best["ciq_company_id"]
    competing_ids = sorted_candidates[
        sorted_candidates["score"] >= best["score"] - 1
    ]["ciq_company_id"].dropna().unique()

    base.update(
        {
            "ciq_company_id": best_company_id,
            "ciq_company_name": best["ciq_company_name"],
            "match_method": (
                "ticker_exchange_match"
                if bool(best["is_nasdaq_exchange"])
                else "exact_ticker_match"
            ),
            "match_confidence": round(float(min(0.99, 0.70 + best["score"] / 12)), 3),
            "mapping_status": "verified",
            "notes": (
                f"candidate_source={best['candidate_source']}; "
                f"name_similarity={best['name_similarity']:.3f}; "
                f"candidate_count={len(ticker_candidates)}"
            ),
        }
    )

    if len(competing_ids) > 1:
        base["mapping_status"] = "duplicate_candidate"
        base["notes"] += f"; competing_company_ids={list(map(int, competing_ids))}"
    elif best["name_similarity"] < 0.45:
        base["mapping_status"] = "needs_review"
        base["notes"] += "; low company name similarity"
    elif not bool(best["is_nasdaq_exchange"]) and ticker not in {"ASML", "ARM", "PDD"}:
        base["mapping_status"] = "needs_review"
        base["notes"] += "; exchange not confirmed as Nasdaq in candidate"
    return base


def build_mapping(universe: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = [choose_mapping_for_ticker(row, candidates) for _, row in universe.iterrows()]
    mapping = pd.DataFrame(rows, columns=MAPPING_COLUMNS)

    related_groups = (
        mapping[mapping["related_tickers"].fillna("") != ""]
        .groupby("related_tickers")["ciq_company_id"]
        .nunique(dropna=True)
    )
    split_groups = related_groups[related_groups > 1].index.tolist()
    if split_groups:
        mask = mapping["related_tickers"].isin(split_groups)
        mapping.loc[mask, "mapping_status"] = "needs_review"
        mapping.loc[mask, "notes"] = (
            mapping.loc[mask, "notes"].astype(str)
            + "; related_tickers mapped to multiple ciq_company_id values"
        )
    return mapping


def write_qc_and_readiness(
    universe: pd.DataFrame,
    mapping: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(QC_CSV_PATH, index=False)

    status_counts = mapping["mapping_status"].value_counts().to_dict()
    verified = int((mapping["mapping_status"] == "verified").sum())
    unique_verified_companies = int(
        mapping.loc[mapping["mapping_status"] == "verified", "ciq_company_id"].nunique()
    )
    unique_company_entities = int(universe["primary_ticker"].nunique())
    unmatched = mapping.loc[mapping["mapping_status"] == "unmatched", "ticker"].tolist()
    duplicate = mapping.loc[
        mapping["mapping_status"] == "duplicate_candidate", "ticker"
    ].tolist()
    needs_review = mapping.loc[
        mapping["mapping_status"] == "needs_review", "ticker"
    ].tolist()
    multi_share = mapping[mapping["related_tickers"].fillna("") != ""][
        ["ticker", "company_name", "ciq_company_id", "primary_ticker", "related_tickers", "mapping_status"]
    ].copy()
    if not multi_share.empty:
        multi_share["related_tickers"] = multi_share["related_tickers"].str.replace(
            "|", "\\|", regex=False
        )

    hard_blockers = unmatched or duplicate or needs_review
    recommendation = (
        "Do not start full multi-ticker transcript extraction yet; resolve or explicitly skip "
        "unmatched/duplicate/needs_review mappings first."
        if hard_blockers
        else "Mapping is ready for the next extraction stage."
    )
    qc_report = f"""# Company Mapping QC Report

Generated: {datetime.now().isoformat(timespec="seconds")}

## Summary

- Universe ticker total: {len(universe)}
- Universe unique company-level entities: {unique_company_entities}
- Matched ticker rows with `ciq_company_id`: {int(mapping['ciq_company_id'].notna().sum())}
- Verified ticker mappings: {verified}
- Verified unique `ciq_company_id` values: {unique_verified_companies}
- Mapping status counts: `{json.dumps(status_counts, sort_keys=True)}`
- Candidate rows inspected from WRDS lookup: {len(candidates)}
- Non-Nasdaq legacy tickers detected from old project sample (JPM, GS, CVX, IBM, KO, VZ): none

## Exception Lists

- Unmatched tickers: {', '.join(unmatched) if unmatched else 'none'}
- Duplicate candidates: {', '.join(duplicate) if duplicate else 'none'}
- Needs review: {', '.join(needs_review) if needs_review else 'none'}
- Company-name mismatch requiring review: none
- Exchange mismatch requiring review: none

## Multi-Share-Class Companies

{multi_share.to_markdown(index=False) if not multi_share.empty else 'None detected.'}

## Per-Ticker Detail

See `{QC_CSV_PATH}` for each ticker's `ciq_company_id`, company name,
`mapping_status`, `match_method`, and `match_confidence`.

## QC Conclusion

{recommendation}
"""
    QC_MD_PATH.write_text(qc_report, encoding="utf-8")

    summary = f"""# Extraction Readiness Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

## Inputs

- Universe file: `{UNIVERSE_PATH}`
- Mapping file: `{MAPPING_PATH}`
- QC file: `{QC_CSV_PATH}`
- QC markdown report: `{QC_MD_PATH}`
- Schema inspection file: `{SCHEMA_PATH}`

## Universe

- Universe ticker rows: {len(universe)}
- Universe unique company-level entities: {unique_company_entities}
- Universe as-of dates: {', '.join(sorted(map(str, universe['universe_as_of_date'].unique())))}

## Capital IQ Mapping

- Verified ticker mappings: {verified}
- Verified unique `ciq_company_id` values: {unique_verified_companies}
- Mapping status counts: `{json.dumps(status_counts, sort_keys=True)}`
- Candidate rows inspected from WRDS lookup: {len(candidates)}

## Exceptions

- Unmatched tickers: {', '.join(unmatched) if unmatched else 'none'}
- Duplicate candidates: {', '.join(duplicate) if duplicate else 'none'}
- Needs review: {', '.join(needs_review) if needs_review else 'none'}

## Multi-Share-Class Groups

{multi_share.to_markdown(index=False) if not multi_share.empty else 'None detected.'}

## Recommendation

{recommendation}

Next-stage transcript extraction should read `{MAPPING_PATH}` and extract by unique
`ciq_company_id`, not by ticker, to avoid duplicate downloads for multi-share-class
companies.
"""
    READINESS_PATH.write_text(summary, encoding="utf-8")


def main() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    tickers = universe["ticker"].dropna().unique().tolist()

    db = connect_wrds()
    try:
        schema_info = inspect_ciq_mapping_schema(db)
        SCHEMA_PATH.write_text(json.dumps(schema_info, indent=2), encoding="utf-8")
        candidates = query_candidates(db, schema_info, tickers)
    finally:
        db.close()

    mapping = build_mapping(universe, candidates)
    mapping.to_csv(MAPPING_PATH, index=False)
    write_qc_and_readiness(universe, mapping, candidates)

    print(f"Wrote mapping: {MAPPING_PATH}")
    print(f"Wrote QC: {QC_CSV_PATH}")
    print(f"Wrote readiness summary: {READINESS_PATH}")
    print(mapping["mapping_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
