"""Pull Bloomberg PX_LAST + BEST_NET_INCOME via blpapi for the current Nasdaq-100.

Replaces ``pull_manual_companies.py`` (which parsed a hand-maintained Excel
workbook whose ``PX_LAST`` series was NOT adjusted for splits/dividends). This
script pulls programmatic Bloomberg data for the **current Nasdaq-100 (~100
tickers)** read from ``data_manual/_meta/nasdaq100_constituents.csv``.

Modes
-----
The ``BLOOMBERG_TERMINAL_AVAILABLE`` flag in ``.env`` selects mode:

- **false / unset (cache mode, default)**: no Bloomberg connection. The script
  just verifies the committed cache at ``data_manual/bbg/`` exists, prints its
  manifest summary, and copies the cache parquets into ``_data/`` so downstream
  tasks resolve their ``file_dep`` paths. Hard-fails with instructions if the
  cache is missing.

- **true (live mode)**: connects via ``blp.BlpQuery``, runs chunked BDH calls
  for ``PX_LAST`` and ``BEST_NET_INCOME`` (``BEST_FPERIOD_OVERRIDE='1BF'`` for
  the consensus field), reshapes to long ``[date, ticker, field, value]``,
  writes the cache parquets at ``data_manual/bbg/``, then copies them into
  ``_data/`` for downstream fast loading.

Cache layout (committed to git)
-------------------------------
    data_manual/bbg/US_Companies_Hist_Data.parquet  # field=PX_LAST
    data_manual/bbg/US_Companies_Forecast.parquet   # field=BEST_NET_INCOME
    data_manual/bbg/bbg_pull_manifest.json

CUSIP fallback path is preserved from the v3 version but is a no-op on main:
``nasdaq100_constituents.csv`` has no ``cusip`` column and current Nasdaq-100
tickers contain no Compustat ``.N`` iid-suffix forms.

Schema matches what the old XLSX parser produced so downstream
``build_returns_monthly.py`` / ``build_revisions_monthly.py`` work unchanged.
Tickers are stored with the ``" US Equity"`` suffix.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from settings import config


# Regex to pull the offending ticker out of a blp securityError message:
#   "Response for 'ATVI.1 US Equity' contains securityError {...}"
_BAD_SEC_RE = re.compile(r"Response for '([^']+)' contains securityError")

DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
META_DIR = MANUAL_DATA_DIR / "_meta"
BBG_CACHE_DIR = MANUAL_DATA_DIR / "bbg"

HIST_FILENAME = "US_Companies_Hist_Data.parquet"
FORECAST_FILENAME = "US_Companies_Forecast.parquet"
MANIFEST_FILENAME = "bbg_pull_manifest.json"

CONSTITUENTS_CSV = META_DIR / "nasdaq100_constituents.csv"

BBG_SUFFIX = " US Equity"
START_DATE = "20050101"
CHUNK_SIZE = 50

HIST_FIELD = "PX_LAST"
FORECAST_FIELD = "BEST_NET_INCOME"

# BDH's default fiscal-period interpretation for BEST_* fields appears to be
# `1FQ` (next-fiscal-quarter), not `1BF` (blended-forward) as the Excel add-in
# defaults to. Forcing `1BF` explicitly matches what the v2 manual workbook
# captured (annual-scale blended-forward consensus, e.g. AAPL ~$100B).
FORECAST_OVERRIDES: list[tuple[str, str]] = [("BEST_FPERIOD_OVERRIDE", "1BF")]

# Cash-dividend / split adjustments. We pin all four flags explicitly so a
# re-pull on a different terminal (whose DPDF / cash-adjustment defaults may
# differ) still produces the same series. The hand-XLSX workbook was UNADJUSTED
# — switching to API + these flags is the whole reason this script exists.
BDH_OPTIONS: dict[str, object] = {
    "periodicitySelection": "DAILY",
    "adjustmentNormal": True,
    "adjustmentSplit": True,
    "adjustmentAbnormal": True,
    # Don't follow terminal Display-Per-Day-Format setting — use the explicit
    # flags above. Keeps the pull deterministic across Bloomberg installs.
    "adjustmentFollowDPDF": False,
}


def load_tickers(constituents_path: Path = CONSTITUENTS_CSV) -> tuple[list[str], list[str]]:
    """Load tickers split by whether Bloomberg can resolve them by symbol.

    Returns
    -------
    clean_tickers : list[str]
        Tickers Bloomberg accepts as plain symbols (e.g. ``AAPL US Equity``).
    dotted_tickers : list[str]
        Compustat iid-suffixed forms (``ATVI.1``, ``DELL.1``, ``SNDK.1``, ...)
        which Bloomberg always rejects. Empty for the current Nasdaq-100.
    """
    if not constituents_path.exists():
        raise FileNotFoundError(
            f"{constituents_path} not found. Run "
            "`python src/build_nasdaq100_universe.py` first."
        )
    df = pd.read_csv(constituents_path, usecols=["ticker"])
    tickers = (
        df["ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    )
    tickers = sorted(t for t in tickers if t and t != "NAN")
    clean = [f"{t}{BBG_SUFFIX}" for t in tickers if "." not in t]
    dotted = [f"{t}{BBG_SUFFIX}" for t in tickers if "." in t]
    return clean, dotted


def _chunked(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _build_cusip_pairs(
    invalid_tickers: list[str],
    constituents_path: Path = CONSTITUENTS_CSV,
) -> list[tuple[str, str]]:
    """Map each invalid ``" US Equity"``-suffixed ticker back to its CUSIP.

    Returns a list of ``(ticker_with_suffix, cusip)`` for tickers that have a
    non-empty CUSIP in the constituents CSV. Tickers without a CUSIP (and the
    common case on main, where the CSV has no ``cusip`` column at all) are
    silently skipped — the CUSIP fallback path is effectively dormant for the
    current Nasdaq-100 universe.
    """
    if not invalid_tickers:
        return []
    try:
        cons = pd.read_csv(constituents_path, usecols=["ticker", "cusip"])
    except (ValueError, KeyError):
        # No `cusip` column on main's constituents CSV — fallback is a no-op.
        return []
    cons["ticker"] = cons["ticker"].astype(str).str.upper().str.strip()
    cons["cusip"] = cons["cusip"].astype(str).str.strip()
    cons = cons[cons["cusip"].ne("") & cons["cusip"].ne("nan")]
    cons = cons.drop_duplicates(subset=["ticker"])
    lookup = dict(zip(cons["ticker"], cons["cusip"]))
    pairs: list[tuple[str, str]] = []
    for full in invalid_tickers:
        bare = full[: -len(BBG_SUFFIX)] if full.endswith(BBG_SUFFIX) else full
        cusip = lookup.get(bare)
        if cusip:
            pairs.append((full, cusip))
    return pairs


def _bdh_to_long(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Reshape a BDH result to long ``[date, ticker, field, value]``.

    The ``blp`` wrapper returns a long frame with columns
    ``[date, security, <field_name>]``. We rename ``security`` → ``ticker``
    and pivot the field column into rows.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "ticker", "field", "value"])
    out = df.rename(columns={"security": "ticker", field: "value"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["field"] = field
    out = out[["date", "ticker", "field", "value"]]
    return out.dropna(subset=["value"]).reset_index(drop=True)


def _bdh_chunk_with_retry(bq, chunk: list[str], field: str, start: str, end: str,
                           overrides: list[tuple[str, str]]) -> tuple[pd.DataFrame, list[str]]:
    """Call BDH for one chunk; on securityError, drop the offending ticker and retry.

    Bloomberg's blp wrapper raises on the FIRST invalid security in a request,
    refusing to return data for the surviving valid tickers. We parse the bad
    ticker out of the error string, remove it, and retry until the chunk
    succeeds or shrinks to empty.

    Returns
    -------
    df : pd.DataFrame
        BDH result (long: ``date, security, <field>``). Empty if chunk exhausted.
    dropped : list[str]
        Bad tickers that were removed from this chunk.
    """
    dropped: list[str] = []
    remaining = list(chunk)
    while remaining:
        try:
            df = bq.bdh(
                remaining,
                [field],
                start_date=start,
                end_date=end,
                options=BDH_OPTIONS,
                overrides=overrides,
            )
            return df, dropped
        except Exception as exc:  # pragma: no cover - network/Bloomberg side
            msg = str(exc)
            m = _BAD_SEC_RE.search(msg)
            if not m:
                # Unknown failure mode — surface it, abandon this chunk.
                print(f"    chunk failed (unhandled): {exc}", flush=True)
                return pd.DataFrame(), dropped
            bad = m.group(1)
            if bad not in remaining:
                # Bloomberg sometimes echoes the security in a slightly different
                # form than what we sent (rare). Give up on this chunk to avoid
                # an infinite loop.
                print(f"    chunk failed: cannot locate '{bad}' in chunk", flush=True)
                return pd.DataFrame(), dropped
            dropped.append(bad)
            remaining.remove(bad)
            print(f"    dropped invalid ticker '{bad}', retrying with "
                  f"{len(remaining)} remaining...", flush=True)
    return pd.DataFrame(), dropped


def pull_field_live(bq, tickers: list[str], field: str, start: str, end: str,
                     overrides: list[tuple[str, str]] | None = None
                     ) -> tuple[pd.DataFrame, list[str]]:
    """Run chunked BDH for one field across all tickers, daily frequency.

    Returns the concatenated long-format frame plus the list of tickers
    Bloomberg rejected as invalid (so the caller can record them in the
    manifest).
    """
    frames: list[pd.DataFrame] = []
    all_dropped: list[str] = []
    overrides = overrides or []
    n_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i, chunk in enumerate(_chunked(tickers, CHUNK_SIZE), 1):
        print(f"  [{field}] chunk {i}/{n_chunks} ({len(chunk)} tickers)...",
              flush=True)
        df, dropped = _bdh_chunk_with_retry(bq, chunk, field, start, end, overrides)
        all_dropped.extend(dropped)
        frames.append(_bdh_to_long(df, field))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "field", "value"]), all_dropped
    return pd.concat(frames, ignore_index=True), all_dropped


def pull_field_via_cusip(bq, ticker_cusip_pairs: list[tuple[str, str]],
                          field: str, start: str, end: str,
                          overrides: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Recover delisted/replaced Compustat tickers (e.g. ``DELL.1``) via CUSIP.

    Bloomberg rejects Compustat's iid-suffix notation but accepts the
    ``/cusip/XXXXXXXXX US Equity`` form. We query one ticker at a time (no
    chunking — per-ticker isolation avoids any bad CUSIP from poisoning the
    rest), then rewrite the ``security`` field of the returned data back to
    the original Compustat ticker so it joins with the rest of the panel.
    """
    overrides = overrides or []
    frames: list[pd.DataFrame] = []
    for ticker, cusip in ticker_cusip_pairs:
        bbg_sec = f"/cusip/{cusip} US Equity"
        print(f"  [{field}] cusip pull: {ticker}  <-  {bbg_sec}", flush=True)
        try:
            df = bq.bdh(
                [bbg_sec],
                [field],
                start_date=start,
                end_date=end,
                options=BDH_OPTIONS,
                overrides=overrides,
            )
        except Exception as exc:  # pragma: no cover - network / Bloomberg side
            print(f"    cusip pull failed for {ticker} ({cusip}): {exc}",
                  flush=True)
            continue
        long = _bdh_to_long(df, field)
        if not long.empty:
            long["ticker"] = ticker  # rewrite back to Compustat form
            frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "field", "value"])
    return pd.concat(frames, ignore_index=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".parquet", dir=path.parent
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".json", dir=path.parent, encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, indent=2, default=str)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def run_live_pull() -> dict:
    """Connect to Bloomberg, pull both fields, write cache + return manifest."""
    from blp import blp  # local import: only needed in live mode

    clean_tickers, dotted_tickers = load_tickers()
    print(f"Live pull: {len(clean_tickers)} clean + {len(dotted_tickers)} "
          f"dotted tickers from {START_DATE} to today.")

    end_date = datetime.today().strftime("%Y%m%d")
    print("Connecting to Bloomberg via blp.BlpQuery()...")
    bq = blp.BlpQuery().start()

    # Pass 1: symbol-based BDH for the clean tickers only.
    hist, hist_invalid = pull_field_live(
        bq, clean_tickers, HIST_FIELD, START_DATE, end_date
    )
    forecast, fcst_invalid = pull_field_live(
        bq, clean_tickers, FORECAST_FIELD, START_DATE, end_date,
        overrides=FORECAST_OVERRIDES,
    )

    # Pass 2: CUSIP-based BDH. Dotted tickers go here unconditionally (Bloomberg
    # never resolves their iid-suffixed symbols). Any clean ticker the symbol
    # pass flagged invalid also gets a CUSIP attempt as a safety net.
    dotted_for_cusip = list(dotted_tickers)
    safety_net = sorted(set(hist_invalid) | set(fcst_invalid))
    for t in safety_net:
        if t not in dotted_for_cusip:
            dotted_for_cusip.append(t)
    cusip_pairs = _build_cusip_pairs(dotted_for_cusip)
    hist_cusip = pd.DataFrame()
    fcst_cusip = pd.DataFrame()
    if cusip_pairs:
        print(f"\nCUSIP pass: {len(cusip_pairs)} tickers "
              f"({len(dotted_tickers)} dotted + {len(safety_net)} safety-net)...")
        hist_cusip = pull_field_via_cusip(
            bq, cusip_pairs, HIST_FIELD, START_DATE, end_date
        )
        fcst_cusip = pull_field_via_cusip(
            bq, cusip_pairs, FORECAST_FIELD, START_DATE, end_date,
            overrides=FORECAST_OVERRIDES,
        )
    if not hist_cusip.empty:
        hist = pd.concat([hist, hist_cusip], ignore_index=True)
    if not fcst_cusip.empty:
        forecast = pd.concat([forecast, fcst_cusip], ignore_index=True)

    BBG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(hist, BBG_CACHE_DIR / HIST_FILENAME)
    _atomic_write_parquet(forecast, BBG_CACHE_DIR / FORECAST_FILENAME)

    requested = set(clean_tickers) | set(dotted_tickers)
    returned = set(hist["ticker"].unique()) | set(forecast["ticker"].unique())
    manifest = {
        "pulled_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "live",
        "ticker_count_requested": len(requested),
        "ticker_count_returned": len(returned),
        "tickers_routed_via_cusip": dotted_tickers,
        "tickers_with_no_data": sorted(requested - returned),
        "fields": [HIST_FIELD, FORECAST_FIELD],
        "start_date": START_DATE,
        "end_date": end_date,
        "frequency": "DAILY",
        "overrides": "BEST_FPERIOD_OVERRIDE=1BF for BEST_NET_INCOME; none for PX_LAST",
        "bdh_options": dict(BDH_OPTIONS),
        "blp_package": "blp (Bloomberg blpapi wrapper)",
    }
    _atomic_write_json(manifest, BBG_CACHE_DIR / MANIFEST_FILENAME)
    print(f"Live pull complete. Cache at {BBG_CACHE_DIR}.")
    return manifest


def hydrate_data_dir() -> None:
    """Copy cache parquets into ``_data/`` so downstream file_deps resolve."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in (HIST_FILENAME, FORECAST_FILENAME):
        src = BBG_CACHE_DIR / fname
        dst = DATA_DIR / fname
        shutil.copyfile(src, dst)
        print(f"  hydrated {dst}")


def assert_cache_present() -> None:
    """Hard-fail with instructions if the committed cache is missing."""
    missing = [
        p for p in (
            BBG_CACHE_DIR / HIST_FILENAME,
            BBG_CACHE_DIR / FORECAST_FILENAME,
        )
        if not p.exists()
    ]
    if not missing:
        return
    raise SystemExit(
        "No cached Bloomberg data at data_manual/bbg/.\n"
        "  Missing: " + ", ".join(str(p) for p in missing) + "\n"
        "Set BLOOMBERG_TERMINAL_AVAILABLE=true in .env to run a live pull, "
        "or obtain the cache files from a teammate."
    )


def print_manifest_summary() -> None:
    path = BBG_CACHE_DIR / MANIFEST_FILENAME
    if not path.exists():
        print(f"  (no manifest at {path})")
        return
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    print(f"Cache manifest ({path}):")
    for key in (
        "pulled_at", "mode", "ticker_count_requested", "ticker_count_returned",
        "fields", "start_date", "end_date", "frequency", "overrides",
    ):
        if key in manifest:
            print(f"  {key}: {manifest[key]}")
    no_data = manifest.get("tickers_with_no_data") or []
    via_cusip = manifest.get("tickers_routed_via_cusip") or []
    print(f"  tickers_with_no_data: {len(no_data)}")
    print(f"  tickers_routed_via_cusip: {len(via_cusip)}")


def main() -> None:
    use_bbg = config(
        "BLOOMBERG_TERMINAL_AVAILABLE", default=False, cast=bool
    )
    if use_bbg:
        run_live_pull()
    else:
        print("Cache mode: BLOOMBERG_TERMINAL_AVAILABLE is false or unset.")
        assert_cache_present()
        print_manifest_summary()

    hydrate_data_dir()
    print("Done.")


if __name__ == "__main__":
    main()
