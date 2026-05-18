"""Run the five backtests across three specifications and persist results.

Driven by the unified signal panel (``_data/signal_panel_monthly.parquet``).
Loops over every strategy that's present in the panel, runs three
specifications (main, stale-call excluded, post-2018 only), and writes:

    _data/results_main.parquet
    _data/results_stale_excl.parquet
    _data/results_post2018.parquet
    _data/metrics_main.json
    _data/metrics_stale_excl.json
    _data/metrics_post2018.json
    _data/ic_timeseries.parquet
    _data/ic_summary.json

The notebook in Phase 7 reads these files. No computation in the notebook.

If a signal column is missing (e.g. transcript-derived signals haven't
been built yet), the orchestrator skips it with a warning and proceeds.
This is intentional so that partial pipelines remain runnable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backtest import (
    compute_ic,
    ic_summary,
    run_backtest,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
PANEL_FILENAME = "signal_panel_monthly.parquet"

STRATEGIES: dict[str, str] = {
    "anchor": "sig_anchor",
    "ridge": "sig_ridge",
    "lm": "sig_lm",
    "momentum": "sig_mom",
    "revisions": "sig_rev",
}

# The stale-call filter only makes sense for the three sentiment strategies
# that carry an earnings-call signal forward each month. Momentum and
# analyst revisions refresh independently every month from Bloomberg, so
# applying days_since_earnings > 60 to them would shrink their universe
# for no methodological reason (and would not match spec section 8, which
# says the filter applies to the "three sentiment strategies").
TRANSCRIPT_STRATEGIES: set[str] = {"anchor", "ridge", "lm"}

SPECIFICATIONS: dict[str, dict] = {
    "main": {},
    "stale_excl": {"drop_stale_gt": 60},  # applied only to TRANSCRIPT_STRATEGIES
    "post2018": {"start_date": "2018-12-31"},
}


def _filters_for(label: str, spec_name: str, base_filters: dict) -> dict:
    """Per-strategy filter resolution.

    The stale-call filter is dropped for non-transcript strategies under
    the ``stale_excl`` spec (see TRANSCRIPT_STRATEGIES rationale above).
    Other filter keys pass through unchanged.
    """
    if spec_name != "stale_excl":
        return base_filters
    if label in TRANSCRIPT_STRATEGIES:
        return base_filters
    return {k: v for k, v in base_filters.items() if k != "drop_stale_gt"}


def _run_one_spec(panel: pd.DataFrame, spec_name: str, filters: dict) -> tuple[pd.DataFrame, dict]:
    """Run all available strategies under one specification."""
    monthly_frames: list[pd.DataFrame] = []
    metrics_block: dict[str, dict] = {}

    for label, col in STRATEGIES.items():
        if col not in panel.columns:
            print(f"  [skip] spec={spec_name} strategy={label}: column {col!r} not in panel")
            continue

        per_strategy_filters = _filters_for(label, spec_name, filters)
        bt = run_backtest(panel, col, filters=per_strategy_filters)
        if bt.monthly.empty:
            print(f"  [skip] spec={spec_name} strategy={label}: no months survived filters")
            continue

        monthly = bt.monthly.copy()
        monthly["strategy"] = label
        monthly["spec"] = spec_name
        monthly_frames.append(monthly)
        metrics_block[label] = bt.metrics
        print(
            f"  [{spec_name}/{label}] n_months={bt.metrics.get('n_months', 0):>3d}"
            f"  ann_ret={bt.metrics.get('ann_ret', float('nan')):+.3f}"
            f"  sharpe={bt.metrics.get('sharpe', float('nan')):+.3f}"
        )

    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    return monthly_all, metrics_block


def _run_ic_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Per-month IC for every available strategy on the main spec."""
    ic_frames: list[pd.DataFrame] = []
    summary: dict[str, dict] = {}
    for label, col in STRATEGIES.items():
        if col not in panel.columns:
            continue
        ic = compute_ic(panel, col)
        if ic.empty:
            continue
        ic["strategy"] = label
        ic_frames.append(ic)
        summary[label] = ic_summary(ic)
        s = summary[label]
        print(
            f"  [IC/{label}] n_months={s.get('ic_n_months', 0):>3d}"
            f"  mean={s.get('ic_mean', float('nan')):+.4f}"
            f"  ir={s.get('ic_ir', float('nan')):+.3f}"
        )
    ic_all = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    return ic_all, summary


def main(data_dir: Path = DATA_DIR) -> None:
    panel_path = data_dir / PANEL_FILENAME
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found. Run `python src/build_signal_panel.py` first."
        )
    panel = pd.read_parquet(panel_path)
    print(f"Loaded panel: {len(panel):,} rows, signals present: "
          f"{sorted(set(STRATEGIES.values()) & set(panel.columns))}")

    for spec_name, filters in SPECIFICATIONS.items():
        print(f"\n--- Running specification: {spec_name} ---")
        monthly, metrics = _run_one_spec(panel, spec_name, filters)
        monthly_out = data_dir / f"results_{spec_name}.parquet"
        metrics_out = data_dir / f"metrics_{spec_name}.json"
        if not monthly.empty:
            monthly.to_parquet(monthly_out, index=False)
        metrics_out.write_text(json.dumps(metrics, indent=2, default=str))
        print(f"  wrote {monthly_out}  +  {metrics_out}")

    print(f"\n--- Computing IC time series (main spec) ---")
    ic_all, ic_summ = _run_ic_panel(panel)
    if not ic_all.empty:
        ic_all.to_parquet(data_dir / "ic_timeseries.parquet", index=False)
    (data_dir / "ic_summary.json").write_text(json.dumps(ic_summ, indent=2, default=str))
    print(f"  wrote ic_timeseries.parquet  +  ic_summary.json")


if __name__ == "__main__":
    main()
