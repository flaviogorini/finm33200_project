"""Render headline charts from the predict_returns_ckx.py outputs.

Produces three PNGs under _output/:
    ckx_cum_returns_1m.png   — cumulative-return curves, fwd_ret_1m, by variant
    ckx_cum_returns_3m.png   — same, fwd_ret_3m
    ckx_ic_auc_bars.png      — pooled OOS IC and AUC bars for every variant × model × target

Run:
    python src/plot_ckx_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def _portfolio_chart(target: str, out_path: Path) -> None:
    pf = pd.read_parquet(OUTPUT_DIR / "ckx_portfolio.parquet")
    pf = pf[pf["target"] == target].copy()
    if pf.empty:
        print(f"  no portfolio rows for {target}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    # V0a uses buy-and-hold; others use long-short tertile. Label each
    # series by variant + model.
    for (variant, model), grp in pf.groupby(["variant", "model"]):
        g = grp.sort_values("date")
        label = f"{variant}/{model}"
        # Make V0a stand out as the benchmark
        kwargs = {"label": label}
        if variant == "v0a":
            kwargs.update(linewidth=2.5, linestyle="--", color="black")
        ax.plot(g["date"], g["cum_ret"], **kwargs)
    ax.set_title(f"Cumulative return by variant/model — target={target} (gross of costs)")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative return")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _metrics_bars(out_path: Path) -> None:
    metrics = json.loads((OUTPUT_DIR / "ckx_metrics.json").read_text())

    rows = []
    for target, target_block in metrics.items():
        for variant, variant_block in target_block.items():
            for model_name, m in variant_block.get("models", {}).items():
                rows.append({
                    "target": target,
                    "variant": variant,
                    "model": model_name,
                    "auc": m.get("auc"),
                    "ic": m.get("ic_spearman"),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    targets = sorted(df["target"].unique())
    for col, target in enumerate(targets):
        sub = df[df["target"] == target].copy()
        sub["label"] = sub["variant"] + "/" + sub["model"]
        sub = sub.sort_values(["variant", "model"])

        axes[0, col].bar(sub["label"], sub["auc"])
        axes[0, col].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        axes[0, col].set_title(f"AUC — {target}")
        axes[0, col].set_ylim(0.4, 0.65)
        axes[0, col].tick_params(axis="x", rotation=45)

        axes[1, col].bar(sub["label"], sub["ic"])
        axes[1, col].axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
        axes[1, col].set_title(f"Spearman IC — {target}")
        axes[1, col].set_ylim(-0.05, 0.25)
        axes[1, col].tick_params(axis="x", rotation=45)

    fig.suptitle("Pooled out-of-sample metrics (13-ticker test rows)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering CKX headline charts...")
    _portfolio_chart("fwd_ret_1m", OUTPUT_DIR / "ckx_cum_returns_1m.png")
    _portfolio_chart("fwd_ret_3m", OUTPUT_DIR / "ckx_cum_returns_3m.png")
    _metrics_bars(OUTPUT_DIR / "ckx_ic_auc_bars.png")


if __name__ == "__main__":
    main()
