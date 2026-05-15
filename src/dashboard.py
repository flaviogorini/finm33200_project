"""Streamlit dashboard for the CKX return-prediction project.

Read-only viewer over the artifacts produced by the pipeline — it never
recomputes anything. Five views:

  1. Ticker forecast      — y_true vs y_pred (+ p_up) for a chosen variant.
  2. Variant ladder       — V0a→V5 AUC / OOS R2 / IC bars from ckx_metrics.json.
  3. 10-Q AI timeline     — generative-AI tone / risk / uncertainty / change
                            scores per filing, with material-change markers.
  4. Cited filing snippets — the LLM's summary + cited evidence per 10-Q.
  5. Portfolio            — cumulative long-short returns per variant.

Run with:
    streamlit run src/dashboard.py

Inputs (guarded — a missing file shows a warning, not a crash):
    _output/ckx_metrics.json
    _output/ckx_predictions.parquet
    _output/ckx_portfolio.parquet
    _data/panel_monthly.parquet   (via build_panel.load_panel)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run src/dashboard.py` executes from the project root; put src/ on
# the path so the project modules import the same way as everywhere else.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_panel import load_panel  # noqa: E402
from settings import config  # noqa: E402

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
METRICS_PATH = OUTPUT_DIR / "ckx_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "ckx_predictions.parquet"
PORTFOLIO_PATH = OUTPUT_DIR / "ckx_portfolio.parquet"

AI_SCORE_COLS = [
    "10q_ai_tone_score",
    "10q_ai_risk_score",
    "10q_ai_uncertainty_score",
    "10q_ai_disclosure_change_score",
]


# ---- cached loaders -----------------------------------------------------


@st.cache_data
def _load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    # json.load accepts the NaN tokens written by the model script.
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def _load_predictions() -> pd.DataFrame | None:
    if not PREDICTIONS_PATH.exists():
        return None
    df = pd.read_parquet(PREDICTIONS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def _load_portfolio() -> pd.DataFrame | None:
    if not PORTFOLIO_PATH.exists():
        return None
    df = pd.read_parquet(PORTFOLIO_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def _load_panel() -> pd.DataFrame | None:
    try:
        return load_panel()
    except FileNotFoundError:
        return None


def _metrics_to_frame(metrics: dict) -> pd.DataFrame:
    """Flatten the nested ckx_metrics.json into a tidy long frame."""
    rows = []
    for target, variants in metrics.items():
        for variant, vinfo in variants.items():
            for model, m in vinfo.get("models", {}).items():
                rows.append({
                    "target": target,
                    "variant": variant,
                    "model": model,
                    "n": m.get("n"),
                    "auc": m.get("auc"),
                    "accuracy": m.get("accuracy"),
                    "oos_r2": m.get("oos_r2"),
                    "ic_spearman": m.get("ic_spearman"),
                })
    return pd.DataFrame(rows)


# ---- views --------------------------------------------------------------


def view_ticker_forecast(preds: pd.DataFrame) -> None:
    st.subheader("Ticker forecast — predicted vs actual returns")
    if preds is None:
        st.warning("`_output/ckx_predictions.parquet` not found. "
                   "Run `doit predict_returns` first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    ticker = c1.selectbox("Ticker", sorted(preds["ticker"].unique()))
    target = c2.selectbox("Target", sorted(preds["target"].unique()))
    variant = c3.selectbox("Variant", sorted(preds["variant"].unique()),
                           index=sorted(preds["variant"].unique()).index("v3")
                           if "v3" in preds["variant"].unique() else 0)
    models = sorted(preds[preds["variant"] == variant]["model"].unique())
    model = c4.selectbox("Model", models)

    sub = preds[
        (preds["ticker"] == ticker)
        & (preds["target"] == target)
        & (preds["variant"] == variant)
        & (preds["model"] == model)
    ].sort_values("date")
    if sub.empty:
        st.info("No predictions for this combination.")
        return

    chart = sub.set_index("date")[["y_true", "y_pred"]]
    st.line_chart(chart)
    st.caption("Out-of-sample walk-forward predictions. `y_true` is the "
               "realized forward return; `y_pred` is the model forecast.")
    st.line_chart(sub.set_index("date")[["p_up"]])
    st.caption("`p_up` — calibrated probability the forward return is positive.")


def view_variant_ladder(metrics: dict) -> None:
    st.subheader("Variant ladder — does each signal layer add value?")
    if metrics is None:
        st.warning("`_output/ckx_metrics.json` not found. "
                   "Run `doit predict_returns` first.")
        return

    mf = _metrics_to_frame(metrics)
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target ", sorted(mf["target"].unique()), key="ladder_target")
    # Headline metrics first: rank IC + AUC are what the return-prediction
    # literature recommends for noisy monthly returns. R² is reported but
    # demoted — monthly return R² is typically near-zero or negative even
    # for working strategies.
    metric = c2.selectbox("Metric", ["ic_spearman", "auc", "accuracy", "oos_r2"])

    sub = mf[mf["target"] == target].copy()
    pivot = sub.pivot_table(index="variant", columns="model", values=metric)
    pivot = pivot.reindex(sorted(pivot.index))
    st.bar_chart(pivot)
    st.caption(
        "V0a/V0b = baselines · V1 = +fundamentals/macro · V2 = +call sentiment · "
        "V3 = +10-Q LM lexicon · V4 = +generative-AI 10-Q (lexicon dropped) · "
        "V5 = LM + generative-AI 10-Q combined. V4/V5 appear once "
        "`doit process_10q:analyze` has run.  \n"
        "**Headline:** rank IC (Spearman) and AUC. Monthly return R² is "
        "noise-bounded near zero; reported for transparency, not as the "
        "primary success metric."
    )
    st.dataframe(
        sub.pivot_table(index="variant", columns="model",
                        values=["ic_spearman", "auc", "accuracy", "oos_r2"]).round(4)
    )


def view_ai_timeline(panel: pd.DataFrame) -> None:
    st.subheader("10-Q generative-AI disclosure timeline")
    if panel is None:
        st.warning("`_data/panel_monthly.parquet` not found. Run `doit build_panel`.")
        return
    have_ai = [c for c in AI_SCORE_COLS if c in panel.columns]
    if not have_ai:
        st.info("No generative-AI 10-Q columns in the panel yet. Run "
                "`doit process_10q:analyze` then `doit process_10q:panel` "
                "and `doit build_panel`.")
        return

    ticker = st.selectbox("Ticker", sorted(panel["ticker"].unique()), key="ai_ticker")
    sub = panel[panel["ticker"] == ticker].sort_values("date")
    # One row per filing — drop the carried-forward monthly duplicates.
    sub = sub.dropna(subset=have_ai, how="all")
    sub = sub.drop_duplicates(subset=["10q_ai_summary", *have_ai], keep="first") \
        if "10q_ai_summary" in sub.columns else sub.drop_duplicates(subset=have_ai)
    if sub.empty:
        st.info(f"No AI-analyzed filings for {ticker}.")
        return

    st.line_chart(sub.set_index("date")[have_ai])
    st.caption("tone (-1..1), risk (0..1), uncertainty (0..1), "
               "disclosure_change (0..1, vs prior filing).")

    if "10q_ai_material_change_flag" in sub.columns:
        flagged = sub[sub["10q_ai_material_change_flag"] == 1]
        st.metric("Filings flagged as a material disclosure change",
                  f"{len(flagged)} / {len(sub)}")

    if "10q_sentiment" in panel.columns:
        cmp = sub.set_index("date")[["10q_ai_tone_score", "10q_sentiment"]]
        st.line_chart(cmp)
        st.caption("Generative-AI tone vs the Loughran-McDonald lexicon "
                   "sentiment for the same filings.")


def view_cited_snippets(panel: pd.DataFrame) -> None:
    st.subheader("Cited filing snippets — what the LLM read")
    if panel is None:
        st.warning("`_data/panel_monthly.parquet` not found. Run `doit build_panel`.")
        return
    if "10q_ai_summary" not in panel.columns:
        st.info("No generative-AI 10-Q analysis in the panel yet. Run "
                "`doit process_10q:analyze` then rebuild the panels.")
        return

    ticker = st.selectbox("Ticker", sorted(panel["ticker"].unique()), key="snip_ticker")
    sub = panel[(panel["ticker"] == ticker)].copy()
    sub = sub.dropna(subset=["10q_ai_summary"])
    sub = sub[sub["10q_ai_summary"] != "ANALYSIS_FAILED"]
    sub = sub.drop_duplicates(subset=["10q_ai_summary"], keep="first")
    sub = sub.sort_values("date", ascending=False)
    if sub.empty:
        st.info(f"No AI-analyzed filings for {ticker}.")
        return

    for _, row in sub.iterrows():
        label = f"{pd.Timestamp(row['date']).date()}"
        if "filing_date" in row and pd.notna(row["filing_date"]):
            label += f"  ·  filed {pd.Timestamp(row['filing_date']).date()}"
        flag = row.get("10q_ai_material_change_flag")
        if flag == 1:
            label += "  ·  ⚠ material change"
        with st.expander(label):
            st.markdown(f"**Summary** — {row['10q_ai_summary']}")
            try:
                evidence = json.loads(row.get("10q_ai_evidence") or "[]")
            except json.JSONDecodeError:
                evidence = []
            for ev in evidence:
                st.markdown(
                    f"> *[{ev.get('section', '?')}]* {ev.get('quote', '')}  \n"
                    f"&nbsp;&nbsp;— {ev.get('why_it_matters', '')}"
                )


def view_portfolio(portfolio: pd.DataFrame) -> None:
    st.subheader("Portfolio — cumulative long-short returns")
    if portfolio is None:
        st.warning("`_output/ckx_portfolio.parquet` not found. "
                   "Run `doit predict_returns` first.")
        return
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target  ", sorted(portfolio["target"].unique()),
                          key="pf_target")
    model = c2.selectbox("Model ", sorted(portfolio["model"].unique()), key="pf_model")
    sub = portfolio[(portfolio["target"] == target) & (portfolio["model"] == model)]
    if sub.empty:
        st.info("No portfolio rows for this combination.")
        return
    pivot = sub.pivot_table(index="date", columns="variant", values="cum_ret")
    st.line_chart(pivot)
    st.caption("Cumulative return of the long-short tertile strategy per "
               "variant (V0a is equal-weight buy-and-hold).")


# ---- main ---------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="CKX Return Prediction", layout="wide")
    st.title("CKX return-prediction dashboard")
    st.caption("Earnings-call transcripts + SEC 10-Q text → monthly return "
               "forecasts. Read-only view over the pipeline outputs.")

    metrics = _load_metrics()
    preds = _load_predictions()
    portfolio = _load_portfolio()
    panel = _load_panel()

    tabs = st.tabs([
        "Ticker forecast",
        "Variant ladder",
        "10-Q AI timeline",
        "Cited snippets",
        "Portfolio",
    ])
    with tabs[0]:
        view_ticker_forecast(preds)
    with tabs[1]:
        view_variant_ladder(metrics)
    with tabs[2]:
        view_ai_timeline(panel)
    with tabs[3]:
        view_cited_snippets(panel)
    with tabs[4]:
        view_portfolio(portfolio)


if __name__ == "__main__":
    main()
