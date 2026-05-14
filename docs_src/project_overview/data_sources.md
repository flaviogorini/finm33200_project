# Data Sources

The unified monthly panel (`_data/panel_monthly.parquet`) joins six feature
families and one label family on `(date, ticker)` at month-end. The 13-ticker
universe — AAPL, AMZN, BA, CVX, GS, HD, IBM, JPM, KO, MSFT, NKE, NVDA, VZ —
is set in [src/settings.py](../../src/settings.py) and applied uniformly
across every stage.

## Datasets

| Dataset | Source | Frequency | Built by | Output |
|---|---|---|---|---|
| Fundamentals (levels + YoY/QoQ growth) | Bloomberg manual download | Daily → month-end | [build_fundamentals_features.py](../../src/build_fundamentals_features.py) | `features_fundamentals_monthly.parquet` |
| Analyst consensus (BEST_*) | Bloomberg manual download | Daily → month-end | [build_consensus_features.py](../../src/build_consensus_features.py) | `features_consensus_monthly.parquet` |
| Macro (VIX, rates, FX, oil) | Bloomberg manual download | Daily → month-end | [build_macro_features.py](../../src/build_macro_features.py) | `features_macro_monthly.parquet` |
| Trailing + forward returns | Bloomberg `PX_LAST` | Daily → month-end | [build_return_labels.py](../../src/build_return_labels.py) | `labels_returns_monthly.parquet` |
| Earnings-call sentiment | WRDS Capital IQ Transcripts + OpenAI `text-embedding-3-small` + anchor-cosine scoring | Per call → month-end | [pull_wrds_earning_transcripts.py](../../src/pull_wrds_earning_transcripts.py) → [embed_transcripts.py](../../src/embed_transcripts.py) → [score_transcript_sentiment.py](../../src/score_transcript_sentiment.py) → [build_sentiment_features.py](../../src/build_sentiment_features.py) | `features_sentiment_monthly.parquet` |
| 10-Q disclosure text (LM lexicon + embedding similarity) | SEC EDGAR HTTP (via `edgartools`) + Loughran-McDonald master dictionary | Per filing → month-end | [pull_sec_10q_filings.py](../../src/pull_sec_10q_filings.py) → [clean_sec_10q_text.py](../../src/clean_sec_10q_text.py) → [score_sec_10q_text.py](../../src/score_sec_10q_text.py) → [embed_sec_10q_text.py](../../src/embed_sec_10q_text.py) → [build_10q_monthly_panel.py](../../src/build_10q_monthly_panel.py) | `sec_10q_monthly_panel.parquet` |

## Point-in-time alignment (no-lookahead)

Each feature family carries its own activation timestamp; the panel join
enforces that nothing from after `date` is visible at `date`:

- **Fundamentals, consensus, macro** — Bloomberg daily values resampled to
  month-end via `.resample("ME").last()`. Each value is inherently
  point-in-time as of the trading day.
- **Trailing returns** — `PX_LAST.pct_change(n)` over month-end prices. No
  future data.
- **Forward returns** — `shift(-n) / px - 1` computed for `n ∈ {1, 3, 6, 12}`.
  Stored in the panel as `fwd_ret_*` and **sequestered via `LABEL_COLS`**;
  every modelling script asserts that no `fwd_*` column enters the feature
  matrix.
- **Earnings-call sentiment** — keyed on `event_date` (the call date).
  `np.searchsorted(side="right") - 1` finds the most recent call with
  `event_date ≤ month_end`. A unit test verifies
  `last_event_date ≤ row_date` for every panel row.
- **10-Q text** — keyed on **filing_date** (when SEC received the filing),
  not on the fiscal `report_period`. `pd.merge_asof(direction="backward")`
  carries the most recent filed 10-Q to each month-end. A runtime
  assertion **fails the build** if any row would use a filing dated after
  its month-end.

The Loughran-McDonald master dictionary
(`data_manual/lm_master_dictionary.csv`, ~80k words) drives the six 10-Q
lexicon columns (`10q_sentiment`, `10q_positive_rate`, `10q_negative_rate`,
`10q_uncertainty`, `10q_litigious`, `10q_constraining`). Without the CSV
on disk, scoring falls back to a small built-in word list (17 positive /
19 negative) — used only for plumbing tests, not for headline results.

## Pipeline runner

The full build is wired into [dodo.py](../../dodo.py); `doit` runs the
stages in dependency order. Caching (`USE_CACHE=true` in `.env`) skips
re-fetching SEC filings, transcripts, and embeddings whose outputs already
exist.

## Caveats

- **Survivorship.** The 13 tickers are hand-curated current large-caps —
  see [goals.md](goals.md) for the explicit non-claim of generalization.
- **Bloomberg manual download.** The `data_manual/` Excel files are not
  in version control; a teammate must download them from a Bloomberg
  terminal. SEC and WRDS Capital IQ pulls are programmatic.
- **Macro publication lag.** Most macro series (VIX, rates, FX) print at
  the close so month-end values are real-time; release-date series (CPI,
  NFP) would need a one-business-day shift. The current macro column set
  is dominated by market-close prints.
