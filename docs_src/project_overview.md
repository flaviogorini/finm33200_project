# Project Overview

## finm33200_project — Generative AI for Monthly Return Forecasting

We forecast monthly stock returns for a 13-ticker universe (AAPL, AMZN,
BA, CVX, GS, HD, IBM, JPM, KO, MSFT, NKE, NVDA, VZ) using a unified
month-end panel that combines six feature families: fundamentals (levels +
YoY/QoQ growth), Bloomberg analyst consensus, macro, trailing returns,
earnings-call sentiment (OpenAI `text-embedding-3-small` + anchor-cosine),
and SEC 10-Q disclosure text (Loughran-McDonald lexicon + embedding
similarity). The research question is whether each successive text-derived
feature family adds incremental predictive content under strict
point-in-time constraints.

The project is structured as a four-step hypothesis-test ladder
(V0a → V0b → V1 → V2 → V3) where every step adds one feature family and
measures the change in out-of-sample R², Spearman IC, and a long-short
tertile portfolio Sharpe. All variants are trained on the same 13-ticker
pooled cross-section so the only thing changing between variants is the
feature columns — see [methodology](project_overview/methodology.md) for
the design and [goals](project_overview/goals.md) for what is and is not
claimed.

| Section | Description |
|---------|-------------|
| [Goals](project_overview/goals.md) | Research question, hypothesis-test ladder, what is NOT claimed |
| [Data Sources](project_overview/data_sources.md) | Six feature families, point-in-time alignment, build pipeline |
| [Methodology](project_overview/methodology.md) | Walk-forward CV, models, portfolio rule, audit fixes |

```{toctree}
:maxdepth: 1
:caption: Project Details

project_overview/goals
project_overview/data_sources
project_overview/methodology
```
