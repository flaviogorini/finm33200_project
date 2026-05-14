# Goals

## Research question

**Does disclosure / text-derived signal add incremental predictive value on
top of price momentum, fundamentals, macro, and earnings-call sentiment,
when monthly stock returns are forecast under strict point-in-time
constraints?**

## Hypothesis testing structure

The project answers the question by stacking five nested feature sets and
comparing them on identical out-of-sample evaluation rows. Each step is a
single hypothesis test:

| Variant | Features | Hypothesis tested |
|---|---|---|
| **V0a** | none — predict zero | Is anything > 0? (sanity check) |
| **V0b** | trailing-return momentum only | The academic bar (Jegadeesh-Titman 1993) |
| **V1**  | V0b + fundamentals (levels + YoY/QoQ growth) + Bloomberg consensus + macro | Does fundamentals + macro add to momentum? |
| **V2**  | V1 + earnings-call sentiment | Does call sentiment add? |
| **V3**  | V2 + SEC 10-Q text features | Does 10-Q disclosure text add? |

V0b is the bar that V1, V2, V3 each need to beat. All variants are trained
on the same 13-ticker pooled cross-section and evaluated on identical test
rows, so the only thing changing between variants is the feature columns.

## Success criteria

A successful run answers each of the four nested hypotheses with a
defensible "yes" or "no":

1. **Statistically meaningful** — out-of-sample R² and Spearman IC for each
   variant, with V0a / V0b as the noise-floor reference. Differences need
   to be large enough to plausibly survive the limited cross-section (13
   tickers) and the overlapping-label uncertainty inherent in `fwd_ret_3m`.
2. **Economically meaningful** — a long-short tertile portfolio backtest,
   gross of transaction costs, that uses the *same* portfolio construction
   rule across V0b/V1/V2/V3 (V0a uses equal-weight buy-and-hold as the
   benchmark).
3. **Robust to horizon** — both `fwd_ret_1m` (clean non-overlapping labels,
   primary) and `fwd_ret_3m` (smoother signal, secondary) reported side by
   side so the reader can see whether conclusions hold across horizons.
4. **Point-in-time safe** — automated tests (`test_panel_no_lookahead.py`,
   `test_10q_point_in_time.py`) confirm no feature row uses information
   from after its `date`.

## What is NOT claimed

- This is a research evaluation, not a trading strategy. Reported portfolio
  Sharpes are **gross of transaction costs** and should be interpreted as
  upper bounds.
- The 13-ticker universe is hand-curated and biased toward current
  large-cap survivors (4 mega-cap tech + 5 industrials + finance, staples,
  energy, telecom). Generalization to a broader universe (e.g., Russell
  1000) is out of scope and not implied.
- The cross-section is too narrow (13 tickers) to support strong
  cross-sectional anomaly claims; results should be read as
  "does feature X add predictive content within this universe" rather than
  "does feature X work universally."
