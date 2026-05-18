# Decision-digest examples & failure cases

The decision digest is the project's tool-layer artifact: given
`(ticker, as_of_date)`, it consumes the returns forecast and a small
retrieved-evidence block (10-Q + transcript chunks), and emits a one-shot
structured digest. The full schema and prompt are documented in
[methodology.md](../project_overview/methodology.md).

Three independent verifiers (see [eval_digest.py](../../src/eval_digest.py)):

- `citation_match_rate` — fraction of cited quotes appearing verbatim in
  the pre-fetched evidence block.
- `numeric_grounding_rate` — fraction of regex-extracted numbers in the
  rationale that appear verbatim in the evidence block.
- `direction_match` — sign-match between `forecast_direction` and the
  realized `fwd_ret_1m`.

The pre-cached digests for the 5 × 4 grid live under `_data/digest_cache/`.
After running `python src/eval_digest.py`, the aggregate numbers land in
`_output/digest_eval_summary.json` and the per-digest detail in
`_output/digest_eval.parquet`.

The two case studies below are placeholders pointing at the cache file paths
that will hold the actual examples once the user runs the pipeline. The
methodology requires picking one success and one failure case — *not*
forcing pre-defined failure-mode boxes. See Olson 2026 (FINM 33200 guest)
on why six contrived failure cases is worse than two honest ones.

## Success case (placeholder)

> _Pick one digest from `_data/digest_cache/` with:_
>
> - `citation_match_rate >= 0.8`
> - `numeric_grounding_rate >= 0.8`
> - `direction_match == 1.0` (forecast direction matched realized fwd_ret_1m)
>
> _Then transcribe the response JSON's `rationale_*` paragraphs and one or
> two evidence items. Frame it as "the digest correctly read the
> disclosure context, and the realized return went the right way."_

## Failure case (placeholder)

> _Pick one digest with at least one of:_
>
> - `direction_match == 0.0` and `confidence > 0.7` (confident-but-wrong)
> - `numeric_grounding_rate < 0.5` (hallucinated numerics in the rationale)
> - Empty evidence (`n_evidence_items < 3`) but a confident direction
>
> _Transcribe the rationale paragraphs and call out which verifier
> caught it. Example framing: "The digest claimed a YoY revenue change
> of 14.2% in `rationale_fundamentals`, but the number 14.2% does not
> appear anywhere in the pre-fetched evidence block — the
> `numeric_grounding_rate` flags this as a hallucinated figure. The
> realized fwd_ret_1m of -3% disagreed with the bullish call. Fix:
> tighten the system prompt's verbatim-numerics rule, or strip
> per-claim numerics that don't pass extract verification before
> emission."_

## How to populate this page

```powershell
# Generate the 20 digests (needs OPENAI_API_KEY; cost ~$2)
python src/generate_digest.py --all

# Run the three verifiers across all cached digests
python src/eval_digest.py

# Inspect the rankings to pick success + failure case
python -c "import pandas as pd; df = pd.read_parquet('_output/digest_eval.parquet'); print(df.sort_values('direction_match').to_string())"
```

For each chosen case, open the corresponding
`_data/digest_cache/{TICKER}__{YYYY-MM-DD}__v1.json` and transcribe the
`response.rationale_*` fields plus the relevant `response.evidence[]` items
into this page. Quote verbatim. The aggregate numbers go in the table below
after running `eval_digest.py`.

## Aggregate results (placeholder)

| Metric | Value |
|---|---|
| Mean citation_match_rate | _from `digest_eval_summary.json`_ |
| Mean numeric_grounding_rate | _from `digest_eval_summary.json`_ |
| Direction accuracy (n digests with realized) | _from `digest_eval_summary.json`_ |

Per-ticker breakdown:

| Ticker | n | Citation | Numeric grounding | Direction match (n realized) |
|---|---|---|---|---|
| _populate from `digest_eval_summary.json` `per_ticker` block_ | | | | |
