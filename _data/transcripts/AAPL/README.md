# Apple Earnings Call Transcript Data

Source: WRDS Capital IQ Transcripts.

Company: Apple Inc. / AAPL / Capital IQ companyId = 24937.

The original target period was 2005-2025. No qualifying Apple earnings call transcript rows were returned for 2005, so the final available sample runs from 2006-01-18 to 2025-10-30.

## Filters

- Event type: Earnings Calls / keyDevEventTypeId = 48
- Transcript presentation type: transcriptPresentationTypeId = 5 / Final transcript
- Proofed Copy is preferred when available.
- If a keyDevId has no Final + Proofed Copy transcript, the latest Final transcript for that keyDevId is selected.

## Output Files

- `aapl_earnings_calls.csv`
- `aapl_transcript_components.csv`
- `aapl_earnings_calls_llm.jsonl`

## Validation Summary

- Call-level earnings calls: 80
- Component-level rows: 5822
- JSONL records: 80
- Duplicate key_devid: 0
- Empty full_text: 0
- Component_count mismatches: 0
- Coverage: complete Q1-Q4 coverage for every year from 2006 through 2025

## Selected Collection Types

- Proofed Copy: 49
- Edited Copy: 18
- SA Edited Copy: 12
- Audited Copy: 1
