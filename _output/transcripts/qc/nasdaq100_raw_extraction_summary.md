# Nasdaq-100 Full Raw Transcript Extraction Summary

Generated: 2026-05-18T01:50:08

## Scope

- Run ID: `nasdaq100_raw_20260518_014157`
- Sample tickers: AAPL, ABNB, ADBE, ADI, ADP, ADSK, AEP, ALNY, AMAT, AMD, AMGN, AMZN, APP, ARM, ASML, AVGO, AXON, BKNG, BKR, CCEP, CDNS, CEG, CHTR, CMCSA, COST, CPRT, CRWD, CSCO, CSGP, CSX, CTAS, CTSH, DASH, DDOG, DXCM, EA, EXC, FANG, FAST, FER, FTNT, GEHC, GILD, GOOG, GOOGL, HON, IDXX, INSM, INTC, INTU, ISRG, KDP, KHC, KLAC, LIN, LRCX, MAR, MCHP, MDLZ, MELI, META, MNST, MPWR, MRVL, MSFT, MSTR, MU, NFLX, NVDA, NXPI, ODFL, ORLY, PANW, PAYX, PCAR, PDD, PEP, PLTR, PYPL, QCOM, REGN, ROP, ROST, SBUX, SHOP, SNDK, SNPS, STX, TMUS, TRI, TSLA, TTWO, TXN, VRSK, VRTX, WBD, WDAY, WDC, WMT, XEL, ZS
- Date range: 2005-01-01 to 2025-12-31
- Extraction unit: unique `ciq_company_id`, not ticker
- Raw metadata output: `/Users/flavio/GitHub/finm33200_project/_data/transcripts/raw/nasdaq100_raw_transcript_metadata.parquet`
- Raw component text output: `/Users/flavio/GitHub/finm33200_project/_data/transcripts/raw/nasdaq100_raw_transcripts.parquet`
- Deduped raw component output: `/Users/flavio/GitHub/finm33200_project/_data/transcripts/raw/nasdaq100_raw_transcripts_deduped.parquet`

This stage did not clean transcript text, create `cleaned_text`, create
`llm_text`, generate embeddings, forecast, or modify the existing AAPL processed
dataset.

## WRDS / Capital IQ Tables

- Metadata table: `ciq.wrds_transcript_detail`
- Component/person table: `ciq.wrds_transcript_person`
- Component text table: `ciq.ciqtranscriptcomponent`
- Join keys: metadata to components on `transcript_id`; component/person to text on `transcriptcomponentid`
- Raw text field: `componenttext`

## Earnings Call Filter

`keydeveventtypeid = 48 OR keydeveventtypename ILIKE '%Earnings%' OR headline ILIKE '%Earnings Call%'`

## Row Counts

- Raw candidate metadata rows: 20467
- Raw component rows: 1396726
- Deduped metadata rows: 6535
- Deduped component rows: 452655

## Per-Ticker QC

| ticker   |   ciq_company_id |   raw_candidate_transcript_rows |   unique_key_devid_count |   deduped_call_count |   missing_text_count |   duplicate_candidate_count | is_duplicate_share_class   | primary_ticker   | related_tickers   |
|:---------|-----------------:|--------------------------------:|-------------------------:|---------------------:|---------------------:|----------------------------:|:---------------------------|:-----------------|:------------------|
| AAPL     |            24937 |                             246 |                       80 |                   80 |                    0 |                         166 | False                      | AAPL             |                   |
| ABNB     |        115705393 |                              49 |                       20 |                   20 |                    0 |                          29 | False                      | ABNB             |                   |
| ADBE     |            24321 |                             230 |                       80 |                   80 |                    0 |                         150 | False                      | ADBE             |                   |
| ADI      |           251411 |                             249 |                       80 |                   80 |                    0 |                         169 | False                      | ADI              |                   |
| ADP      |           126269 |                             237 |                       80 |                   80 |                    0 |                         157 | False                      | ADP              |                   |
| ADSK     |           119902 |                             239 |                       79 |                   79 |                    0 |                         160 | False                      | ADSK             |                   |
| AEP      |           135470 |                             230 |                       75 |                   75 |                    0 |                         155 | False                      | AEP              |                   |
| ALNY     |          2796659 |                             213 |                       72 |                   72 |                    0 |                         141 | False                      | ALNY             |                   |
| AMAT     |           251230 |                             241 |                       80 |                   80 |                    0 |                         161 | False                      | AMAT             |                   |
| AMD      |           168864 |                             243 |                       80 |                   80 |                    0 |                         163 | False                      | AMD              |                   |
| AMGN     |            24816 |                             236 |                       76 |                   76 |                    0 |                         160 | False                      | AMGN             |                   |
| AMZN     |            18749 |                             239 |                       80 |                   80 |                    0 |                         159 | False                      | AMZN             |                   |
| APP      |        231651802 |                              39 |                       19 |                   19 |                    0 |                          20 | False                      | APP              |                   |
| ARM      |        667281196 |                              25 |                        9 |                    9 |                    0 |                          16 | False                      | ARM              |                   |
| ASML     |           388904 |                             299 |                      101 |                  101 |                    0 |                         198 | False                      | ASML             |                   |
| AVGO     |         25016048 |                             199 |                       60 |                   60 |                    0 |                         139 | False                      | AVGO             |                   |
| AXON     |           885779 |                             207 |                       73 |                   73 |                    2 |                         134 | False                      | AXON             |                   |
| BKNG     |            33254 |                             236 |                       78 |                   78 |                    0 |                         158 | False                      | BKNG             |                   |
| BKR      |        425005479 |                             105 |                       33 |                   33 |                    0 |                          72 | False                      | BKR              |                   |
| CCEP     |        109130811 |                             187 |                       54 |                   54 |                    0 |                         133 | False                      | CCEP             |                   |
| CDNS     |            25941 |                             237 |                       78 |                   78 |                    0 |                         159 | False                      | CDNS             |                   |
| CEG      |          3136719 |                              31 |                       13 |                   13 |                    0 |                          18 | False                      | CEG              |                   |
| CHTR     |            19609 |                             213 |                       67 |                   67 |                    0 |                         146 | False                      | CHTR             |                   |
| CMCSA    |           173341 |                             241 |                       79 |                   79 |                    0 |                         162 | False                      | CMCSA            |                   |
| COST     |            92817 |                             238 |                       80 |                   80 |                    0 |                         158 | False                      | COST             |                   |
| CPRT     |            27023 |                             225 |                       70 |                   70 |                    0 |                         155 | False                      | CPRT             |                   |
| CRWD     |        420347413 |                              89 |                       27 |                   27 |                    0 |                          62 | False                      | CRWD             |                   |
| CSCO     |            19691 |                             243 |                       79 |                   79 |                    0 |                         164 | False                      | CSCO             |                   |
| CSGP     |           387127 |                             232 |                       71 |                   71 |                    0 |                         161 | False                      | CSGP             |                   |
| CSX      |           257948 |                             249 |                       74 |                   74 |                    0 |                         175 | False                      | CSX              |                   |
| CTAS     |           260725 |                             227 |                       76 |                   76 |                    0 |                         151 | False                      | CTAS             |                   |
| CTSH     |           386024 |                             226 |                       77 |                   77 |                    0 |                         149 | False                      | CTSH             |                   |
| DASH     |        243735719 |                              46 |                       20 |                   20 |                    0 |                          26 | False                      | DASH             |                   |
| DDOG     |        134521275 |                              74 |                       25 |                   25 |                    0 |                          49 | False                      | DDOG             |                   |
| DXCM     |           114039 |                             220 |                       71 |                   71 |                    0 |                         149 | False                      | DXCM             |                   |
| EA       |            27963 |                             240 |                       79 |                   79 |                    0 |                         161 | False                      | EA               |                   |
| EXC      |           296181 |                             218 |                       75 |                   75 |                    0 |                         143 | False                      | EXC              |                   |
| FANG     |        170245760 |                             164 |                       51 |                   51 |                    0 |                         113 | False                      | FANG             |                   |
| FAST     |           270747 |                             237 |                       73 |                   73 |                    0 |                         164 | False                      | FAST             |                   |
| FER      |           883809 |                             194 |                       50 |                   50 |                    0 |                         144 | False                      | FER              |                   |
| FTNT     |          2689126 |                             316 |                       92 |                   92 |                    0 |                         224 | False                      | FTNT             |                   |
| GEHC     |       1804385326 |                              28 |                       12 |                   12 |                    0 |                          16 | False                      | GEHC             |                   |
| GILD     |            29002 |                             242 |                       78 |                   78 |                    0 |                         164 | False                      | GILD             |                   |
| GOOG     |            29096 |                             260 |                       79 |                   79 |                    0 |                         181 | True                       | GOOGL            | GOOG|GOOGL        |
| GOOGL    |            29096 |                             260 |                       79 |                   79 |                    0 |                         181 | False                      | GOOGL            | GOOG|GOOGL        |
| HON      |          1340740 |                             239 |                       74 |                   74 |                    0 |                         165 | False                      | HON              |                   |
| IDXX     |            29729 |                             231 |                       70 |                   70 |                    0 |                         161 | False                      | IDXX             |                   |
| INSM     |            29978 |                             180 |                       55 |                   55 |                    0 |                         125 | False                      | INSM             |                   |
| INTC     |            21127 |                             235 |                       80 |                   80 |                    0 |                         155 | False                      | INTC             |                   |
| INTU     |            21171 |                             222 |                       78 |                   78 |                    0 |                         144 | False                      | INTU             |                   |
| ISRG     |            30239 |                             238 |                       74 |                   74 |                    0 |                         164 | False                      | ISRG             |                   |
| KDP      |           334622 |                             178 |                       61 |                   61 |                    0 |                         117 | False                      | KDP              |                   |
| KHC      |           278212 |                             235 |                       79 |                   79 |                    0 |                         156 | False                      | KHC              |                   |
| KLAC     |           282225 |                             224 |                       77 |                   77 |                    0 |                         147 | False                      | KLAC             |                   |
| LIN      |           318049 |                             218 |                       69 |                   69 |                    0 |                         149 | False                      | LIN              |                   |
| LRCX     |            30655 |                             239 |                       77 |                   77 |                    0 |                         162 | False                      | LRCX             |                   |
| MAR      |            31148 |                             234 |                       71 |                   71 |                    0 |                         163 | False                      | MAR              |                   |
| MCHP     |            31500 |                             237 |                       76 |                   76 |                    0 |                         161 | False                      | MCHP             |                   |
| MDLZ     |           739693 |                             236 |                       74 |                   74 |                    0 |                         162 | False                      | MDLZ             |                   |
| MELI     |         34322384 |                             209 |                       70 |                   70 |                    0 |                         139 | False                      | MELI             |                   |
| META     |         20765463 |                             187 |                       54 |                   54 |                    0 |                         133 | False                      | META             |                   |
| MNST     |           343729 |                             227 |                       72 |                   72 |                    0 |                         155 | False                      | MNST             |                   |
| MPWR     |          3051626 |                             208 |                       69 |                   69 |                    0 |                         139 | False                      | MPWR             |                   |
| MRVL     |            31158 |                             226 |                       78 |                   78 |                    0 |                         148 | False                      | MRVL             |                   |
| MSFT     |            21835 |                             250 |                       80 |                   80 |                    0 |                         170 | False                      | MSFT             |                   |
| MSTR     |           384976 |                             161 |                       46 |                   46 |                    0 |                         115 | False                      | MSTR             |                   |
| MU       |           289030 |                             261 |                       91 |                   91 |                    0 |                         170 | False                      | MU               |                   |
| NFLX     |            32012 |                             259 |                       80 |                   80 |                    0 |                         179 | False                      | NFLX             |                   |
| NVDA     |            32307 |                             290 |                       79 |                   79 |                    0 |                         211 | False                      | NVDA             |                   |
| NXPI     |           934467 |                             180 |                       53 |                   53 |                    0 |                         127 | False                      | NXPI             |                   |
| ODFL     |           319404 |                             227 |                       71 |                   71 |                    0 |                         156 | False                      | ODFL             |                   |
| ORLY     |           324289 |                             225 |                       74 |                   74 |                    0 |                         151 | False                      | ORLY             |                   |
| PANW     |         25460099 |                             191 |                       54 |                   54 |                    0 |                         137 | False                      | PANW             |                   |
| PAYX     |           295368 |                             244 |                       76 |                   76 |                    0 |                         168 | False                      | PAYX             |                   |
| PCAR     |           294721 |                             241 |                       73 |                   73 |                    0 |                         168 | False                      | PCAR             |                   |
| PDD      |        572577901 |                              95 |                       30 |                   30 |                    0 |                          65 | False                      | PDD              |                   |
| PEP      |            32854 |                             248 |                       78 |                   78 |                    0 |                         170 | False                      | PEP              |                   |
| PLTR     |         43580005 |                              67 |                       21 |                   21 |                    0 |                          46 | False                      | PLTR             |                   |
| PYPL     |           112732 |                             150 |                       41 |                   41 |                    0 |                         109 | False                      | PYPL             |                   |
| QCOM     |            33493 |                             238 |                       79 |                   79 |                    0 |                         159 | False                      | QCOM             |                   |
| REGN     |            33715 |                             200 |                       56 |                   56 |                    0 |                         144 | False                      | REGN             |                   |
| ROP      |            22751 |                             234 |                       72 |                   72 |                    0 |                         162 | False                      | ROP              |                   |
| ROST     |            33926 |                             230 |                       80 |                   80 |                    0 |                         150 | False                      | ROST             |                   |
| SBUX     |            34745 |                             252 |                       80 |                   80 |                    0 |                         172 | False                      | SBUX             |                   |
| SHOP     |         84159238 |                             142 |                       42 |                   42 |                    0 |                         100 | False                      | SHOP             |                   |
| SNDK     |       1860586153 |                               9 |                        3 |                    3 |                    0 |                           6 | False                      | SNDK             |                   |
| SNPS     |            35028 |                             250 |                       79 |                   79 |                    0 |                         171 | False                      | SNPS             |                   |
| STX      |          3738520 |                             257 |                       80 |                   80 |                    0 |                         177 | False                      | STX              |                   |
| TMUS     |            93339 |                             185 |                       50 |                   50 |                    0 |                         135 | False                      | TMUS             |                   |
| TRI      |           515275 |                             225 |                       70 |                   70 |                    0 |                         155 | False                      | TRI              |                   |
| TSLA     |         27444752 |                             221 |                       61 |                   61 |                    0 |                         160 | False                      | TSLA             |                   |
| TTWO     |           371281 |                             243 |                       77 |                   77 |                    0 |                         166 | False                      | TTWO             |                   |
| TXN      |           140283 |                             240 |                       80 |                   80 |                    0 |                         160 | False                      | TXN              |                   |
| VRSK     |          1027055 |                             207 |                       63 |                   63 |                    0 |                         144 | False                      | VRSK             |                   |
| VRTX     |            36235 |                             248 |                       79 |                   79 |                    0 |                         169 | False                      | VRTX             |                   |
| WBD      |         22666093 |                             214 |                       64 |                   64 |                    0 |                         150 | False                      | WBD              |                   |
| WDAY     |         23815047 |                             172 |                       53 |                   53 |                    0 |                         119 | False                      | WDAY             |                   |
| WDC      |           314057 |                             226 |                       75 |                   75 |                    0 |                         151 | False                      | WDC              |                   |
| WMT      |           313055 |                             213 |                       72 |                   72 |                    0 |                         141 | False                      | WMT              |                   |
| XEL      |           527542 |                             226 |                       79 |                   79 |                    0 |                         147 | False                      | XEL              |                   |
| ZS       |         58838228 |                             104 |                       31 |                   31 |                    0 |                          73 | False                      | ZS               |                   |

## Deduplication Rule

All raw candidate transcript versions are preserved in `/Users/flavio/GitHub/finm33200_project/_data/transcripts/raw/nasdaq100_raw_transcript_metadata.parquet` and
`/Users/flavio/GitHub/finm33200_project/_data/transcripts/raw/nasdaq100_raw_transcripts.parquet`. The deduped validation file keeps one transcript per
`ciq_company_id` + `key_devid` using:

1. Final presentation type first, when available.
2. Collection priority based on observed collection names: Audited, Proofed,
   Edited, Corrected, Spellchecked, then other/unknown.
3. Latest `transcriptcreationdate_utc` + `transcriptcreationtime_utc`.
4. Larger raw text character count and component count.

No duplicate candidates are silently discarded; the candidate counts are in the
QC file.

## AAPL Benchmark

- Old AAPL processed call count: 80
- New AAPL raw candidate metadata rows: 246
- New AAPL unique `key_devid` count: 80
- New AAPL deduped call count: 80

The new deduped AAPL count is expected to be close to, but not forced to equal,
the old processed count. Differences can come from broader earnings-call
metadata filters, Capital IQ transcript version choices, and the old pipeline's
additional Final/Q1-Q4/headline filters.

## Share-Class Handling

- Requested tickers: 101
- Unique `ciq_company_id` extracted: 100
- Duplicate share-class ticker rows shown in QC but not separately extracted:
  GOOG

## Low Coverage / Special Cases

| ticker   |   ciq_company_id | ciq_company_name    |   raw_candidate_transcript_rows |   unique_key_devid_count |   unique_transcript_id_count |   deduped_call_count | first_transcript_date   | last_transcript_date   |   years_covered | event_type_distribution   | transcript_version_distribution                               | failed_extraction   |   missing_text_count |   duplicate_candidate_count | is_duplicate_share_class   | primary_ticker   | related_tickers   | dedupe_rule_used                                                                                          | notes   |
|:---------|-----------------:|:--------------------|--------------------------------:|-------------------------:|-----------------------------:|---------------------:|:------------------------|:-----------------------|----------------:|:--------------------------|:--------------------------------------------------------------|:--------------------|---------------------:|----------------------------:|:---------------------------|:-----------------|:------------------|:----------------------------------------------------------------------------------------------------------|:--------|
| SNDK     |       1860586153 | Sandisk Corporation |                               9 |                        3 |                            9 |                    3 | 2025-05-07              | 2025-11-06             |            2025 | {"Earnings Calls": 9}     | {"Edited Copy": 4, "Proofed Copy": 1, "Spellchecked Copy": 4} | False               |                    0 |                           6 | False                      | SNDK             |                   | presentation_final_then_collection_audited_proofed_edited_corrected_then_latest_creation_then_text_length |         |

## Recommendation

Full Nasdaq-100 raw extraction succeeded. The raw candidate and deduped raw datasets can be frozen after reviewing the AXON missing-text components and expected low-coverage companies. It is safe to proceed to cleaning and section-parsing design; do not start embeddings until cleaned views are defined.
