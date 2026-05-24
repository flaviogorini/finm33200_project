# Handoff — Write-up revision pass + anchor bug + repo cleanup (2026-05-23)

## Goal

Revise `reports/writeup.qmd` (and mirror in METHODOLOGY.md) per user
feedback after reading the rendered HTML — expand per-signal sections,
correct factual claims, strip dev-facing artefacts. Fix one substantive
bug uncovered along the way and finish trimming dead code from the repo.

## Decisions made

- **Anchor monthly-grid bug = real data bug, fix not just document.** The
  user explicitly chose "fix and re-render" over "document and leave it".
- **Keep all 5 signals as first-class, not "1+4 baselines".** Strategy 2
  was previously framed in writeup + METHODOLOGY as "the central LLM test"
  with Strategy 1 as "cheap baseline". User overruled — both are
  legitimate methodological tests, comparison between them is itself a
  result.
- **`BEst Net Income` is Bloomberg `1BF` blended-forward, not raw FY1.**
  Confirmed by reading the workbook `Info` sheet under `data_manual/`.
  Old wording in both writeup and METHODOLOGY was wrong; corrected.
- **Strip all repo-internal links from the write-up.** Markdown links to
  `../src/...`, `../METHODOLOGY.md`, `../data_manual/...` do not work for
  someone opening the rendered HTML. Keep `_output/99_*.png` references
  only — Quarto inlines those as base64. Explanations stayed; only the
  link syntax went.
- **Math notation: stop using `\text{multi_word_name}` constructs.** GitHub
  and IDE markdown previewers strip `\_` inside math delimiters before the
  KaTeX renderer sees them. Replaced every `\text{sentiment\_diff}` /
  `\text{sig\_anchor}` / `\text{BEst\_NI}` etc. with short single-letter
  symbols ($s$, $\Delta s$, $L$, $\Delta L$, $M$, $B$, $R$, $\tau$) defined
  inline in prose. Panel column names (`sig_anchor`, `sig_lm`, …) now
  written as plain inline code outside the math.
- **"No event-cluster taxonomy" is a scope choice, not a limitation.**
  Removed every framing of "we did not do event-cluster" from writeup,
  METHODOLOGY (limitations + future-work + intro + prior-work), and
  bibliography entry for S&P. Kept "Anchor sentences unvalidated" — user
  agreed that one is a real limitation.
- **`gen ai project.md` deleted.** Was the original project-scope doc;
  no longer needed (METHODOLOGY + writeup cover everything). Removed from
  README "Where the story lives" list.
- **`src/misc_tools.py` + `src/test_misc_tools.py` deleted (1,072 lines).**
  Cookiecutter template leftover; grep confirmed `misc_tools` is not
  imported anywhere in the project. `src/test_backtest_engine.py` is real
  and kept (12 tests, all pass).
- **README "ships with embeddings" claim deleted.** `_data/` is fully
  gitignored, including `embeddings_transcripts.parquet`. Rewrote
  Prerequisites bullets to honestly require WRDS + OpenAI API key.

## What was done

4 commits on `main` (commits `9d93f69` → `5b6d3d5`):

1. **`9d93f69`** — Anchor month-end bug fix in [src/build_sentiment_features.py](../../src/build_sentiment_features.py).
2. **`9914602`** — Dead-code + README cleanup: delete `misc_tools.py`,
   `test_misc_tools.py`, `gen ai project.md`; fix `dodo.py` task_run_pytest
   `file_dep` list; rewrite README's Prerequisites bullets.
3. **`243eb9e`** — Big writeup + METHODOLOGY editorial pass.
4. **`5b6d3d5`** — Prior handoff doc (`2026-05-23-writeup-and-cleanup.md`).

Pipeline re-run after the bug fix:
`build_sentiment_features` → `build_signal_panel` → `run_backtests` →
`joint_regression` → `run_notebooks:99_results.ipynb.py` → `write_report`.
Charts in `_output/99_*.png`, metrics JSON in `_data/metrics_*.json`,
notebook in `_output/99_results.{ipynb,html}`, writeup in
`reports/writeup.html` all reflect post-fix numbers.

## What's left

- Push `main` to `origin/main` (4 commits ahead, user-driven push).
- Optional follow-ups raised but not done:
  - Re-run FM regression restricted to stale-excluded rows only —
    anchor's t-stat is currently `-0.15` in the main-spec FM because
    stale carry-forward months wash out the post-earnings signal. Cleanest
    way to verify "anchor cosine is genuinely event-driven" beyond just
    the metrics-table comparison.
  - Cite an analyst-revisions foundational paper (Stickel 1991 or
    Chan/Jegadeesh/Lakonishok 1996) — bibliography currently has none.

## Gotchas

- **Anchor numbers changed materially after the bug fix.** Main-spec
  Sharpe 0.673 → 0.364; stale-excl Sharpe 1.122 → 0.744; post-2018
  Sharpe 0.871 → 0.322. The prior numbers were flattered: weekend
  calendar month-ends were silently dropped from the anchor merge — a
  non-random ~28 % of months that happened to be better than average.
  Anchor non-null obs went from 9,617 → 19,359 (now matches LM's 19,353).
  Headline write-up commentary now reflects the corrected, less rosy
  picture. **Do not cite earlier "Sharpe 0.67 anchor" numbers**; they are
  from the buggy panel.
- **Stale-excl is now the cleanest anchor result.** anchor 0.74 Sharpe vs
  ridge 0.49 / LM 0.44 — meaningful gap. The user's intuition that "anchor
  is fundamentally an event-driven signal" is supported by anchor's main
  → stale-excl Sharpe roughly doubling while ridge barely moves (ridge
  already encodes `days_since_earnings` as an explicit feature, so its
  predictions are already implicitly freshness-weighted).
- **Math in METHODOLOGY.md uses single-letter symbols, not `\text{}`.**
  If you add a new equation, follow the established convention:
  $s_{i,t}$ (anchor diff), $L_{i,t}$ (LM), $M_{i,m}$ (momentum),
  $B_{i,m}$ (BEst NI), $R_{i,m}$ (revisions), $\tau_{i,t}$
  (days_since_earnings), FM signals indexed by $k \in \{a, r, l, p, v\}$.
  Don't reintroduce `\text{sig\_xxx}` — it breaks GitHub + IDE markdown
  rendering.
- **Writeup link policy: no clickable repo paths in the prose.** The only
  `../` references in `reports/writeup.qmd` are to `_output/99_*.png`
  charts, which Quarto inlines. If you add anything else, write it as
  prose ("the methodology spec", "the Bloomberg workbook's `Info` sheet")
  rather than as a markdown link.
- **`_data/` and `_output/` chart PNGs are gitignored**; the
  `reports/writeup.html` is the only render artefact tracked in git.
- **`src/test_backtest_engine.py` is the only real test file**;
  `task_run_pytest`'s `file_dep` list in [dodo.py](../../dodo.py) was
  updated to drop the stale references to `misc_tools.py`,
  `test_misc_tools.py`, and `conftest.py` (the last was already deleted
  in a prior handoff but the `file_dep` line still pointed at it).
- **`.claude/settings.local.json` is intentionally uncommitted** —
  local permissions allowlist, kept out of git per prior session
  decision.
