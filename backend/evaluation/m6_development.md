# M6 development evaluation — 2026-11-02

Issue57; branch `feat/57-m6-grounded-comparison`, M5 base `a802165`.
Engineering regression and public synthetic diagnostics; **not independent Gold,
human review or demonstrated real-world retrieval/answer improvement**.

## Reproducible boundaries

`evaluation/generation_benchmark.py` reads six hand-authored claim labels from
`evaluation/fixtures/m6_claims.json`. Three deterministic counterexamples cover
causation, direction and sample size; an unstated qualitative result requires
semantic interpretation. Two supported controls permit measurement of false
rejections. The same candidates, registry, guards and budgets feed both
`whole_review` and `atomic_batch`. The first generation interface call is replayed;
only the subsequent semantic model call is actual inference. Report these
separately rather than counting two real model calls. No automatic overall
quality gain is assumed.

The five-paper demo uses the real comparison workflow, SQLite originals and a
fixed retrieval/model boundary. Four papers have synthetic abstract text; the
fifth has no evidence. Repeated dimension cells are plumbing fixtures, not
substantive study comparisons. Missing scientific fields remain incomparable or
insufficient. It performs no model inference or live Web retrieval.

Actual-answer fixtures cover a native locally created PDF with real page/span
parsing, a sealed public OA full-text fixture, an abstract asked for an absent
sample size, and no answer. Web originals are injected public fixtures, **not
downloaded papers or live academic retrieval**. DeepEval only receives actual
verified generated text from these runs. Missing/failed verified output is
`not_evaluated`, never a made-up zero/one. Local model/Judge results are dependent
automatic opinions and have not been human calibrated.

## Runtime and measured results

The existing configured loopback Ollama endpoint was offline and no native
binary was found. Restored official Ollama0.40.13 portable CPU binaries into
ignored `data/runtime/ollama`, bound `127.0.0.1:11434`, and isolated downloaded
weights in `data/runtime/ollama-models`. This is a user-scoped background process,
not an installed startup service. The 4B weight download was slow and paused
with partial data retained. Project default remains `qwen3:4b-instruct`; an
explicit smaller-model diagnostic is recorded below after execution. No new
paid account, API key or cloud generation was used. Local CPU/time/network/disk
consumption is not monetarily measured.

Artifacts: ignored `data/evaluation/m6_20261102/` contains configuration/pipeline
and fixture SHA-256, original fixture outputs, model decisions/budgets, matrix
demo and `generation_deepeval_results.jsonl`. Actual results follow after execution.

## Regression and review evidence

- Registry/anchors/Web focused27passed; science/generation/comparability focused
  46passed; API/workflow/M5 focused55passed. Individual RED/GREEN details remain
  in the ignored task ledger.
- First old QA compatibility run:34passed/5failed. Two old function mocks,
  two evidence-only/empty-message expectations were updated; a missing per-paper
  retrieval-budget Trace field was reproduced and fixed. No old generation path
  was re-enabled in production to satisfy compatibility tests.
- First complete backend:1080passed,14warnings,619.09s. Frontend39passed,13.61s;
  ESLint passed. Next production build stopped at the existing dependency
  security-upgrade notice; retry succeeded. Dependency security maintenance
  remains recommended separately; no dependencies were upgraded in M6.
- Self-check reproduced authoritative local year and nested citation status
  issues (4failed), fixed and23focused checks passed.
- One fresh whole-branch reviewer found three P2 issues: negated reduction
  escaped the deterministic direction check; a pronoun's previous identity was
  announced without actually scoping retrieval; retired index generations could
  pass registry revalidation. English/Chinese polarity, actual API scope and
  retired-source regressions failed before fixes. Fixed by broader simple
  polarity matching, authoritative SQLite reference→paper scope and rejecting
  retired generation for new findings.35focused checks passed/4.48s.
- Edge production UI at1280×900 and390×844 with fixed public API fixtures:
  verified claim/original quote, Web abstract type with no invented page,
  unknown cell, same-title identity selectors, OR/RR incomparability and
  bibliographic access records. The real backend was not used for this rendering
  check. Comparison table scrolls horizontally on narrow screens. Existing
  compact global navigation extends15px past the mobile viewport; the new table
  is contained. Screenshots under ignored `output/playwright/`.

Final full regression/privacy and exact-head CI results will be recorded after
execution. No merge.
