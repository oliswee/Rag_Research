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
downloaded papers or live academic retrieval**. DeepEval receives actual
generated text, with `metric_target` separating published verified answers from
unpublished drafts rejected by production checks. Draft scores do not establish
end-to-end QA success. Missing/failed model output is `not_evaluated`, never a
made-up zero/one. Local model/Judge results are dependent
automatic opinions and have not been human calibrated.

## Runtime and measured results

The existing configured loopback Ollama endpoint was offline and no native
binary was found. Restored official Ollama0.40.13 portable CPU binaries into
ignored `data/runtime/ollama`, bound `127.0.0.1:11434`, and isolated downloaded
weights in `data/runtime/ollama-models`. This is a user-scoped background process,
not an installed startup service. The 4B weight download was initially slow and paused, then resumed and
completed with SHA-256 verification. Both local model manifests are available. Project default remains `qwen3:4b-instruct`; an
explicit smaller-model diagnostic and default-model results are recorded below. No new
paid account, API key or cloud generation was used. Local CPU/time/network/disk
consumption is not monetarily measured.

Artifacts: ignored `data/evaluation/m6_20261102/` contains configuration/pipeline
and fixture SHA-256, original fixture outputs, model decisions/budgets, matrix
demo and `generation_deepeval_results.jsonl`. Measured results are listed below.

### Actual small-model results

Ollama0.40.13 / qwen3:0.6b, explicit diagnostic override, temperature0. Four
nonempty answer cases produced actual drafts but **0/4 passed production
citation identity checks**. They confused paper IDs/child IDs or copied the
schema's placeholder quote key. An explicit child-ID catalogue and explanation
was then added;30related regressions passed/2.23s, but repeating these four cases
still yielded0/4 published findings. No-answer correctly abstained without a
model call. Keep both runs; this is a model-capability limitation, not a quality
pass or proof that the prompt improved performance.

The six fixed-candidate ablation cases contain four negative labels and two
supported controls. Each mode replays six generation calls and makes three real
semantic calls: **additional real calls for atomic batch =0**. First functioning
run: whole-review false acceptance1/4 negatives, false rejection0/2 controls;
atomic false acceptance0/4 and false rejection0/2. Repeated run with the explicit
ID prompt: whole1/4 and0/2; atomic0/4 and**1/2 false rejection**. The unstated
"excellent accuracy" result was admitted by whole-review. Judgements varied
between runs despite temperature0; do not generalize this small reused sample.

DeepEval4.2.8 judged actual **unpublished drafts**, not a successful published
answer, with default templates/threshold0.5/temperature0 and60seconds per metric:

| Actual draft | Faithfulness | Answer Relevancy | Status |
| --- | ---: | ---: | --- |
| Local native PDF,95% |0.333333|1.0|both evaluated|
| Web OA full-text fixture,95% |1.0|1.0|both evaluated|
| Abstract with missing sample size |1.0|1.0|both evaluated|
| Two conflicting original accuracies |0.5|null|faithfulness evaluated; relevancy timed out|

The inconsistent scores for the simple95% statement illustrate the tiny Judge's
limitations. Seven metrics evaluated and one timed out; no reference answer or
independent calibration was used. Four actual generation calls consumed observed
2243input/1767output tokens; semantic verification was not called for malformed
citations. Judge HTTP subcalls/tokens/total resource cost remain unknown/null.
Framework cost0 only denotes the adapter's monetary accounting. Intermediate
generation/semantic/runtime probes are separate runs, not included in these
four-call totals. Raw failures, latencies and fingerprints remain in the artifacts.

### Default4B model: CPU and GPU follow-through

Restored `qwen3:4b-instruct` (manifest
`0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`,
2497293803bytes) without changing the project's model setting. On CPU, the first
native-PDF request exceeded the30-second business-call limit and safely returned
evidence_only, with null scores. The subsequent warm abstract request passed
citation/semantic checks; Faithfulness1.0, Answer Relevancy0.5. This failure is
retained in `default_4b_final`, not erased by the GPU rerun.

The machine has an RTX4060 Laptop GPU/8GiB. Added the official portable Vulkan
backend (about12MiB compressed dependencies) and restarted only the helper server
created in this task. Ollama reported37/37 layers offloaded. This follows
[Ollama's Windows portable CLI](https://docs.ollama.com/windows) and
[Vulkan hardware support](https://docs.ollama.com/gpu); no driver, paid service or
system startup configuration was changed. Local service remains bound to
127.0.0.1:11434, with weights under `data/runtime/ollama-models`.

Actual default-model GPU answers, both **published_grounded_answer**, not drafts:

| Case | Generation+verification ms | Calls | Observed input/output tokens | Citation page | Faithfulness | Answer Relevancy |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Local native PDF |19749.516|2|883/234|1|1.0|1.0|
| Abstract, accuracy plus missing sample size question |4970.491|2|664/158|null|1.0|0.5|

Four business model calls,1547input/392output tokens observed. Judge metric
latencies respectively14372.646/10130.425ms and12253.459/11447.026ms. Judge
HTTP subcalls/tokens/total cost remain unknown. No paid cloud calls. The abstract
answer supports the reported accuracy only; it does not answer the absent sample
size, hence no claim of complete question coverage. **Two synthetic answers are
not a calibrated accuracy estimate, a retrieval gain or a latency benchmark.**
Web evidence is still an injected fixture. `default_4b_gpu` preserves JSONL,
raw candidate versus published answer, actual citations, configuration/fixture
fingerprints and timings. Its App fingerprint equals the delivered App.

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

## Final application verification

- Full backend1086passed,13existing warnings,439.17s, after the review fixes.
  The later prompt-only identity clarification passed30related regressions/2.23s;
  its exact implementation commit passed the complete remote CI.
- Frontend39passed/13.61s; ESLint and production build passed on retry after the
  existing Next dependency security notice. Ruff check/format192files and
  whitespace checks passed. Final whole-branch privacy gate46files passed, including the separate
  metrics report; no configured keys/contact emails/raw corpus were included.
- Implementation head `ae1fdaacd5a53eea65145160a805e6f80c0a9696` passed both
  Backend Tests and Frontend Build in CI41397523708. Dependent Draft PR58 targets
  M5; no merge. The final report/runner commit CI is recorded in PR58 after it
  finishes, avoiding a documentation-only commit loop.
- Final offline and actual-draft diagnostic App fingerprints match the delivered
  application fingerprint. Small-model manifests explicitly differ from the
  default model configuration; no hidden provider substitution.
