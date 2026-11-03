# Verified QA and per-paper comparison (M6)

`POST /api/query` and `POST /api/research/compare` use the same evidence registry,
atomic claim checks and bounded generation service. Comparison accepts two to
five distinct SQLite paper IDs. Every paper gets its own Hybrid → rerank →
parent context; all local searches finish before any permitted academic fallback.
Missing findings remain unknown in their own columns. Academic records do not
automatically enter the library.

## Identity and original evidence

Before model use, local child text, paper, source version, spans and pages are
re-read from SQLite. Archived papers, retired generations and altered same-ID
retrieval results are rejected. A retained historical citation can still be
opened through the source API; it cannot establish a new finding from a retired
index generation. Validated parent text is background, never a substitute for
the cited child quote. Legacy text has an excerpt digest and null PDF pages.

Temporary Web evidence must match an immutable snapshot sealed by the current
M5 academic pool. Quotes must be exact substrings of the original child.
Full-text OA, abstract-only and bibliographic metadata remain distinct. Metadata
cannot establish experimental results. Abstract-only sources have no PDF page.
Native pages must come from validated original spans; cross-page quotations do
not invent one page. The registry retains DOI, original origin, URL, version,
retrieval date and licence when available. Local publication year comes from
SQLite publication metadata, never a generated or caller-supplied year.

## Generation and scientific checks

The model proposes atomic claims with exact child IDs and original quotes.
Deterministic checks reject missing/cross-paper citations, unsupported numbers,
units, numeric roles, OR/RR/HR changes, simple polarity flips, association upgraded
to causation and abstract upgraded to full-text access. A subsequent **single
batched semantic call** checks each surviving claim and each scientific field.
Only a supported claim's text is displayed. Timeout, malformed output or budget
exhaustion preserves original evidence, without publishing unverified findings.
There is no factual token streaming before verification. Nested citation support
states follow the claim's semantic decision.

Quoted, separately supported population, design, outcome, exposure, effect
measure, adjustment and direction fields support pairwise comparison. Matching
independent study identities and supported characteristics are necessary for
`comparable_agreement` or `comparable_conflict`. Different measures/populations
produce `not_comparable`; missing fields or independence produce
`insufficient_evidence`. The same DOI is the same study, including different
versions, and does not count as independent corroboration. These are conservative
rules plus a fallible model opinion, not scientific truth or human validation.

## Budgets and privacy

The entire QA or comparison model stage shares at most 10 interface calls,
8192 reserved output tokens, 131072 serialized input bytes, 30 seconds per call
and a 180-second model-stage deadline. Reserved output limits are not observed
token usage. Per-paper comparison uses one generation plus one batched check
when both succeed, not a call per field. SDK retries are disabled in this stage.
The shared Web budget remains at most two providers/eight HTTP attempts/one
licensed PDF and 45 seconds across the columns. Model and Web deadlines are
stage limits, not a promised whole-request latency.

Local-only/uploaded-only enforcement precedes retrieval and model construction;
the local endpoint must be loopback. M5 scope inheritance remains in effect.
An unresolved pronoun or same-title identity asks one scope question. A pronoun
with previous local child references is bound through SQLite to exactly one
paper before retrieval and generation; missing/multiple identities are clarified.
The resolved scope is inherited by later turns. Only permitted provider queries
leave the machine; local excerpts and history never become academic search terms.
Trace keeps source IDs, hashes, decisions, timings, model usage and budgets;
private source text and error bodies are excluded.

## Development diagnostics

Run from `backend`:

```powershell
uv run python -m evaluation.generation_benchmark --output data/evaluation/m6
uv run python -m evaluation.generation_benchmark --output data/evaluation/m6-live --live
```

The first command runs public synthetic guards and a five-paper orchestration
demo with fixed retrieval/model boundaries. It does not measure retrieval or
model quality. `--live` permits only configured loopback Ollama, replays fixed
candidates for whole-review/atomic ablation, then generates actual answers and
applies DeepEval Faithfulness and Answer Relevancy to actual generated outputs.
Each metric record identifies a published grounded answer or an **unpublished
candidate draft**. Draft scores never establish successful end-to-end QA: the
production verifier can reject a semantically accurate draft with invalid IDs.
`--model` explicitly overrides the local diagnostic model only. Outputs go to
ignored local artifacts, including `generation_deepeval_results.jsonl`.
`--skip-ablation` avoids repeating a completed replay; `--cases` selects a bounded
subset of answer fixtures. These flags do not relax production citation checks.
No credentials or paid cloud model is needed. Keep each run in a fresh output
directory; JSONL appends preserve previous records. Scores, failures, versions,
model choice and limitations belong in `backend/evaluation/m6_development.md`
and the separate `backend/evaluation/metrics_history.md` section11.

These fixtures are visible during implementation, use a tiny designed sample
and injected evidence boundaries, and are **not independent Gold**, human PDF
audit, real Web-answer improvement, latency SLA or broadly calibrated accuracy.
Independent calibration and audit remain M8.
