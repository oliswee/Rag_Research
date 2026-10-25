# Multi-query Hybrid retrieval (M3)

Normal source requests and follow-ups use Hybrid by default. Uploaded PDFs and
online imports share this path. A library without source chunks can still use
the legacy Pattern workflow. The web UI exposes paper/year/provider scope;
technical retriever modes are available only through internal API/evaluation
options. The original question is always used to compose the answer.

## Query planning

Q0 is the literal original question. One bounded model call may produce Q1
(standalone rewrite) and Q2 (keyword/synonym query), independently searched.
Variants retain original content tokens and numbers; token-set duplicates are
removed before retrieval. Q1/Q2 have stable request-local IDs, types and the
original parent question. Recent user history may resolve follow-up references.

Exact identifiers, exclusions/negations, quoted phrases and ambiguous numeric
conditions conservatively use Q0 only, with a recorded reason. A single calendar
year can remain literal in variants; decimals, inequalities, units and multiple
numeric roles do not get rewritten. New names, numeric/negative conditions or
missing research terms are rejected. These checks favor safe false rejections;
they are not a general semantic equivalence proof. Model absence, bad JSON,
provider error or timeout also returns Q0. Explicit BM25/Dense baselines always
use literal Q0, even for follow-ups, matching the development benchmark.

Paper IDs, year range, provider and local_only are request-owned fields. A model
does not receive the ability to modify them. Existing local_only enforcement
requires cached FastEmbed and loopback Ollama/Qdrant; generated text cannot
authorize Web access or change the corpus scope.

## Branches and budgets

| Parameter | Default / upper bound |
| --- | --- |
| Queries | 3 |
| Ranked lists | 6: queries × BM25/Dense |
| Per-list K | 30 |
| Final K | 10 |
| RRF k | 60 |
| Request branch concurrency | 3 |
| Rewrite timeout | 20s, one model attempt |
| Branch deadline including queue wait | 15s |

Validated internal configuration can lower these values. Embeddings for valid
queries are batched once. Independent branch tasks overlap lexical work with
embedding/vector work; all operations on the request's AsyncSession are
serialized. A semaphore bounds active branches. This is one layer of RRF over
independent lists; Hybrid is the fusion result, not another retriever.

SQLite execution uses a native progress-handler interruption and bounded busy
timeout. Async cancellation is used while queued, not inside an active SQLite
query; the lock stays held through cleanup. This preserves the connection and
the in-memory test corpus after a deadline. Native operations that cannot yield
(for example a blocking SQLite user-defined function) finish their current
operation before interruption can be observed.

Synchronous Qdrant reads run in worker threads. Worker-owned clients close in
those threads after reads finish, including after a branch timeout. Benchmark
readers have independent clients with the same benchmark dataset scope; the
parent benchmark client may close without racing those readers.

Dense IDs are checked against SQLite active/ready generations and paper/year/
provider scope before they can vote in RRF. Source resolution checks active
generation and immutable original spans again. Duplicated IDs vote once per
list. Retired, pending, missing and foreign IDs do not become evidence.

## Failure reporting and Trace

- Rewrite rejected/unavailable: single-query Hybrid, with the planning reason.
- One retriever unavailable: bm25_fallback or dense_fallback.
- Some variant branches fail: hybrid_degraded, surviving branches retained.
- Every branch fails: retrieval_failed; it is distinct from an empty successful
  search. Baseline Dense can fall back to BM25 if no valid dense evidence remains.

Trace records query_id/type/parent-question hash, retriever, rank/score, fused
rank, branch status/error class, queue-inclusive timing, budgets, final IDs and
source anchors. Private queries are hashes/placeholders; no source excerpts or
exception messages enter Trace. Single-query BM25/Dense ranking aliases remain
available for prior tooling. Public QASPER traces may contain sanitized public
questions. API query_count and retrieval_status explain the planning outcome;
retrieval_mode describes retriever degradation separately.

## Development evaluation

From backend/:

```powershell
uv run python -m evaluation.multiquery_benchmark --output-dir data/evaluation/m3_verified_20261024 --judge
```

The evaluator checks the frozen QASPER40 manifest/chunk/label byte hashes before
any index work. It saves actual local generated plans, generation traces and
timings, and validates cache question/model/validator identity before replay.
Use a new output directory after an incompatible planning change. Development
plans revalidated after a stricter safety fix retain their original generation
fingerprint and a separate validation record; changed queries require retrieval
to be rerun.

Compare BM25, Dense, Q0 Hybrid and multi-query Hybrid on the same indexed text.
Small development-only sweeps use K10 vs30 and RRF20 vs60. Each run saves per-case
Recall/Hit/MRR/nDCG, latency and Trace; no-answer questions are excluded from
relevance averages. Retrieval timing excludes saved planning, which is reported
separately. The optional bounded local DeepEval smoke measures Contextual
Relevancy for two cases with one context each; failures are not_evaluated.

QASPER uses existing extracted-text children to preserve published-label
comparability, not the real PDFs or a new human Gold set. Structured current
children are verified separately on the local library. This development set is
used for diagnosis; no final semantic quality or statistical improvement claim
is made. Model-weight revisions may remain unknown; model ID/collection
namespace and dimensional checks prevent mixing different configured models.
Exact measured results and remaining limits are appended to
[metrics_history.md](../backend/evaluation/metrics_history.md).
