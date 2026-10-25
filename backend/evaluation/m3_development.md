# M3 QASPER development comparison

Date: 2026-10-24 (Australia/Sydney). Execution head `e89444f` plus the then-dirty
review-fix worktree; later commit IDs are not backfilled as experiment versions.
Exact per-stage pipeline/config hashes are saved in the reports and Trace.

## Fixed configuration

- QASPER dev0.3:40 papers,1,892 extracted-text children;32 answerable and8
  paper-scoped no-answer cases. Same frozen byte hashes as qasper_dev_config.yaml.
- BGE `BAAI/bge-small-en-v1.5`,384 dimensions, cosine Qdrant;
  benchmark scope `benchmark:c0d182c5013c0edd`, separate from the actual library.
  Embedding weight revision unknown; configured model ID/collection/dimension
  checks prevent incompatible model reuse.
- Q0 literal question; at most3 independent validated queries, concurrency3,
  per-list K30, final K10, single-layer RRF k60. Rewrite20s/one attempt,
  branch15s including queue wait. Explicit baselines never rewrite Q0.
- Planner and optional Judge: local Ollama `qwen3:4b-instruct`, observed digest
  `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`.
  No cloud-provider requests or answer generation in this experiment.

## Actual retrieval runs

Quality means use only32 public evidence-labelled cases. Latency uses40 cases
and includes the retrieval pipeline/Trace, excluding cached query planning.
Runs are sequential and had development workloads on the machine; do not infer
that Multi-query is faster from these latency differences.

| Mode | Recall@10 / Hit@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.78125 | 0.348996 | 0.447931 | 11.53 | 13.11 |
| Dense | 0.75000 | 0.338690 | 0.437090 | 3895.26 | 4364.92 |
| Single-query Hybrid | 0.78125 | 0.384462 | 0.477571 | 4366.75 | 4479.28 |
| Multi-query Hybrid | 0.78125 | 0.368837 | 0.466038 | 3898.59 | 4439.68 |

Multi-query changed mean Recall by0 and MRR by−0.015625 against Single-query.
Recall wins/ties/losses:0/32/0. This run does not show a quality gain. The v6
mainline feature remains with explicit safe fallbacks; semantic improvement and
parameter selection require later work, not a small-sample success claim.

## Actual planning and parameter replay

40 local plans:35 have Q0 only,4 have2 queries,1 has3 queries. Six questions
used the constraint guard without a model call;29 model outputs produced no
valid distinct variants. In total62 candidate variants were rejected. Strict
content-token preservation causes false rejections; this is recorded rather
than substituting handcrafted variants into the measured Multi-query run.

34 successful planner calls,4,420 input tokens and884 output tokens; total
planning91,367.83ms. Per-case planning latency and complete generation Trace
remain in query_plans.jsonl. Those calls are separate from retrieval timing.

After the final stricter numeric-role fix, all40 saved plans were revalidated:
**zero query changes**. Original generation fingerprints/traces/timings were
preserved alongside the final validator key in plan_revalidation.json. No new
model output or generation timestamp was invented.

The small parameter sweep replays the saved validated branch rankings. It does
not call embeddings or retrieve new candidates, and has no retrieval latency:

| Replay | Recall@10 | MRR@10 | nDCG@10 | p50 / p95 |
| --- | ---: | ---: | ---: | --- |
| First10 results per list, RRF60 | 0.81250 | 0.372346 | 0.475275 | null |
| First30 results per list, RRF20 | 0.78125 | 0.368093 | 0.465323 | null |

The K10 replay is exploratory on the same development set. Defaults stay K30 /
RRF60; no held-out parameter-selection or significance claim is made. An aborted
reretrieval K10 attempt is retained separately and is not used for these scores.

## Local DeepEval smoke

DeepEval4.2.8, Contextual Relevancy only; two first public cases, one retrieved
context each, threshold0.5, temperature0, default DeepEval prompt. Actual output
is empty because no answer-generation metric is being evaluated.

| Case order | Status | Score | Judge latency ms |
| --- | --- | ---: | ---: |
| 1 | evaluated | 0.0 | 43766.063 |
| 2 | evaluated | 1.0 | 17407.344 |

These two opinions are not an overall quality estimate or human calibration.
Judge API request count/tokens remain null. Framework cost fields are0 for the
local adapter, not a measured electricity/hardware cost. Telemetry/update
reporting is disabled for this local smoke.

## Engineering and actual library

The final review reproduced/fixed shared SQLite timeout invalidation, numeric
operator/decimal drift, baseline Q0 replacement, benchmark client-lifetime
races and permissive budgets. Native SQLite execution deadlines retain the
session/lock; vector worker clients own their teardown. Numeric-role swapping
has an additional failing-before/passing-after regression.

A later full run exposed an asynchronous SQLite-interrupt/finalization race
(891 passed /1 failed). The extra interrupt timer was removed; native progress
callbacks and busy-timeout deadlines remain, with CPU-read/rollback regression.
Final delivery verification (2026-10-25): **893 backend tests passed**,17
warnings,209.98s;26 focused checks passed,4.77s. Ruff check and format passed
for149 files. Regression warning details remain in the saved local test log.

One actual-library fixture query used3 explicit development queries and6
branches, returning10 active structured sources with verified raw anchors and
publication revision;5,199.803ms,0 LLM calls. This checks connectivity and
provenance, not model-generated expansion quality or a latency distribution.

Raw files are Git-ignored under `data/evaluation/m3_verified_20261024/`:
four actual mode reports/Trace, query plans with generation/validation lineage,
two replay reports, summary.json and interrupted-run records. Actual-library
check: `data/evaluation/m3_library_20261024/`. Initial service-down failure and
interrupted planning artifacts are retained in sibling directories.

The frontend's31 tests and build pass. The first local build exited with an
existing Next.js dependency-security upgrade notice; retry completed with the
notice. Dependencies were not changed by M3. This remains a maintenance item,
not a security-certification claim.

Reproduction and runtime contract: [multiquery_hybrid.md](../../docs/multiquery_hybrid.md).
Separate cumulative measurements: [metrics_history.md](metrics_history.md).
