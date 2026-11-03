# M7 reliability development evidence

Date: 2026-11-03, Australia/Sydney. Issue59; base main
`f98d2559373cd9d85d5ec3315c842cc54564f335`. M1–M6 merge commits are on main;
its final integration CI41592777776 passed. Old PR14/20/22/23 were untouched.

This report separates engineering checks, injected boundary failures and actual
local hardware measurements. None establishes independent human Gold quality,
real Web answer gains, service capacity or a production SLA.

Incremental publication checks cover new/duplicate/changed/deleted/interrupted
documents, model and memo corruption, invalid physical points, legacy migration,
foreign dataset collisions and idempotent repair. The focused Task1 suite passed
78 tests in 32.95s. Actual FTS postings use tokenizer-instance comparison;
external-content row counts alone cannot establish index consistency.

Publication metadata checks cover matching DOI/event direction, explicit dated
notices, preprint/formal relations, unknown/conflicting/reinstated states, stale
refresh and unknown-safe UI. Task2 backend43 and frontend9 focused tests passed.
These are synthetic provider records; no live Crossref service-quality claim.
Crossref's `updates:{doi}` filter returns notice records, whose `update-to` DOI
identifies the updated work. See the primary
[API format](https://github.com/CrossRef/rest-api-doc/blob/master/api_format.md)
and [REST documentation](https://github.com/CrossRef/rest-api-doc/blob/master/README.md).

Feedback checks cover six reasons, known run/source identity, minimal persistence,
paper deletion and three explicitly synthetic feedback→guard→fix-reference
roundtrips. Task3 backend24 and frontend7 focused tests passed. The Web guard
rejects a metadata-only fixture; it does not reproduce real topical relevance.
User-supplied fix references are labels, not verified code commits or evidence
that the user's original semantic complaint has been corrected.

Request and privacy checks include atomic reservations, retry charging, unknown
and explicit-price ceilings, total timeout/context reset, concurrent isolation,
six malicious title/abstract/native-PDF cases with six legitimate controls,
secret-bearing originals and an approving semantic Judge. An alphabetic
disposable credential initially escaped the semantic boundary; deterministic
admission and final-response guards now reject it. Task4 focused120 tests passed
in 7.79s, with one existing AsyncMock warning. Actual provider-client HTTP429,
timeout, connection-unavailable, exhausted-verification-budget and global
deadline checks are injected transport tests, separate from real latency.

Runtime preparation exposed a missing temporary BGE cache. The model was
prepared into persistent `data/models/embedding`; Local-only remains offline.
Windows cache symlink fallback was observed and retained in the ignored log.
Ollama0.40.13 uses cached qwen3:4b-instruct and RTX4060 Laptop Vulkan; the default
provider/model were retained. Cloud paid model calls are not required.

The local SQLite database was backed up before migration from b72c1a to e95c2a.
Both existing paper rows remain; `PRAGMA foreign_key_check` returned no violations.
The runtime benchmark uses separate in-memory SQLite/Qdrant, never this library.

Actual 30-request measurements, final verification and review outcomes are
appended below after the runs finish. Raw artifacts live under ignored
`data/evaluation/m7_runtime_20261103/`; reproduction instructions are in
[`docs/reliability_maintenance.md`](../../docs/reliability_maintenance.md).

## Actual local runtime results

30 serial requests over six synthetic documents (three native one-page PDFs and three abstracts), ten each BM25/Dense/Hybrid. All returned supported answers with the expected synthetic quote; 60 real generation/verification model calls, 27,575 input and 8,275 output tokens. External attempts0, fake PDF pages0, observed request failures0. This is a deliberately simple runtime probe, not an answer-accuracy estimate.

| Stage | Observations | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| observed_wall_ms | 30 | 7753.766 | 9167.209 |
| total_ms | 30 | 7744.707 | 9153.761 |
| Q0.bm25 | 20 | 8.638 | 11.316 |
| Q0.dense | 20 | 383.072 | 443.551 |
| rerank | 10 | 11.176 | 241.682 |
| packing | 30 | 0.249 | 0.558 |
| generation | 30 | 4654.595 | 5567.958 |
| verification | 30 | 960.081 | 1119.159 |

First request wall16501.730ms; remaining29 wall p507736.223ms/p958939.62ms. BGE indexing warmed its cache before requests; first request is not an all-components cold benchmark. Total trace elapsed ends before optional feedback persistence; wall time includes the returned response. Inherited selected/selected_children markers have unmeasured zero placeholders and are excluded from latency summaries. Some backend/frontend checks overlapped the final requests, so hardware numbers include ordinary local contention. Query rewriting/Web were disabled and not assigned fake timings.

Observed configuration: `sha256:f2d647a6a3278fefa4fa64cd9931bbd3f7d124c93b700bb1eb79434c11fd1cbe`; pipeline `sha256:8c42b83ecbcc10e42f0347dc36e5eec4c7ac88a151a259e2c85bf210b60d7db3`. Qdrant is the actual local engine with an explicit in-memory storage adapter; Docker/server transport was not measured. No claim about larger real corpora or concurrent users follows.

Final branch review found one P2: a failed retired-generation cleanup could delete deterministic IDs reactivated by later reimport. Reproduced RED; cleanup now rechecks current SQLite active membership and retires obsolete work. Lifecycle/maintenance integration16 tests passed in1.98s; independent review subset53 passed. No other Important/Critical finding. The prior full suite was interrupted to restart against this fix; no partial full-suite success is claimed.

The30-request runtime predates this maintenance-only fix; its recorded config/pipeline hashes are retained above. Final-source config `sha256:0ad1be38f36c945ca63cd8063272b5f5aa75960c7e76535a3e6cf53701d95ad0`, pipeline `sha256:bf06a383141b44ed3e9bf753d850a72d19bd09c6430f0ade02eaa97163c22853` differs, so those hardware timings are not asserted to come from the final source hash. Request/retrieval/generation paths were unchanged by the cleanup fix.
