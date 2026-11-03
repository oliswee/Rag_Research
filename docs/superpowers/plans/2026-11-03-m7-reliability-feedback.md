# M7 reliability and local feedback implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, task by task. One fresh whole-branch review at the end; no per-task delegation.

**Goal:** Make the existing single-user local RAG recoverable, diagnosable and bounded without treating development fixtures as quality certification.

**Architecture:** Retain immutable SQLite sources and M2 publication certificates. Add scoped reconciliation and durable vector-deletion work, reuse validated embeddings by model/text hash, and keep request/run/feedback metadata in local SQLite. Source-status information and feedback are separate from temporary Web originals. A request budget encloses existing bounded stages; no autonomous tool loop or unverified streaming.

**Tech stack:** Existing FastAPI/SQLAlchemy/Alembic/SQLite FTS5/Qdrant, BGE FastEmbed, local Ollama4B, Next/React/Vitest. No new paid account, model credential or runtime dependency.

**Spec:** User-adopted v6 checklist at `D:/Download_edge/FolioTrace_M1-M8_%E4%BC%98%E5%8C%96%E6%89%A7%E8%A1%8C%E6%B8%85%E5%8D%95_v6_DeepEval%E8%87%AA%E5%8A%A8%E8%AF%84%E6%B5%8B_2026-10-14.md`, M7 K4/K5/K6/S2/F2/F3/L1. Issue59, base merged main `f98d2559373cd9d85d5ec3315c842cc54564f335`, branch `feat/59-m7-reliability-feedback`.

## Global constraints and rulings

- User authorized only M1-M6 integration (PR48/50/52/54/56/58), now merged with final main CI41592777776 success and tree identical to M6. Legacy PR14/20/22/23 untouched. M7/M8 work is separately requested; no authorization to merge these future branches is inferred.
- v6 is the approved design; repeated continue requests and prior waiver authorize routine choices without another design/worktree approval. Use this existing dedicated checkout and feature branch. Preserve all original/citation history and configured secrets.
- SQLite authoritative; embedding reuse is valid only for the same model/dimensions and unchanged normalized embedding input with integrity checks. Changed segmentation may require a whole-paper rebuild; no promise of minimal arbitrary rechunking.
- Reconciliation and deletion are limited to this configured dataset/collection. Never delete another dataset's points, entire collections, immutable source history or user files. Recheck ownership before deletion. Repair is serial/idempotent and records partial failures.
- A paper deletion can leave durable vector cleanup work; source rows disappear only through the existing explicit user delete action. Tests and hardware diagnostics use separate synthetic datasets.
- Unknown retraction is not unretracted. Positive status needs an explicit verifiable provider event and provenance; source versions do not imply independent studies or scientific authority.
- Feedback stores only IDs/DOI/safe URLs/granularity/reasons/config/time, not arbitrary external full text or secrets. Synthetic regression guards are development fixtures, not automatic reproduction/correction of a real user's semantic judgement or training labels.
- Cost ceilings are reserved token/call budgets with explicitly unknown currency where provider billing is unmeasured. Optional monetary caps require supplied unit prices; local monetary cost does not mean zero CPU/time/energy.
- M8 requires first independent human annotation/PDF review. Do all authorized engineering and evaluation work, but never upgrade QASPER development labels or Codex decisions into human Gold. Keep acceptance provenance explicit.

## Review focus

1. Repeated import after an interrupted publication or corrupted memo must recover rather than reuse a stale ready certificate (Task1).
2. A deleted paper or Qdrant ID collision must not erase a different library's point or lose the retry record (Task1).
3. Retraction/version metadata with a missing DOI, wrong update direction, private URL or no explicit event stays unknown (Task2).
4. Feedback must refer to this stored run's original source identity; no fabricated Web record, untrusted raw text, unrelated paper or server-secret output (Tasks3/4).
5. Timeout/cancellation and concurrent requests must release their own resource clients and reset budget/privacy ContextVars; no extra model/network calls or unverified findings (Task4).

## Task1 — Incremental publication and scoped check/repair

**Files:** create `app/models/maintenance.py`, `app/services/incremental_index.py`, `app/services/index_maintenance.py`, `evaluation/check_index.py`; add an additive Alembic migration and model exports; integrate ingestion/discovery/reindex/delete boundaries. Existing `versioned_vectors.publish_generation` remains the full verification publisher.

**Interfaces:** `sync_generation(db,build,*,embedder=None,vector_index=None,force=False)->dict`; model/text-hash memo wrapper implementing `generate_text_batch`; `check_index(db,index)->dict`; `repair_index(db,index,*,embedder=None)->dict`; durable `queue_source_deletion` and `retry_source_deletions`. Check actual FTS postings, not external-content SELECT row counts. Verify payload/vector hashes against the stored manifest; missing/corrupt vectors lose readiness before repair. Retired/orphan vectors are removable within own dataset while historical SQLite citations stay resolvable.

- [x] RED five lifecycle scenarios: new, duplicate same content, changed content/stable untouched unit, explicit deletion and failed-publication retry; memo tamper/model/dimension changes; zero new embedding work for an unchanged valid generation.
- [x] RED actual Qdrant local-client/SQLite FTS fixtures: missing postings, missing/mismatched vector, orphan/retired vector, foreign scope collision, partial deletion failure and second repair idempotency.
- [x] Implement validated memo reuse, exact existing-source ready check and safe full-paper fallback. Integrate normal ingestion/import/reindex so the feature is usable, not only a benchmark helper.
- [x] Add serial bounded CLI dry check and explicit repair; sanitize errors and retain durable cleanup jobs across paper deletion/restart. No unverified vector can become a citation.
- [x] GREEN focused publication/reindex/delete/migration suites; commit and ledger evidence.

## Task2 — Version/retraction provenance

**Files:** add lifecycle metadata model/schema/service/router and paper-detail response/UI; use existing DOI/arXiv normalizers and M6 comparison checks.

**Interfaces:** `publication_status(db,paper_id)->dict` defaults unknown; explicit `refresh_publication_status` uses a bounded official metadata request, never query-triggered network access. Parse verified relation/retraction events separately from titles. Store checked time/source URL/related DOI or arXiv/version and raw-event digest, not full external documents.

- [x] RED preprint/formal-version relation, wrong DOI/update direction, population mismatch, comparable conflict and unknown retraction fixtures; unsafe URL/missing identity/provider failure remains unknown.
- [x] Implement explicit source provenance, metadata retrieval deadline/size/attempt limit and unknown-safe UI; no similarity authority ranking, automatic import or background query network fetch.
- [x] GREEN source/compare/metadata regression; commit.

## Task3 — Local answer feedback and synthetic regression roundtrips

**Files:** create `app/models/feedback.py`, schemas/services/router and additive migration; record bounded minimum run manifests from QA/compare; add an English feedback control near returned evidence/cells and API/types/tests.

**Interfaces:** `record_run_manifest(db,trace,response)` stores run/config/scope/minimal canonical source refs; `POST /api/feedback` accepts known run/source and one of six reasons; `create_feedback_regression` / `run_feedback_regression` / fix-reference record form a local roundtrip. Wrong-paper, unsupported-citation and web-result-irrelevant templates are labelled synthetic guards. Real semantic feedback without labels stays pending review.

- [x] RED local/Web feedback identity, unknown run/foreign source, bounded storage, no raw external full text/keys, all six reason choices and deletion-retained diagnostic identity.
- [x] RED three synthetic feedback→regression→automatic guard result→fix record roundtrips; no arbitrary code execution, HTTP fetching, training or external issue posting.
- [x] Implement SQLite persistence/API and accessible English UI acknowledgement/errors; preserve scope and source granularity.
- [x] GREEN backend/UI focused checks; commit.

## Task4 — Request budgets, stage accounting and injection regression

**Files:** add `app/services/request_budget.py`, config/Trace changes and bounded QA/compare wrappers; integrate actual LLM attempts and academic HTTP accounting; add scientific-output credential guard and six end-to-end malicious/legitimate pairs.

**Interfaces:** request-local deadline/call/input-output reservation/external-attempt counters via ContextVar; existing rewrite/retrieval/rerank/packing/Web/generation/verification remain sub-budgets. Exceeding a budget returns an explicit safe diagnostic and preserves already admitted source evidence where available. Every client closes and context resets on cancellation/error. Stage summaries include p50/p95 only from observed samples, separate reservations from usage and generation from Judge costs.

- [ ] RED timeout/call/token/network/currency-unknown limits, retries, concurrent isolation, context reset, missing services and partial failure; no post-exhaustion call or unchecked fact.
- [ ] RED at least six malicious title/abstract/PDF content cases plus legitimate controls: rule override, secret request, invented citation, cross-paper scope, unauthorised write instruction and unsafe link. Use disposable canaries/fixtures, never real keys.
- [ ] Implement bounded integrations and sanitization without granting tools to document text or relaxing M5 privacy/M6 original-citation checks.
- [ ] GREEN targeted M3-M6/API/Trace/security suites; commit.

## Task5 — Actual local diagnostics and delivery

**Files:** `evaluation/reliability_benchmark.py`, `evaluation/m7_development.md`, separate `metrics_history.md` section12, `docs/reliability_maintenance.md`; ignored raw artifacts/30-request Trace summaries. Read frontend AGENTS/installed Next docs before UI edits.

- [ ] Run >=30 real local-hardware requests with actual BGE, Qdrant engine, reranker/Ollama when used; isolate synthetic corpus/dataset, record warm/cold/mode/failures/p50/p95/model calls/tokens/network attempts. No substituted fake latency claims.
- [ ] Measure 429/timeout/unavailable/partial-service/budget exhaustion and all adversarial/feedback fixtures separately; report injected boundaries explicitly. Back up local SQLite before migration/maintenance; never rebuild the user's library just to obtain a benchmark.
- [ ] One fresh whole-branch reviewer after implementation; reproduce/fix important findings once. Full appropriate backend/frontend/lint/build/migration/privacy checks and source fingerprint evidence.
- [ ] Push own branch, draft PR targeting merged main, exact-head CI, issue59/root plan/ledger/metrics update. Notify M7 progress and continue directly to separately planned M8; no intermediate approval prompt.

## M8 handoff

Freeze the verified M7 implementation/config/corpus for M8. Use a separate plan/branch and retain untouched final test labels. Prepare annotation guidelines/review materials and explicit unresolved human-audit gates, then run all reproducible automatic comparisons/report/demo work. Do not quietly call delegated AI review independent human annotation. Final user notification should distinguish engineering completion from any unfulfilled formal human-quality acceptance.
