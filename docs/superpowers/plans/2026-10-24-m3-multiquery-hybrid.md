# M3 Multi-query Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make bounded, constraint-preserving multi-query BM25/Dense retrieval the normal source-query path.

**Architecture:** Keep Q0 verbatim; one bounded local rewrite call may provide independent Q1/Q2. Validate and deduplicate before launching at most six branches. Serialize access to the request's AsyncSession, overlap BM25 with batched query embeddings and threaded Qdrant reads, then validate candidates against authoritative SQLite before single-layer RRF and source resolution.

**Tech Stack:** Existing Python/FastAPI, asyncio, SQLite/FTS5, Qdrant, BGE FastEmbed, Ollama. No new dependencies.

**Spec:** User-approved FolioTrace v6, M3 Q2/Q3/R3/R4/R6; issue #51. Depends on M2 draft #50, commit a1f5772. The user already approved this architecture and continued execution; implementation proceeds inline without another approval prompt.

## Global Constraints

- At most3 deduplicated queries, Q0 always present; exact identifiers and uncertain negation/quoted constraints may use Q0 only with reason.
- Paper/year/provider/local_only remain request-owned scope; generated text cannot change them.
- Missing/invalid/timed-out rewriting returns Q0. Defaults local BGE/Ollama; cloud calls remain opt-in.
- At most6 ranked lists, per-list K30, final K10, RRF k60, concurrency3, branch timeout15s and rewrite timeout20s; internal validated configuration may lower these values.
- SQLite active generations and ready dense generations remain mandatory; legacy points only for papers without a structured active generation.
- No concurrent access to one AsyncSession. No private query/source text or credentials in persisted Trace.
- Preserve old citations. No production vector collection reset. Draft delivery allowed; merge needs separate authorization.
- QASPER is public development data, not M8 Gold. Append measurements/limits to metrics_history.md.

## Review Focus

1. A slow threaded Qdrant request cannot close its client while still running or mutate Trace after timeout (task2 timeout regression).
2. Duplicate variants or duplicate IDs cannot amplify RRF; skipped/failed branches remain visible (tasks1/2).
3. Rewriting cannot erase research objects, exact terms, numbers, negative constraints or paper scope (task1 table-driven tests).
4. A stale/foreign/out-of-scope vector cannot influence fusion before SQLite revalidation (task2 real SQLite test).
5. Default first query and follow-up use the source path for uploaded as well as imported chunks; no hidden Dense-only default (task3 API tests).

### Task1: Bounded query plans

**Files:** create app/workflows/query_expansion.py and tests/test_workflows/test_query_expansion.py; extend retrieval_trace.py/config.py/trace_versions.py.

**Interfaces:** frozen RetrievalQuery(query_id, query_type, text, parent_question); QueryPlan(queries,status,reason,rejected_count). async expand_queries(question,history,enabled=True,trace=None)->QueryPlan. Existing rewrite_for_retrieval stays compatible for legacy callers.

- [x] Write failing tests for independent Q0/Q1/Q2, semantic/token dedup, eight constraint/follow-up cases, unavailable/timeout/429/malformed failure and private Trace hashes.
- [x] Run uv run pytest tests/test_workflows/test_query_expansion.py -q; observe missing-module/behavior failures.
- [x] Implement one bounded JSON call; retain all original content tokens/numbers, reject new negation/named identifiers; conservative Q0-only on explicit exclusions/exact identifiers/quoted terms. History supplies context only to variants.
- [x] Run the same tests, then legacy rewrite/Trace tests; record RED/GREEN in progress.md and commit.

### Task2: Concurrent branches and authoritative fusion

**Files:** create app/services/retrieval_branches.py and tests/test_services/test_retrieval_branches.py; modify source_query.py.

**Interfaces:** async collect_rankings(db,queries,active_versions,scope,mode,trace,vector_index,per_query_k,timeout,concurrency)->list[BranchResult]; BranchResult carries query_id/retriever/ranked IDs/status/duration. retrieve adds optional queries, per_query_k, rrf_k parameters and keeps its (evidence,actual_mode) return contract.

- [x] Write failing six-list/event-barrier concurrency, duplicate-ID, per-branch timeout/failure, no concurrent DB, stale-vector and immutable scope tests.
- [x] Run the targeted files and observe failures.
- [x] Batch embeddings once per request; bound branches with semaphore; serialize lexical session access; execute synchronous vector reads in threads with client lifetime owned there. Store results after awaited completion, never inside abandoned workers.
- [x] Revalidate dense IDs and all scope constraints in SQLite before RRF; fuse only valid independent lists and re-resolve active sources. Record query_id/retriever/rank/fused_rank, timeouts and actual degradation; retain single-query ranking aliases.
- [x] Run branch/source-index/reindex/Trace regressions, record results and commit.

### Task3: Defaults and product routing

**Files:** modify schemas/query.py, routers/query.py, frontend query page/tests and source-query API tests.

- [x] Failing tests: omitted API mode invokes BM25+Dense+RRF; Q0 retained through rewrite; follow-up source routing; uploaded-only corpus; partial/total failure transparent; frontend no mode picker and default hybrid request.
- [x] Default Hybrid/rewrite enabled for normal source requests; expose query_count and degradation reason, preserve original question for answering. Keep explicit modes for internal baseline use; legacy pattern path only when no source chunks exist.
- [x] Remove normal UI's technical mode selector; keep source/year/paper scope. Read frontend AGENTS/local Next guide before editing.
- [x] Run relevant backend/API/frontend tests and frontend build; record results and commit.

### Task4: Development benchmark, review and draft delivery

**Files:** extend evaluation/isolated_corpus.py or add evaluation/multiquery_benchmark.py, its tests; docs/multiquery_hybrid.md; metrics_history.md.

- [x] Reuse frozen40-case QASPER corpus and isolated vector scope. Cache actual local generated plans with query/config hashes and rejection reasons for replay; no fabricated variants or labels.
- [x] Save dense/BM25/single-Hybrid/multi-Hybrid per-case Recall/MRR/nDCG, branch timing/count/errors, rewrite calls/tokens; compare a small K/RRF sweep on development groups only. Optional DeepEval local Judge failure remains not_evaluated.
- [x] Run complete backend suite/lint and frontend build; safe staged secret/privacy check. One fresh scoped whole-branch reviewer; one RED/GREEN fix pass for material findings, then final suite.
- [ ] Commit/push dependent draft PR based on M2, verify exact-head CI, update issue and persistent plan. Report completion only with evidence; no merge or M8 quality claim.

## Execution evidence (2026-10-25)

- Task1 commit8683e72:24 expansion tests RED;38 expansion/legacy rewrite/Trace checks GREEN.
- Task2 commitc4ae5b4:6-list overlap/session safety and source-version/scope/RRF regressions.
- Task3 commite89444f:uploaded-only and followup default API RED/GREEN;frontend31 tests and production build pass. Existing dependency-upgrade notice retained as maintenance.
- Task4 actual40-case QASPER comparison and two local DeepEval smoke scores saved; no default-parameter quality gain. Report m3_development.md and separate metrics history disclose results and limits.
- Ruling: K/RRF development sweeps replay saved independent ranked lists, with no retrieval latency claim; avoids repeating identical local embeddings. Actual core four-mode runs retained separately.
- One fresh whole-branch reviewer found5 P2 issues; root reproduced7 failures plus numeric-role regression and fixed them. Borrowed-client lifetime has a regression. No second review loop.
- Later full run891passed/1failed exposed asynchronous SQLite interrupt racing rollback; removed that timer, retaining native progress/busy deadlines and a CPU-read/rollback regression.
- Final backend893passed/17warnings/209.98s;26 targeted checks passed4.77s;Ruff149 files check/format pass. Private stage check19 files passed. Final push/PR/exact-head CI remain below until confirmed; no merge.
