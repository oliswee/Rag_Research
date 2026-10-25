# M4 Ranking and Context Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn validated M3 child candidates into locally reranked, bounded parent context with immutable citations and complete selection attribution.

**Architecture:** Keep `retrieve` as a baseline-compatible child retriever. The normal Hybrid API requests 30 candidates, scores them with a cached local FastEmbed cross-encoder, selects at most 10 children with explicit paper coverage, then packs validated original parent windows. Prompt context and citable child excerpts remain distinct; quote validation continues against original child text.

**Tech Stack:** Python, FastEmbed ONNX CPU, SQLite/SQLAlchemy, existing Qdrant/BGE, pytest, local Ollama/DeepEval.

**Spec:** User-approved FolioTrace v6, M4 P2/R5/C2/C3 and global privacy/fallback rules, in D:/Download_edge/FolioTrace_M1-M8_%E4%BC%98%E5%8C%96%E6%89%A7%E8%A1%8C%E6%B8%85%E5%8D%95_v6_DeepEval%E8%87%AA%E5%8A%A8%E8%AF%84%E6%B5%8B_2026-10-14.md.

## Global Constraints

- Original question is the cross-encoder input; expansions never replace it.
- Runtime model loading uses cached files only, CPU, one in-flight worker; timeout/busy/error retains RRF order. Explicit setup may download public model weights, never private text.
- Child citation text/IDs/spans stay unchanged. Parent context is background, never an entire parent disguised as a child citation.
- Defaults: 30 candidates (configurable 10–50), 10 selected (5–10), 15s rerank timeout, 8192 conservative token units and 2048 per-parent units. Units are UTF-8 bytes of the exact rendered prompt context, a conservative BPE token bound; no model-specific token accuracy claim.
- Whole child excerpts are admitted or dropped with a reason, never cut inside numbers/units/quotes. Parent windows use complete sentence/line boundaries and mapped raw spans.
- First reserve child coverage before spending spare budget on parent background. Explicit 2–5 papers get an evidence/unknown status for each; unsupported semantic conclusion coverage remains unknown until M6.
- Keep same DOI/version duplicates mapped; distinct versions remain separate. Mixed local/Web records use the same packing budget, without enabling Web routing before M5.
- Private text/errors/secrets stay out of Trace and Git. No paid calls, human Gold claim or merge.

## Review Focus

1. A timed-out native CPU job must not permit unbounded new workers or mutate a finished Trace.
2. A child with missing/corrupt parent retains validated child evidence, never foreign parent content.
3. Disjoint/cross-page numeric spans keep exact quotation anchors after windowing and deduplication.
4. One large parent cannot consume the quota needed for other explicitly requested papers.
5. Prompt IDs, actual rendered context, coverage and dropped reasons must agree even on answer failure/no LLM.

### Task 1: Bounded local reranking

**Files:** Create `backend/app/services/child_reranker.py`; modify config/trace_versions; test `backend/tests/test_services/test_child_reranker.py`.

**Interfaces:** Consumes validated source dictionaries and original question. Produces `rerank_children(question, sources, *, trace=None, scorer=None, timeout=None) -> list[dict]`, retaining every unique candidate; `select_children(sources, limit, paper_ids=None) -> list[dict]`. Local scorer exposes model/fingerprint and `score(question, texts)`.

- [ ] Write RED tests for original question, score order/stable ties, all candidates retained, invalid/NaN/partial scores, timeout/error/busy fallback and 2–5 paper coverage.
- [ ] Implement local cached `Xenova/ms-marco-MiniLM-L-6-v2`, CPU threads2, guarded worker lifetime and safe Trace metadata. No runtime download.
- [ ] Run targeted tests and config/version regressions; commit task.

### Task 2: Parent windows and bounded packing

**Files:** Create `backend/app/services/context_packing.py`; modify source_anchors; test `backend/tests/test_services/test_context_packing.py` and existing anchor tests.

**Interfaces:** `pack_context(sources, *, paper_ids=None, budget=None, parent_budget=None, expand=True, trace=None) -> PackedContext`; result holds `sources`, `blocks`, rendered `text`, `coverage`, exact `token_units`. `resolve_child` accepts explicit safe-parent-fallback mode for retrieval while retaining strict document/child integrity.

- [ ] Write RED tests for same-parent merge/all child mappings, missing/corrupt parent, large parent window, overlapping spans, cross-page decimal/unit preservation, child over budget, explicit 2–5 paper quotas, duplicate DOI/version and mixed origins/abstracts.
- [ ] Implement child-first admission followed by bounded parent expansion; map window slices to original spans. Validate parent/child matching; fall back to original child. Track every candidate/drop/paper status without source text in Trace.
- [ ] Run packing/anchors regressions; commit task.

### Task 3: Normal Hybrid integration and actual prompt attribution

**Files:** Modify source_query, query router, schemas and source retrieval integration tests; create `backend/app/services/evidence_context.py` orchestration if necessary.

**Interfaces:** `retrieve(..., candidate_limit=None)` preserves current top10 baseline by default, permits bounded M4 candidate pool up to50. `prepare_evidence` orchestrates rerank/select/pack. `answer_from_evidence(..., packed_context=None)` renders prepared text but validates quotes only in original source excerpts.

- [ ] Write RED source/API tests proving default Hybrid uses original question and30 candidates, explicit baselines remain flat, prompt IDs equal admitted sources, and uncitable parent-only quotes fail validation.
- [ ] Implement integration for initial and followup source queries; return per-paper context coverage, retain exact citations and baseline behavior. Trace records prompt only when sent to generation.
- [ ] Run relevant API/workflow tests and full backend suite; commit task.

### Task 4: Development evaluation and delivery

**Files:** Create `backend/evaluation/ranking_packing_benchmark.py`, `docs/ranking_packing.md`, `backend/evaluation/m4_development.md`; append `backend/evaluation/metrics_history.md`.

**Interfaces:** Frozen QASPER manifest/labels and saved M3 branch rankings supply up to30 candidates by replay of validated lists; no claim of new retrieval timing. Actual local cross-encoder scores and flat/parent packing run per case. Public QASPER legacy chunks have no native M2 parents: use explicit adjacent-chunk development parents with full provenance, and separately test actual M2 library parent anchors.

- [ ] RED tests for replay/evaluation separation, deterministic metric computation, input hashes and parent provenance.
- [ ] Run actual40-case development reranking/packing with model digest, CPU p50/p95, candidate/prompt evidence coverage and rank metrics. Run bounded local DeepEval Contextual Relevancy/Precision smoke with public reference; failures stay not_evaluated. Save raw data ignored, reports tracked.
- [ ] Final backend tests, Ruff, frontend gates if touched, privacy check; one fresh whole-branch reviewer and one RED/GREEN fix pass for important findings.
- [ ] Commit/push own branch, dependent Draft PR against M3, verify exact-head remote CI and notify user M4 engineering completion. Do not merge.

## Decisions and preflight

- Existing approved v6 supplies design; repeated continue authorizes inline execution and routine implementation choices. Keep current dedicated checkout with new feature branch, matching prior milestones.
- Lightweight MiniLM uses existing FastEmbed dependency (official Qdrant documentation and installed0.9.0 source checked); BGE embeddings unchanged.
- Task1 → Task3: ordered validated sources; Task2 → Task3: `PackedContext.text/sources` are authoritative prompt material; Task3 → Task4: baselines remain opt-in flat and public replay is explicitly distinguished from fresh retrieval.
- Coverage means inclusion in prompt; semantic support is unknown until answer validation, never inferred from selection alone.
