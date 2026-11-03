# M1 Trace and DeepEval Implementation Plan

> **Execution:** Implement natively, test first, and review the complete branch before PR. This plan implements the user-supplied FolioTrace M1–M8 v6 checklist, M1 only.

**Goal:** Make local retrieval decisions and automated evaluation reproducible without treating judge output or draft labels as human Gold.

**Architecture:** A small trace recorder stores versioned JSONL in the ignored local data directory. The query route and retriever add IDs, stages, fallbacks, timings and support outcome. An optional DeepEval adapter consumes sanitized trace records, configures its judge explicitly, and writes separate evaluation results. A fixed QASPER config drives the public-label retrieval baseline.

**Tech stack:** Python 3.12, FastAPI, SQLite FTS5, Qdrant, pytest, optional DeepEval.

**Spec:** User-provided FolioTrace v6 checklist (2026-10-14), sections B1 and M1.

## Global constraints

- M1 has no human Gold or PDF quality audit.
- Never mark a missing or failed judge score as zero or one.
- Cloud calls require an explicit judge selection and usable provider credentials.
- Keep full source text, keys and exception messages out of trace output.
- Keep 40 local draft questions unverified.

## Review focus

- A dense failure must show BM25 fallback and still produce a trace.
- A judge timeout or HTTP 429 must leave retrieval metrics intact and mark judge results not_evaluated.
- An unanswerable QASPER case must not be scored as evidence Recall@K.
- The source version and paper scope must be present in public benchmark metadata.
- Trace files must not include any API key or a whole evidence excerpt.

### Task 1: Trace schema and recorder

Create `app/services/retrieval_trace.py` with a typed recorder and append-only JSONL writer. Test that it saves stage IDs/ranks/timings, fallback reason classes and final support status; rejects arbitrary text payloads and secret-like fields. Default path is under ignored `backend/data/`.

### Task 2: Route and retriever instrumentation

Pass one recorder from the query API to `retrieve`. Capture original and effective query, scope, BM25/dense candidate IDs and ranks, merged IDs, prompt evidence IDs, generation outcome and stage durations. Wrap route completion so failures still flush a safe trace. Keep response schema backward compatible, adding only a run ID if useful. Exercise real SQLite retrieval and a simulated dense failure.

### Task 3: Optional DeepEval adapter

Create `evaluation/deepeval_adapter.py` with trace-to-`LLMTestCase` mapping, metric selection by reference availability, explicit judge provider/model configuration, error handling and JSONL result writing. Keep the package optional and cloud evaluation off by default. Test answerable, unanswerable, missing key and judge failure without a billable call.

### Task 4: QASPER baseline and documentation

Add `evaluation/qasper_dev_config.yaml` and a reproducible runner for 40 published-label cases. Run BM25, dense and single-query hybrid when keys/Qdrant are available; record actual fallback and mark unavailable modes as unavailable. Store derived results under ignored `backend/data/evaluation/`; document source hashes, model versions, limitations, resource budgets and unverified local draft status in `baseline_v1.md`.

### Task 5: Verification and delivery

Run focused tests, full backend pytest, Ruff check and format check, inspect the diff and tracked files for secrets. Commit a scoped M1 change, push and open a draft PR. Review it. Do not merge without separate authorization.

### Task 6: Runtime version records (2026-10-23 continuation)

Replace runtime `unversioned` placeholders with deterministic SQLite corpus and
allowlisted configuration fingerprints. Hash actual chunk/pattern content and
citation/filter metadata in a stable row order; persist only digests. Include
pipeline source and dependency versions in the configuration manifest. Secrets,
emails, database paths and service URLs are excluded. Stable input must produce
stable versions across fresh recorder instances; edits, deletions and model
changes must change the appropriate version. Explicitly labelled public corpus
versions remain intact in the evaluation runner.

Index records must distinguish a logical version derived from corpus/config and
collection scope from a verified physical vector revision. Do not invent a
physical Qdrant revision or claim that a configuration digest proves index
consistency. Preserve existing retrieval behavior and collect snapshot timing.
Validate both query endpoints, empty libraries, version changes and safe failures.
Run targeted tests RED/GREEN, full backend tests and Ruff before committing and
updating the existing draft PR. Parent/Web fields stay pending their milestones.

Ruling: compute the runtime corpus fingerprint from persisted SQLite rows on each
request rather than a process-local counter. This detects edits made outside the
app and survives restarts; it costs a linear scan on the current small library,
whose time is recorded. A cached transactional revision can replace it under M7
only if all mutation paths are covered.
