# M6 Grounded QA and Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return only verified atomic findings with resolvable local/Web citations, independent paper coverage and explicit scientific comparability states.

**Architecture:** A request-local immutable evidence registry re-resolves local SQLite children and seals Web evidence at the M5 boundary. A common bounded generator validates original quotes and scientific quantities before one batched atomic semantic verification. Comparison runs each selected paper's local pipeline before scoped M5 fallback, then generates/verifies four dimensions per paper and compares only independently supported study characteristics. Existing text/evidence response fields remain; new claims/citations/status fields provide authoritative structured results.

**Tech Stack:** Existing FastAPI/Pydantic/SQLAlchemy, BGE/ONNX/Ollama, httpx/PyMuPDF, pytest, React/Next. No paid model signup or new runtime dependency.

**Spec:** Approved user v6 M6 Q4/G3/G4/G5/G6/G7 and exits, `D:/Download_edge/FolioTrace_M1-M8_%E4%BC%98%E5%8C%96%E6%89%A7%E8%A1%8C%E6%B8%85%E5%8D%95_v6_DeepEval%E8%87%AA%E5%8A%A8%E8%AF%84%E6%B5%8B_2026-10-14.md`, lines224–252. Existing user preferences authorize inline implementation, development audits by Codex, own feature branches/push/draft PR/issue updates and separate metrics; no merge.

## Global Constraints

- Preserve M5 local_only/only_uploaded/paper/year/provider/version bounds before inference or external I/O. Local BGE and loopback Ollama remain defaults. No automatic corpus/vector import, no inferred PDF pages, no factual streaming before verification.
- Compare2–5 distinct selected local identities, no title-based identity substitution. All selected papers receive independent local Hybrid→rerank→parent runs before any Web lookup. Missing local/fulltext remains explicit; identity-scoped Web can supplement only with legal matching evidence.
- One comparison-wide M5 Web budget:2 providers/8HTTP attempts/45s,1PDF/3MiB/50pages/2same-host redirects,10s fetch/15s parser. One correction per paper maximum. Budget exhaustion preserves partial verified local answers.
- Source schema: claims[],citations[],paper_id,doi,source_origin local|web,evidence_type full_text|abstract_only|metadata_only,source_url,raw_quote/span,pdf_page int|null,version,support_status,uncertainty. No raw quotes/claims in persisted Trace. Local sources recheck SQLite; Web references require the current request's immutable sealed snapshots, never a global cache.
- Metadata may display bibliography but cannot support a finding. Abstracts cannot become full-text facts. Parent background cannot support an unrelated child citation. Keep same-DOI version/origin provenance; duplicate same-study sources are not independent studies.
- Max20 atomic proposals/request; comparison max4 findings/paper. Generation+semantic verification bounded to10 calls,8192 output tokens,131072 serialized input bytes,30s/call and180s model-stage deadline, lower configurable limits allowed. No hidden model retries (max_retries=1); close owned clients. Compare uses800generation+600verification output tokens/paper; QA1200+800. Keep stage budgets independent of DeepEval.
- Atomic support statuses: supported/contradicted/insufficient_evidence/unsupported/unknown. Deterministic failures suppress factual text even if semantic judge says supported. Numeric/unit/OR-vs-RR/direction/sample-size and causal-escalation checks are conservative disclosed heuristics, not a proof of scientific truth.
- Study characteristics population,study_design,outcome,exposure,effect_measure,adjustment,direction,year,version must retain supporting citations/quotes. Missing characteristics produce insufficient_evidence; distinct measures/design/population/outcomes/versions produce not_comparable; explicit comparable matched characteristics allow agreement/conflict. No two-independent-study claim from same DOI/identity.
- M1–M7 use development fixtures/public labels/local DeepEval; no new human Gold or claims of measured real Web gains. Record actual generated-answer Faithfulness and Answer Relevancy separately from deterministic checks, including costs/errors/missing data.

## Review Focus

1. A same-ID mutated excerpt/span/page/version or retired/archived local paper must fail closed at generation, not inherit retrieval trust.
2. One missing selected paper must remain unknown even when another paper has strong evidence, including Web budget exhaustion and same-title/version collisions.
3. Model-provided scientific characteristics can be fabricated separately from an otherwise valid claim; compare only quote-validated and semantically checked fields.
4. Parser/source data can contain instructions, unsupported numbers, swapped denominators, units and negations; final schema/semantic status cannot override deterministic checks.
5. Generation failure, timeout, cancellation or invalid JSON must close clients and preserve safe excerpts/unknown states without leaking private text in Trace or making more model calls than budgeted.

### Task 1: Unified immutable evidence and citation schema

**Files:** Create `backend/app/schemas/evidence.py`, `backend/app/services/evidence_registry.py`, `backend/tests/test_services/test_evidence_registry.py`; extend Web pool/routing result with sealed snapshots.

**Interfaces:** Frozen `EvidenceSnapshot` captures one original excerpt/identity/spans/pages/version/source URL; `EvidenceRegistry` maps source IDs to validated snapshots; `build_registry(db,sources,*,web_snapshots=(),paper_ids=None,local_only=False)`. `Citation`, `AtomicClaim`, `GroundedAnswer`, `StudyCharacteristics` are Pydantic response contracts. `seal_web_sources(sources)` only at the trusted temporary-pool boundary, not from caller-provided model data.

- [ ] RED local original/corrupt/legacy/null-page/archived and Web current-snapshot/tamper/metadata/version/unknown-ID/cross-paper fixtures.
- [ ] Implement SQLite authoritative re-resolution and immutable normalized registry; cap50 sources, preserve all original source origins/versions, canonical local citation endpoints and safe public Web URLs. No DB writes or persistent Web cache.
- [ ] GREEN registry + source anchor/Web regressions; commit and ledger evidence.

### Task 2: Bounded atomic generation, scientific checks and comparability

**Files:** Create `backend/app/services/claim_verification.py`, `backend/app/services/grounded_generation.py`, `backend/app/services/study_comparison.py` and focused tests; extend config/trace allowlist.

**Interfaces:** `VerificationBudget.call(client,system,payload,*,max_tokens,trace)`; `check_proposal(proposal,registry,*,paper_id=None) -> AtomicClaim`; `generate_grounded(question,registry,packed,trace,*,dimensions=None,budget=None,client=None,verification_mode='atomic_batch') -> GroundedAnswer`; `compare_studies(left,right) -> ComparisonAssessment`. Characteristics are quoted evidence fields, not unverified model labels.

- [ ] RED supported/contradicted/insufficient verdicts, wrong/missing citations, partial quotes/parent-only spans,95→99,OR→RR,ms→s,direction flip,n/sample-size,association→causal,abstract→fulltext and metadata counterexamples.
- [ ] Implement deterministic checks + one batched semantic call; reject multi-sentence non-atomic proposals conservatively; suppress rejected finding text, preserve reason codes/uncertainty and evidence-only/abstain fallback. Close owned client with cancellation-safe cleanup. Trace only IDs/counts/status/reasons/budgets.
- [ ] RED/GREEN comparable agreement/conflict,OR/RR mismatch,preprint/native-version mismatch,missing adjustment and same-study independence cases. Test strict call/output/input/time limits and no extra calls on failure.
- [ ] GREEN focused suites; commit.

### Task 3: Independent paper retrieval and verified comparison workflow

**Files:** Create `backend/app/workflows/comparison.py`, `backend/app/schemas/research.py`; update research router and small M5 budget/initial-local reuse interfaces; test `backend/tests/test_workflows/test_comparison.py` and API comparison regressions.

**Interfaces:** `run_comparison(db,request,trace,*,local_runner=None,searcher=None,generator=None) -> ComparisonResponse`; M5 `run_local_web(...,initial_local=None)` reuses already completed local stage; `AcademicSearch(...,budget=None)` and Web pool share one `WebBudget` including provider/PDF allocation. Matrix rows retain legacy paper_id/text/evidence and add claims/citations/support/uncertainty/comparisons/coverage/run_id.

- [ ] RED2/3/5paper all-local-before-Web, one missing/scoped external match, no legal external text, duplicate title/version, same DOI local+Web, shared budget exhaustion, upload/local privacy and model timeout fixtures.
- [ ] Implement one deterministic per-paper query covering four dimensions, rewrite off; independent scope/coverage and at most one local correction; generate/verify each column within shared model budget. Missing paper identity404; existing paper without usable evidenceunknown. Route constraints/local endpoint checks apply before DB/inference.
- [ ] Keep reference compatibility fields, emit explicit comparison assessments and atomic source provenance, safe partial failures and sanitized Trace; GREEN integration/privacy tests, commit.

### Task 4: Verified QA schema and necessary scope clarification

**Files:** Integrate query router/schema with `answer_grounded(...)` using shared registry/generator; create `backend/app/services/query_clarification.py`; update obsolete API boundary mocks and add QA regressions. Keep historical low-level source-query helpers for explicit legacy/evaluation callers; production QA/compare share the new verifier.

**Interfaces:** `answer_grounded(db,question,packed,web_snapshots,trace,*,paper_ids=None,local_only=False) -> GroundedAnswer`; `clarify_scope(question,papers,history,policy) -> Clarification|None`. Existing response/returned evidence preserved; add authoritative claims/citations/support_status/uncertainty/clarification/discovery_url.

- [ ] RED API actual-source recheck, mixed-origin claims, unknown pages/versions, lost paper coverage, absent model/no evidence/failure, no unchecked factual streaming, followup privacy inheritance and exact prompt IDs after registry pruning.
- [ ] Implement one necessary clarification for unbound pronouns/duplicate titles/undefined latest time scope or clearly stated bounded assumption; coverage shortage links existing Discover page without imports. Selected IDs override title ambiguity. No extra agent or clarification model.
- [ ] GREEN QA/M3–M5 source/session regressions; commit.

### Task 5: Matrix/source UI, generated-answer diagnostics and delivery

**Files:** Update comparison/query types/API/UI and relevant React tests; add `docs/grounded_generation.md`, `backend/evaluation/m6_development.md`, `backend/evaluation/generation_benchmark.py`, deterministic public regression fixtures and separate metrics_history section11; ignored actual generation/DeepEval JSONL artifacts.

- [ ] Read installed Next client docs and React/UI skills. RED/GREEN matrix explicit unknown/unsupported/comparability/source level/quotes/citation links/clarification/privacy controls; preserve selected scopes and avoid unsafe URLs/raw HTML. Browser desktop/mobile public fixtures.
- [ ] Run labelled claim/citation matrix and whole-review versus atomic-batch ablation on fixed generated candidates, recording additional calls/rejection/false rejection with development-label limitations. Actual bounded local Ollama answers on local/fulltext,Web abstract,partial/no-answer and conflict fixtures; DeepEval Faithfulness+Answer Relevancy with model/version/template/temperature/threshold/status/tokens/cost/time; failure stays not_evaluated.
- [ ] One fresh whole-branch reviewer after all tasks, one RED/GREEN important-fix pass, final full backend/frontend/lint/build/privacy checks. Preserve failures; no redundant full baseline (M5 exact-head CI already green).
- [ ] Own commits/push and dependent Draft PR targeting M5; verify exact-head CI; update issue57/root milestone/ledger. Notify M6 engineering completion only with recorded evidence; no merge or M8 Gold claim.

## Execution rulings

- User's repeated continue instructions and waived individual review/approval carry forward; v6 already provides the approved design. Implement inline without repeating design/worktree permission questions, in the existing dedicated checkout on `feat/57-m6-grounded-comparison` from verified M5 `a802165`.
- Local legacy chunks remain citable through SQLite original text, but no native snapshot means PDF pages are null; compute an excerpt version hash rather than pretending a PDF byte hash exists.
- QA and comparison production endpoints switch to the common verified path. Historical helper functions remain tested compatibility boundaries; do not falsely call those legacy evaluations M6 verification.
- No model-based extraction may establish scientific comparability without quote/support validation; conservative unknown is the chosen default. Exact semantic independence/calibration stays M8.
