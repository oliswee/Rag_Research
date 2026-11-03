# M8 final evaluation and delivery implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline, task by task; one fresh whole-branch reviewer at the end. No per-task delegation.

**Goal:** Finish reproducible final evaluation tooling, source-grounded development diagnostics and delivery, while preserving unmet independent human acceptance as an explicit gate.

**Architecture:** Freeze merged M1–M7 source/config; reuse existing retrieval, packing, DeepEval and index-maintenance implementations. Add a strict evaluation provenance/anchor contract, a new40-question public-source candidate corpus, paired actual retrieval ablations and bounded local generation/Judge cases. Reports distinguish imported dataset references, AI audit, injected fixtures and independently reviewed Gold.

**Tech stack:** Existing Python/SQLite FTS/Qdrant/BGE/reranker/Ollama/DeepEval; no new paid credentials. Existing Next app for real screenshots.

**Spec:** User-adopted v6 M8 E2/E2-PDF/E3/E4/E5/E6/E7/E8/C4/T1/D1 at the original Download_edge checklist. Issue61; base2b48206285f3237ce7590b3e5593c415bd199bda (merged PR60, tree identical to tested M7).

## Global constraints and rulings

- Latest user explicitly authorizes M7 merge and M8 execution, superseding the prior stop. PR60 merged after exact-head green CI41634257131 and empty reviews/threads. Old four PRs remain untouched; no M8 merge inferred.
- Prior user delegated development audit to Codex. AI visual/source decisions must retain reviewer_type=model_assisted and cannot satisfy independent human Gold/PDF/claim-support requirements. Do all autonomous work without repeated review prompts, report formal gates pending; never promote QASPER40unverified or public dataset references to project Gold.
- Freeze new40questions from QASPER train papers absent prior dev40. Dataset evidence can drive separately labelled reference development metrics; questions/answers/fulltext and raw traces remain ignored. Gold file has pending-review rows, not fabricated human approvals. Text-source offsets are exact; PDF pages remain null for extracted dataset text.
- Runtime dependencies restored without deleting Docker volumes, source history or resetting user settings. Back up existing SQLite before any real reindex. Actual service benchmark and in-memory replay always labelled separately.
- Every comparison shares corpus/embed/reranker/generation/Judge config. Timings are actual measured stages; replay is not retrieval latency. Bootstrap is paired by case with fixed seed; missing evidence/labels remain null, never success/zero by default.
- No new production default based on small development effects. Model/Judge usage, timeout and unavailable states preserved; local zero provider bill is not zero resource cost.
- No invented user-study participants, human error labels or final Gold. AI audit and automated attribution10+ can be delivered; formal human10case/50PDF/30–50Gold gates remain pending.

## Review focus

1. Forged review provenance, changed raw quotes/offsets/pages, stale freeze hashes and reused prior questions must fail closed.
2. Unanswerable questions must not contribute zero-valued fake recall or get a guessed reference answer; duplicate evidence is not independent support.
3. Pairing by position or post-hoc changing corpus/model/Judge must not create false paired wins or confidence intervals.
4. No unverified semantic support or unmeasured zero latency/cost may become a release gate. All seven funnel stages must distinguish unknown from observed failure.
5. Runtime recovery and stress tests must preserve actual user data and foreign Qdrant datasets, and cannot claim real multiuser capacity.

## Task1 — Runtime readiness, provenance contract and new frozen materials

**Files:** create evaluation/final_contracts.py, evaluation/prepare_final.py, tests/test_evaluation/test_final_contracts.py; docs/evaluation/annotation_guidelines.md and ignored data/evaluation/m8_final/. Use existing parser50materials with checksummed provenance; new pdf_parser_human_audit.csv remains pending while AI audit stored separately.

**Interfaces:** validate_annotation(row,documents,require_human=False); freeze_manifest(paths,config); prepare(output,count=40) deterministic32answerable+8unanswerable on40newpapers with exact raw quotes/document offsets and PDF null. Preserve separate development_reference.jsonl and gold_annotations.jsonl pending status. Prepare50PDFaudit anchors/screenshots with cohort provenance; no AI-to-human promotion.

- [x] RED annotation provenance/offset/page/hash/scope failures, case-ID uniqueness and old-corpus separation; implement then GREEN.
- [x] Start actual Qdrant dependency without resetting Docker; dry check, backed-up scoped repair if required, report exact outcome.
- [x] Generate frozen candidate/review materials and model/config/source hashes; commit contract/material-preparation code.

## Task2 — Paired retrieval ablations, funnel and replay stress

**Files:** evaluation/final_benchmark.py, evaluation/final_statistics.py, tests/test_evaluation/test_final_statistics.py; docs/evaluation/ablation.csv and error_analysis.md outputs.

**Interfaces:** paired_summary(baseline,candidate,metric,seed=61,draws=2000) key by case_id, reject mismatches; funnel(case,trace,reference) seven-stage evidence-based attribution with unknown states. Actual BM25/Dense/single/multi Hybrid on same native plain-text corpus/real BGE/Qdrant; multi-query plans actual local model, save once. Rerank/flat/parent/no-parent on shared actual candidates, long-context only fitting documents, Web subset explicitly injected/licensed-public fixtures or actual allowed network, never silently substituted.

- [x] RED paired alignment/null handling/bootstrap/funnel/precision/coverage; implement then GREEN.
- [x] Run40case actual retrieval and ablations, preserve per-case traces/raw offsets; report wins/ties/losses/paired uncertainty with no unsupported gain claim.
- [x] Generate10+AI-attributed failure records distinctly from human review. Replay larger synthetic point sets with actual local hardware latency/memory; no quality/capacity extrapolation.

## Task3 — Bounded generation, DeepEval and citation truthfulness

**Files:** evaluation/final_generation.py and tests/test_evaluation/test_final_generation.py; outputs deepeval_final_results.jsonl, long_context.csv, web_fallback.csv, evidence_order.csv, freeze_config.json.

**Interfaces:** reuse grounded_generation/evidence_registry and deepeval_adapter; local bounded Judge includes3reference-free and2reference metrics when valid public reference exists, with metric target/provenance/config/tokens/calls distinct from pipeline usage. Per-metric timeout and malformed/unavailable errors explicit. Six local/OA/abstract/metadata/fault/conflict cases, deterministic ID/DOI/URL/quote/page checks; human claim-support stays null.

- [x] RED no-answer reference omission, unsupported/malformed citations, Judge failure/missingusage, synthetic labels versus human gates; GREEN.
- [x] Run actual bounded generation/Judge on frozen public development cases, suitable full-paper vs RAG and fixed-evidence-order comparison; label injected Web boundaries and actual network separately.
- [x] Preserve all failures/unknowns; gate0fakepages/0Local-onlyWeb/0credentialleaks on actual fixtures and existing security suites. No changed production defaults without evidence.

## Task4 — Final reports, architecture, walkthrough and real screenshots

**Files:** docs/evaluation/evaluation_final.md, annotation_guidelines.md, error_analysis.md, ablation.csv, pdf_parser_human_audit.csv, gold_annotations.jsonl, deepeval_final_results.jsonl, freeze_config.json; docs/architecture_decisions.md, docs/demo_script.md; README and metrics_history section13.

- [x] Publish sanitized schema/templates and machine-generated results, source/config/dataset hashes, full provenance and pending human/user-study gates.
- [x] Describe SQLite/FTS/Qdrant vs alternatives, actual local resources and migration, deterministic Local-first/CRAG-inspired fallback and deferred agent/GraphRAG/SelfRAG/HyDE work.
- [ ] Provide discovery→import→retrieval→source→compare walkthrough, real tested screenshots without private text/keys. No fabricated3–5user observations.
- [x] Keep engineering completion separate from formal M8 human acceptance; README claims link to measured artifacts.

## Task5 — Final review, verification and delivery

- [ ] One fresh whole-branch review, reproduce/fix Important/Critical findings once with RED/GREEN. Full appropriate backend/frontend/lint/build/migration/privacy and exact-head CI.
- [ ] Push own feature branch/draft PR, issue61 and ignored ledgers updated; M8 PR merge not authorized.
- [ ] Final notification describes completed deliverables and any unmet independent human acceptance/service prerequisites candidly, with no invented approval or user-study results.

## Execution evidence (November4)

Tasks1–3 and autonomous report/architecture/README work are delivered. Full backend1180passed before late guard additions; evaluation65 and security/maintenance32 passed after fixes, Ruff238files clean. Whole-branch review found3P2, all reproduced/fixed; bounded confirmation14passed and no remaining Important/Critical. Final actual user-library dry check consistent=true, FTS/vector mismatch counts0. See metrics_history section13 and docs/evaluation.

Task4 screenshot item remains partial: real API/source/feedback/compare walkthrough completed in an isolated synthetic database, but automatic approval blocked frontend preview startup with only blocked-by-policy detail. Existing real screenshot retained; no new screenshot or browser/user-study claim. User-delegated AI review is complete as labelled; independent human Gold/PDF/claim/error gates remain pending, formal_m8_accepted=false. Initial model weight binding was posthoc and disclosed, not retrospectively prospective. Draft PR/exact-head CI and final notification remain until verified.
