# M8 final development evaluation and delivery

Date:2026-11-04 (Australia/Sydney). M7 PR60 merged as`2b48206285f3237ce7590b3e5593c415bd199bda`; M1–M7 production source/config frozen for this run. M8 adds evaluation/delivery tooling. **Independent human M8 acceptance is not complete.** The engineering results below use public dataset references, explicitly controlled source fixtures and separately labelled AI review. No overall scientific-answer accuracy is certified.

## Corpus and methods

40 new QASPER train papers/questions,32 answerable/8 unanswerable dataset references; earlier dev40 excluded.929395 text characters map to849 native children. Native provided-text spans/pages-null are checked against exact raw offsets. Dataset labels, model pretraining exposure and scientific entailment were not independently certified. Paper Hit is conditional on a supplied paper scope, not whole-library paper discovery.

Actual local SQLite FTS, BGE-small-en-v1.5(384), Qdrant1.16.2 and cached cross-encoder were used. Seven variants share corpus/models/budgets. The initial run made39 actual rewrite calls:13 plans applied,26 rejected to Q0 and1 disabled by constraint guard. Final replay preserves those exact plans, reruns actual retrieval/reranking with no new rewrite calls, and records complete R0/ranking diagnostics. Initial artifacts remain intact.

Packing is scored separately from rank lists. The table below uses admitted child Recall@10 after packing. Full original-paragraph character coverage is a reference proxy, not answer completeness; visible parent background is not automatically citable child evidence. Missing/unanswerable metrics remain null. Retrieval timings exclude rewrite/model-generation cost; derived packing variants share the same measured retrieval. Some diagnostic jobs overlapped, so timings are workstation observations, not an isolated SLA.

| Variant | Packed Recall@10 | MRR | Citable coverage | Visible coverage | Retrieval p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25_flat | 0.484375 | 0.361979 | 0.492289 | 0.492289 | 306.94/541.75 |
| dense_flat | 0.562500 | 0.423958 | 0.580377 | 0.580377 | 1030.44/1310.67 |
| single_hybrid_flat | 0.546875 | 0.432292 | 0.561113 | 0.561113 | 1062.55/1366.62 |
| multi_hybrid_flat | 0.546875 | 0.418750 | 0.561113 | 0.561113 | 1029.22/1317.65 |
| multi_parent | 0.546875 | 0.418750 | 0.561113 | 0.561113 | 1029.22/1317.65 |
| multi_rerank_flat | 0.718750 | 0.508333 | 0.743809 | 0.743809 | 1029.22/1317.65 |
| multi_rerank_parent | 0.718750 | 0.508333 | 0.743809 | 0.743809 | 1029.22/1317.65 |

Compared with single Hybrid flat, final packed recall mean delta=0.171875; paired wins/ties/losses={'win': 10, 'tie': 19, 'loss': 3}; percentile bootstrap95% interval=[0.015625, 0.328125] (32 answerable pairs,2000 draws,seed61). This is descriptive reference development, not a confirmatory significance/quality claim. See paired_results.jsonl and per_case.csv. No production default was changed to fit these questions.

## Generation, citations and Judge

Eight actual public-reference generations and six controlled local-PDF/Web-OA/abstract/metadata/fault/conflict scenarios are recorded separately. Synthetic Web originals are injected licensed fixtures, not actual Web retrieval gains. Declared metadata/fault cases have no factual evidence. Claim support is verified by the production pipeline but remains uncalibrated against humans. Exact citation ID/quote/DOI-field/URL/page binding is independently checked; DOI-field equality is not verification that a DOI is registered or that a publication is scientifically true.

Public-reference statuses:{'evidence_only': 5, 'partial': 2, 'supported': 1}; controlled statuses:{'supported': 4, 'abstain': 2}. Public pipeline calls=12; controlled pipeline calls=8. Judge attempts=119, separately from pipeline and rewrite calls. Detailed token counts, failed/unknown usage, metric targets and DeepEval4.2.8/local4B settings are in deepeval_final_results.jsonl. Three reference-free metrics plus two reference metrics apply only when valid reference answers exist. Bounded Judge timeout/malformed results remain not_evaluated; averages count scored observations only. Safe evidence-only output is not silently scored as a successfully answered question. Human claim support and unsupported-claim rate remain null. Local provider API currency is0; total CPU/GPU/energy cost is unmeasured.

Long-context eligibility uses the same serialized8192-unit context ceiling and all native source children. Parser-filtered children are not raw unfiltered full-paper equivalence; eligibility/exclusions are retained in long_context.csv. Fixed-evidence ordering is a small synthetic controlled test with human completeness unmeasured; it does not justify changing evidence order.

A separate matched comparison actually generated from the full untruncated original provided paper plus a canonical citation-span catalogue versus RAG, under the same8192 context ceiling and production verifier. Two fitting papers produced four results (long_context_comparison.csv): the answerable case was supported with RAG and evidence_only with raw full paper; the no-answer-reference case was evidence_only in both modes. Human answer completeness and paired Judge scores are unmeasured; these two pairs do not establish a general long-context advantage. The earlier all-native-child run is retained separately in the cost ledger.

Citation checks are stratified by local/Web and full-text/abstract evidence in citation_checks.csv, including actual citation counts, exact-quote hashes and canonical page bindings. Zero-citation abstentions are not successful scientific claims. DOI fields are absent in these particular generation samples; registered-DOI existence was not measured. Reference-no-answer and reference-answerable output statuses are in reference_abstention.csv; human correct/false-refusal decisions remain unknown. One reference-no-answer question produced a partial related statement, which does not establish a complete answer.

## Web, PDF and resources

M5 routing fixtures rerun:{'route_count': 16, 'wrong_web_triggers': 1, 'missed_web_triggers': 1, 'local_first_count': 16, 'max_corrections': 1, 'faults_passed': 7, 'fault_count': 7}. The deliberately challenging fixture set retains wrong/missed routes; engineering/privacy gates do not certify semantic routing accuracy. Two actual connectivity probes are separate:[('openalex', 8, [200]), ('semantic_scholar', 0, [429])]. Connectivity does not establish answer improvement or full-text availability.

An actual production Web adapter smoke test used an isolated empty local library and run-ID-correlated traces. Observed modes:[(True, 'abstain', 0, 0), (False, 'evidence_only', 4, 5)]. Local-only abstained with0 external attempts. Allowed Web made4 external attempts, recovered from Semantic Scholar429 to200, received OpenAlex200, and handled a temporary-PDF timeout;16 Web records yielded5 evidence entries but no supported answer or model generation. This is actual acquisition/fault handling, not an answer-quality gain. The first unguarded Windows helper was re-executed by PDF worker spawning; its mixed traces are archived and excluded from paired results, with total duplicated usage unknown.

Historical PDF packet:{'samples': 50, 'suspicious': 30, 'ordinary': 20, 'document_hash_matches': 50, 'old_offsets_match_current': 27, 'same_text_block_present': 38, 'independent_human_reviewed': 0, 'current_parser_review_packet': 50, 'notes': 'Changed offsets require audit; not automatic error labels.'}. All50 original PDF hashes match; old offsets/blocks can differ after earlier parser fixes and are not automatic new errors. A new current-parser30-suspicious/20-ordinary packet is separately frozen and published with human decisions blank. Codex visually inspected double columns, simple/complex tables, captions, formulas and a page boundary; scans retain typed unsupported behavior and abstracts have null geometry. AI/source checks are not human layout/cleaning certification.

Synthetic replay:1000 points×384 dimensions,32 searches,query p50/p95=28.48/42.98ms; Qdrant memory after=196.5MiB / 7.626GiB, indexed vectors=0. These random-vector runs measure actual local search/storage observations; no embedding inference, answer quality, ANN or concurrent-user capacity claim. Collections are separate from user data.

Synthetic replay:10000 points×384 dimensions,32 searches,query p50/p95=24.53/40.02ms; Qdrant memory after=244.3MiB / 7.626GiB, indexed vectors=0. These random-vector runs measure actual local search/storage observations; no embedding inference, answer quality, ANN or concurrent-user capacity claim. Collections are separate from user data.

Docker27.3.1 was restored by preserving/recreating only its inaccessible stale runtime directory, without deleting volumes. Backed-up scoped maintenance removed181 retired vectors; current93 valid vectors and1910 foreign-scope vectors were preserved, and a second repair was idempotent. SQLite retains both papers and274 historical chunks. User data was not used for benchmark questions or screenshots.

## Acceptance and delivery limits

Engineering deliverables include reproducible tools, frozen candidates, paired ablations, original-anchor checks, AI error review, bounded Judge results, stress/connectivity diagnostics, English architecture/install/maintenance documentation and a recording/walkthrough script. The existing real UI screenshot is retained. Automatic approval rejected new frontend preview startup with only blocked-by-policy detail, so no new UI screenshot or browser walkthrough is claimed. API/developer observations are not a3–5-person user study.

Formal M8 gates remain unmet:0 independent human Gold,0/50 human PDF reviews,0/10 human error reviews and unmeasured human claim support for six scenario classes. AI review delegated by the user is labelled AI. Read annotation_guidelines.md and release_gate.json; no synthetic/development result is promoted to independent human acceptance. Final review, test and exact-head CI evidence are recorded in the PR and metrics history.

The initial local-PDF fixture was rejected before generation because in-memory bbox tuples differed from persisted JSON arrays. A RED test reproduced the fixture-boundary error; the corrected fixture uses the actual JSON wire representation and retains its original PDF/source database. All six scenarios were rerun; prior results/costs remain separate in diagnostic_cost_ledger.json. This correction did not change the frozen production parser/registry or the40-question corpus.

Actual model files/digests are now checked against a bound identity manifest:{'actual_model_identity_verified_now': True, 'models_in_initial_prospective_freeze': False, 'model_binding_sha256': '0074d55ffa8f2883679e4afdd425d6d08fcc82b173af6edc600429034e0226ac'}. This initial development run recorded weight identities after its first diagnostics, so prospective every-stage weight verification is not retrospectively claimed. Fresh preparation now includes model_freeze.json in the initial manifest, and retrieval/generation/Judge/report stages reject cache changes or same-name retags. Model names/config alone are insufficient.

Actual isolated synthetic API walkthrough:{'review_type': 'developer_API_walkthrough_not_user_study', 'real_human_participants': 0, 'synthetic_papers': 2, 'health_http': 200, 'query_http': 200, 'query_support_status': 'supported', 'source_jumps_verified': 1, 'feedback_review_status': 'needs_review', 'comparison_http': 200, 'comparison_rows': 4, 'comparison_statuses': ['supported', 'supported', 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', 'unknown'], 'local_only': True, 'frontend_preview': 'automatic_approval_rejected_startup', 'observed_pipeline_calls': 6, 'observed_model_attempts': 6}. This covers backend calls/source jumps/feedback/comparison only, with0 human participants and no new browser screenshot.
