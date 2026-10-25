# M4 development results — 2026-10-25

## Scope and provenance

Public QASPER0.3 development subset:40 papers,1,892 chunks,40 questions (32 answerable,8 paper-scoped no-answer). Same frozen hashes as `qasper_dev_config.yaml` and M3. Rank averages use32 answerable cases; CPU timing uses40. This is a public development comparison, not independent FolioTrace Gold, a significance test or a production SLA.

Retrieval is **saved-list replay** from `data/evaluation/m3_verified_20261024/trace_multi_hybrid.jsonl`: only independent validated Q0/Q1/Q2 BM25/Dense lists vote in RRF60, at most30 per list; resolve into30 candidates. No aliases double-count. Query/ID/scope alignment checked. No fresh retrieval, embedding, query expansion or answer generation is timed or claimed in this run.

Each case actually runs the cached local cross-encoder against the original question and child text/section. `Xenova/ms-marco-MiniLM-L-6-v2`, FastEmbed0.9.0, CPU2 threads, artifact digest `sha256:7d8561a043ed799d7c810903011deab75fd98ce67dec33f436a80cc4268d140f`. Model loading is cached-only; explicit preparation downloaded public weights. BGE embedding choice is unchanged.

QASPER chunks lack native M2 PDF parents. The versioned development adapter groups consecutive triples within one paper/section, retains exact public text anchors and **null PDF pages**. These adjacent-text parents cannot establish PDF parsing quality. Native M2 library checks are reported separately below.

Raw ignored artifacts: `data/evaluation/m4_20261025/{report.json,trace_ranking_packing.jsonl,public_contexts.json,judge_smoke.json}`. The report retains corpus hashes, saved retrieval Trace SHA256, previous retrieval pipeline fingerprints, current runtime configuration/pipeline fingerprint, model digest, case IDs/ranks/timing and context results. Do not replace earlier raw failure logs or retrospectively assign the final commit to an earlier execution.

## Actual ranking and packing

| Stage | Recall/Hit@10 | MRR@10 | nDCG@10 | Fully visible evidence fraction |
| --- | ---: | ---: | ---: | ---: |
| RRF first10 from30 candidates |0.78125|0.368837|0.466038|not measured before packing|
| Cross-encoder selected10 |0.87500|0.497991|0.591944|not measured before packing|
| RRF + flat bounded context |0.71875|0.361024|0.446321|0.71875|
| Rerank + flat bounded context |0.84375|0.493527|0.581527|0.84375|
| Rerank + parent bounded context |0.84375|0.493527|0.581527|0.84375|

All40 actual reranks succeeded; no timeout/busy/error fallback. On this fixed development set, reranking retrieves28/32 labelled evidence chunks versus25/32 with RRF first10. Packing retains27/32 versus23/32 for RRF packing. **Parent expansion adds no further fully visible labelled evidence in this experiment.** Ranking alone is insufficient to show what reaches the context; one reranked hit is still lost under the budget.

Context budget8192 UTF-8-byte conservative BPE units, per-parent2048. Mean rendered context units: RRF flat7807.00; rerank flat7739.35; rerank parent7990.50. Mean admitted fraction of the selected10 children:0.735/0.730/0.730. Parent windows were used124 times across40 cases. Every candidate selection/drop has an ID/reason; Trace separates assembled context from actual sent prompt IDs. This retrieval/packing-only experiment sends **no generation prompt**, so actual prompt coverage is0 and semantic support remains unknown.

CPU rerank p50 **4195.401ms**, p95 **5743.242ms**, separate model load **888.918ms**. Quantiles use sorted index `floor((n−1)*p)` over40 requests, including the first inference. They exclude retrieval, rewriting, packing and Judge. Measurements are from a development Windows machine, without a dedicated-load SLA. Do not compare them to M3 total retrieval timings as a speedup. The final scored run followed the earlier full regression; other lightweight development checks occurred separately.

## Native M2 library and numeric diagnostics

Fixed development question, actual SQLite BM25 candidate pool30 and actual local reranker:5 original children packed into3 blocks,7939 units, all child quotations exact,3 block-to-original-anchor checks passed and the corpus fingerprint stayed unchanged. Rerank statusok,8841.879ms in the later native smoke. Default budgets produced **0 parent-expanded blocks** in this case; they safely retained children.

A separately labelled single-child probe with parent budget4096 produced a native parent window of4335 rendered units and passed exact original span reconstruction. This is an integrity check with an explicit override, not the default configuration's quality result. No Qdrant reindex, source rewrite or LLM generation was performed.

Two fixed model diagnostic fixtures selected the intended answer for a1.5ms-versus5.1ms latency question and a20MB-versus200MB memory question. These two examples check numeric/entity discrimination only; they do not establish general numerical accuracy. The latency logits were close, so do not treat the scores as calibrated certainty. Raw numeric scores and expected/observed indices remain in `data/evaluation/m4_library_20261025/report.json`.

## Local Judge smoke

One available public-reference question, first assembled block only, comparing RRF flat / rerank flat / rerank parent, with DeepEval4.2.8 Contextual Relevancy and Contextual Precision. Ollama `qwen3:4b-instruct`, temperature0, threshold0.5, default DeepEval prompt,60s per metric, telemetry disabled. Reference input SHA256 retained. This is a small judge integration check, not answer generation or final quality certification. API subcall/token accounting may be unknown; a framework-reported local cost0 is not full compute cost.

All6 metrics completed within their deadlines:

| Context mode | Contextual Relevancy | Contextual Precision | Relevancy / Precision ms |
| --- | ---: | ---: | ---: |
| RRF flat |0.000000|1.000000|42407.153 /19954.842|
| Rerank flat |0.166667|1.000000|27531.665 /18131.925|
| Rerank parent |0.166667|1.000000|28323.249 /19251.984|

Contextual Relevancy remains below the0.5 threshold for all3 contexts. Precision on one context is a limited integration check and cannot establish ranking quality or compensate for low relevance. The deterministic development ranking gain and these low judge scores are reported together. No overall answer-quality improvement is claimed. Ollama model digest was verified as `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`. All6 framework cost fields are0; API subcall/token accounting remains null. No paid cloud calls.

## Verification and limitations

- Three fresh-review P2 findings were reproduced and fixed: paper/title identity missing from prompts; partially overlapping parents duplicating child text; parent expansion carrying stale child-only page ranges. All3 RED with16 controls, then41 focused checks passed12.87s. Original-span union now rejects conflicting overlaps and preserves each child's citation ranges.
- Development adapter initially expected `id` instead of the corpus's `paper_id`; a real-schema regression reproduced this. Initial helper reuse then exposed a shadowed local name; it was corrected with an explicit alias. These failed attempts produced no scored report and are preserved in ignored logs.
- Initial full backend verification:929 passed,17 warnings,11662.01s as reported by pytest. A later run had931 passed/17 warnings/1 setup error in397.46s: Windows socket error10055 while pytest-asyncio created a Proactor event-loop self-pipe, before the metadata test body. Microsoft defines this as [unavailable socket buffer space](https://github.com/MicrosoftDocs/win32/blob/docs/desktop-src/WinSock/windows-sockets-error-codes-2.md); this observation does not establish port exhaustion as its cause. The metadata file retried16/16 successfully in0.52s. Final serial full regression **932 passed,17 warnings,432.89s**, including review fixes, DOI provenance and the deterministic inference boundary. Both failed and successful logs remain: `output/m4_final_tests.log` and `output/m4_serial_retry_tests.log`. No application or system-network settings were changed to conceal the setup error.
- Final Ruff check/format passed158 files; whole feature-branch24-file privacy gate passed. Current runtime source fingerprint matches the actual scored benchmark. Remote exact-head CI is verified in the dependent PR delivery record.
- Unit/API tests use an equal-score inference double, while direct local-loader/timeout tests cover runtime boundaries and this CLI performs the actual model evaluation. Offline CI never downloads models or calls a paid judge.
- Existing HTTP422 deprecations and AsyncMock coroutine warnings remain. No frontend code or dependencies changed; the previous Next dependency upgrade notice remains recorded maintenance work.
- UTF-8-byte accounting is a conservative bound for BPE context tokens, not an exact generator tokenizer or whole-request budget. Indivisible children can be dropped; titles/metadata consume budget. Full M8 independent Gold/PDF review and held-out ablations remain unperformed.
