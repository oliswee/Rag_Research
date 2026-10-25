# M5 development evaluation — 2026-10-25

Issue55; branch `feat/55-m5-local-web-routing`, base M4 `418bd8e`. Engineering verification and labelled development diagnostics, **not independent Gold, human audit, or a measured real-world Web-answer gain**.

## Reproducible routing / fault diagnostics

Runner: `evaluation/local_web_benchmark.py`. It creates an in-memory SQLite corpus, injects fixed local retrieval outputs and an empty academic boundary for routing cases. It calls the real routing/packing/pool workflow; no model inference or real provider traffic is included in this matrix. Hand-authored labels include privacy conflicts, absent papers, abstract-only evidence, missing numbers/subquestions/anchors, conflicting findings and two semantic challenges. Labels are known during implementation; these are development checks.

| Measure | Observed |
| --- | ---: |
| Labelled route cases |16|
| Local retrieval first |16/16|
| Maximum local corrections |1|
| Wrong Web triggers |1/16|
| Missed Web triggers |1/16|
| Forbidden Web triggers (privacy/baseline subset) |0|
| Provider fault matrix |7/7 safe partial failures|
| Metadata/factual model calls in routing matrix |0/0|

The wrong trigger is a semantically relevant synonym passage with insufficient lexical overlap. The missed trigger is a passage with question terms that only describes a proposed evaluation heading, without a finding. Do not interpret 14/16 agreement as general precision/recall: the fixture distribution is designed, small and reused. Sufficiency remains a heuristic requiring M8 independent labels/calibration. RRF is never interpreted as a probability.

Fault matrix uses actual httpx MockTransport/provider parsing: 429 with one retry, timeout,403blocked, network failure, >512KiB response, malicious private redirect and malformed JSON. The second provider remains available. Each matrix row made at most3 mock HTTP attempts and preserved one bibliographic candidate. Separate workflow regressions inject the same faults jointly with local evidence and assert original local text survives, no local excerpts reach URLs/bodies, no new/dirty corpus rows, and errors contain no provider secret text.

Additional evidence fixtures cover licensed OA originals with exact anchors, abstracts, metadata, unknown licences, no DOI, duplicate/versioned results, unavailable/malformed PDFs, cross-host/private redirects, byte/type limits, strict selected-paper versions, explicit arXiv-version bypass, native worker termination and released slot, per-request PDF deadline, no automatic imports, abstract unsupported-number/full-text claims and metadata exclusion.

Raw ignored artifacts: `data/evaluation/m5_20261025/final_verified_matrix.json` (configuration/pipeline and fixture SHA-256 recorded at execution), initial `summary.json` (contains the earlier matrix and live connectivity), `judge_smoke.json`. Earlier matrix snapshots remain available; measured fingerprints are not retroactively rewritten. Matrix latency is injected-fixture execution time, not retrieval performance/SLA; no p95 is claimed.

## Two real free-provider connectivity checks

Fixed public search “retrieval augmented generation”, no local library content/history, no PDFs, each provider limited to one HTTP attempt:

| Provider | Result | Latency ms |
| --- | --- | ---: |
| Anonymous OpenAlex |HTTP200,8 candidates|6961.120|
| Semantic Scholar, existing configured credential |HTTP200,8 candidates|5814.102|

This verifies connectivity and bounded parsing only. It does not validate provider relevance, publication coverage, licence validity of a PDF, citation truth, or Web-assisted answer improvement. Academic quota/billing/resource consumption was not measured. No paid model calls or key creation occurred.

## Local DeepEval diagnostic

DeepEval4.2.8, local Ollama `qwen3:4b-instruct`, temperature0, Faithfulness threshold0.5, default template,60-second deadline per diagnostic. Synthetic public fixture: context says95%; one answer repeats95%, the negative control says99%. Answers are fixed controls, not production-generated answers.

| Control | Status | Faithfulness | Latency ms |
| --- | --- | ---: | ---: |
| Supported abstract statement |evaluated|1.0|38851.476|
| Unsupported numeric control |evaluated|0.0|30999.156|

These two automatic judgements distinguish this specific number change only. No human calibration or overall answer-quality estimate. Actual judge HTTP subcall counts/tokens are unknown (`null`); framework evaluation_cost0 is not total energy/resource cost. Generation calls0; no paid cloud model calls.

## Review / regression record

- Initial focused backend85passed/28.70s. Policy/joint-provider additions48passed/3.89s. Native parser lifecycle plus scope reset16passed/31.00s.
- Initial complete backend **994passed /1failed /14warnings /471.90s**. The failure was the obsolete Pattern-empty-library message expectation. Updated it to assert evidence abstention, Web route and single correction; no production fallback was disabled to pass the test.
- Fresh independent whole-branch review found4 actionable issues, all reproduced before fixes: UI privacy reset across followup/reload (P1), retained native version lost without active index, explicit arXiv version bypass via DOI/provider identity, invalid Scholar year range syntax (P2). The index explanation was refined: vector-pending alone remains active; retained documents without an active generation still require version binding. Fixed via original hash binding/fail-closed ambiguity, strict version checks, hyphen ranges, synchronized persisted UI scope. No per-task agent delegation.
- UI34tests passed before the privacy regression; the new followup/reload regression failed before synchronization and passed after. Lint passed. First Next production build stopped at an existing dependency security-upgrade notice; retry built successfully with that notice retained. M5 does not update dependencies. Dependency security upgrade remains recommended separately.
- Final full suite, exact-head CI and privacy results are appended below after execution. No merge.

Browser check: production frontend in Edge at desktop1280×720 and mobile390×844 with fixed public API response fixtures (real backend was offline). Verified Web abstract/licence/retrieval date display, metadata disclosure and privacy checkboxes retained after reload. Screenshots and route fixture stay under ignored `output/playwright/`. This is a UI rendering check, not a live end-to-end retrieval experiment. Initial default Chrome launch failed because Chrome was absent; the installed Edge succeeded. The existing compact mobile navigation extends near the viewport edge; the new controls wrap without horizontal overflow.
