# M1 baseline_v1 — QASPER development subset

Run date: 2026-10-16 (Australia/Sydney). Issue: [#47](https://github.com/oliswee/Rag_Research/issues/47).

## Data and method

The local QASPER v0.3 development subset has 40 papers, 1,892 application chunks and 40 published-label questions. Thirty-two questions have a mapped evidence paragraph; eight are unanswerable within their specified paper. The text is QASPER's S2ORC extraction, not output from FolioTrace's PDF parser. See `qasper_dev_config.yaml` for corpus checksums. The corpus files and per-question results stay under Git-ignored `backend/data/evaluation/`.

To reproduce from a fresh checkout, run these commands from `backend/`. The checksum checks stop on a changed upstream release. The tracked preparation script validates the dev JSON and deterministically recreates the 40-case subset. Compare the three generated file hashes with `qasper_dev_config.yaml` before running the baseline.

```powershell
New-Item -ItemType Directory -Force data/evaluation/qasper | Out-Null
Invoke-WebRequest https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz -OutFile data/evaluation/qasper/release.tgz
if ((Get-FileHash data/evaluation/qasper/release.tgz -Algorithm SHA256).Hash.ToLower() -ne 'a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a') { throw 'QASPER archive checksum mismatch' }
tar -xzf data/evaluation/qasper/release.tgz -C data/evaluation/qasper
if ((Get-FileHash data/evaluation/qasper/qasper-dev-v0.3.json -Algorithm SHA256).Hash.ToLower() -ne '2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93') { throw 'QASPER dev checksum mismatch' }
uv run python -m evaluation.prepare_qasper
Get-FileHash data/evaluation/qasper_benchmark/papers.json, data/evaluation/qasper_benchmark/chunks.jsonl, data/evaluation/qasper_benchmark/labels.jsonl -Algorithm SHA256
```

Run from `backend/`:

```powershell
uv run python -m evaluation.isolated_corpus --papers data/evaluation/qasper_benchmark/papers.json --chunks data/evaluation/qasper_benchmark/chunks.jsonl --labels data/evaluation/qasper_benchmark/labels.jsonl --output data/evaluation/qasper_benchmark/baseline_v1_bm25.json --trace-output data/evaluation/qasper_benchmark/retrieval_trace_bm25.jsonl --corpus-version qasper-v0.3-dev-40 --public-corpus --mode bm25
uv run python -m evaluation.isolated_corpus --papers data/evaluation/qasper_benchmark/papers.json --chunks data/evaluation/qasper_benchmark/chunks.jsonl --labels data/evaluation/qasper_benchmark/labels.jsonl --output data/evaluation/qasper_benchmark/baseline_v1_dense.json --trace-output data/evaluation/qasper_benchmark/retrieval_trace_dense.jsonl --corpus-version qasper-v0.3-dev-40 --public-corpus --mode dense
uv run python -m evaluation.isolated_corpus --papers data/evaluation/qasper_benchmark/papers.json --chunks data/evaluation/qasper_benchmark/chunks.jsonl --labels data/evaluation/qasper_benchmark/labels.jsonl --output data/evaluation/qasper_benchmark/baseline_v1_hybrid.json --trace-output data/evaluation/qasper_benchmark/retrieval_trace_hybrid.jsonl --corpus-version qasper-v0.3-dev-40 --public-corpus --mode hybrid
```

Each run uses the application's retrieval path and an isolated in-memory SQLite database. Dense and hybrid use local FastEmbed `BAAI/bge-small-en-v1.5` (384 dimensions) and Qdrant, filtered to benchmark dataset `benchmark:c0d182c5013c0edd`; benchmark vectors never match the app library's dataset filter. All 40 traces per mode record the public query, hashed paper scope, ranked IDs and scores, selected and resolved IDs, and stage timing. Hybrid additionally records RRF. Trace records contain no paper text. Private API requests store query hashes and a placeholder instead of plaintext. The 40 local full-text draft questions separately carry `verification_status: unverified`; they are not Gold.

## Observed result

| Mode | Available in this run | Evidence Recall@10 | Hit@10 | MRR@10 | nDCG@10 | p50 / p95 retrieval ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | Yes | 0.78125 | 0.78125 | 0.348996 | 0.447931 | 3.98 / 4.87 |
| Dense (BGE) | Yes; no fallback | 0.75000 | 0.75000 | 0.338690 | 0.437090 | 745.54 / 830.92 |
| Single-query Hybrid | Yes; no fallback | 0.78125 | 0.78125 | 0.384462 | 0.477571 | 709.75 / 817.09 |

Relevance metrics use the 32 evidence-labelled questions only; the eight paper-scoped no-answer questions are excluded. Latency includes all 40 questions on this local machine with Trace enabled and, for Dense/Hybrid, local BGE query embedding and Qdrant. Index build time is excluded. The three runs were sequential, so latency differences are descriptive rather than a controlled speed comparison. Hybrid improved nDCG/MRR on this subset but not Recall. This public development subset has already been inspected in prior work and is not an untouched final test set.

At the case level, Hybrid raised reciprocal rank on 11 answerable questions, lowered it on 7, and tied on 14 versus BM25. It recovered annotated evidence for cases 13 and 17 that BM25 missed, while dropping cases 15 and 21 out of the top ten. Seven of 32 answerable cases still missed the annotated paragraph. These are development diagnostics, not a tuned improvement claim.

Generation uses local Ollama `qwen3:4b-instruct` and the application's `answer_from_evidence` path. A four-case generation pilot takes the first three answerable and first paper-scoped no-answer question in the frozen label order, with five hybrid excerpts supplied to each answer. Public QASPER references are attached to the three answerable cases in ignored `data/evaluation/qasper_benchmark/qasper_generation_inputs.jsonl`. Its full five-metric local Judge run was stopped after over seven minutes without a result, so no score is reported for those four cases. The no-answer case generated a cited answer despite QASPER's unanimous no-answer annotation; this is a failure case, not evidence of correct answering.

A smaller automatic Judge smoke run uses the first answerable and first no-answer question with one hybrid excerpt each. Both generated answers abstained. References are explicitly omitted from this smoke run to evaluate the three no-reference metrics only; reference-based metrics have a separate live run below. Inputs and results stay under ignored `data/evaluation/qasper_benchmark/`. Neither pilot is Gold or a human citation audit.

DeepEval 4.2.8 with local `qwen3:4b-instruct` completed both smoke cases. The answerable case scored Faithfulness 0.00, Answer Relevancy 0.00, Contextual Relevancy 0.25; the no-answer case scored 0.00, 0.00, 0.00. Both had `status=evaluated` and no judge error. Per-case Judge latency was 78.77 s and 59.38 s. These are local model opinions about two abstentions, with no reference-based score; they cannot establish overall answer quality or calibration. Token counts and monetary cost were not returned by this judge adapter and remain `null` in the result file.

One separate answerable-case run supplied the published QASPER reference to all five DeepEval metrics. Its actual answer cited the top excerpt, and the local Judge returned Faithfulness 1.00, Answer Relevancy 1.00, Contextual Relevancy 0.25, Contextual Precision 1.00, and Contextual Recall 1.00 (`status=evaluated`, 95.61 s). The same query and one-excerpt prompt had produced an abstention in the previous smoke generation despite temperature 0. This observed output variation means the single high-scoring run is not a stable quality estimate. The full five-metric 40-question run remains unmeasured.

```powershell
uv run python -m evaluation.qasper_generation --report data/evaluation/qasper_benchmark/baseline_v1_hybrid.json --traces data/evaluation/qasper_benchmark/retrieval_trace_hybrid.jsonl --papers data/evaluation/qasper_benchmark/papers.json --chunks data/evaluation/qasper_benchmark/chunks.jsonl --labels data/evaluation/qasper_benchmark/labels.jsonl --source data/evaluation/qasper/qasper-dev-v0.3.json --output data/evaluation/qasper_benchmark/qasper_generation_smoke_inputs.jsonl --answerable 1 --no-answer 1 --evidence-limit 1 --omit-reference
uv run --extra evaluation python -m evaluation.deepeval_adapter --input data/evaluation/qasper_benchmark/qasper_generation_smoke_inputs.jsonl --output data/evaluation/qasper_benchmark/eval_results_smoke.jsonl --provider ollama --model qwen3:4b-instruct
uv run python -m evaluation.qasper_generation --report data/evaluation/qasper_benchmark/baseline_v1_hybrid.json --traces data/evaluation/qasper_benchmark/retrieval_trace_hybrid.jsonl --papers data/evaluation/qasper_benchmark/papers.json --chunks data/evaluation/qasper_benchmark/chunks.jsonl --labels data/evaluation/qasper_benchmark/labels.jsonl --source data/evaluation/qasper/qasper-dev-v0.3.json --output data/evaluation/qasper_benchmark/qasper_generation_reference_inputs.jsonl --answerable 1 --no-answer 0 --evidence-limit 1
uv run --extra evaluation python -m evaluation.deepeval_adapter --input data/evaluation/qasper_benchmark/qasper_generation_reference_inputs.jsonl --output data/evaluation/qasper_benchmark/eval_results_reference.jsonl --provider ollama --model qwen3:4b-instruct
```

The adapter and DeepEval 4.2.8 also have offline CI tests. A previous missing-key smoke created two synthetic `not_evaluated/missing_key` records with null scores; they are not QASPER quality measurements. M1 remains open until the remaining R0 acceptance evidence exists.

## Implementation review on 2026-10-23

The latency table above uses the final October 16 rerun. Relevance scores were unchanged. Generation pilots used the earlier hybrid run, preserved in `retrieval_trace_hybrid_generation.jsonl`; their input records also embed that trace. Historical generation records reused retrieval IDs and did not measure generation tokens/calls/time; their embedded zero call counts describe retrieval only and must not be used as generation accounting. The original artifacts were retained without rewriting measurements.

New generation runs validate question hashes, paper scope, ranked evidence and published-question identity before calling the model. They require loopback Ollama, assign a separate generation run ID linked to the retrieval run, and record generated tokens, model calls, returned citations and generation duration. Benchmark indexing checks all candidate IDs before any upsert and rejects collisions with another dataset. Run index builds serially: this preflight is not a database transaction against concurrent external writers.

API requests with `local_only=true` require FastEmbed and loopback Ollama/Qdrant, disable proxy/redirect handling for these clients, and only load cached embedding models. Missing model files cannot trigger a download in that request. Query rewriting shares the request Trace. Retry counts cover the instrumented retry loop; SDK internal retries are disabled. A missing usage value keeps the aggregate unknown (`null`).

Runtime Trace schema 2 adds deterministic corpus/config versions. The corpus
digest streams persisted paper, chunk and pattern content plus citation/filter
metadata; edits are detected even if a stored chunk checksum was not updated.
The config manifest records model IDs, normalized pipeline source revision,
dependency versions and retry/timeout settings. Endpoints are hashed; credentials,
emails, local paths and source text are excluded from the manifest. Snapshot time
is recorded as `corpus_snapshot`; the current implementation scans the small
library per request, with no large-library performance claim. Avoid concurrent
corpus writes while collecting a reproducible run.

`index_version` identifies a logical manifest of corpus/config/collection scope;
`index_locator` and `attempted_index_locators` retain used/attempted locations,
including BM25 fallback. `physical_index_revision` and model weight revisions
remain `null`: a logical manifest cannot certify the actual Qdrant point contents
or immutable weights behind a model tag. This also means the newly instrumented
runtime overhead is not part of the historical retrieval-only latency table.

Remaining R0 work includes parent and prompt spans (M2–M4), verified physical
index revisions, and actual Web events (M5). Web Trace currently reports
`not_implemented`; no Web fallback is claimed. Full 40-question generated-answer
quality and local Judge calibration remain unmeasured.

## Budget record format

For each later run, record `corpus_version`, `index_version`, parser/chunker/embedding/generator/judge model IDs, DeepEval version, metric thresholds, prompt template version, query count, stage p50/p95 latency, generated and judge token counts, provider request counts, API errors, currency/cost and whether private context upload was explicitly allowed. Unknown measurements are `null`, never zero. No SLA is inferred from this run.
