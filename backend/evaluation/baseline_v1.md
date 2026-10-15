# M1 baseline_v1 — QASPER development subset

Run date: 2026-10-15 (Australia/Sydney). Issue: [#47](https://github.com/oliswee/Rag_Research/issues/47).

## Data and method

The local QASPER v0.3 development subset has 40 papers, 1,892 application chunks and 40 published-label questions. Thirty-two questions have a mapped evidence paragraph; eight are unanswerable within their specified paper. The text is QASPER's S2ORC extraction, not output from FolioTrace's PDF parser. See `qasper_dev_config.yaml` for corpus checksums and `backend/data/evaluation/DATASET_NOTES.md` for preparation provenance. The corpus files and per-question results stay under Git-ignored `backend/data/evaluation/`.

Run from `backend/`:

```powershell
uv run python -m evaluation.isolated_corpus --papers data/evaluation/qasper_benchmark/papers.json --chunks data/evaluation/qasper_benchmark/chunks.jsonl --labels data/evaluation/qasper_benchmark/labels.jsonl --output data/evaluation/qasper_benchmark/baseline_v1_bm25.json --trace-output data/evaluation/qasper_benchmark/retrieval_trace.jsonl --corpus-version qasper-v0.3-dev-40 --public-corpus
```

The run uses the application's SQLite FTS5 keyword retrieval in an isolated in-memory database. All 40 Trace records contain original query, paper scope, BM25 ranked IDs and raw SQLite rank score, selected IDs, resolved IDs and stage timing. No RRF fusion ran in this BM25-only baseline. Trace records contain no paper text. The 40 local full-text draft questions separately carry `verification_status: unverified`; they are not Gold.

## Observed result

| Mode | Available in this run | Evidence Recall@10 | Hit@10 | MRR@10 | nDCG@10 | p50 / p95 retrieval ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | Yes | 0.78125 | 0.78125 | 0.348996 | 0.447931 | 3.483 / 4.531 |
| Dense | No: OpenAI embedding key and Qdrant unavailable | — | — | — | — | — |
| Single-query Hybrid | No: dense path unavailable | — | — | — | — | — |

Relevance means use the 32 evidence-labelled questions only; the eight paper-scoped no-answer questions are excluded. Latency includes all 40 questions, on this local machine, with Trace enabled. This public development subset has already been inspected in prior work and is not an untouched final test set.

Generation and DeepEval judging were **not run**: no Anthropic or OpenAI key is configured in this environment. The adapter and optional DeepEval 4.2.8 package were tested offline. Running `uv run --extra evaluation python -m evaluation.deepeval_adapter --input evaluation/fixtures/deepeval_smoke.jsonl --output data/evaluation/eval_results.jsonl --provider anthropic --model claude-sonnet-4-20250514` produced two synthetic smoke records marked `not_evaluated/missing_key`, with null scores. They are not QASPER quality measurements. M1 remains open until the remaining baseline and E0 acceptance evidence exists.

## Budget record format

For each later run, record `corpus_version`, `index_version`, parser/chunker/embedding/generator/judge model IDs, DeepEval version, metric thresholds, prompt template version, query count, stage p50/p95 latency, generated and judge token counts, provider request counts, API errors, currency/cost and whether private context upload was explicitly allowed. Unknown measurements are `null`, never zero. No SLA is inferred from this run.
