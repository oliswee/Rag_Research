# Final evaluation artifacts

This directory publishes sanitized M8 development results and review templates.
`evaluation_final.md` is the entry point. Independent human Gold, PDF judgements
and claim-support calibration are separate acceptance gates; an automated
development score does not satisfy them.

Raw inputs/results stay under ignored
`backend/data/evaluation/m8_final_20261103/`. They include40 new QASPER train
questions, original-source spans, frozen hashes, native SQLite sources, actual
Qdrant traces, generation outputs and Judge observations. The earlier QASPER
dev40 is excluded. Provided dataset text has null PDF pages. Questions/quotes
are not duplicated in these public summaries.

Prepare/download QASPER v0.3 and the M2 PDF review packet first; see the parser
audit report and annotation guidelines. From `backend/`, reproduce in a fresh
directory, preserving failed runs:

```powershell
uv sync --extra evaluation
uv run python -m evaluation.prepare_final --output data/evaluation/m8_new
uv run python -m evaluation.final_benchmark --root data/evaluation/m8_new
uv run python -m evaluation.final_benchmark --root data/evaluation/m8_new --replay
uv run python -m evaluation.final_generation --root data/evaluation/m8_new
uv run python -m evaluation.final_long_context --root data/evaluation/m8_new
uv run python -m evaluation.final_fixture_generation --root data/evaluation/m8_new
uv run python -m evaluation.final_web_flow --root data/evaluation/m8_new
uv run python -m evaluation.final_pdf_audit --root data/evaluation/m8_new
uv run python -m evaluation.final_stress --root data/evaluation/m8_new
uv run python -m evaluation.local_web_benchmark --output data/evaluation/m8_new/web_routes.json --live
```

Before publication, perform and save the separately labelled ten-case source/
context error review as `ai_error_review.jsonl` (or independent human review
under its own provenance). The report requires this review and a model freeze;
it does not silently generate reviewer decisions. Optional API walkthrough
observations are recorded separately. Export sanitized results with
`uv run python -m evaluation.final_reports --root data/evaluation/m8_new --destination ../docs/evaluation_new`.
Use a fresh output directory and preserve completed reviewer forms.

Replay reuses frozen actual query plans/native sources, performs actual
retrieval/reranking again and preserves raw ranking/R0 attribution. It makes no
new rewrite calls or hidden attempts to improve rejected rewrites. Initial runs
remain intact. Stress creates new diagnostic collections and refuses to replace
existing collections. It uses random vectors, not relevance labels or embedding
throughput measurements.

The bounded local Judge runs DeepEval's three reference-free metrics and two
reference metrics when an actual reference answer exists. ID/quote/DOI-field/
page binding checks are independent of Judge scores. Counts separate generation,
rewrite and Judge work; failed/unmeasured usage stays unknown. No paid provider
is required.

Fresh preparation captures actual cached model-file hashes and the Ollama tag
digest in the initial freeze. Later stages reject altered weights or same-name
retags. The historical November3 development run bound these identities after
its first diagnostics; its report explicitly records that this was not a
prospective every-stage model freeze.

`long_context_comparison.csv` contains the two actual matched RAG/raw-full-paper
pairs. `actual_web_flow.csv` contains a guarded production-adapter smoke test;
its Local-only mode has no external attempts. Neither small diagnostic certifies
human answer completeness. The diagnostic cost ledger includes failed fixture
rounds, extra comparisons and a Windows helper spawn failure with unknown total
duplicated usage. Preserve those logs rather than counting only successful work.

`gold_annotations.jsonl` is intentionally empty until independent human Gold
exists. Pending candidate forms remain in the ignored frozen packet.
`pdf_parser_human_audit.csv` contains current-parser fields with empty human
decisions. Follow `annotation_guidelines.md` to create a new reviewed freeze.
AI visual/source/error review is separately labelled in `error_analysis.md`.
Developer walkthrough observations are not a3–5-person user study.

The existing real UI screenshot is retained. Automatic approval blocked a new
frontend preview during this run; no new UI screenshot is claimed.
