# M4 local ranking and context packing

## Runtime

Normal Hybrid source queries and followups request up to30 validated child candidates, rank them against the **original question** using a local cross-encoder, select at most10, then pack bounded original parent windows. Explicit BM25/Dense requests retain the previous flat baseline. Embeddings remain local BGE.

The initial CPU model is `Xenova/ms-marco-MiniLM-L-6-v2`, provided by the existing FastEmbed dependency. Model selection follows [Qdrant's FastEmbed reranker documentation](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/); the installed0.9.0 API was checked. Setup is explicit:

```powershell
uv run --directory backend python -m app.services.child_reranker --prepare
```

Only public weights/configuration are downloaded during setup. Requests use cached files only with `CPUExecutionProvider`,2 ONNX threads and one native scoring worker. Error, invalid scores, timeout15s or an already-busy worker preserves RRF order and records the reason. Timeout stops waiting; it does not kill an in-flight ONNX operation. The operation retains its slot until completion, preventing unbounded native jobs. No LLM call is used for scoring.

Configuration: `reranker_candidate_limit`30 (10–50), `reranker_selected_limit`10 (5–10), `reranker_timeout_seconds`15 (at most15), `context_token_budget`8192 (at most16384), `context_parent_budget`2048 (at most4096). These are development defaults, not measured service-level guarantees.

The parent budget applies to expanded window text; its serialized metadata also counts against the global context budget. An indivisible child can exceed the parent-window budget and still be admitted under the global budget without expansion.

## Citations and packing

- Every child is validated against the original document/version/spans and retained checksum. Missing/corrupt parent content is omitted while a validated child remains resolvable. Document/child corruption still fails closed. No history is rewritten.
- Select paper representatives before filling other slots. Reserve **whole children** before spending spare space on parent background, so one parent cannot consume other requested papers' minimum coverage. A child larger than the available budget is dropped with a reason, never cut through a value or quote.
- Parent windows are mapped from original anchors, using complete sentence/line units around selected children. Cross-page anchors retain their original page fields; abstracts/public extracted text keep unknown pages null. A large indivisible sentence falls back to the original child.
- Merge same-parent and overlapping-parent contexts; retain every selected child's ID and citable character ranges. Same DOI + known document version + exact matching excerpt may share text across origins; different versions stay separate. Differing excerpts are not discarded just because the DOI matches.
- Render context as JSON blocks with decoded text, citable ranges, a budgeted paper ID/title map, per-child paper identity, and parent/version/scope/origin/pages. Titles are capped at200 characters; metadata counts toward the budget. Offsets are characters in decoded block text. Expansion pages come from admitted original spans while child page fields remain unchanged. Generation can use background for understanding, but supporting quotes must still occur in the original child excerpt. Parent text never replaces a child citation.
- Count UTF-8 bytes of the **exact serialized context**, including metadata, as conservative BPE token units. This is deliberately stricter than typical generation tokenization; it is not an exact Qwen token count. System/question/history and generated output are outside this context-only budget. No budget guarantee is claimed for arbitrary non-BPE tokenizers.

## Attribution

Trace holds RRF candidate ranks, resolved children, rerank status/model/artifact digest/scores/timing, selected children, packed span anchors, every selection/drop reason and per-paper coverage. Private source text and exception messages are excluded. Requested paper coverage keys are hashed in persisted Trace.

`context_count/context_status` describe assembled material. `prompt_count/prompt_status` are updated only when generation is called; `prompt_evidence_ids` survives generation failure. `actual_prompt_evidence_coverage` is based on sent IDs, separately from assembled coverage. Missing selected papers are named in the answer with unknown results. Semantic conclusion support remains unknown in M4.

`background_context_fraction` measures the byte fraction of assembled text outside selected child ranges. It is a structural expansion measure, **not a semantic noise/relevance score**. DeepEval Contextual Relevancy/Precision are recorded separately as model judgments.

## Development reproduction

```powershell
uv run --directory backend python -m evaluation.ranking_packing_benchmark --output data/evaluation/m4_20261025
uv run --directory backend python -m evaluation.ranking_packing_benchmark --output data/evaluation/m4_20261025 --judge-only
```

The benchmark verifies the frozen40-case QASPER files, replays M3's validated independent branch lists into30 candidates, then actually runs the local cross-encoder and flat/parent packing. Retrieval latency is null because retrieval is replayed. CPU ranking latency is measured separately from model load, packing and LLM planning.

QASPER's public text chunks have no native M2 PDF parents. An explicitly versioned adapter groups consecutive triples within each paper/section, with original text spans and no invented PDF pages. These are development adjacent-text parents; separately verify native parents against the actual M2 library. Child ranking scores and full visible public evidence-span coverage are distinct metrics. Local Judge smoke uses one available public-reference case, first context block,3 modes and2 metrics with60s deadlines; scores/failures are not independent human certification.

Results and warnings: `backend/evaluation/m4_development.md`; separate metric history: `backend/evaluation/metrics_history.md`. M8 independent Gold/PDF audits and formal ablations remain future work.
