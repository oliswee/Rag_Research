# M5 — Local-first academic Web fallback

Issue [55](https://github.com/oliswee/Rag_Research/issues/55), based on M4 `418bd8e`. This is a bounded deterministic workflow inspired by corrective retrieval; it does not reproduce CRAG training or implement an autonomous research agent.

## Request contract

`POST /api/query` and `/api/query/followup` accept the existing question/paper/year/provider controls plus `intent` (`factual`, `compare`, `discover`, `latest`), `latest`, `local_only`, and `only_uploaded`.

1. Freeze literal constraints before rewriting or inference. Recognized English/Chinese privacy cues can strengthen controls. Natural-language parsing is a finite rule set; use the explicit controls for reliable privacy policy.
2. Omitted followup constraints inherit from the session; explicit null/false resets them. `latest:false` also clears inherited latest intent unless intent is explicitly supplied. A literal request for latest work still sets latest intent.
3. `local_only` or `only_uploaded` requires local BGE and loopback Ollama/Qdrant before inference, and prohibits academic API/DNS/PDF/resolver calls. Uploaded provenance means `Paper.import_source IS NULL`; intersect selected paper IDs, never broaden an empty intersection.
4. UI reflects effective privacy from the response and stores a validated primitive scope snapshot tied to its session ID. A legacy session without a matching scope snapshot is not silently resumed. No Web pool is stored in the session.

## Routing

Every normal request, including an empty library, executes local Multi-query Hybrid → original-question Child rerank → Parent packing first. Explicit BM25/Dense remain local flat evaluation paths. The former empty-library Pattern shortcut is removed; the legacy Pattern workflow remains available internally for its independent tests.

| Local assessment / constraint | Next step |
| --- | --- |
| Privacy restriction | Answer local or abstain; explain unavailable freshness |
| Latest/discovery, external access permitted | Academic search after local retrieval |
| Sufficient | Answer local; zero academic calls |
| Uncertain / retrieval failure | At most one targeted local correction, then reassess |
| Known corpus gap / still insufficient after correction | Academic search if permitted |

Assessment states are `sufficient`, `retrieval_failure`, `corpus_gap`, `uncertain`. Known gaps require observed missing requested papers, missing indexed chunks or abstract-only content. Unreturned evidence is only a **suspected** retrieval miss. Span availability, question terms/subquestions/numbers, simple opposed findings and unavailable reranking contribute deterministic reasons. These are conservative heuristics, not calibrated relevance probabilities. RRF scores never become sufficiency probabilities. Arbitrary semantic contradictions and paraphrases remain limitations, demonstrated in the development report.

The correction preserves literal Q0 and adds one content-first query; it can narrow to one uncovered requested paper already known to have chunks. No recursive retry. Initial/correction rankings, query variants and reranking status remain separate in `local_routing_attempts`; actual prompt IDs remain separate from assembled context.

## Academic adapter and budgets

`academic_search.py` reuses provider normalizers/parsers with new bounded transport. It does not use DiscoveryService's global query cache or three-attempt transport.

- At most two providers: topic routes select Semantic Scholar/OpenAlex/arXiv/PubMed; default latest uses date-ordered OpenAlex + Crossref. Explicit provider is respected. Scholar year uses documented hyphen ranges. Its explicit latest search is relevance-limited, then date-sorted among returned records; no claim to exhaustively find the newest publication.
- Fixed official endpoint hosts; redirects disabled. Existing S2/NCBI credentials only go to their exact provider. Anonymous OpenAlex and public Crossref require no new account/key. No paid LLM provisioning.
- Up to 8 HTTP attempts shared with PDF redirects, one retry per provider, up to 8 candidates/provider, metadata ≤512KiB, request ≤10s and whole Web deadline ≤45s. PubMed's search/fetch both consume attempts. arXiv requests are paced by a shared lock. Lower configuration limits are allowed.
- API search receives the original redacted question or verified public DOI/arXiv/provider identifier, never local excerpts or history. No raw provider exception body/URL parameters enter Trace. Provider failures preserve local results.
- Known DOI/arXiv/provider identity is mandatory for selected-paper supplements. Explicit arXiv `vN` cannot be bypassed by a DOI/provider match. Native version binding includes retained documents without an active index; unresolved/ambiguous versions fail closed. Hash matching can reject legitimate differently encoded copies of the same publication; this false-negative tradeoff is intentional.

## Evidence and access

| Level | Factual use | PDF pages |
| --- | --- | --- |
| `full_text` | Licensed original PDF excerpts, exact structured-transcript block/page offsets and document SHA | Actual parsed pages |
| `abstract_only` | Only statements actually supported by original abstract quotes | null |
| `metadata_only` | Deterministic bibliographic display; excluded from factual model input | null |

Each Web record includes its temporary paper ID, DOI/provider identity, public source URL, retrieval time, origin, evidence level, access/licence and version status. Unknown licence is not permission. Only explicit CC0/CC-BY/CC-BY-SA OA locations can trigger automatic temporary PDF reading; noncommercial/restricted/unknown terms remain abstract or metadata. No automatic Unpaywall resolution is added in M5; existing explicit-import resolution remains separate.

At most one PDF, ≤3MiB, ≤50 pages and ≤2 same-host redirects. Reuse public-address validation, DNS pinning, SNI, type/magic and streamed size checks. Download including DNS/redirects is also bounded by the request deadline. A single spawned native parser process is terminated/joined on deadline/cancellation; busy slot falls back. Cleaner flags exclude bibliography/repeated margins/page numbers. Excerpts preserve whole original sentence spans; overlong indivisible text is omitted, not rewritten into evidence.

The pool has no database/import/vector dependency. It is destroyed after the request; only returned evidence excerpts remain in the response/browser conversation. Search records are not imported. Selected native PDF mismatches remain metadata; unknown pages are never derived from journal pagination. Existing model quote validation and a support check apply; abstract claims additionally reject unsupported numbers and full-text-access assertions. This does not certify arbitrary semantic truth; M6 expands answer/comparison grounding.

Local and Web factual sources share M4's serialized context budget (8192 conservative UTF-8 byte units by default). Metadata never consumes the factual prompt. Evidence rejected by selection/packing is not recorded as sent to the model. UI displays origin, evidence level, licence, retrieval date and routing/privacy notices with safe text/links.

## Verification and reproduction

See [development report](../backend/evaluation/m5_development.md) and separate [metrics history](../backend/evaluation/metrics_history.md).

```powershell
uv run --directory backend pytest -q
uv run --directory backend python -m evaluation.local_web_benchmark --output data/evaluation/m5/final_matrix.json
# Optional fixed-public metadata connectivity (2 HTTP attempts, no PDF/LLM):
uv run --directory backend python -m evaluation.local_web_benchmark --output data/evaluation/m5/live.json --live
# Optional two synthetic local DeepEval diagnostics, 60s each:
uv run --directory backend python -m evaluation.local_web_benchmark --output data/evaluation/m5/judge.json --judge-only
```

Provider reference contracts: [Semantic Scholar](https://api.semanticscholar.org/api-docs/), [OpenAlex API](https://help.openalex.org/api/), [arXiv manual](https://info.arxiv.org/help/api/user-manual.html), [Crossref access](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/). Metadata licences do not grant permission to download arbitrary linked PDFs.
