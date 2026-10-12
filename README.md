# FolioTrace

A local research assistant for discovering papers and tracing answers to source excerpts. FolioTrace includes a FastAPI backend, a Next.js frontend, SQLite with FTS5, and Qdrant.

![FolioTrace Discover page running locally](docs/images/foliotrace-discover.png)

## Current features

- Discover papers through Semantic Scholar, OpenAlex, arXiv, and PubMed. The search chooses sources from English topic terms: general queries use Semantic Scholar and OpenAlex, AI/CS queries add arXiv, and medical queries use OpenAlex and PubMed (mixed queries use all four). This is a deterministic routing rule, not AI classification; users do not need to choose a mode. Search shows up to 20 papers at a time; **Load more papers** fetches the next set from each selected source without a fixed library limit. arXiv adds preprints and their public PDFs; PubMed adds biomedical records and MeSH or source keywords. Select topic chips to filter the papers already loaded (multiple selections match any topic). Results show source-based abstract extracts, citation counts where available, and a topic-match explanation. Retrieved results are reranked within each loaded batch by query terms in the title, abstract, and topics, with provider rank, citation count, and recency as additional signals. OpenAlex references connect papers within the visible result set. Recent, highly cited, and potential foundation labels are relative signals, not quality ratings. A provider rate limit leaves other sources' results visible.
- Open **Explore metadata and citations** on a result to fetch Crossref DOI metadata, up to five citing papers from Semantic Scholar, and up to five outgoing OpenAlex references. These details load only when requested. A missing DOI or rate-limited provider leaves the available context visible; citation counts and links retain their source labels.
- When you add a paper to the library, the importer first tries its public PDF, then checks Unpaywall, CORE, and Europe PMC for an open copy. Europe PMC article XML is used only when an explicit Creative Commons or public domain license is present. If no usable full text is found, the importer keeps the abstract or metadata and shows the outcome in the import status. The **Abstract preview** label in search describes the result card; it does not predict what the importer will find.
- Click **Add to library** on one result or select several for a batch import. Open PDFs are parsed by page; unavailable PDFs fall back to abstracts or metadata. DOI, arXiv and provider IDs prevent duplicate imports. The library has no fixed paper count limit.
- Upload PDFs or paste text, and review generated claims where API keys are available.
- Ask questions against local source excerpts, filter by paper, year or provider, and compare 2–5 papers in an evidence matrix.
- Use SQLite FTS5 keyword search, Qdrant dense search, or experimental RRF hybrid search. Dense retrieval falls back to keyword search when vectors are unavailable. Each answer cites only SQLite excerpts; abstract excerpts have no invented page number.
- Optionally rewrite a question into a standalone search query using the configured Anthropic API key. The answer still addresses the original question, and the effective search query is shown with the result. If rewriting is unavailable, search uses the current question with the previous question for context.

## Run locally

### Native app with Qdrant in Docker

Install Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js 20+, and Docker Desktop. From the repository root, run these PowerShell commands:

```powershell
docker compose up -d qdrant
if (!(Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
New-Item -ItemType Directory -Force backend/data | Out-Null
```

Add keys to `backend/.env` when needed. `ANTHROPIC_API_KEY` enables generated answers and claim extraction; `OPENAI_API_KEY` enables dense indexing and search. Paper discovery and SQLite keyword search work without either key. `SEMANTIC_SCHOLAR_API_KEY` and `NCBI_API_KEY` are optional; `NCBI_EMAIL` can identify your NCBI requests. Set `UNPAYWALL_EMAIL` to an email address to enable Unpaywall DOI lookups, and `CORE_API_KEY` to a CORE API key to enable CORE lookups. Europe PMC open-access lookup needs no key. Environment files are ignored by Git. Full text is never guaranteed; provider limits, access rights, missing identifiers, or extraction failures can leave a paper at abstract or metadata scope. [NCBI E-utilities guidance](https://www.ncbi.nlm.nih.gov/books/NBK25497/) describes its request limits and [NCBI's disclaimer](https://www.ncbi.nlm.nih.gov/About/disclaimer.html) applies to PubMed data.

Start the API in one terminal:

```powershell
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal, from the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

Open [http://localhost:3000/discover](http://localhost:3000/discover). The header should show a green backend connection indicator. [http://localhost:8000/health](http://localhost:8000/health) reports the SQLite and Qdrant status. Use `localhost` for the frontend: the API currently allows that origin on port 3000.

### All services in Docker

To let Compose build and run the frontend, API, and Qdrant together instead, run this from the repository root:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
docker compose up --build
```

Open [http://localhost:3000/discover](http://localhost:3000/discover). Compose runs SQLite migrations at startup and stores its database and Qdrant data in Docker volumes. This SQLite database is separate from `backend/data/research.db` used by the native API, so papers imported through one setup do not automatically appear in the other.

After adding an OpenAI key to a library created without one, run `uv run python -m evaluation.reindex` from `backend/` to rebuild source vectors. Keywords remain searchable while indexing is unavailable.

Retrieval quality is not yet certified: the spec requires 40 independently labelled queries before hybrid search becomes the default. Run `uv run python -m evaluation.evaluate labels.jsonl --output report.json` from `backend/` after creating that dataset. Add `--judge` and install DeepEval to include Faithfulness scoring. The script also checks citation IDs and reports Recall@10, nDCG@10, and latency for each retrieval mode.

## Discovery evaluation

The discovery API returns `searched_sources`, `failed_sources`, and `search_latency_ms` with each search. Save search responses for a fixed set of queries, label the relevant DOIs independently, and place them in a JSON file shaped like `{"cases": [{"query": "...", "relevant_dois": ["10.1000/example"], "result": { ...search response... }}]}`. From `backend/`, run `uv run python -m evaluation.discovery_metrics judgments.json` to report unique relevant papers@50, Recall@50, search latency, and provider failure rate. To measure **actual** full-text availability, first import the papers and add each resulting `content_scope` (`full_text`, `abstract`, or `metadata`) to its saved candidate; otherwise that metric is reported as unavailable. These metrics need human judgments and saved runs before they can support a before/after quality claim. The query router uses English topic cues and may miss unfamiliar or non-English terminology; the displayed reranking applies to fetched results, not the entire scholarly corpus.

## License

This code is distributed under the MIT License. See [LICENSE](LICENSE) for the original copyright notice and license terms.
