# FolioTrace

A local research assistant for discovering papers and tracing answers to source excerpts. FolioTrace includes a FastAPI backend, a Next.js frontend, SQLite with FTS5, and Qdrant.

![FolioTrace Discover page running locally](docs/images/foliotrace-discover.png)

## Current features

- Discover papers through Semantic Scholar and OpenAlex (up to 20 results per search). Results mix both providers when available and show provider topics, source-based abstract extracts, related themes, citation counts, and a topic-match explanation. OpenAlex references connect papers within the visible result set. Recent, highly cited, and potential foundation labels are relative signals, not quality ratings. A provider rate limit leaves the other provider's results visible.
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

Add keys to `backend/.env` when needed. `ANTHROPIC_API_KEY` enables generated answers and claim extraction; `OPENAI_API_KEY` enables dense indexing and search. Paper discovery and SQLite keyword search work without either key. `SEMANTIC_SCHOLAR_API_KEY` is optional. Environment files are ignored by Git.

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

## License

This code is distributed under the MIT License. See [LICENSE](LICENSE) for the original copyright notice and license terms.
