# FolioTrace

A local research assistant for discovering papers and tracing answers to source excerpts. FolioTrace includes a FastAPI backend, a Next.js frontend, SQLite with FTS5, and Qdrant.

## Current features

- Discover papers through Semantic Scholar and OpenAlex (up to 20 results per search).
- Select papers to import in the background. Open PDFs are parsed by page; unavailable PDFs fall back to abstracts or metadata. DOI, arXiv and provider IDs prevent duplicate imports. The library has no fixed paper count limit.
- Upload PDFs or paste text, and review generated claims where API keys are available.
- Ask questions against local source excerpts, filter by paper, year or provider, and compare 2–5 papers in an evidence matrix.
- Use SQLite FTS5 keyword search, Qdrant dense search, or experimental RRF hybrid search. Dense retrieval falls back to keyword search when vectors are unavailable. Each answer cites only SQLite excerpts; abstract excerpts have no invented page number.

## Run locally

1. Copy `.env.example` to `.env`. Add `ANTHROPIC_API_KEY` for generated answers and `OPENAI_API_KEY` for dense indexing/search. Discovery and keyword search work without either key. The `.env` file is ignored by Git.
2. Start the services with `docker compose up --build`.
3. Open `http://localhost:3000`.

The frontend is in `frontend/`; the API and tests are in `backend/`. The API is exposed at `http://localhost:8000`. Docker upgrades SQLite on startup and saves SQLite and Qdrant in local volumes. The local Python backend can be started from `backend/` with `uv run alembic upgrade head` and `uv run uvicorn app.main:app --reload` while Qdrant runs separately.

After adding an OpenAI key to a library created without one, run `uv run python -m evaluation.reindex` from `backend/` to rebuild source vectors. Keywords remain searchable while indexing is unavailable.

Retrieval quality is not yet certified: the spec requires 40 independently labelled queries before hybrid search becomes the default. Run `uv run python -m evaluation.evaluate labels.jsonl --output report.json` from `backend/` after creating that dataset. Add `--judge` and install DeepEval to include Faithfulness scoring. The script also checks citation IDs and reports Recall@10, nDCG@10, and latency for each retrieval mode.

## License

This code is distributed under the MIT License. See [LICENSE](LICENSE) for the original copyright notice and license terms.
