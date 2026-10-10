# Rag_Research

A research assistant for collecting papers, extracting structured findings, and asking questions across a personal library. The current application is based on Research Vault and includes a FastAPI backend, a Next.js frontend, SQLite, and Qdrant.

## Current features

- Upload a PDF or paste text.
- Extract claims, evidence, and context for review.
- Browse saved papers and findings.
- Query the library and synthesize answers with source references.

## Run locally

1. Copy `.env.example` to `.env` and add your Anthropic and OpenAI API keys. The `.env` file is ignored by Git.
2. Start the services with `docker compose up --build`.
3. Open `http://localhost:3000`.

The frontend is in `frontend/`; the API and tests are in `backend/`. The API is exposed at `http://localhost:8000`.

## License

This code is distributed under the MIT License. See [LICENSE](LICENSE) for the original copyright notice and license terms.
