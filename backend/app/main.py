"""FastAPI application entry point."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text

from app.config import settings
from app.db import async_session_maker
from app.models import ImportJob
from app.routers import (
    paper_patterns_router,
    papers_router,
    patterns_router,
    query_router,
)
from app.routers.discovery import router as discovery_router
from app.routers.research import router as research_router
from app.routers.researchers import router as researchers_router
from app.services.session_manager import session_manager
from app.services.source_index import SourceVectorIndex
from app.services.vector_store import VectorStoreService
from app.workflows.discovery_import import process_job

logger = logging.getLogger(__name__)


def _configure_langsmith_tracing() -> bool:
    """Configure LangSmith tracing from settings.

    LangGraph reads tracing configuration from environment variables.
    This function exports settings values to os.environ so LangGraph
    can pick them up, and logs the tracing status.

    Returns:
        True if tracing is enabled and configured, False otherwise.
    """
    if not settings.langchain_tracing_v2:
        logger.info("LangSmith tracing: disabled")
        return False

    if not settings.langchain_api_key:
        logger.warning(
            "LangSmith tracing: enabled but LANGCHAIN_API_KEY not set. "
            "Tracing will not work. Set LANGCHAIN_API_KEY to enable tracing."
        )
        return False

    # Export settings to environment variables for LangGraph
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    logger.info(
        "LangSmith tracing: enabled (project=%s)",
        settings.langchain_project,
    )
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    Path("data").mkdir(exist_ok=True)

    # Configure LangSmith tracing
    _configure_langsmith_tracing()

    # Validate API keys (warn if missing, don't fail)
    if not settings.anthropic_api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set - paper ingestion will fail. "
            "Set this environment variable to enable LLM features."
        )
    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY not set - embeddings will fail. "
            "Set this environment variable to enable vector search."
        )

    # Ensure Qdrant collection exists
    vector_store = VectorStoreService()
    if vector_store.ensure_collection():
        logger.info("Qdrant collection 'patterns' ready")
    else:
        logger.warning(
            "Failed to ensure Qdrant collection - vector search may not work. "
            "Check that Qdrant is running."
        )

    try:
        SourceVectorIndex().ensure_collection()
    except Exception:
        logger.warning("Source vector index unavailable; BM25 search remains available")

    # Jobs survive a process restart. Retry unfinished items when the app returns.
    try:
        async with async_session_maker() as db:
            pending = (
                (
                    await db.execute(
                        select(ImportJob.id).where(
                            ImportJob.status.in_(["queued", "running"])
                        )
                    )
                )
                .scalars()
                .all()
            )
        for job_id in pending:
            asyncio.create_task(process_job(job_id))
    except Exception:
        logger.warning("Could not resume pending import jobs")

    # Start session cleanup task
    await session_manager.start_cleanup_task()

    yield

    # Shutdown
    await session_manager.stop_cleanup_task()


# OpenAPI tag descriptions
openapi_tags = [
    {
        "name": "papers",
        "description": (
            "Paper management: upload PDFs/text, list papers, and manage pattern "
            "extraction. Supports PDF and plain text uploads with automatic metadata "
            "and pattern extraction using Claim/Evidence/Context schema."
        ),
    },
    {
        "name": "patterns",
        "description": (
            "Pattern CRUD operations: update pattern content (triggers re-embedding "
            "if content fields change) and delete patterns from both SQLite and Qdrant."
        ),
    },
    {
        "name": "query",
        "description": (
            "Natural language query interface: submit queries to search patterns via "
            "semantic search and synthesize responses with Claude. Supports "
            "conversation sessions for follow-up queries."
        ),
    },
]

app = FastAPI(
    title="FolioTrace API",
    description=(
        "An AI research assistant that ingests papers/articles, extracts structured "
        "research patterns using the Claim/Evidence/Context schema, stores them in a "
        "searchable knowledge base, and answers cross-paper questions through "
        "synthesis.\n\n"
        "## Key Features\n"
        "- **Paper Ingestion**: Upload PDFs or text, extract metadata and patterns\n"
        "- **Pattern Management**: Review, approve, update, and delete patterns\n"
        "- **Semantic Search**: Query patterns with natural language and filters\n"
        "- **Synthesis**: Claude generates grounded responses citing sources\n"
        "- **Sessions**: Maintain conversation context for follow-up queries"
    ),
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=openapi_tags,
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint with database and Qdrant status."""
    db_ok = False
    qdrant_ok = False

    # Check database connection
    try:
        async with async_session_maker() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        pass

    # Check Qdrant connection
    try:
        vector_store = VectorStoreService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        qdrant_ok = vector_store.health_check()
    except Exception:
        pass

    # Check LangSmith tracing status
    tracing_enabled = os.environ.get(
        "LANGCHAIN_TRACING_V2"
    ) == "true" and os.environ.get("LANGCHAIN_API_KEY")

    # Determine overall status
    if db_ok and qdrant_ok:
        status = "ok"
    elif db_ok:
        status = "degraded"
    else:
        status = "error"

    return {
        "status": status,
        "database": "ok" if db_ok else "error",
        "qdrant": "ok" if qdrant_ok else "error",
        "sessions": session_manager.session_count,
        "tracing": "enabled" if tracing_enabled else "disabled",
    }


# Include routers
app.include_router(papers_router)
app.include_router(paper_patterns_router)
app.include_router(patterns_router)
app.include_router(query_router)
app.include_router(discovery_router)
app.include_router(research_router)
app.include_router(researchers_router)
