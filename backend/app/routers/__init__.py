"""API routers for Research Vault."""

from app.routers.papers import router as papers_router
from app.routers.patterns import paper_patterns_router, patterns_router
from app.routers.query import router as query_router

__all__ = [
    "papers_router",
    "paper_patterns_router",
    "patterns_router",
    "query_router",
]
