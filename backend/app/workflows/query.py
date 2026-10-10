"""Query workflow for searching and synthesizing answers."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Paper, Pattern
from app.schemas.session import ConversationTurn, SourceReference
from app.services import EmbeddingService, LLMClient, SessionManager, VectorStoreService
from app.services.prompts.synthesis import (
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
    PatternContext,
    SynthesisResult,
    build_conversation_history,
    build_pattern_context,
    parse_synthesis_result,
)

logger = logging.getLogger(__name__)


@dataclass
class QueryInput:
    """Input for query workflow."""

    query: str
    session_id: str | None = None  # None = new session
    limit: int = 10  # Max patterns to retrieve


@dataclass
class QueryResult:
    """Result of query workflow."""

    answer: str = ""
    sources: list[SourceReference] = field(default_factory=list)
    confidence: str = "medium"
    follow_up_suggestions: list[str] = field(default_factory=list)
    session_id: str = ""
    success: bool = True
    error: str | None = None


class QueryWorkflow:
    """Workflow for querying the pattern library.

    Flow:
    1. Get/create session (SessionManager)
    2. Generate query embedding (EmbeddingService)
    3. Search vector store (VectorStoreService)
    4. Retrieve pattern details (database)
    5. Build pattern context (synthesis.build_pattern_context)
    6. Synthesize answer (LLMClient + synthesis prompt)
    7. Update session with turn
    """

    def __init__(
        self,
        db: AsyncSession,
        session_manager: SessionManager | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStoreService | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize query workflow.

        Args:
            db: Async database session.
            session_manager: Session manager. Creates default if not provided.
            embedding_service: Embedding service. Creates default if not provided.
            vector_store: Vector store service. Creates default if not provided.
            llm_client: LLM client. Creates default if not provided.
        """
        self.db = db
        self._session_manager = session_manager
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_client = llm_client

    @property
    def session_manager(self) -> SessionManager:
        """Lazy initialization of session manager."""
        if self._session_manager is None:
            self._session_manager = SessionManager()
        return self._session_manager

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy initialization of embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStoreService:
        """Lazy initialization of vector store service."""
        if self._vector_store is None:
            self._vector_store = VectorStoreService()
        return self._vector_store

    @property
    def llm_client(self) -> LLMClient:
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    async def run(self, input: QueryInput) -> QueryResult:
        """Execute the query workflow.

        Args:
            input: Query input with query text and optional session ID.

        Returns:
            QueryResult with answer, sources, and session ID.
        """
        # Step 1: Get/create session
        if input.session_id:
            session = self.session_manager.get_session(input.session_id)
            if not session:
                # Session expired or doesn't exist, create new one
                logger.info(f"Session {input.session_id} not found, creating new")
                session = self.session_manager.create_session()
        else:
            session = self.session_manager.create_session()

        # Step 2: Generate query embedding
        try:
            embedding_result = await self.embedding_service.generate_embedding(
                input.query
            )
            query_embedding = embedding_result.embedding
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return QueryResult(
                session_id=session.session_id,
                success=False,
                error=f"Failed to generate query embedding: {e}",
            )

        # Step 3: Search vector store
        try:
            search_results = await self.vector_store.search(
                query_vector=query_embedding,
                limit=input.limit,
            )
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return QueryResult(
                session_id=session.session_id,
                success=False,
                error=f"Pattern search failed: {e}",
            )

        # Handle no results
        if not search_results:
            return self._create_no_results_response(session.session_id, input.query)

        # Step 4: Retrieve pattern details
        pattern_ids = [r["id"] for r in search_results]
        patterns_with_papers = await self._get_patterns_with_papers(pattern_ids)

        if not patterns_with_papers:
            return self._create_no_results_response(session.session_id, input.query)

        # Step 5: Build pattern context
        pattern_contexts = self._build_pattern_contexts(
            patterns_with_papers, search_results
        )
        context_str = build_pattern_context(pattern_contexts)

        # Build conversation history
        history_dicts = [
            {"role": turn.role, "content": turn.content} for turn in session.history
        ]
        history_str = build_conversation_history(history_dicts)

        # Step 6: Synthesize answer
        try:
            synthesis_result = await self._synthesize(
                input.query, context_str, history_str
            )
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return QueryResult(
                session_id=session.session_id,
                success=False,
                error=f"Answer synthesis failed: {e}",
            )

        # Step 7: Update session with turn
        now = datetime.now(UTC)

        # Add user query turn
        user_turn = ConversationTurn(
            role="user",
            content=input.query,
            timestamp=now,
        )
        self.session_manager.update_session(session.session_id, user_turn)

        # Add assistant response turn
        sources = [
            SourceReference(
                pattern_id=s.pattern_id,
                summary=s.paper_title,
            )
            for s in synthesis_result.sources
        ]
        assistant_turn = ConversationTurn(
            role="assistant",
            content=synthesis_result.answer,
            sources=sources,
            timestamp=now,
        )
        self.session_manager.update_session(session.session_id, assistant_turn)

        return QueryResult(
            answer=synthesis_result.answer,
            sources=sources,
            confidence=synthesis_result.confidence,
            follow_up_suggestions=synthesis_result.follow_up_suggestions,
            session_id=session.session_id,
            success=True,
        )

    async def _get_patterns_with_papers(
        self, pattern_ids: list[str]
    ) -> list[tuple[Pattern, Paper | None]]:
        """Fetch patterns with their associated papers.

        Args:
            pattern_ids: List of pattern IDs to fetch.

        Returns:
            List of (pattern, paper) tuples.
        """
        if not pattern_ids:
            return []

        # Fetch patterns
        result = await self.db.execute(
            select(Pattern).where(Pattern.id.in_(pattern_ids))
        )
        patterns = result.scalars().all()

        # Create a map for ordering
        pattern_map = {p.id: p for p in patterns}

        # Fetch papers for all patterns
        paper_ids = [p.paper_id for p in patterns if p.paper_id]
        papers_map: dict[str, Paper] = {}
        if paper_ids:
            paper_result = await self.db.execute(
                select(Paper).where(Paper.id.in_(paper_ids))
            )
            papers_map = {p.id: p for p in paper_result.scalars().all()}

        # Return in original order with papers
        results = []
        for pid in pattern_ids:
            pattern = pattern_map.get(pid)
            if pattern:
                paper = papers_map.get(pattern.paper_id) if pattern.paper_id else None
                results.append((pattern, paper))

        return results

    def _build_pattern_contexts(
        self,
        patterns_with_papers: list[tuple[Pattern, Paper | None]],
        search_results: list,
    ) -> list[PatternContext]:
        """Build PatternContext objects for synthesis.

        Args:
            patterns_with_papers: List of (pattern, paper) tuples.
            search_results: Original search results (for ordering).

        Returns:
            List of PatternContext objects.
        """
        contexts = []
        for pattern, paper in patterns_with_papers:
            contexts.append(
                PatternContext(
                    pattern_id=pattern.id,
                    paper_title=paper.title if paper else "Unknown",
                    name=pattern.name,
                    claim=pattern.claim,
                    evidence=pattern.evidence,
                    context=pattern.context,
                )
            )
        return contexts

    async def _synthesize(
        self,
        query: str,
        pattern_context: str,
        conversation_history: str,
    ) -> SynthesisResult:
        """Synthesize an answer using LLM.

        Args:
            query: User's query.
            pattern_context: Formatted pattern context.
            conversation_history: Formatted conversation history.

        Returns:
            SynthesisResult with answer and sources.
        """
        user_prompt = SYNTHESIS_USER.format(
            query=query,
            pattern_context=pattern_context,
            conversation_history=conversation_history,
        )

        response = await self.llm_client.complete_json(
            system_prompt=SYNTHESIS_SYSTEM,
            user_prompt=user_prompt,
        )

        return parse_synthesis_result(response)

    def _create_no_results_response(self, session_id: str, query: str) -> QueryResult:
        """Create a response when no patterns match the query.

        Args:
            session_id: Current session ID.
            query: User's query.

        Returns:
            QueryResult with appropriate message.
        """
        return QueryResult(
            answer=(
                "I couldn't find any patterns in the library that are relevant to "
                "your question. This could mean:\n"
                "- The topic hasn't been covered in any ingested papers yet\n"
                "- Try rephrasing your question with different terms\n"
                "- Consider uploading papers related to this topic"
            ),
            confidence="low",
            session_id=session_id,
            success=True,
        )
