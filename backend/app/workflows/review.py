"""Review workflow for processing pattern review decisions."""

import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Paper, PaperStatus, Pattern, PatternStatus
from app.schemas.pattern import PatternDecision
from app.services import EmbeddingService, VectorStoreService

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of review workflow."""

    approved_count: int = 0
    rejected_count: int = 0
    errors: list[str] = field(default_factory=list)
    paper_status_updated: bool = False
    success: bool = True


class ReviewWorkflow:
    """Workflow for processing pattern review decisions.

    Flow:
    1. Validate pattern IDs exist and belong to paper
    2. For approved patterns: embed → persist to vector store → update status
    3. For rejected patterns: update status only
    4. Update paper status if all patterns are reviewed
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStoreService | None = None,
    ) -> None:
        """Initialize review workflow.

        Args:
            db: Async database session.
            embedding_service: Embedding service. Creates default if not provided.
            vector_store: Vector store service. Creates default if not provided.
        """
        self.db = db
        self._embedding_service = embedding_service
        self._vector_store = vector_store

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

    async def run(
        self,
        paper_id: str,
        decisions: list[PatternDecision],
    ) -> ReviewResult:
        """Execute the review workflow.

        Args:
            paper_id: ID of the paper being reviewed.
            decisions: List of pattern review decisions.

        Returns:
            ReviewResult with counts and any errors.
        """
        result = ReviewResult()

        # Step 1: Validate paper exists
        paper = await self._get_paper(paper_id)
        if not paper:
            result.success = False
            result.errors.append(f"Paper {paper_id} not found")
            return result

        # Process each decision
        for decision in decisions:
            try:
                await self._process_decision(paper_id, decision, result)
            except Exception as e:
                logger.error(f"Failed to process decision {decision.temp_id}: {e}")
                result.errors.append(f"Pattern {decision.temp_id}: {e}")

        # Step 4: Update paper status if all patterns reviewed
        try:
            updated = await self._update_paper_status_if_complete(paper)
            result.paper_status_updated = updated
        except Exception as e:
            logger.error(f"Failed to update paper status: {e}")
            result.errors.append(f"Paper status update failed: {e}")

        # Commit transaction
        try:
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            await self.db.rollback()
            result.success = False
            result.errors.append(f"Database commit failed: {e}")

        return result

    async def _get_paper(self, paper_id: str) -> Paper | None:
        """Fetch paper by ID."""
        result = await self.db.execute(select(Paper).where(Paper.id == paper_id))
        return result.scalar_one_or_none()

    async def _get_pattern(self, paper_id: str, pattern_id: str) -> Pattern | None:
        """Fetch pattern by ID, validating it belongs to the paper."""
        result = await self.db.execute(
            select(Pattern).where(
                Pattern.id == pattern_id,
                Pattern.paper_id == paper_id,
            )
        )
        return result.scalar_one_or_none()

    async def _process_decision(
        self,
        paper_id: str,
        decision: PatternDecision,
        result: ReviewResult,
    ) -> None:
        """Process a single pattern decision."""
        pattern = await self._get_pattern(paper_id, decision.temp_id)
        if not pattern:
            raise ValueError("Pattern not found or doesn't belong to paper")

        if decision.approved:
            await self._approve_pattern(pattern, result)
        else:
            await self._reject_pattern(pattern, result)

    async def _approve_pattern(self, pattern: Pattern, result: ReviewResult) -> None:
        """Approve a pattern: embed, persist to vector store, update status."""
        # Generate embedding
        try:
            embedding_text = self.embedding_service.prepare_embedding_text(
                {
                    "name": pattern.name,
                    "claim": pattern.claim,
                    "evidence": pattern.evidence,
                    "context": pattern.context,
                }
            )
            embedding_result = await self.embedding_service.generate_embedding(
                embedding_text
            )
            embedding = embedding_result.embedding
        except Exception as e:
            logger.error(f"Embedding generation failed for {pattern.id}: {e}")
            result.errors.append(f"Pattern {pattern.id} embedding failed: {e}")
            # Still approve but note the error - pattern is valid
            embedding = None

        # Persist to vector store if embedding succeeded
        if embedding:
            try:
                await self.vector_store.upsert(
                    pattern_id=pattern.id,
                    embedding=embedding,
                    payload={
                        "paper_id": pattern.paper_id,
                        "name": pattern.name,
                        "claim": pattern.claim,
                        "evidence": pattern.evidence,
                        "context": pattern.context,
                        "tags": pattern.tags or [],
                    },
                )
            except Exception as e:
                logger.error(f"Vector store upsert failed for {pattern.id}: {e}")
                result.errors.append(f"Pattern {pattern.id} vector store failed: {e}")

        # Update status to approved
        pattern.status = PatternStatus.approved
        result.approved_count += 1
        logger.info(f"Approved pattern {pattern.id}")

    async def _reject_pattern(self, pattern: Pattern, result: ReviewResult) -> None:
        """Reject a pattern and best-effort remove any previously indexed vector."""
        if pattern.status == PatternStatus.approved:
            try:
                self.vector_store.delete_pattern(pattern.id)
            except Exception as exc:
                logger.warning(
                    "Failed to remove rejected pattern vector %s: %s",
                    pattern.id,
                    type(exc).__name__,
                )
                result.errors.append(
                    f"Pattern {pattern.id} vector removal pending: {type(exc).__name__}"
                )
        pattern.status = PatternStatus.rejected
        result.rejected_count += 1
        logger.info(f"Rejected pattern {pattern.id}")

    async def _update_paper_status_if_complete(self, paper: Paper) -> bool:
        """Update paper status to 'processed' if all patterns are reviewed.

        Returns:
            True if paper status was updated, False otherwise.
        """
        # Check if any patterns are still pending
        pending_result = await self.db.execute(
            select(Pattern.id)
            .where(
                Pattern.paper_id == paper.id,
                Pattern.status == PatternStatus.pending,
            )
            .limit(1)
        )
        has_pending = pending_result.scalar_one_or_none() is not None

        if not has_pending and paper.status == PaperStatus.ready_for_review:
            paper.status = PaperStatus.active
            logger.info(f"Updated paper {paper.id} status to active")
            return True

        return False
