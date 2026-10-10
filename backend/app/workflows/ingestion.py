"""Ingestion workflow for processing uploaded PDFs."""

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Paper, PaperStatus, Pattern, PatternStatus
from app.schemas.paper import PaperMetadataInput
from app.schemas.pattern import ExtractedPatternResponse
from app.services import (
    MetadataExtractor,
    PatternExtractor,
    PDFParseError,
    PDFParserService,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionInput:
    """Input for ingestion workflow.

    Either pdf_bytes OR text must be provided (not both, not neither).
    """

    pdf_bytes: bytes | None = None
    text: str | None = None
    user_metadata: PaperMetadataInput | None = None

    def __post_init__(self):
        """Validate that exactly one of pdf_bytes or text is provided."""
        has_pdf = self.pdf_bytes is not None
        has_text = self.text is not None and self.text.strip()
        if not has_pdf and not has_text:
            raise ValueError("Either pdf_bytes or text must be provided")
        if has_pdf and has_text:
            raise ValueError("Only one of pdf_bytes or text should be provided")


@dataclass
class IngestionResult:
    """Result of ingestion workflow."""

    paper_id: str
    pattern_count: int
    patterns: list[ExtractedPatternResponse] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None


class IngestionWorkflow:
    """Workflow for ingesting PDFs and extracting patterns.

    Flow:
    1. Parse PDF (PDFParserService)
    2. Extract metadata (MetadataExtractor)
    3. Create Paper record (database)
    4. Extract patterns (PatternExtractor)
    5. Create Pattern records (database, status=pending)
    """

    def __init__(
        self,
        db: AsyncSession,
        pdf_parser: PDFParserService | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        pattern_extractor: PatternExtractor | None = None,
    ) -> None:
        """Initialize ingestion workflow.

        Args:
            db: Async database session.
            pdf_parser: PDF parser service. Creates default if not provided.
            metadata_extractor: Metadata extractor. Creates default if not provided.
            pattern_extractor: Pattern extractor. Creates default if not provided.
        """
        self.db = db
        self._pdf_parser = pdf_parser
        self._metadata_extractor = metadata_extractor
        self._pattern_extractor = pattern_extractor

    @property
    def pdf_parser(self) -> PDFParserService:
        """Lazy initialization of PDF parser."""
        if self._pdf_parser is None:
            self._pdf_parser = PDFParserService()
        return self._pdf_parser

    @property
    def metadata_extractor(self) -> MetadataExtractor:
        """Lazy initialization of metadata extractor."""
        if self._metadata_extractor is None:
            self._metadata_extractor = MetadataExtractor()
        return self._metadata_extractor

    @property
    def pattern_extractor(self) -> PatternExtractor:
        """Lazy initialization of pattern extractor."""
        if self._pattern_extractor is None:
            self._pattern_extractor = PatternExtractor()
        return self._pattern_extractor

    async def run(self, input: IngestionInput) -> IngestionResult:
        """Execute the ingestion workflow.

        Args:
            input: Ingestion input with PDF bytes OR text and optional user metadata.

        Returns:
            IngestionResult with paper_id, patterns, and any warnings/errors.
        """
        warnings: list[str] = []
        paper_id = str(uuid.uuid4())

        # Step 1: Get text content (either parse PDF or use provided text)
        if input.pdf_bytes is not None:
            # Parse PDF to extract text
            try:
                parse_result = self.pdf_parser.parse(input.pdf_bytes)
                logger.info(f"Parsed PDF: {parse_result.page_count} pages")
                text_content = parse_result.text
                pdf_metadata = parse_result.metadata
            except PDFParseError as e:
                logger.error(f"PDF parse failed: {e}")
                return IngestionResult(
                    paper_id=paper_id,
                    pattern_count=0,
                    success=False,
                    error=f"PDF parsing failed: {e}",
                )
        else:
            # Use provided text directly
            text_content = input.text
            pdf_metadata = None
            logger.info(f"Using provided text: {len(text_content)} chars")

        # Step 2: Extract metadata
        try:
            metadata = await self.metadata_extractor.extract(
                text=text_content,
                pdf_hints=pdf_metadata,
                user_metadata=input.user_metadata,
            )
            logger.info(f"Extracted metadata: {metadata.title}")
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            warnings.append(f"Metadata extraction failed: {e}")
            # Use defaults from user metadata or fallback
            metadata = None

        # Step 3: Create Paper record
        try:
            paper = self._create_paper(
                paper_id, text_content, pdf_metadata, metadata, input
            )
            self.db.add(paper)
            await self.db.flush()  # Get paper.id before creating patterns
            logger.info(f"Created paper: {paper.id}")
        except Exception as e:
            logger.error(f"Failed to create paper: {e}")
            await self.db.rollback()
            return IngestionResult(
                paper_id=paper_id,
                pattern_count=0,
                success=False,
                error=f"Database error: {e}",
            )

        # Step 4: Extract patterns
        extracted_patterns: list[ExtractedPatternResponse] = []
        try:
            metadata_context = self._build_metadata_context(paper)
            extraction_result = await self.pattern_extractor.extract(
                text=text_content,
                metadata_context=metadata_context,
            )
            warnings.extend(extraction_result.warnings)

            if not extraction_result.is_successful:
                warnings.append("Pattern extraction had issues - manual review needed")

            logger.info(f"Extracted {extraction_result.pattern_count} patterns")
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            warnings.append(f"Pattern extraction failed: {e}")
            extraction_result = None

        # Step 5: Create Pattern records
        if extraction_result and extraction_result.patterns.patterns:
            try:
                extracted_patterns = await self._create_patterns(
                    paper.id, extraction_result
                )
                logger.info(f"Created {len(extracted_patterns)} pattern records")
            except Exception as e:
                logger.error(f"Failed to create patterns: {e}")
                warnings.append(f"Failed to save patterns: {e}")

        # Commit transaction
        try:
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            await self.db.rollback()
            return IngestionResult(
                paper_id=paper_id,
                pattern_count=0,
                success=False,
                error=f"Database commit failed: {e}",
            )

        return IngestionResult(
            paper_id=paper.id,
            pattern_count=len(extracted_patterns),
            patterns=extracted_patterns,
            warnings=warnings,
            success=True,
        )

    def _create_paper(
        self,
        paper_id: str,
        text_content: str,
        pdf_metadata: dict | None,
        metadata,
        input: IngestionInput,
    ) -> Paper:
        """Create Paper model from extraction results."""
        # Determine values from metadata, user input, or defaults
        if metadata:
            title = metadata.title
            authors = metadata.authors or []
            source = metadata.source
            published_date = metadata.published_date
            paper_type = metadata.type
        elif input.user_metadata:
            title = input.user_metadata.title or "Untitled"
            authors = input.user_metadata.authors or []
            source = input.user_metadata.source
            published_date = input.user_metadata.published_date
            paper_type = input.user_metadata.type
        else:
            if pdf_metadata:
                title = pdf_metadata.get("title", "Untitled")
            else:
                title = "Untitled"
            authors = []
            source = None
            published_date = None
            paper_type = None

        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            source=source,
            published_date=published_date,
            type=paper_type,
            content_text=text_content,
            status=PaperStatus.ready_for_review,
        )

    def _build_metadata_context(self, paper: Paper) -> str:
        """Build metadata context string for pattern extraction."""
        parts = [f"Title: {paper.title}"]
        if paper.authors:
            parts.append(f"Authors: {', '.join(paper.authors)}")
        if paper.source:
            parts.append(f"Source: {paper.source}")
        if paper.type:
            parts.append(f"Type: {paper.type.value}")
        return " | ".join(parts)

    async def _create_patterns(
        self,
        paper_id: str,
        extraction_result,
    ) -> list[ExtractedPatternResponse]:
        """Create Pattern records from extraction results."""
        patterns_response: list[ExtractedPatternResponse] = []

        for extracted in extraction_result.patterns.patterns:
            pattern_id = str(uuid.uuid4())

            # Create database record
            pattern = Pattern(
                id=pattern_id,
                paper_id=paper_id,
                name=extracted.name,
                claim=extracted.claim,
                evidence=extracted.evidence,
                context=extracted.context,
                tags=extracted.tags or [],
                status=PatternStatus.pending,
            )
            self.db.add(pattern)

            # Create response for UI
            # Determine confidence level
            if extracted.confidence:
                confidence = extracted.confidence.value
            else:
                confidence = "medium"

            patterns_response.append(
                ExtractedPatternResponse(
                    temp_id=pattern_id,
                    name=extracted.name,
                    claim=extracted.claim,
                    evidence=extracted.evidence,
                    context=extracted.context,
                    tags=extracted.tags or [],
                    confidence=confidence,
                )
            )

        return patterns_response
