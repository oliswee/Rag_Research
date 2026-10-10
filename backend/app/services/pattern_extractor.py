"""Pattern extraction service using 3-pass evidence-anchored pipeline.

For documents > 40K tokens, uses chunked extraction with merge/dedupe.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.config import ModelID
from app.services.document_chunker import DocumentChunk, DocumentChunker
from app.services.llm_client import LLMClient, LLMError
from app.services.prompts.evidence_inventory import (
    EVIDENCE_INVENTORY_SYSTEM,
    EVIDENCE_INVENTORY_USER,
    EvidenceInventory,
    EvidenceItem,
    parse_evidence_inventory,
)
from app.services.prompts.extraction import (
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
    ExtractedPatternRaw,
    ExtractionResult,
    parse_extraction_result,
)
from app.services.prompts.verification import (
    VERIFICATION_SYSTEM,
    VERIFICATION_USER,
    VerificationResult,
    VerificationStatus,
    compute_verification_status,
    parse_verification_result,
)

logger = logging.getLogger(__name__)

# Temperature settings for each pass
EVIDENCE_TEMP = 0.2  # Low for consistent extraction
EXTRACTION_TEMP = 0.2  # Low for reproducibility
VERIFICATION_TEMP = 0.1  # Very low for consistency

# Max concurrent chunk extractions to avoid rate limiting
DEFAULT_MAX_CONCURRENT_CHUNKS = 1


@dataclass
class PatternExtractionResult:
    """Complete result from 3-pass extraction pipeline."""

    patterns: ExtractionResult
    evidence_inventory: EvidenceInventory
    verification: VerificationResult
    warnings: list[str]
    was_chunked: bool = False
    chunk_count: int = 1

    @property
    def is_successful(self) -> bool:
        """Return True if extraction completed without critical errors."""
        return self.verification.verification_status != VerificationStatus.fail

    @property
    def pattern_count(self) -> int:
        """Return number of extracted patterns."""
        return len(self.patterns.patterns)


class PatternExtractor:
    """Extracts patterns from papers using 3-pass evidence-anchored pipeline.

    Pass 1 (Haiku): Extract evidence inventory from paper
    Pass 2 (Sonnet): Extract patterns with [E#] citations
    Pass 3 (Haiku): Verify extraction quality

    For documents > 40K tokens, uses chunked extraction:
    - Phase A: Extract from each chunk (Pass 1 + 2)
    - Phase B: Merge and deduplicate patterns
    - Phase C: Synthesis pass for cross-chunk deduplication
    """

    def __init__(
        self,
        haiku_client: LLMClient | None = None,
        sonnet_client: LLMClient | None = None,
        chunker: DocumentChunker | None = None,
        max_concurrent_chunks: int | None = None,
    ) -> None:
        """Initialize pattern extractor.

        Args:
            haiku_client: LLM client for Pass 1 and 3. Creates default if not provided.
            sonnet_client: LLM client for Pass 2. Creates default if not provided.
            chunker: Document chunker for long documents. Default if not provided.
            max_concurrent_chunks: Max chunks to process in parallel. Limits API
                request rate to avoid 429 errors. Defaults to 1 (sequential).
        """
        self._haiku_client = haiku_client
        self._sonnet_client = sonnet_client
        self._chunker = chunker
        self._max_concurrent_chunks = (
            max_concurrent_chunks or DEFAULT_MAX_CONCURRENT_CHUNKS
        )

    def _ensure_haiku(self) -> LLMClient:
        """Lazily initialize Haiku client."""
        if self._haiku_client is None:
            self._haiku_client = LLMClient(model=ModelID.HAIKU_3_5)
        return self._haiku_client

    def _ensure_sonnet(self) -> LLMClient:
        """Lazily initialize Sonnet client."""
        if self._sonnet_client is None:
            self._sonnet_client = LLMClient(model=ModelID.SONNET_4)
        return self._sonnet_client

    def _ensure_chunker(self) -> DocumentChunker:
        """Lazily initialize document chunker."""
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker

    async def extract(
        self,
        text: str,
        metadata_context: str | None = None,
    ) -> PatternExtractionResult:
        """Extract patterns from paper text using 3-pass pipeline.

        For small documents (< 40K tokens):
            Pass 1 → Pass 2 → Pass 3

        For large documents (> 40K tokens):
            Phase A: Chunk + parallel extraction
            Phase B: Merge + deduplicate
            Phase C: Final synthesis

        Args:
            text: Paper content text.
            metadata_context: Optional metadata to include in extraction context.

        Returns:
            PatternExtractionResult with patterns, evidence, and verification.
        """
        # Check if chunking is needed
        chunker = self._ensure_chunker()
        chunking_result = chunker.chunk(text)

        if not chunking_result.was_chunked:
            # Small document: standard 3-pass extraction
            return await self._extract_small_document(text, metadata_context)
        else:
            # Large document: chunked extraction
            return await self._extract_large_document(
                chunking_result.chunks,
                chunking_result.document_outline,
                metadata_context,
            )

    async def _extract_small_document(
        self,
        text: str,
        metadata_context: str | None = None,
    ) -> PatternExtractionResult:
        """Standard 3-pass extraction for small documents.

        Args:
            text: Paper content text.
            metadata_context: Optional metadata context.

        Returns:
            PatternExtractionResult.
        """
        warnings: list[str] = []
        doc_tokens = len(text) // 4  # Rough estimate
        logger.info(f"Document size: ~{doc_tokens:,} tokens (no chunking needed)")

        # Pass 1: Evidence Inventory (Haiku)
        try:
            evidence_inventory = await self._extract_evidence(text)
            item_count = len(evidence_inventory.evidence_items)
            logger.info(f"Pass 1 complete: {item_count} evidence items")
        except LLMError as e:
            logger.error(f"Pass 1 failed: {e}")
            return self._failed_result(str(e))

        # Pass 2: Pattern Extraction (Sonnet)
        try:
            extraction_result = await self._extract_patterns(
                text, evidence_inventory, metadata_context
            )
            logger.info(f"Pass 2 complete: {len(extraction_result.patterns)} patterns")
        except LLMError as e:
            logger.error(f"Pass 2 failed: {e}")
            return self._failed_result(str(e), evidence_inventory)

        # Pass 3: Verification (Haiku)
        try:
            verification_result = await self._verify_extraction(
                evidence_inventory, extraction_result
            )
            logger.info(
                f"Pass 3 complete: status={verification_result.verification_status}"
            )
        except LLMError as e:
            logger.warning(f"Pass 3 failed: {e}")
            # Verification failure is non-fatal - return patterns with warning
            warnings.append(f"Verification failed: {e}")
            verification_result = VerificationResult(
                verification_status=VerificationStatus.warn,
                improvement_suggestions=[
                    "Verification pass failed - manual review recommended"
                ],
            )

        # Build warnings from verification
        warnings.extend(self._build_warnings(verification_result))

        return PatternExtractionResult(
            patterns=extraction_result,
            evidence_inventory=evidence_inventory,
            verification=verification_result,
            warnings=warnings,
            was_chunked=False,
            chunk_count=1,
        )

    async def _extract_large_document(
        self,
        chunks: list[DocumentChunk],
        document_outline: str,
        metadata_context: str | None = None,
    ) -> PatternExtractionResult:
        """Chunked extraction for large documents.

        Phase A: Extract from each chunk (Pass 1 + 2)
        Phase B: Merge and deduplicate patterns
        Phase C: Final synthesis pass

        Args:
            chunks: Document chunks.
            document_outline: Outline of document structure.
            metadata_context: Optional metadata context.

        Returns:
            PatternExtractionResult.
        """
        warnings: list[str] = []
        chunk_count = len(chunks)
        logger.info(f"Processing large document in {chunk_count} chunks...")
        warnings.append(f"Document processed in {chunk_count} chunks")

        # Phase A: Extract from chunks with concurrency limit
        semaphore = asyncio.Semaphore(self._max_concurrent_chunks)

        async def extract_with_limit(chunk: DocumentChunk) -> dict[str, Any]:
            async with semaphore:
                return await self._extract_chunk(chunk, document_outline)

        chunk_results = await asyncio.gather(
            *[extract_with_limit(chunk) for chunk in chunks],
            return_exceptions=True,
        )

        # Collect all patterns and evidence
        all_patterns: list[ExtractedPatternRaw] = []
        all_evidence_inventories: list[EvidenceInventory] = []
        extraction_errors: list[str] = []

        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                extraction_errors.append(f"Chunk {i + 1} failed: {result}")
                continue
            all_patterns.extend(result.get("patterns", []))
            if result.get("evidence_inventory"):
                all_evidence_inventories.append(result["evidence_inventory"])

        if extraction_errors:
            warnings.extend(extraction_errors)

        if not all_patterns:
            return self._failed_result(
                f"No patterns extracted from {chunk_count} chunks"
            )

        # Phase B: Merge and deduplicate
        logger.info(
            f"Merging {len(all_patterns)} patterns from {chunk_count} chunks..."
        )
        merged_patterns = self._merge_patterns(all_patterns)
        logger.info(f"After deduplication: {len(merged_patterns)} patterns")

        # Phase C: Final synthesis (identify semantic duplicates)
        logger.info("Running synthesis pass...")
        synthesis_result = await self._synthesize_patterns(
            merged_patterns, document_outline
        )

        # Apply synthesis decisions
        final_patterns = self._apply_synthesis(merged_patterns, synthesis_result)
        logger.info(f"After synthesis: {len(final_patterns)} patterns")

        # Combine evidence inventories
        combined_evidence = self._merge_evidence_inventories(all_evidence_inventories)

        # Build final result
        extraction_result = ExtractionResult(
            patterns=final_patterns,
            gaps_identified=synthesis_result.get("gaps", []),
        )

        # Skip verification for chunked docs - synthesis serves similar purpose
        verification_result = VerificationResult(
            verification_status=VerificationStatus.warn,
            overall_quality="medium",
            improvement_suggestions=[
                "Chunked extraction - patterns may benefit from manual review"
            ],
        )

        return PatternExtractionResult(
            patterns=extraction_result,
            evidence_inventory=combined_evidence,
            verification=verification_result,
            warnings=warnings,
            was_chunked=True,
            chunk_count=chunk_count,
        )

    async def _extract_chunk(
        self,
        chunk: DocumentChunk,
        document_outline: str,
    ) -> dict[str, Any]:
        """Extract patterns from a single chunk.

        Runs Pass 1 + 2 only (verification skipped for chunks).

        Args:
            chunk: Document chunk.
            document_outline: Full document outline for context.

        Returns:
            Dict with patterns and evidence_inventory.
        """
        logger.info(f"Extracting chunk {chunk.chunk_index + 1}/{chunk.total_chunks}...")

        # Build context for this chunk
        chunk_context = f"""## Document Context
{document_outline}

## Current Section
Chunk {chunk.chunk_index + 1} of {chunk.total_chunks}
Sections: {", ".join(chunk.section_titles) if chunk.section_titles else "N/A"}

## Content
{chunk.text}
"""

        # Pass 1: Extract evidence inventory
        evidence_inventory = await self._extract_evidence(chunk_context)

        # Pass 2: Extract patterns with citations
        extraction = await self._extract_patterns(chunk_context, evidence_inventory)

        return {
            "patterns": extraction.patterns,
            "evidence_inventory": evidence_inventory,
        }

    async def _extract_evidence(self, text: str) -> EvidenceInventory:
        """Pass 1: Extract evidence inventory from paper."""
        client = self._ensure_haiku()

        user_prompt = EVIDENCE_INVENTORY_USER.format(text=text)

        response = await client.complete_json(
            system_prompt=EVIDENCE_INVENTORY_SYSTEM,
            user_prompt=user_prompt,
            temperature=EVIDENCE_TEMP,
        )

        return parse_evidence_inventory(response)

    async def _extract_patterns(
        self,
        text: str,
        evidence: EvidenceInventory,
        metadata_context: str | None = None,
    ) -> ExtractionResult:
        """Pass 2: Extract patterns with evidence citations."""
        client = self._ensure_sonnet()

        # Format evidence inventory for prompt
        evidence_str = self._format_evidence_inventory(evidence)

        # Add metadata context if provided
        full_text = text
        if metadata_context:
            full_text = f"[Metadata: {metadata_context}]\n\n{text}"

        user_prompt = EXTRACTION_USER.format(
            evidence_inventory=evidence_str,
            text=full_text,
        )

        response = await client.complete_json(
            system_prompt=EXTRACTION_SYSTEM,
            user_prompt=user_prompt,
            temperature=EXTRACTION_TEMP,
        )

        return parse_extraction_result(response)

    async def _verify_extraction(
        self,
        evidence: EvidenceInventory,
        extraction: ExtractionResult,
    ) -> VerificationResult:
        """Pass 3: Verify extraction quality."""
        client = self._ensure_haiku()

        # Format inputs for verification prompt
        evidence_str = self._format_evidence_inventory(evidence)
        patterns_str = self._format_patterns(extraction)

        user_prompt = VERIFICATION_USER.format(
            evidence_inventory=evidence_str,
            patterns=patterns_str,
        )

        response = await client.complete_json(
            system_prompt=VERIFICATION_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=4096,  # Verification output is small, but add margin for safety
            temperature=VERIFICATION_TEMP,
        )

        result = parse_verification_result(response)

        # Validate/override LLM status with computed status
        computed_status = compute_verification_status(result)
        if computed_status != result.verification_status:
            logger.info(
                f"Overriding LLM status {result.verification_status} "
                f"with computed status {computed_status}"
            )
            result.verification_status = computed_status

        return result

    async def _synthesize_patterns(
        self,
        patterns: list[ExtractedPatternRaw],
        document_outline: str,
    ) -> dict[str, Any]:
        """Synthesis pass for chunked documents.

        Reviews patterns for cross-chunk connections and redundancy.

        Args:
            patterns: Merged patterns from all chunks.
            document_outline: Document structure overview.

        Returns:
            Synthesis result with merge/remove decisions.
        """
        if not patterns:
            return {"merge": [], "remove": [], "gaps": []}

        # Summarize patterns to reduce prompt size
        def fmt_pattern(p: ExtractedPatternRaw) -> str:
            claim = p.claim[:80] + "..." if len(p.claim) > 80 else p.claim
            return f"- {p.name}: {claim}"

        pattern_list = "\n".join(fmt_pattern(p) for p in patterns)
        count = len(patterns)

        synthesis_prompt = f"""## Patterns from chunked extraction ({count} total)
{pattern_list}

## Task
Identify which patterns should be MERGED (similar concepts) or REMOVED (duplicates).

Return JSON:
```json
{{
  "merge": [{{"keep": "Pattern A name", "remove": "Pattern B name"}}],
  "remove": ["Pattern C name"],
  "gaps": ["Topics not covered that might be important"]
}}
```

If no changes needed, return: {{"merge": [], "remove": [], "gaps": []}}"""

        try:
            client = self._ensure_haiku()
            synthesis_system = (
                "You identify duplicate or similar patterns "
                "from chunked document extraction."
            )
            result = await client.complete_json(
                system_prompt=synthesis_system,
                user_prompt=synthesis_prompt,
                max_tokens=1024,
                temperature=0.1,
            )
            return result
        except LLMError as e:
            logger.warning(f"Synthesis failed ({e}), skipping")
            return {"merge": [], "remove": [], "gaps": []}

    def _merge_patterns(
        self,
        patterns: list[ExtractedPatternRaw],
    ) -> list[ExtractedPatternRaw]:
        """Merge patterns from multiple chunks, removing exact duplicates.

        Args:
            patterns: All patterns from chunks.

        Returns:
            Deduplicated pattern list.
        """
        seen_names: set[str] = set()
        unique_patterns: list[ExtractedPatternRaw] = []

        for pattern in patterns:
            name = pattern.name.lower().strip() if pattern.name else ""
            if name and name not in seen_names:
                seen_names.add(name)
                unique_patterns.append(pattern)
            elif not name:
                # Keep patterns without names
                unique_patterns.append(pattern)

        return unique_patterns

    def _apply_synthesis(
        self,
        patterns: list[ExtractedPatternRaw],
        synthesis_result: dict[str, Any],
    ) -> list[ExtractedPatternRaw]:
        """Apply synthesis decisions to pattern list.

        Args:
            patterns: Deduplicated patterns.
            synthesis_result: Synthesis pass output.

        Returns:
            Final filtered pattern list.
        """
        # Build set of patterns to remove
        remove_set: set[str] = set()
        for name in synthesis_result.get("remove", []):
            if isinstance(name, str):
                remove_set.add(name.lower().strip())

        for merge in synthesis_result.get("merge", []):
            if isinstance(merge, dict) and merge.get("remove"):
                remove_set.add(merge["remove"].lower().strip())

        # Filter patterns
        final_patterns = []
        for pattern in patterns:
            name = pattern.name.lower().strip() if pattern.name else ""
            if name not in remove_set:
                final_patterns.append(pattern)

        return final_patterns

    def _merge_evidence_inventories(
        self,
        inventories: list[EvidenceInventory],
    ) -> EvidenceInventory:
        """Merge evidence inventories from multiple chunks.

        Args:
            inventories: Evidence inventories from each chunk.

        Returns:
            Combined evidence inventory.
        """
        if not inventories:
            return EvidenceInventory(paper_title="Unknown")

        if len(inventories) == 1:
            return inventories[0]

        # Use first non-empty title
        paper_title = "Unknown"
        for inv in inventories:
            if inv.paper_title and inv.paper_title != "Unknown":
                paper_title = inv.paper_title
                break

        # Combine evidence items with chunk-prefixed IDs
        combined_items: list[EvidenceItem] = []
        for i, inv in enumerate(inventories):
            for item in inv.evidence_items:
                # Prefix ID with chunk number (e.g., E1 -> C1-E1)
                prefixed_item = EvidenceItem(
                    id=f"C{i + 1}-{item.id}",
                    quote=item.quote,
                    location=item.location,
                    type=item.type,
                    importance=item.importance,
                )
                combined_items.append(prefixed_item)

        return EvidenceInventory(
            paper_title=paper_title,
            evidence_items=combined_items,
        )

    def _format_evidence_inventory(self, evidence: EvidenceInventory) -> str:
        """Format evidence inventory for inclusion in prompts."""
        lines = [f"Paper: {evidence.paper_title}", ""]
        for item in evidence.evidence_items:
            lines.append(f"[{item.id}] ({item.type}, {item.importance})")
            lines.append(f"  Location: {item.location}")
            lines.append(f'  Quote: "{item.quote}"')
            lines.append("")
        return "\n".join(lines)

    def _format_patterns(self, extraction: ExtractionResult) -> str:
        """Format extracted patterns for verification prompt."""
        patterns_data = []
        for p in extraction.patterns:
            patterns_data.append(
                {
                    "rank": p.rank,
                    "name": p.name,
                    "claim": p.claim,
                    "evidence": p.evidence,
                    "context": p.context,
                    "evidence_ids": p.evidence_ids,
                }
            )
        return json.dumps(patterns_data, indent=2)

    def _build_warnings(self, verification: VerificationResult) -> list[str]:
        """Build warning messages from verification result."""
        warnings = []

        # Add citation issues as warnings
        for issue in verification.citation_issues:
            warnings.append(
                f"Pattern {issue.pattern_rank} ({issue.field}): {issue.issue}"
            )

        # Add accuracy concerns as warnings
        for concern in verification.accuracy_concerns:
            msg = f"Pattern {concern.pattern_rank} [{concern.evidence_id}]: "
            msg += concern.concern
            warnings.append(msg)

        # Add unused evidence warning if significant
        if len(verification.unused_evidence) > 3:
            warnings.append(
                f"Unused evidence: {', '.join(verification.unused_evidence[:5])}"
                + ("..." if len(verification.unused_evidence) > 5 else "")
            )

        return warnings

    def _failed_result(
        self,
        error_message: str,
        evidence: EvidenceInventory | None = None,
    ) -> PatternExtractionResult:
        """Create a failed extraction result."""
        return PatternExtractionResult(
            patterns=ExtractionResult(patterns=[], gaps_identified=[error_message]),
            evidence_inventory=evidence or EvidenceInventory(paper_title="Unknown"),
            verification=VerificationResult(
                verification_status=VerificationStatus.fail,
                improvement_suggestions=[f"Extraction failed: {error_message}"],
            ),
            warnings=[f"Extraction failed: {error_message}"],
        )
