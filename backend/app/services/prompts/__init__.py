"""LLM prompt templates for extraction and synthesis."""

from app.services.prompts.evidence_inventory import (
    EVIDENCE_INVENTORY_SYSTEM,
    EVIDENCE_INVENTORY_USER,
    EvidenceImportance,
    EvidenceInventory,
    EvidenceItem,
    EvidenceType,
    parse_evidence_inventory,
)
from app.services.prompts.extraction import (
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
    ExtractedPatternRaw,
    ExtractionResult,
    PatternConfidence,
    extract_evidence_ids,
    parse_extraction_result,
)
from app.services.prompts.metadata import (
    METADATA_EXTRACTION_SYSTEM,
    METADATA_EXTRACTION_USER,
)
from app.services.prompts.synthesis import (
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
    PatternContext,
    SourceReference,
    SynthesisResult,
    build_conversation_history,
    build_pattern_context,
    parse_synthesis_result,
    sanitize_markdown,
)
from app.services.prompts.verification import (
    VERIFICATION_SYSTEM,
    VERIFICATION_USER,
    AccuracyConcern,
    CitationIssue,
    IssueSeverity,
    VerificationResult,
    VerificationStatus,
    compute_verification_status,
    parse_verification_result,
)

__all__ = [
    # Evidence Inventory (Pass 1)
    "EVIDENCE_INVENTORY_SYSTEM",
    "EVIDENCE_INVENTORY_USER",
    "EvidenceImportance",
    "EvidenceInventory",
    "EvidenceItem",
    "EvidenceType",
    "parse_evidence_inventory",
    # Pattern Extraction (Pass 2)
    "EXTRACTION_SYSTEM",
    "EXTRACTION_USER",
    "ExtractionResult",
    "ExtractedPatternRaw",
    "PatternConfidence",
    "extract_evidence_ids",
    "parse_extraction_result",
    # Verification (Pass 3)
    "VERIFICATION_SYSTEM",
    "VERIFICATION_USER",
    "AccuracyConcern",
    "CitationIssue",
    "IssueSeverity",
    "VerificationResult",
    "VerificationStatus",
    "compute_verification_status",
    "parse_verification_result",
    # Metadata Extraction
    "METADATA_EXTRACTION_SYSTEM",
    "METADATA_EXTRACTION_USER",
    # Synthesis
    "SYNTHESIS_SYSTEM",
    "SYNTHESIS_USER",
    "PatternContext",
    "SourceReference",
    "SynthesisResult",
    "build_conversation_history",
    "build_pattern_context",
    "parse_synthesis_result",
    "sanitize_markdown",
]
