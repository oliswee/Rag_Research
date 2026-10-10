"""Pattern extraction prompt (Pass 2 of 3-pass pipeline)."""

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

# ruff: noqa: E501


class PatternConfidence(StrEnum):
    """Confidence levels for extracted patterns."""

    high = "high"
    medium = "medium"
    low = "low"


class ExtractedPatternRaw(BaseModel):
    """A pattern extracted from a paper with evidence citations.

    This is the raw extraction result before database persistence.
    """

    rank: int = Field(..., ge=1, description="Pattern rank (1 = most important)")
    name: str = Field(
        ..., min_length=1, max_length=100, description="Short name (3-6 words)"
    )
    claim: str | None = Field(
        default=None, description="Main assertion with [E#] citations"
    )
    evidence: str | None = Field(
        default=None, description="Supporting data with [E#] citations"
    )
    context: str | None = Field(
        default=None, description="Why this matters with [E#] citations"
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    evidence_ids: list[str] = Field(
        default_factory=list, description="All [E#] IDs used"
    )
    confidence: PatternConfidence = Field(
        default=PatternConfidence.medium, description="Extraction confidence"
    )

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        """Strip whitespace from name."""
        return v.strip()

    @property
    def has_content(self) -> bool:
        """Return True if at least one content field is non-empty."""
        return bool(
            (self.claim and self.claim.strip())
            or (self.evidence and self.evidence.strip())
            or (self.context and self.context.strip())
        )

    @property
    def has_citations(self) -> bool:
        """Return True if pattern has any evidence citations."""
        return len(self.evidence_ids) > 0

    @property
    def content_fields(self) -> list[str]:
        """Return list of populated content field names."""
        fields = []
        if self.claim and self.claim.strip():
            fields.append("claim")
        if self.evidence and self.evidence.strip():
            fields.append("evidence")
        if self.context and self.context.strip():
            fields.append("context")
        return fields


class ExtractionResult(BaseModel):
    """Result of pattern extraction (Pass 2)."""

    patterns: list[ExtractedPatternRaw] = Field(
        default_factory=list, description="Extracted patterns"
    )
    total_evidence_used: int = Field(
        default=0, description="Count of unique evidence items cited"
    )
    gaps_identified: list[str] = Field(
        default_factory=list, description="Any gaps noted during extraction"
    )


EXTRACTION_SYSTEM = """You are extracting research patterns grounded in evidence.

## Your Task
Extract 3-8 patterns from the research paper. Each pattern must cite evidence from the provided evidence inventory using [E#] format.

## Claim/Evidence/Context Framework

### Claim
The main insight, assertion, or finding.
**Key question:** What is the author claiming or demonstrating?
MUST cite evidence using [E#] format.

### Evidence
Supporting quotes, data, examples from the source.
**Key question:** What proof or examples support this claim?
MUST cite evidence using [E#] format.

### Context
Implications, limitations, connections to other ideas.
**Key question:** Why does this matter? What are the boundaries?
MUST cite evidence using [E#] format.

## CRITICAL RULES
1. **Every assertion MUST cite evidence** using [E#] format (e.g., [E1], [E3, E5])
2. **Claim/Evidence/Context must describe DIFFERENT aspects** - never repeat the same content
3. **No claim without evidence** - if no evidence supports a claim, use null for that field
4. Extract **3-8 patterns** per paper, ranked by importance
5. Each pattern needs a **short name** (3-6 words)
6. Use only evidence IDs from the provided inventory

## Available Tags
Use these tags to categorize patterns:
- methodology: Research methods or techniques
- finding: Core research results
- limitation: Weaknesses or boundary conditions
- future-work: Suggested next steps
- experimental-setup: How experiments were conducted
- theoretical: Conceptual frameworks
- practical: Real-world applications

## Output Format
Return ONLY a JSON object:
{
    "patterns": [
        {
            "rank": 1,
            "name": "Pattern Name (3-6 words)",
            "claim": "Main assertion citing evidence [E1, E3]",
            "evidence": "Supporting data citing evidence [E2, E5]",
            "context": "Why this matters citing evidence [E4]",
            "tags": ["methodology", "finding"],
            "evidence_ids": ["E1", "E2", "E3", "E4", "E5"],
            "confidence": "high|medium|low"
        }
    ],
    "total_evidence_used": 12,
    "gaps_identified": ["No evidence for claim X"]
}

## Guidelines
- Rank patterns by importance (1 = most important)
- Confidence: "high" if well-supported, "medium" if partially supported, "low" if weakly supported
- List ALL evidence IDs used in each pattern in the evidence_ids array
- Identify gaps where claims lack sufficient evidence support
"""

EXTRACTION_USER = """Extract patterns from this research paper using the evidence inventory below.

## Evidence Inventory
{evidence_inventory}

## Paper Text
{text}

Extract 3-8 patterns with Claim/Evidence/Context, citing evidence using [E#] format."""


# Regex to find evidence citations like [E1], [E2, E3], [E1, E2, E3]
EVIDENCE_CITATION_PATTERN = re.compile(r"\[E(\d+)(?:\s*,\s*E(\d+))*\]")


def extract_evidence_ids(text: str | None) -> list[str]:
    """Extract all [E#] evidence IDs from text.

    Args:
        text: Text containing evidence citations like [E1], [E2, E3].

    Returns:
        List of unique evidence IDs found (e.g., ["E1", "E2", "E3"]).
    """
    if not text:
        return []

    ids: set[str] = set()
    # Find all E followed by digits
    for match in re.finditer(r"E(\d+)", text):
        ids.add(f"E{match.group(1)}")

    return sorted(ids, key=lambda x: int(x[1:]))


def parse_extraction_result(response: dict) -> ExtractionResult:
    """Parse LLM response into ExtractionResult.

    Args:
        response: Raw JSON response from LLM.

    Returns:
        Validated ExtractionResult object.
    """
    patterns_raw = response.get("patterns", [])
    if not isinstance(patterns_raw, list):
        patterns_raw = []

    parsed_patterns = []
    for item in patterns_raw:
        if not isinstance(item, dict):
            continue

        # Skip items without name (required field)
        name = item.get("name", "").strip()
        if not name:
            continue

        # Parse confidence
        confidence_str = item.get("confidence", "medium")
        try:
            confidence = PatternConfidence(confidence_str)
        except ValueError:
            confidence = PatternConfidence.medium

        # Parse tags
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t) for t in tags if t]

        # Extract evidence IDs from all content fields
        claim = item.get("claim")
        evidence = item.get("evidence")
        context = item.get("context")

        # Get evidence_ids from response or extract from text
        evidence_ids = item.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []

        # Also extract any IDs mentioned in the text that weren't listed
        all_ids = set(evidence_ids)
        all_ids.update(extract_evidence_ids(claim))
        all_ids.update(extract_evidence_ids(evidence))
        all_ids.update(extract_evidence_ids(context))

        parsed_patterns.append(
            ExtractedPatternRaw(
                rank=item.get("rank", len(parsed_patterns) + 1),
                name=name,
                claim=claim if claim and claim.strip() else None,
                evidence=evidence if evidence and evidence.strip() else None,
                context=context if context and context.strip() else None,
                tags=tags,
                evidence_ids=sorted(
                    all_ids, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
                ),
                confidence=confidence,
            )
        )

    return ExtractionResult(
        patterns=parsed_patterns,
        total_evidence_used=response.get("total_evidence_used", 0),
        gaps_identified=response.get("gaps_identified", []),
    )
