"""Evidence inventory extraction prompt (Pass 1 of 3-pass pipeline)."""

from enum import StrEnum

from pydantic import BaseModel, Field

# ruff: noqa: E501


class EvidenceType(StrEnum):
    """Types of evidence that can be extracted."""

    finding = "finding"
    definition = "definition"
    mechanism = "mechanism"
    limitation = "limitation"
    comparison = "comparison"
    claim = "claim"
    methodology = "methodology"
    result = "result"


class EvidenceImportance(StrEnum):
    """Importance levels for evidence items."""

    high = "high"
    medium = "medium"
    low = "low"


class EvidenceItem(BaseModel):
    """A single piece of evidence extracted from a paper."""

    id: str = Field(..., description="Unique identifier (E1, E2, etc.)")
    quote: str = Field(..., description="Exact quote from the paper")
    location: str = Field(..., description="Section name or location indicator")
    type: EvidenceType = Field(..., description="Type of evidence")
    importance: EvidenceImportance = Field(
        default=EvidenceImportance.medium, description="Importance level"
    )


class EvidenceInventory(BaseModel):
    """Complete evidence inventory from a paper."""

    paper_title: str = Field(..., description="Title of the paper")
    evidence_items: list[EvidenceItem] = Field(
        default_factory=list, description="List of evidence items"
    )
    paper_type: str = Field(
        default="empirical", description="Type of paper (empirical, theoretical, etc.)"
    )
    core_contribution_summary: str = Field(
        default="", description="1-2 sentence summary of core contribution"
    )


EVIDENCE_INVENTORY_SYSTEM = """You are extracting concrete evidence from a research paper.

## Your Task
Extract 12-25 distinct pieces of evidence from the paper. Each piece must be a direct quote or specific finding that can be cited later.

## What Counts as Evidence
- Direct quotes from the paper (use exact wording)
- Quantitative results (numbers, percentages, measurements)
- Experimental findings with specific outcomes
- Defined terms or concepts with their definitions
- Methodology descriptions with specific details
- Limitations or failure modes explicitly mentioned
- Comparisons between approaches with specific metrics

## What Does NOT Count as Evidence
- Your interpretations or inferences
- General statements without specifics
- Background information not central to the paper's contribution
- Vague claims without supporting details

## Evidence Types
- **finding**: A research result or discovery
- **definition**: A term or concept being defined
- **mechanism**: How something works
- **limitation**: A weakness or boundary condition
- **comparison**: Comparison between methods/approaches
- **claim**: An assertion made by the authors
- **methodology**: A specific method or technique description
- **result**: A quantitative or qualitative outcome

## Output Format
Return ONLY a JSON object:
{
    "paper_title": "Title of the paper",
    "evidence_items": [
        {
            "id": "E1",
            "quote": "exact quote from paper",
            "location": "section name (e.g., Abstract, Methods, Results)",
            "type": "finding|definition|mechanism|limitation|comparison|claim|methodology|result",
            "importance": "high|medium|low"
        }
    ],
    "paper_type": "empirical|theoretical|system|survey|position",
    "core_contribution_summary": "1-2 sentence summary of the paper's main contribution"
}

## Guidelines
- Use sequential IDs: E1, E2, E3, etc.
- Quote exactly - do not paraphrase
- Mark importance as "high" for core findings, "medium" for supporting evidence, "low" for minor details
- Aim for 12-25 evidence items to ensure comprehensive coverage
- Focus on the paper's novel contributions, not background/related work
"""

EVIDENCE_INVENTORY_USER = """Extract evidence from this research paper:

{text}

Return a JSON evidence inventory with 12-25 evidence items."""


def parse_evidence_inventory(response: dict) -> EvidenceInventory:
    """Parse LLM response into EvidenceInventory.

    Args:
        response: Raw JSON response from LLM.

    Returns:
        Validated EvidenceInventory object.

    Raises:
        ValueError: If response cannot be parsed.
    """
    # Handle missing or malformed evidence_items
    evidence_items = response.get("evidence_items", [])
    if not isinstance(evidence_items, list):
        evidence_items = []

    parsed_items = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue

        # Skip items missing required fields
        if not item.get("id") or not item.get("quote"):
            continue

        # Parse type with fallback
        type_str = item.get("type", "finding")
        try:
            evidence_type = EvidenceType(type_str)
        except ValueError:
            evidence_type = EvidenceType.finding

        # Parse importance with fallback
        importance_str = item.get("importance", "medium")
        try:
            importance = EvidenceImportance(importance_str)
        except ValueError:
            importance = EvidenceImportance.medium

        parsed_items.append(
            EvidenceItem(
                id=item["id"],
                quote=item["quote"],
                location=item.get("location", "Unknown"),
                type=evidence_type,
                importance=importance,
            )
        )

    return EvidenceInventory(
        paper_title=response.get("paper_title", "Unknown"),
        evidence_items=parsed_items,
        paper_type=response.get("paper_type", "empirical"),
        core_contribution_summary=response.get("core_contribution_summary", ""),
    )
