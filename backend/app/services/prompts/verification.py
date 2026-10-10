"""Verification prompt (Pass 3 of 3-pass pipeline)."""

from enum import StrEnum

from pydantic import BaseModel, Field

# ruff: noqa: E501


class VerificationStatus(StrEnum):
    """Overall verification result status."""

    passed = "pass"
    warn = "warn"
    fail = "fail"


class IssueSeverity(StrEnum):
    """Severity level for verification issues."""

    error = "error"
    warning = "warning"


class CitationIssue(BaseModel):
    """A single citation issue found during verification."""

    pattern_rank: int = Field(..., description="Rank of the pattern with the issue")
    field: str = Field(..., description="Field with the issue (claim/evidence/context)")
    issue: str = Field(..., description="Description of the issue")
    severity: IssueSeverity = Field(
        default=IssueSeverity.warning, description="Issue severity"
    )


class AccuracyConcern(BaseModel):
    """A concern about citation accuracy."""

    pattern_rank: int = Field(..., description="Rank of the pattern")
    evidence_id: str = Field(..., description="Evidence ID in question")
    concern: str = Field(..., description="Description of the accuracy concern")


class VerificationResult(BaseModel):
    """Result of extraction verification (Pass 3)."""

    verification_status: VerificationStatus = Field(
        default=VerificationStatus.passed, description="Overall verification result"
    )
    citation_issues: list[CitationIssue] = Field(
        default_factory=list, description="Issues with citations"
    )
    unused_evidence: list[str] = Field(
        default_factory=list, description="Evidence IDs not cited in any pattern"
    )
    accuracy_concerns: list[AccuracyConcern] = Field(
        default_factory=list, description="Concerns about citation accuracy"
    )
    overall_quality: str = Field(
        default="medium", description="Overall extraction quality (high/medium/low)"
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for improving extraction"
    )

    @property
    def has_errors(self) -> bool:
        """Return True if any error-level issues exist."""
        return any(
            issue.severity == IssueSeverity.error for issue in self.citation_issues
        )

    @property
    def has_warnings(self) -> bool:
        """Return True if any warning-level citation issues exist."""
        return any(
            issue.severity == IssueSeverity.warning for issue in self.citation_issues
        )

    @property
    def issue_count(self) -> int:
        """Return total number of issues."""
        return len(self.citation_issues) + len(self.accuracy_concerns)


VERIFICATION_SYSTEM = """You are verifying the quality of pattern extraction from a research paper.

## Your Task
Review the extracted patterns against the evidence inventory and identify any issues.

## Checks to Perform

### 1. Citation Validity
- Does each [E#] reference exist in the evidence inventory?
- Flag any citations to non-existent evidence as errors.

### 2. Citation Accuracy
- Does the claim/evidence/context accurately represent what the cited evidence says?
- Flag any misrepresentations or overstatements.

### 3. Evidence Coverage
- Which evidence items from the inventory were NOT used?
- High-importance evidence that isn't cited may indicate gaps.

### 4. Component Separation
- Are claim/evidence/context truly different content?
- Flag if the same content is repeated across fields.

## Output Format
Return ONLY a JSON object with EXACTLY these fields (no additional fields):
{
    "verification_status": "pass|warn|fail",
    "citation_issues": [
        {
            "pattern_rank": 1,
            "field": "claim|evidence|context",
            "issue": "E99 does not exist in evidence inventory",
            "severity": "error|warning"
        }
    ],
    "unused_evidence": ["E8", "E12"],
    "accuracy_concerns": [
        {
            "pattern_rank": 1,
            "evidence_id": "E3",
            "concern": "Pattern claims X but evidence only shows Y"
        }
    ],
    "overall_quality": "high|medium|low",
    "improvement_suggestions": ["Consider citing E8 for additional support"]
}

CRITICAL INSTRUCTIONS:
- Return ONLY the JSON object, nothing else
- Do NOT add any text, commentary, or observations before or after the JSON
- Do NOT add any fields beyond those shown above
- Keep improvement_suggestions concise (max 3 items, one sentence each)
- Your entire response must be valid JSON that can be parsed directly

## Verification Status Guidelines
- **pass**: No errors, few or no warnings, good evidence coverage
- **warn**: No errors but has warnings OR significant unused evidence
- **fail**: Has errors (invalid citations) OR major accuracy concerns
"""

VERIFICATION_USER = """Verify this pattern extraction against the evidence inventory.

## Evidence Inventory
{evidence_inventory}

## Extracted Patterns
{patterns}

Check for citation validity, accuracy, and coverage. Return verification results as JSON."""


def parse_verification_result(response: dict) -> VerificationResult:
    """Parse LLM response into VerificationResult.

    Args:
        response: Raw JSON response from LLM.

    Returns:
        Validated VerificationResult object.
    """
    # Parse verification status
    status_str = response.get("verification_status", "pass")
    try:
        status = VerificationStatus(status_str)
    except ValueError:
        status = VerificationStatus.passed

    # Parse citation issues
    citation_issues_raw = response.get("citation_issues", [])
    if not isinstance(citation_issues_raw, list):
        citation_issues_raw = []

    citation_issues = []
    for item in citation_issues_raw:
        if not isinstance(item, dict):
            continue
        if not item.get("pattern_rank") or not item.get("issue"):
            continue

        severity_str = item.get("severity", "warning")
        try:
            severity = IssueSeverity(severity_str)
        except ValueError:
            severity = IssueSeverity.warning

        citation_issues.append(
            CitationIssue(
                pattern_rank=item["pattern_rank"],
                field=item.get("field", "unknown"),
                issue=item["issue"],
                severity=severity,
            )
        )

    # Parse unused evidence
    unused_evidence = response.get("unused_evidence", [])
    if not isinstance(unused_evidence, list):
        unused_evidence = []
    unused_evidence = [str(e) for e in unused_evidence if e]

    # Parse accuracy concerns
    accuracy_concerns_raw = response.get("accuracy_concerns", [])
    if not isinstance(accuracy_concerns_raw, list):
        accuracy_concerns_raw = []

    accuracy_concerns = []
    for item in accuracy_concerns_raw:
        if not isinstance(item, dict):
            continue
        if not item.get("pattern_rank") or not item.get("concern"):
            continue

        accuracy_concerns.append(
            AccuracyConcern(
                pattern_rank=item["pattern_rank"],
                evidence_id=item.get("evidence_id", "unknown"),
                concern=item["concern"],
            )
        )

    # Parse overall quality
    quality = response.get("overall_quality", "medium")
    if quality not in ("high", "medium", "low"):
        quality = "medium"

    # Parse improvement suggestions
    suggestions = response.get("improvement_suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []
    suggestions = [str(s) for s in suggestions if s]

    return VerificationResult(
        verification_status=status,
        citation_issues=citation_issues,
        unused_evidence=unused_evidence,
        accuracy_concerns=accuracy_concerns,
        overall_quality=quality,
        improvement_suggestions=suggestions,
    )


def compute_verification_status(result: VerificationResult) -> VerificationStatus:
    """Compute verification status from result contents.

    This can be used to override or validate LLM-provided status.

    Args:
        result: VerificationResult to analyze.

    Returns:
        Computed VerificationStatus.
    """
    # Fail if any error-level issues
    if result.has_errors:
        return VerificationStatus.fail

    # Fail if many accuracy concerns
    if len(result.accuracy_concerns) >= 3:
        return VerificationStatus.fail

    # Warn if any warnings or unused high-importance evidence
    if result.has_warnings:
        return VerificationStatus.warn

    # Warn if significant unused evidence (more than 30%)
    if len(result.unused_evidence) > 5:
        return VerificationStatus.warn

    return VerificationStatus.passed
