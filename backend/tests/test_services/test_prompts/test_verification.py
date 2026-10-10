"""Tests for verification prompt and parsing (Pass 3)."""

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


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """All verification statuses should be defined."""
        assert VerificationStatus.passed == "pass"
        assert VerificationStatus.warn == "warn"
        assert VerificationStatus.fail == "fail"


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_all_severities_exist(self) -> None:
        """All severity levels should be defined."""
        assert IssueSeverity.error == "error"
        assert IssueSeverity.warning == "warning"


class TestCitationIssue:
    """Tests for CitationIssue model."""

    def test_create_valid_issue(self) -> None:
        """Should create issue with all fields."""
        issue = CitationIssue(
            pattern_rank=1,
            field="claim",
            issue="E99 does not exist",
            severity=IssueSeverity.error,
        )
        assert issue.pattern_rank == 1
        assert issue.field == "claim"
        assert issue.issue == "E99 does not exist"
        assert issue.severity == IssueSeverity.error

    def test_default_severity_is_warning(self) -> None:
        """Should default to warning severity."""
        issue = CitationIssue(
            pattern_rank=1,
            field="evidence",
            issue="Minor issue",
        )
        assert issue.severity == IssueSeverity.warning


class TestAccuracyConcern:
    """Tests for AccuracyConcern model."""

    def test_create_valid_concern(self) -> None:
        """Should create concern with all fields."""
        concern = AccuracyConcern(
            pattern_rank=2,
            evidence_id="E5",
            concern="Pattern overstates the evidence",
        )
        assert concern.pattern_rank == 2
        assert concern.evidence_id == "E5"
        assert "overstates" in concern.concern


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_create_valid_result(self) -> None:
        """Should create result with all fields."""
        result = VerificationResult(
            verification_status=VerificationStatus.passed,
            citation_issues=[],
            unused_evidence=["E10", "E11"],
            accuracy_concerns=[],
            overall_quality="high",
            improvement_suggestions=["Consider citing E10"],
        )
        assert result.verification_status == VerificationStatus.passed
        assert result.unused_evidence == ["E10", "E11"]
        assert result.overall_quality == "high"

    def test_default_values(self) -> None:
        """Should use defaults for all fields."""
        result = VerificationResult()
        assert result.verification_status == VerificationStatus.passed
        assert result.citation_issues == []
        assert result.unused_evidence == []
        assert result.accuracy_concerns == []
        assert result.overall_quality == "medium"
        assert result.improvement_suggestions == []

    def test_has_errors_with_error_issue(self) -> None:
        """has_errors should return True if error-level issues exist."""
        result = VerificationResult(
            citation_issues=[
                CitationIssue(
                    pattern_rank=1,
                    field="claim",
                    issue="Invalid citation",
                    severity=IssueSeverity.error,
                )
            ]
        )
        assert result.has_errors is True

    def test_has_errors_with_only_warnings(self) -> None:
        """has_errors should return False if only warnings."""
        result = VerificationResult(
            citation_issues=[
                CitationIssue(
                    pattern_rank=1,
                    field="claim",
                    issue="Minor issue",
                    severity=IssueSeverity.warning,
                )
            ]
        )
        assert result.has_errors is False

    def test_has_warnings_with_warning_issue(self) -> None:
        """has_warnings should return True if warning-level issues exist."""
        result = VerificationResult(
            citation_issues=[
                CitationIssue(
                    pattern_rank=1,
                    field="claim",
                    issue="Minor issue",
                    severity=IssueSeverity.warning,
                )
            ]
        )
        assert result.has_warnings is True

    def test_has_warnings_with_unused_evidence(self) -> None:
        """has_warnings returns False for only unused evidence (no issues)."""
        result = VerificationResult(unused_evidence=["E8", "E9"])
        # has_warnings only checks citation issues, not unused_evidence
        assert result.has_warnings is False

    def test_has_warnings_empty_result(self) -> None:
        """has_warnings should return False for clean result."""
        result = VerificationResult()
        assert result.has_warnings is False

    def test_issue_count(self) -> None:
        """issue_count should sum citation issues and accuracy concerns."""
        result = VerificationResult(
            citation_issues=[
                CitationIssue(pattern_rank=1, field="claim", issue="Issue 1"),
                CitationIssue(pattern_rank=2, field="evidence", issue="Issue 2"),
            ],
            accuracy_concerns=[
                AccuracyConcern(pattern_rank=1, evidence_id="E1", concern="Concern")
            ],
        )
        assert result.issue_count == 3


class TestParseVerificationResult:
    """Tests for parse_verification_result function."""

    def test_parse_valid_response(self) -> None:
        """Should parse complete valid response."""
        response = {
            "verification_status": "pass",
            "citation_issues": [
                {
                    "pattern_rank": 1,
                    "field": "claim",
                    "issue": "Citation E99 not found",
                    "severity": "error",
                }
            ],
            "unused_evidence": ["E8", "E12"],
            "accuracy_concerns": [
                {
                    "pattern_rank": 2,
                    "evidence_id": "E3",
                    "concern": "Overstatement",
                }
            ],
            "overall_quality": "medium",
            "improvement_suggestions": ["Add more citations"],
        }

        result = parse_verification_result(response)

        assert result.verification_status == VerificationStatus.passed
        assert len(result.citation_issues) == 1
        assert result.citation_issues[0].severity == IssueSeverity.error
        assert result.unused_evidence == ["E8", "E12"]
        assert len(result.accuracy_concerns) == 1
        assert result.overall_quality == "medium"

    def test_parse_warn_status(self) -> None:
        """Should parse warn status correctly."""
        response = {"verification_status": "warn"}
        result = parse_verification_result(response)
        assert result.verification_status == VerificationStatus.warn

    def test_parse_fail_status(self) -> None:
        """Should parse fail status correctly."""
        response = {"verification_status": "fail"}
        result = parse_verification_result(response)
        assert result.verification_status == VerificationStatus.fail

    def test_parse_invalid_status_defaults_to_pass(self) -> None:
        """Should default to pass for invalid status."""
        response = {"verification_status": "invalid"}
        result = parse_verification_result(response)
        assert result.verification_status == VerificationStatus.passed

    def test_parse_skips_invalid_citation_issues(self) -> None:
        """Should skip malformed citation issues."""
        response = {
            "citation_issues": [
                {"pattern_rank": 1},  # Missing issue
                {"issue": "Problem"},  # Missing pattern_rank
                "not a dict",
                {"pattern_rank": 2, "issue": "Valid issue"},
            ]
        }

        result = parse_verification_result(response)

        assert len(result.citation_issues) == 1
        assert result.citation_issues[0].pattern_rank == 2

    def test_parse_invalid_severity_defaults_to_warning(self) -> None:
        """Should default to warning for invalid severity."""
        response = {
            "citation_issues": [
                {
                    "pattern_rank": 1,
                    "field": "claim",
                    "issue": "Test",
                    "severity": "critical",
                }
            ]
        }

        result = parse_verification_result(response)

        assert result.citation_issues[0].severity == IssueSeverity.warning

    def test_parse_skips_invalid_accuracy_concerns(self) -> None:
        """Should skip malformed accuracy concerns."""
        response = {
            "accuracy_concerns": [
                {"pattern_rank": 1},  # Missing concern
                {"concern": "Problem"},  # Missing pattern_rank
                123,
                {"pattern_rank": 2, "concern": "Valid concern"},
            ]
        }

        result = parse_verification_result(response)

        assert len(result.accuracy_concerns) == 1
        assert result.accuracy_concerns[0].pattern_rank == 2

    def test_parse_handles_non_list_unused_evidence(self) -> None:
        """Should handle unused_evidence that isn't a list."""
        response = {"unused_evidence": "E8"}
        result = parse_verification_result(response)
        assert result.unused_evidence == []

    def test_parse_handles_non_list_suggestions(self) -> None:
        """Should handle improvement_suggestions that isn't a list."""
        response = {"improvement_suggestions": "single suggestion"}
        result = parse_verification_result(response)
        assert result.improvement_suggestions == []

    def test_parse_invalid_quality_defaults_to_medium(self) -> None:
        """Should default to medium for invalid quality."""
        response = {"overall_quality": "excellent"}
        result = parse_verification_result(response)
        assert result.overall_quality == "medium"

    def test_parse_empty_response(self) -> None:
        """Should handle empty response."""
        response = {}
        result = parse_verification_result(response)
        assert result.verification_status == VerificationStatus.passed
        assert result.citation_issues == []
        assert result.unused_evidence == []

    def test_parse_missing_field_uses_unknown(self) -> None:
        """Should use 'unknown' for missing field in citation issue."""
        response = {"citation_issues": [{"pattern_rank": 1, "issue": "Test issue"}]}

        result = parse_verification_result(response)

        assert result.citation_issues[0].field == "unknown"


class TestComputeVerificationStatus:
    """Tests for compute_verification_status function."""

    def test_pass_for_clean_result(self) -> None:
        """Should return pass for clean result."""
        result = VerificationResult()
        assert compute_verification_status(result) == VerificationStatus.passed

    def test_fail_for_error_issues(self) -> None:
        """Should return fail if error-level issues exist."""
        result = VerificationResult(
            citation_issues=[
                CitationIssue(
                    pattern_rank=1,
                    field="claim",
                    issue="Invalid",
                    severity=IssueSeverity.error,
                )
            ]
        )
        assert compute_verification_status(result) == VerificationStatus.fail

    def test_fail_for_many_accuracy_concerns(self) -> None:
        """Should return fail if 3+ accuracy concerns."""
        result = VerificationResult(
            accuracy_concerns=[
                AccuracyConcern(pattern_rank=1, evidence_id="E1", concern="C1"),
                AccuracyConcern(pattern_rank=2, evidence_id="E2", concern="C2"),
                AccuracyConcern(pattern_rank=3, evidence_id="E3", concern="C3"),
            ]
        )
        assert compute_verification_status(result) == VerificationStatus.fail

    def test_warn_for_warning_issues(self) -> None:
        """Should return warn if warning-level issues exist."""
        result = VerificationResult(
            citation_issues=[
                CitationIssue(
                    pattern_rank=1,
                    field="claim",
                    issue="Minor",
                    severity=IssueSeverity.warning,
                )
            ]
        )
        assert compute_verification_status(result) == VerificationStatus.warn

    def test_warn_for_unused_evidence(self) -> None:
        """Should return warn if significant unused evidence."""
        result = VerificationResult(
            unused_evidence=["E1", "E2", "E3", "E4", "E5", "E6"]
        )
        assert compute_verification_status(result) == VerificationStatus.warn

    def test_pass_with_few_unused_evidence(self) -> None:
        """Should return pass if only few unused evidence."""
        result = VerificationResult(unused_evidence=["E1", "E2"])
        assert compute_verification_status(result) == VerificationStatus.passed


class TestPromptTemplates:
    """Tests for prompt template constants."""

    def test_system_prompt_exists(self) -> None:
        """System prompt should exist and have content."""
        assert VERIFICATION_SYSTEM
        assert len(VERIFICATION_SYSTEM) > 100

    def test_system_prompt_mentions_checks(self) -> None:
        """System prompt should mention verification checks."""
        assert "Citation Validity" in VERIFICATION_SYSTEM
        assert "Citation Accuracy" in VERIFICATION_SYSTEM
        assert "Evidence Coverage" in VERIFICATION_SYSTEM

    def test_system_prompt_specifies_output_format(self) -> None:
        """System prompt should specify JSON output."""
        assert "JSON" in VERIFICATION_SYSTEM
        assert "verification_status" in VERIFICATION_SYSTEM

    def test_user_prompt_has_placeholders(self) -> None:
        """User prompt should have required placeholders."""
        assert "{evidence_inventory}" in VERIFICATION_USER
        assert "{patterns}" in VERIFICATION_USER

    def test_user_prompt_can_be_formatted(self) -> None:
        """User prompt should be formattable."""
        formatted = VERIFICATION_USER.format(
            evidence_inventory="E1: Quote one",
            patterns='[{"rank": 1, "name": "Test"}]',
        )
        assert "E1: Quote one" in formatted
        assert "Test" in formatted

    def test_system_prompt_explains_status_levels(self) -> None:
        """System prompt should explain pass/warn/fail criteria."""
        assert "pass" in VERIFICATION_SYSTEM.lower()
        assert "warn" in VERIFICATION_SYSTEM.lower()
        assert "fail" in VERIFICATION_SYSTEM.lower()
