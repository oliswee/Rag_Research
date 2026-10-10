"""Tests for pattern extraction prompt and parsing (Pass 2)."""

from app.services.prompts.extraction import (
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
    ExtractedPatternRaw,
    ExtractionResult,
    PatternConfidence,
    extract_evidence_ids,
    parse_extraction_result,
)


class TestPatternConfidence:
    """Tests for PatternConfidence enum."""

    def test_all_confidence_levels_exist(self) -> None:
        """All confidence levels should be defined."""
        assert PatternConfidence.high == "high"
        assert PatternConfidence.medium == "medium"
        assert PatternConfidence.low == "low"


class TestExtractedPatternRaw:
    """Tests for ExtractedPatternRaw model."""

    def test_create_valid_pattern(self) -> None:
        """Should create pattern with all fields."""
        pattern = ExtractedPatternRaw(
            rank=1,
            name="ReAct Reasoning Method",
            claim="ReAct combines reasoning and acting [E1, E2]",
            evidence="Achieves 94% accuracy on HotpotQA [E3]",
            context="Enables more interpretable agent behavior [E4]",
            tags=["methodology", "finding"],
            evidence_ids=["E1", "E2", "E3", "E4"],
            confidence=PatternConfidence.high,
        )
        assert pattern.rank == 1
        assert pattern.name == "ReAct Reasoning Method"
        assert pattern.claim is not None
        assert "E1" in pattern.claim
        assert pattern.confidence == PatternConfidence.high

    def test_pattern_has_content_all_fields(self) -> None:
        """has_content should return True when all fields set."""
        pattern = ExtractedPatternRaw(
            rank=1,
            name="Test",
            claim="Claim [E1]",
            evidence="Evidence [E2]",
            context="Context [E3]",
        )
        assert pattern.has_content is True

    def test_pattern_has_content_claim_only(self) -> None:
        """has_content should return True with only claim."""
        pattern = ExtractedPatternRaw(
            rank=1,
            name="Test",
            claim="Claim [E1]",
        )
        assert pattern.has_content is True

    def test_pattern_has_content_empty(self) -> None:
        """has_content should return False with no content."""
        pattern = ExtractedPatternRaw(
            rank=1,
            name="Test",
        )
        assert pattern.has_content is False

    def test_pattern_has_content_whitespace_only(self) -> None:
        """has_content should return False with whitespace-only content."""
        pattern = ExtractedPatternRaw(
            rank=1,
            name="Test",
            claim="   ",
            evidence="\t\n",
        )
        assert pattern.has_content is False

    def test_pattern_has_citations(self) -> None:
        """has_citations should check evidence_ids."""
        pattern_with = ExtractedPatternRaw(
            rank=1,
            name="Test",
            claim="Claim [E1]",
            evidence_ids=["E1"],
        )
        pattern_without = ExtractedPatternRaw(
            rank=1,
            name="Test",
            claim="Claim without citation",
        )
        assert pattern_with.has_citations is True
        assert pattern_without.has_citations is False

    def test_pattern_content_fields(self) -> None:
        """content_fields should list populated fields."""
        pattern = ExtractedPatternRaw(
            rank=1,
            name="Test",
            claim="Claim [E1]",
            context="Context [E2]",
        )
        assert pattern.content_fields == ["claim", "context"]

    def test_pattern_default_values(self) -> None:
        """Should use defaults for optional fields."""
        pattern = ExtractedPatternRaw(rank=1, name="Minimal")
        assert pattern.claim is None
        assert pattern.evidence is None
        assert pattern.context is None
        assert pattern.tags == []
        assert pattern.evidence_ids == []
        assert pattern.confidence == PatternConfidence.medium

    def test_pattern_name_stripped(self) -> None:
        """Name should be stripped of whitespace."""
        pattern = ExtractedPatternRaw(rank=1, name="  Padded Name  ")
        assert pattern.name == "Padded Name"


class TestExtractEvidenceIds:
    """Tests for extract_evidence_ids function."""

    def test_extract_single_citation(self) -> None:
        """Should extract single [E#] citation."""
        result = extract_evidence_ids("This is supported by evidence [E1].")
        assert result == ["E1"]

    def test_extract_multiple_separate_citations(self) -> None:
        """Should extract multiple separate citations."""
        result = extract_evidence_ids("See [E1] and [E5] for details.")
        assert result == ["E1", "E5"]

    def test_extract_grouped_citations(self) -> None:
        """Should extract grouped citations like [E1, E2]."""
        result = extract_evidence_ids("Multiple sources support this [E1, E2, E3].")
        assert result == ["E1", "E2", "E3"]

    def test_extract_mixed_citations(self) -> None:
        """Should handle mix of single and grouped citations."""
        result = extract_evidence_ids("See [E1] and [E3, E4] also [E2].")
        assert result == ["E1", "E2", "E3", "E4"]

    def test_extract_no_citations(self) -> None:
        """Should return empty list when no citations."""
        result = extract_evidence_ids("No evidence here.")
        assert result == []

    def test_extract_none_input(self) -> None:
        """Should handle None input."""
        result = extract_evidence_ids(None)
        assert result == []

    def test_extract_empty_string(self) -> None:
        """Should handle empty string."""
        result = extract_evidence_ids("")
        assert result == []

    def test_extract_deduplicates(self) -> None:
        """Should deduplicate repeated citations."""
        result = extract_evidence_ids("[E1] and again [E1] and [E2, E1].")
        assert result == ["E1", "E2"]

    def test_extract_sorts_numerically(self) -> None:
        """Should sort IDs numerically."""
        result = extract_evidence_ids("[E10] before [E2] and [E1]")
        assert result == ["E1", "E2", "E10"]


class TestParseExtractionResult:
    """Tests for parse_extraction_result function."""

    def test_parse_valid_response(self) -> None:
        """Should parse complete valid response."""
        response = {
            "patterns": [
                {
                    "rank": 1,
                    "name": "ReAct Method",
                    "claim": "ReAct combines reasoning and acting [E1]",
                    "evidence": "Achieves 94% accuracy [E2]",
                    "context": "Enables interpretability [E3]",
                    "tags": ["methodology", "finding"],
                    "evidence_ids": ["E1", "E2", "E3"],
                    "confidence": "high",
                }
            ],
            "total_evidence_used": 3,
            "gaps_identified": [],
        }

        result = parse_extraction_result(response)

        assert isinstance(result, ExtractionResult)
        assert len(result.patterns) == 1
        assert result.patterns[0].name == "ReAct Method"
        assert result.patterns[0].confidence == PatternConfidence.high
        assert result.total_evidence_used == 3

    def test_parse_multiple_patterns(self) -> None:
        """Should parse multiple patterns."""
        response = {
            "patterns": [
                {"rank": 1, "name": "Pattern One", "claim": "Claim [E1]"},
                {"rank": 2, "name": "Pattern Two", "claim": "Claim [E2]"},
                {"rank": 3, "name": "Pattern Three", "claim": "Claim [E3]"},
            ]
        }

        result = parse_extraction_result(response)

        assert len(result.patterns) == 3
        assert result.patterns[0].rank == 1
        assert result.patterns[1].rank == 2
        assert result.patterns[2].rank == 3

    def test_parse_extracts_evidence_ids_from_text(self) -> None:
        """Should extract evidence IDs from claim/evidence/context text."""
        response = {
            "patterns": [
                {
                    "rank": 1,
                    "name": "Test Pattern",
                    "claim": "The method works [E1, E2]",
                    "evidence": "Supported by [E3]",
                    "context": "Relevant because [E4, E5]",
                    # evidence_ids intentionally omitted
                }
            ]
        }

        result = parse_extraction_result(response)

        assert set(result.patterns[0].evidence_ids) == {"E1", "E2", "E3", "E4", "E5"}

    def test_parse_merges_evidence_ids(self) -> None:
        """Should merge explicit evidence_ids with extracted ones."""
        response = {
            "patterns": [
                {
                    "rank": 1,
                    "name": "Test",
                    "claim": "Claim [E1]",
                    "evidence_ids": ["E1", "E5"],  # E5 not in text
                }
            ]
        }

        result = parse_extraction_result(response)

        assert "E1" in result.patterns[0].evidence_ids
        assert "E5" in result.patterns[0].evidence_ids

    def test_parse_skips_pattern_without_name(self) -> None:
        """Should skip patterns missing name."""
        response = {
            "patterns": [
                {"rank": 1, "claim": "Claim [E1]"},  # No name
                {"rank": 2, "name": "Valid Pattern", "claim": "Claim [E2]"},
            ]
        }

        result = parse_extraction_result(response)

        assert len(result.patterns) == 1
        assert result.patterns[0].name == "Valid Pattern"

    def test_parse_skips_empty_name(self) -> None:
        """Should skip patterns with empty name."""
        response = {
            "patterns": [
                {"rank": 1, "name": "", "claim": "Claim"},
                {"rank": 2, "name": "   ", "claim": "Claim"},
                {"rank": 3, "name": "Valid", "claim": "Claim"},
            ]
        }

        result = parse_extraction_result(response)

        assert len(result.patterns) == 1
        assert result.patterns[0].name == "Valid"

    def test_parse_handles_invalid_confidence(self) -> None:
        """Should fallback to medium for invalid confidence."""
        response = {
            "patterns": [{"rank": 1, "name": "Test", "confidence": "very_high"}]
        }

        result = parse_extraction_result(response)

        assert result.patterns[0].confidence == PatternConfidence.medium

    def test_parse_handles_non_list_tags(self) -> None:
        """Should handle tags that isn't a list."""
        response = {"patterns": [{"rank": 1, "name": "Test", "tags": "methodology"}]}

        result = parse_extraction_result(response)

        assert result.patterns[0].tags == []

    def test_parse_handles_empty_patterns(self) -> None:
        """Should handle empty patterns list."""
        response = {"patterns": [], "total_evidence_used": 0}

        result = parse_extraction_result(response)

        assert result.patterns == []
        assert result.total_evidence_used == 0

    def test_parse_handles_missing_patterns(self) -> None:
        """Should handle missing patterns key."""
        response = {"total_evidence_used": 5}

        result = parse_extraction_result(response)

        assert result.patterns == []

    def test_parse_handles_non_list_patterns(self) -> None:
        """Should handle patterns that isn't a list."""
        response = {"patterns": "not a list"}

        result = parse_extraction_result(response)

        assert result.patterns == []

    def test_parse_handles_non_dict_pattern_items(self) -> None:
        """Should skip non-dict items in patterns list."""
        response = {
            "patterns": [
                "string",
                123,
                {"rank": 1, "name": "Valid", "claim": "Claim [E1]"},
                None,
            ]
        }

        result = parse_extraction_result(response)

        assert len(result.patterns) == 1
        assert result.patterns[0].name == "Valid"

    def test_parse_assigns_default_rank(self) -> None:
        """Should assign sequential rank if missing."""
        response = {
            "patterns": [
                {"name": "First"},
                {"name": "Second"},
            ]
        }

        result = parse_extraction_result(response)

        assert result.patterns[0].rank == 1
        assert result.patterns[1].rank == 2

    def test_parse_handles_whitespace_content(self) -> None:
        """Should convert whitespace-only content to None."""
        response = {
            "patterns": [
                {
                    "rank": 1,
                    "name": "Test",
                    "claim": "   ",
                    "evidence": "\t\n",
                    "context": "",
                }
            ]
        }

        result = parse_extraction_result(response)

        assert result.patterns[0].claim is None
        assert result.patterns[0].evidence is None
        assert result.patterns[0].context is None

    def test_parse_gaps_identified(self) -> None:
        """Should parse gaps_identified list."""
        response = {
            "patterns": [{"rank": 1, "name": "Test"}],
            "gaps_identified": ["Missing evidence for claim X", "No support for Y"],
        }

        result = parse_extraction_result(response)

        assert len(result.gaps_identified) == 2
        assert "Missing evidence" in result.gaps_identified[0]

    def test_parse_empty_response(self) -> None:
        """Should handle completely empty response."""
        response = {}

        result = parse_extraction_result(response)

        assert result.patterns == []
        assert result.total_evidence_used == 0
        assert result.gaps_identified == []


class TestPromptTemplates:
    """Tests for prompt template constants."""

    def test_system_prompt_exists(self) -> None:
        """System prompt should exist and have content."""
        assert EXTRACTION_SYSTEM
        assert len(EXTRACTION_SYSTEM) > 100

    def test_system_prompt_mentions_claim_evidence_context(self) -> None:
        """System prompt should mention all three content fields."""
        assert "Claim" in EXTRACTION_SYSTEM
        assert "Evidence" in EXTRACTION_SYSTEM
        assert "Context" in EXTRACTION_SYSTEM

    def test_system_prompt_specifies_citation_format(self) -> None:
        """System prompt should specify [E#] citation format."""
        assert "[E" in EXTRACTION_SYSTEM
        assert "E#" in EXTRACTION_SYSTEM or "E1" in EXTRACTION_SYSTEM

    def test_system_prompt_specifies_pattern_count(self) -> None:
        """System prompt should specify 3-8 patterns."""
        assert "3-8" in EXTRACTION_SYSTEM

    def test_user_prompt_has_placeholders(self) -> None:
        """User prompt should have required placeholders."""
        assert "{evidence_inventory}" in EXTRACTION_USER
        assert "{text}" in EXTRACTION_USER

    def test_user_prompt_can_be_formatted(self) -> None:
        """User prompt should be formattable."""
        formatted = EXTRACTION_USER.format(
            evidence_inventory="E1: Quote one\nE2: Quote two",
            text="Paper content here",
        )
        assert "E1: Quote one" in formatted
        assert "Paper content here" in formatted

    def test_system_prompt_mentions_json_output(self) -> None:
        """System prompt should specify JSON output."""
        assert "JSON" in EXTRACTION_SYSTEM

    def test_system_prompt_lists_tags(self) -> None:
        """System prompt should list available tags."""
        assert "methodology" in EXTRACTION_SYSTEM
        assert "finding" in EXTRACTION_SYSTEM
        assert "limitation" in EXTRACTION_SYSTEM
