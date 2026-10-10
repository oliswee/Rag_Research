"""Tests for evidence inventory prompt and parsing."""

from app.services.prompts.evidence_inventory import (
    EVIDENCE_INVENTORY_SYSTEM,
    EVIDENCE_INVENTORY_USER,
    EvidenceImportance,
    EvidenceInventory,
    EvidenceItem,
    EvidenceType,
    parse_evidence_inventory,
)


class TestEvidenceType:
    """Tests for EvidenceType enum."""

    def test_all_evidence_types_exist(self) -> None:
        """All expected evidence types should be defined."""
        expected = [
            "finding",
            "definition",
            "mechanism",
            "limitation",
            "comparison",
            "claim",
            "methodology",
            "result",
        ]
        for type_name in expected:
            assert EvidenceType(type_name) == type_name

    def test_evidence_type_is_string(self) -> None:
        """EvidenceType should be usable as a string."""
        assert str(EvidenceType.finding) == "finding"
        assert EvidenceType.mechanism == "mechanism"


class TestEvidenceImportance:
    """Tests for EvidenceImportance enum."""

    def test_all_importance_levels_exist(self) -> None:
        """All importance levels should be defined."""
        assert EvidenceImportance.high == "high"
        assert EvidenceImportance.medium == "medium"
        assert EvidenceImportance.low == "low"


class TestEvidenceItem:
    """Tests for EvidenceItem model."""

    def test_create_valid_evidence_item(self) -> None:
        """Should create item with all required fields."""
        item = EvidenceItem(
            id="E1",
            quote="This is a direct quote from the paper.",
            location="Abstract",
            type=EvidenceType.finding,
            importance=EvidenceImportance.high,
        )
        assert item.id == "E1"
        assert item.quote == "This is a direct quote from the paper."
        assert item.location == "Abstract"
        assert item.type == EvidenceType.finding
        assert item.importance == EvidenceImportance.high

    def test_evidence_item_default_importance(self) -> None:
        """Should default to medium importance."""
        item = EvidenceItem(
            id="E2",
            quote="Another quote",
            location="Methods",
            type=EvidenceType.methodology,
        )
        assert item.importance == EvidenceImportance.medium


class TestEvidenceInventory:
    """Tests for EvidenceInventory model."""

    def test_create_valid_inventory(self) -> None:
        """Should create inventory with all fields."""
        item = EvidenceItem(
            id="E1",
            quote="Test quote",
            location="Results",
            type=EvidenceType.result,
        )
        inventory = EvidenceInventory(
            paper_title="Test Paper",
            evidence_items=[item],
            paper_type="empirical",
            core_contribution_summary="A test contribution.",
        )
        assert inventory.paper_title == "Test Paper"
        assert len(inventory.evidence_items) == 1
        assert inventory.paper_type == "empirical"

    def test_inventory_default_values(self) -> None:
        """Should use defaults for optional fields."""
        inventory = EvidenceInventory(paper_title="Minimal Paper")
        assert inventory.evidence_items == []
        assert inventory.paper_type == "empirical"
        assert inventory.core_contribution_summary == ""


class TestParseEvidenceInventory:
    """Tests for parse_evidence_inventory function."""

    def test_parse_valid_response(self) -> None:
        """Should parse a complete valid response."""
        response = {
            "paper_title": "ReAct: Synergizing Reasoning and Acting",
            "evidence_items": [
                {
                    "id": "E1",
                    "quote": "ReAct prompts LLMs to generate reasoning traces.",
                    "location": "Abstract",
                    "type": "mechanism",
                    "importance": "high",
                },
                {
                    "id": "E2",
                    "quote": "The method achieves 94% accuracy on HotpotQA.",
                    "location": "Results",
                    "type": "result",
                    "importance": "high",
                },
            ],
            "paper_type": "empirical",
            "core_contribution_summary": "A method for combining reasoning and acting.",
        }

        result = parse_evidence_inventory(response)

        assert isinstance(result, EvidenceInventory)
        assert result.paper_title == "ReAct: Synergizing Reasoning and Acting"
        assert len(result.evidence_items) == 2
        assert result.evidence_items[0].id == "E1"
        assert result.evidence_items[0].type == EvidenceType.mechanism
        assert result.evidence_items[1].importance == EvidenceImportance.high

    def test_parse_response_with_missing_optional_fields(self) -> None:
        """Should handle missing optional fields with defaults."""
        response = {
            "paper_title": "Test Paper",
            "evidence_items": [
                {
                    "id": "E1",
                    "quote": "A test quote.",
                    # location missing
                    # type missing
                    # importance missing
                }
            ],
            # paper_type missing
            # core_contribution_summary missing
        }

        result = parse_evidence_inventory(response)

        assert result.paper_title == "Test Paper"
        assert len(result.evidence_items) == 1
        assert result.evidence_items[0].location == "Unknown"
        assert result.evidence_items[0].type == EvidenceType.finding
        assert result.evidence_items[0].importance == EvidenceImportance.medium
        assert result.paper_type == "empirical"
        assert result.core_contribution_summary == ""

    def test_parse_response_with_invalid_type(self) -> None:
        """Should fallback to 'finding' for invalid evidence type."""
        response = {
            "paper_title": "Test",
            "evidence_items": [
                {
                    "id": "E1",
                    "quote": "Quote",
                    "location": "Section",
                    "type": "invalid_type",
                    "importance": "high",
                }
            ],
        }

        result = parse_evidence_inventory(response)

        assert result.evidence_items[0].type == EvidenceType.finding

    def test_parse_response_with_invalid_importance(self) -> None:
        """Should fallback to 'medium' for invalid importance."""
        response = {
            "paper_title": "Test",
            "evidence_items": [
                {
                    "id": "E1",
                    "quote": "Quote",
                    "location": "Section",
                    "type": "finding",
                    "importance": "very_high",  # Invalid
                }
            ],
        }

        result = parse_evidence_inventory(response)

        assert result.evidence_items[0].importance == EvidenceImportance.medium

    def test_parse_response_skips_items_without_id(self) -> None:
        """Should skip evidence items missing required 'id' field."""
        response = {
            "paper_title": "Test",
            "evidence_items": [
                {"quote": "No ID here", "location": "Abstract", "type": "finding"},
                {
                    "id": "E1",
                    "quote": "Has ID",
                    "location": "Methods",
                    "type": "finding",
                },
            ],
        }

        result = parse_evidence_inventory(response)

        assert len(result.evidence_items) == 1
        assert result.evidence_items[0].id == "E1"

    def test_parse_response_skips_items_without_quote(self) -> None:
        """Should skip evidence items missing required 'quote' field."""
        response = {
            "paper_title": "Test",
            "evidence_items": [
                {"id": "E1", "location": "Abstract", "type": "finding"},  # No quote
                {
                    "id": "E2",
                    "quote": "Has quote",
                    "location": "Methods",
                    "type": "finding",
                },
            ],
        }

        result = parse_evidence_inventory(response)

        assert len(result.evidence_items) == 1
        assert result.evidence_items[0].id == "E2"

    def test_parse_response_handles_empty_evidence_list(self) -> None:
        """Should handle empty evidence_items list."""
        response = {
            "paper_title": "Empty Paper",
            "evidence_items": [],
            "paper_type": "theoretical",
        }

        result = parse_evidence_inventory(response)

        assert result.paper_title == "Empty Paper"
        assert result.evidence_items == []

    def test_parse_response_handles_missing_evidence_items(self) -> None:
        """Should handle missing evidence_items key."""
        response = {
            "paper_title": "No Evidence",
            "paper_type": "survey",
        }

        result = parse_evidence_inventory(response)

        assert result.evidence_items == []

    def test_parse_response_handles_non_list_evidence_items(self) -> None:
        """Should handle evidence_items that is not a list."""
        response = {
            "paper_title": "Bad Format",
            "evidence_items": "not a list",
        }

        result = parse_evidence_inventory(response)

        assert result.evidence_items == []

    def test_parse_response_skips_non_dict_items(self) -> None:
        """Should skip evidence items that are not dictionaries."""
        response = {
            "paper_title": "Mixed",
            "evidence_items": [
                "string item",
                123,
                {"id": "E1", "quote": "Valid", "location": "A", "type": "finding"},
                None,
            ],
        }

        result = parse_evidence_inventory(response)

        assert len(result.evidence_items) == 1
        assert result.evidence_items[0].id == "E1"

    def test_parse_response_handles_completely_empty_response(self) -> None:
        """Should handle completely empty response dict."""
        response = {}

        result = parse_evidence_inventory(response)

        assert result.paper_title == "Unknown"
        assert result.evidence_items == []
        assert result.paper_type == "empirical"

    def test_parse_all_evidence_types(self) -> None:
        """Should correctly parse all valid evidence types."""
        types_to_test = [
            "finding",
            "definition",
            "mechanism",
            "limitation",
            "comparison",
            "claim",
            "methodology",
            "result",
        ]

        evidence_items = [
            {"id": f"E{i + 1}", "quote": f"Quote {i}", "location": "Test", "type": t}
            for i, t in enumerate(types_to_test)
        ]

        response = {"paper_title": "Type Test", "evidence_items": evidence_items}

        result = parse_evidence_inventory(response)

        assert len(result.evidence_items) == 8
        for i, expected_type in enumerate(types_to_test):
            assert result.evidence_items[i].type == EvidenceType(expected_type)


class TestPromptTemplates:
    """Tests for prompt template constants."""

    def test_system_prompt_exists_and_not_empty(self) -> None:
        """System prompt should exist and have content."""
        assert EVIDENCE_INVENTORY_SYSTEM
        assert len(EVIDENCE_INVENTORY_SYSTEM) > 100
        assert "evidence" in EVIDENCE_INVENTORY_SYSTEM.lower()

    def test_user_prompt_has_text_placeholder(self) -> None:
        """User prompt should have {text} placeholder."""
        assert "{text}" in EVIDENCE_INVENTORY_USER

    def test_user_prompt_can_be_formatted(self) -> None:
        """User prompt should be formattable."""
        formatted = EVIDENCE_INVENTORY_USER.format(text="Sample paper content")
        assert "Sample paper content" in formatted

    def test_system_prompt_mentions_json_output(self) -> None:
        """System prompt should specify JSON output format."""
        assert "JSON" in EVIDENCE_INVENTORY_SYSTEM

    def test_system_prompt_specifies_evidence_count(self) -> None:
        """System prompt should specify expected evidence count."""
        assert "12-25" in EVIDENCE_INVENTORY_SYSTEM or "12" in EVIDENCE_INVENTORY_SYSTEM
