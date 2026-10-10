"""Tests for synthesis prompt and utilities."""

from app.services.prompts.synthesis import (
    MAX_CONTEXT_CHARS,
    MAX_PATTERN_CHARS,
    PatternContext,
    SourceReference,
    SynthesisResult,
    build_conversation_history,
    build_pattern_context,
    format_pattern_for_context,
    parse_synthesis_result,
    sanitize_markdown,
    truncate_text,
)


class TestSanitizeMarkdown:
    """Tests for sanitize_markdown function."""

    def test_escapes_triple_backticks(self) -> None:
        """Should escape triple backticks."""
        text = "Here is code: ```python\nprint('hello')\n```"
        result = sanitize_markdown(text)
        assert "\\`\\`\\`" in result
        assert "```" not in result

    def test_handles_empty_string(self) -> None:
        """Should return empty string for empty input."""
        assert sanitize_markdown("") == ""

    def test_handles_none(self) -> None:
        """Should return empty string for None input."""
        assert sanitize_markdown(None) == ""  # type: ignore

    def test_preserves_normal_text(self) -> None:
        """Should preserve text without backticks."""
        text = "Normal text with *markdown* and `single backticks`"
        result = sanitize_markdown(text)
        assert result == text

    def test_escapes_multiple_code_blocks(self) -> None:
        """Should escape all triple backticks."""
        text = "```first``` and ```second```"
        result = sanitize_markdown(text)
        assert result.count("\\`\\`\\`") == 4
        assert "```" not in result


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_returns_short_text_unchanged(self) -> None:
        """Should not truncate text under limit."""
        text = "Short text"
        result = truncate_text(text, 100)
        assert result == text

    def test_truncates_long_text(self) -> None:
        """Should truncate text over limit."""
        text = "This is a long text that needs to be truncated"
        result = truncate_text(text, 20)
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")

    def test_truncates_at_word_boundary(self) -> None:
        """Should truncate at word boundary when possible."""
        text = "one two three four five six seven"
        result = truncate_text(text, 15)
        # Should truncate at a space, not mid-word
        assert not result.rstrip(".").endswith("thre")

    def test_handles_empty_string(self) -> None:
        """Should return empty string for empty input."""
        assert truncate_text("", 100) == ""

    def test_handles_none(self) -> None:
        """Should return empty string for None input."""
        assert truncate_text(None, 100) == ""  # type: ignore

    def test_handles_text_at_exact_limit(self) -> None:
        """Should not truncate text at exact limit."""
        text = "exact"
        result = truncate_text(text, 5)
        assert result == text


class TestPatternContext:
    """Tests for PatternContext model."""

    def test_creates_with_required_fields(self) -> None:
        """Should create with required fields."""
        pattern = PatternContext(
            pattern_id="abc123",
            name="Test Pattern",
        )
        assert pattern.pattern_id == "abc123"
        assert pattern.name == "Test Pattern"
        assert pattern.paper_title == "Unknown"

    def test_creates_with_all_fields(self) -> None:
        """Should create with all fields."""
        pattern = PatternContext(
            pattern_id="abc123",
            paper_title="Test Paper",
            name="Test Pattern",
            claim="Main claim",
            evidence="Supporting evidence",
            context="Additional context",
        )
        assert pattern.claim == "Main claim"
        assert pattern.evidence == "Supporting evidence"
        assert pattern.context == "Additional context"


class TestFormatPatternForContext:
    """Tests for format_pattern_for_context function."""

    def test_formats_minimal_pattern(self) -> None:
        """Should format pattern with only required fields."""
        pattern = PatternContext(
            pattern_id="abc123",
            name="Test Pattern",
        )
        result = format_pattern_for_context(pattern)
        assert "[P:abc123]" in result
        assert "Test Pattern" in result
        assert "Source: Unknown" in result

    def test_formats_full_pattern(self) -> None:
        """Should format pattern with all fields."""
        pattern = PatternContext(
            pattern_id="abc123",
            paper_title="Test Paper",
            name="Test Pattern",
            claim="Main claim",
            evidence="Supporting evidence",
            context="Additional context",
        )
        result = format_pattern_for_context(pattern)
        assert "Claim: Main claim" in result
        assert "Evidence: Supporting evidence" in result
        assert "Context: Additional context" in result

    def test_truncates_long_claim(self) -> None:
        """Should truncate long claim."""
        pattern = PatternContext(
            pattern_id="abc123",
            name="Test",
            claim="x" * (MAX_PATTERN_CHARS + 100),
        )
        result = format_pattern_for_context(pattern)
        assert "..." in result

    def test_sanitizes_markdown(self) -> None:
        """Should sanitize markdown in pattern fields."""
        pattern = PatternContext(
            pattern_id="abc123",
            name="Test ```code``` Pattern",
            claim="Claim with ```code```",
        )
        result = format_pattern_for_context(pattern)
        assert "```" not in result


class TestBuildPatternContext:
    """Tests for build_pattern_context function."""

    def test_builds_context_from_patterns(self) -> None:
        """Should build context string from patterns."""
        patterns = [
            PatternContext(pattern_id="p1", name="Pattern 1", claim="Claim 1"),
            PatternContext(pattern_id="p2", name="Pattern 2", claim="Claim 2"),
        ]
        result = build_pattern_context(patterns)
        assert "[P:p1]" in result
        assert "[P:p2]" in result
        assert "Pattern 1" in result
        assert "Pattern 2" in result

    def test_handles_empty_list(self) -> None:
        """Should return placeholder for empty list."""
        result = build_pattern_context([])
        assert "No patterns available" in result

    def test_truncates_when_exceeding_limit(self) -> None:
        """Should truncate when total context exceeds limit."""
        # Create many patterns to exceed limit
        patterns = [
            PatternContext(
                pattern_id=f"p{i}",
                name=f"Pattern {i}",
                claim="x" * 1000,
                evidence="y" * 1000,
                context="z" * 1000,
            )
            for i in range(50)
        ]
        result = build_pattern_context(patterns)
        assert len(result) <= MAX_CONTEXT_CHARS + 100  # Some buffer for truncation msg
        assert "truncated" in result.lower()


class TestBuildConversationHistory:
    """Tests for build_conversation_history function."""

    def test_builds_history_from_turns(self) -> None:
        """Should build history string from turns."""
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = build_conversation_history(turns)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result

    def test_handles_empty_list(self) -> None:
        """Should return placeholder for empty list."""
        result = build_conversation_history([])
        assert "New conversation" in result

    def test_limits_to_max_turns(self) -> None:
        """Should limit to max_turns most recent."""
        turns = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
        result = build_conversation_history(turns, max_turns=3)
        assert "Message 7" in result
        assert "Message 8" in result
        assert "Message 9" in result
        assert "Message 0" not in result

    def test_truncates_long_messages(self) -> None:
        """Should truncate long messages."""
        turns = [{"role": "user", "content": "x" * 1000}]
        result = build_conversation_history(turns)
        assert len(result) < 600  # 500 + role prefix + ellipsis

    def test_sanitizes_content(self) -> None:
        """Should sanitize markdown in content."""
        turns = [{"role": "user", "content": "```code```"}]
        result = build_conversation_history(turns)
        assert "```" not in result


class TestParseSynthesisResult:
    """Tests for parse_synthesis_result function."""

    def test_parses_valid_response(self) -> None:
        """Should parse complete valid response."""
        response = {
            "answer": "The answer is [P:abc123]",
            "sources": [
                {
                    "pattern_id": "abc123",
                    "paper_title": "Test Paper",
                    "relevance": "supporting",
                }
            ],
            "confidence": "high",
            "follow_up_suggestions": ["What about X?"],
        }
        result = parse_synthesis_result(response)
        assert result.answer == "The answer is [P:abc123]"
        assert len(result.sources) == 1
        assert result.sources[0].pattern_id == "abc123"
        assert result.confidence == "high"
        assert len(result.follow_up_suggestions) == 1

    def test_provides_default_answer(self) -> None:
        """Should provide default answer if missing."""
        response = {}
        result = parse_synthesis_result(response)
        assert "Unable to synthesize" in result.answer

    def test_handles_invalid_sources(self) -> None:
        """Should skip invalid sources."""
        response = {
            "answer": "Test",
            "sources": [
                {"pattern_id": "valid"},
                "not a dict",
                {"no_id": "invalid"},
            ],
        }
        result = parse_synthesis_result(response)
        assert len(result.sources) == 1
        assert result.sources[0].pattern_id == "valid"

    def test_normalizes_invalid_relevance(self) -> None:
        """Should normalize invalid relevance values."""
        response = {
            "answer": "Test",
            "sources": [{"pattern_id": "p1", "relevance": "invalid"}],
        }
        result = parse_synthesis_result(response)
        assert result.sources[0].relevance == "supporting"

    def test_normalizes_invalid_confidence(self) -> None:
        """Should normalize invalid confidence values."""
        response = {"answer": "Test", "confidence": "invalid"}
        result = parse_synthesis_result(response)
        assert result.confidence == "medium"

    def test_limits_follow_up_suggestions(self) -> None:
        """Should limit follow-up suggestions to 5."""
        response = {
            "answer": "Test",
            "follow_up_suggestions": [f"Question {i}" for i in range(10)],
        }
        result = parse_synthesis_result(response)
        assert len(result.follow_up_suggestions) == 5

    def test_handles_non_list_sources(self) -> None:
        """Should handle sources that isn't a list."""
        response = {"answer": "Test", "sources": "not a list"}
        result = parse_synthesis_result(response)
        assert result.sources == []

    def test_handles_non_list_suggestions(self) -> None:
        """Should handle suggestions that isn't a list."""
        response = {"answer": "Test", "follow_up_suggestions": "not a list"}
        result = parse_synthesis_result(response)
        assert result.follow_up_suggestions == []


class TestSynthesisResult:
    """Tests for SynthesisResult model."""

    def test_creates_with_defaults(self) -> None:
        """Should create with default values."""
        result = SynthesisResult(answer="Test answer")
        assert result.answer == "Test answer"
        assert result.sources == []
        assert result.confidence == "medium"
        assert result.follow_up_suggestions == []


class TestSourceReference:
    """Tests for SourceReference model."""

    def test_creates_with_defaults(self) -> None:
        """Should create with default values."""
        ref = SourceReference(pattern_id="abc123")
        assert ref.pattern_id == "abc123"
        assert ref.paper_title == "Unknown"
        assert ref.relevance == "supporting"
