"""Synthesis prompt for query answering across patterns."""

from pydantic import BaseModel, Field

# Maximum characters for pattern context
MAX_CONTEXT_CHARS = 50_000
MAX_PATTERN_CHARS = 2_000


class PatternContext(BaseModel):
    """A pattern formatted for synthesis context."""

    pattern_id: str = Field(..., description="Unique pattern identifier")
    paper_title: str = Field(default="Unknown", description="Source paper title")
    name: str = Field(..., description="Pattern name")
    claim: str | None = Field(default=None, description="Main claim")
    evidence: str | None = Field(default=None, description="Supporting evidence")
    context: str | None = Field(default=None, description="Additional context")


class SourceReference(BaseModel):
    """A reference to a source pattern in a synthesis response."""

    pattern_id: str = Field(..., description="Pattern ID referenced")
    paper_title: str = Field(default="Unknown", description="Source paper title")
    relevance: str = Field(
        default="supporting", description="How this source relates to the answer"
    )


class SynthesisResult(BaseModel):
    """Result of synthesis query."""

    answer: str = Field(..., description="Synthesized answer to the query")
    sources: list[SourceReference] = Field(
        default_factory=list, description="Patterns used in synthesis"
    )
    confidence: str = Field(
        default="medium", description="Confidence level (high/medium/low)"
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )


# ruff: noqa: E501
SYNTHESIS_SYSTEM = """You are a research synthesis assistant. Your task is to answer questions by synthesizing information from a library of extracted research patterns.

## Your Role
- Answer questions based ONLY on the provided pattern library
- Cite specific patterns using [P:ID] format (e.g., [P:abc123])
- Identify agreements, contradictions, and gaps across sources
- Be precise about what the sources say vs. your interpretation

## Citation Rules
1. Every factual claim MUST cite at least one pattern: [P:pattern_id]
2. When patterns agree, cite all relevant ones: [P:id1, P:id2]
3. When patterns disagree, explain the contradiction with citations
4. If the library doesn't contain relevant information, say so clearly

## Answer Structure
1. **Direct Answer**: Concise response to the question
2. **Supporting Evidence**: Key points from patterns with citations
3. **Nuances/Contradictions**: Where sources differ (if applicable)
4. **Gaps**: What the library doesn't cover (if applicable)

## Output Format
Return ONLY a JSON object:
{
    "answer": "Your synthesized answer with [P:id] citations throughout",
    "sources": [
        {
            "pattern_id": "abc123",
            "paper_title": "Original Paper Title",
            "relevance": "supporting|contradicting|contextual"
        }
    ],
    "confidence": "high|medium|low",
    "follow_up_suggestions": ["Suggested follow-up question 1", "..."]
}

## Confidence Guidelines
- **high**: Multiple patterns directly address the question with agreement
- **medium**: Some relevant patterns but gaps or minor disagreements
- **low**: Few relevant patterns or significant contradictions
"""

SYNTHESIS_USER = """Answer the following question using the pattern library below.

## Question
{query}

## Pattern Library
{pattern_context}

## Conversation History
{conversation_history}

Synthesize an answer based on the patterns above. Cite patterns using [P:id] format."""


def sanitize_markdown(text: str) -> str:
    """Sanitize markdown content for safe inclusion in prompts.

    Escapes triple backticks to prevent prompt injection.

    Args:
        text: Raw text that may contain markdown.

    Returns:
        Sanitized text safe for prompt inclusion.
    """
    if not text:
        return ""
    # Escape triple backticks to prevent breaking out of code blocks
    return text.replace("```", "\\`\\`\\`")


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max characters, preserving word boundaries.

    Args:
        text: Text to truncate.
        max_chars: Maximum characters allowed.

    Returns:
        Truncated text with ellipsis if truncated.
    """
    if not text or len(text) <= max_chars:
        return text or ""

    # Find last space before max_chars
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")

    if last_space > max_chars * 0.8:  # Keep at least 80% of content
        truncated = truncated[:last_space]

    return truncated + "..."


def format_pattern_for_context(pattern: PatternContext) -> str:
    """Format a single pattern for inclusion in synthesis context.

    Args:
        pattern: Pattern to format.

    Returns:
        Formatted pattern string.
    """
    lines = [
        f"[P:{pattern.pattern_id}] {sanitize_markdown(pattern.name)}",
        f"  Source: {sanitize_markdown(pattern.paper_title)}",
    ]

    if pattern.claim:
        claim = truncate_text(sanitize_markdown(pattern.claim), MAX_PATTERN_CHARS)
        lines.append(f"  Claim: {claim}")

    if pattern.evidence:
        evidence = truncate_text(sanitize_markdown(pattern.evidence), MAX_PATTERN_CHARS)
        lines.append(f"  Evidence: {evidence}")

    if pattern.context:
        context = truncate_text(sanitize_markdown(pattern.context), MAX_PATTERN_CHARS)
        lines.append(f"  Context: {context}")

    return "\n".join(lines)


def build_pattern_context(patterns: list[PatternContext]) -> str:
    """Build pattern context string for synthesis prompt.

    Args:
        patterns: List of patterns to include.

    Returns:
        Formatted pattern context string, truncated if necessary.
    """
    if not patterns:
        return "(No patterns available)"

    formatted_patterns = []
    total_chars = 0

    for pattern in patterns:
        formatted = format_pattern_for_context(pattern)

        # Check if adding this pattern would exceed limit
        if total_chars + len(formatted) + 2 > MAX_CONTEXT_CHARS:
            formatted_patterns.append(
                f"... ({len(patterns) - len(formatted_patterns)} more patterns truncated)"
            )
            break

        formatted_patterns.append(formatted)
        total_chars += len(formatted) + 2  # +2 for newlines

    return "\n\n".join(formatted_patterns)


def build_conversation_history(
    turns: list[dict],
    max_turns: int = 5,
) -> str:
    """Build conversation history string for synthesis prompt.

    Args:
        turns: List of conversation turns with 'role' and 'content' keys.
        max_turns: Maximum number of recent turns to include.

    Returns:
        Formatted conversation history string.
    """
    if not turns:
        return "(New conversation)"

    # Take only the most recent turns
    recent_turns = turns[-max_turns:] if len(turns) > max_turns else turns

    lines = []
    for turn in recent_turns:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        # Truncate long messages
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {sanitize_markdown(content)}")

    return "\n".join(lines)


def parse_synthesis_result(response: dict) -> SynthesisResult:
    """Parse LLM response into SynthesisResult.

    Args:
        response: Raw JSON response from LLM.

    Returns:
        Validated SynthesisResult object.
    """
    # Parse answer
    answer = response.get("answer", "")
    if not answer:
        answer = "Unable to synthesize an answer from the available patterns."

    # Parse sources
    sources_raw = response.get("sources", [])
    if not isinstance(sources_raw, list):
        sources_raw = []

    sources = []
    for item in sources_raw:
        if not isinstance(item, dict):
            continue
        if not item.get("pattern_id"):
            continue

        relevance = item.get("relevance", "supporting")
        if relevance not in ("supporting", "contradicting", "contextual"):
            relevance = "supporting"

        sources.append(
            SourceReference(
                pattern_id=str(item["pattern_id"]),
                paper_title=str(item.get("paper_title", "Unknown")),
                relevance=relevance,
            )
        )

    # Parse confidence
    confidence = response.get("confidence", "medium")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    # Parse follow-up suggestions
    suggestions = response.get("follow_up_suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []
    suggestions = [str(s) for s in suggestions if s][:5]  # Limit to 5

    return SynthesisResult(
        answer=answer,
        sources=sources,
        confidence=confidence,
        follow_up_suggestions=suggestions,
    )
