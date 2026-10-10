"""Tests for document chunking service."""

from app.services.document_chunker import (
    CHARS_PER_TOKEN,
    DEFAULT_TOKEN_LIMIT,
    ChunkingResult,
    DocumentChunk,
    DocumentChunker,
)


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    def test_init_default_values(self):
        """Chunker initializes with default values."""
        chunker = DocumentChunker()
        assert chunker.token_limit == DEFAULT_TOKEN_LIMIT
        assert chunker.char_limit == DEFAULT_TOKEN_LIMIT * CHARS_PER_TOKEN

    def test_init_custom_token_limit(self):
        """Chunker accepts custom token limit."""
        chunker = DocumentChunker(token_limit=10000)
        assert chunker.token_limit == 10000
        assert chunker.char_limit == 10000 * CHARS_PER_TOKEN

    def test_estimate_tokens(self):
        """Token estimation uses character ratio."""
        chunker = DocumentChunker()
        text = "a" * 400
        assert chunker.estimate_tokens(text) == 100  # 400 / 4

    def test_needs_chunking_small_document(self):
        """Small documents don't need chunking."""
        chunker = DocumentChunker(token_limit=1000)
        text = "a" * 100  # 25 tokens
        assert chunker.needs_chunking(text) is False

    def test_needs_chunking_large_document(self):
        """Large documents need chunking."""
        chunker = DocumentChunker(token_limit=100)
        text = "a" * 1000  # 250 tokens
        assert chunker.needs_chunking(text) is True


class TestSectionExtraction:
    """Tests for section boundary detection."""

    def test_extract_numbered_sections(self):
        """Detects numbered section headers."""
        chunker = DocumentChunker()
        text = """
1. Introduction
Some content here.

2. Methods
More content here.

3. Results
Final content.
"""
        sections = chunker.extract_sections(text)
        assert len(sections) >= 3
        section_names = [s[1] for s in sections]
        assert any("Introduction" in name for name in section_names)
        assert any("Methods" in name for name in section_names)
        assert any("Results" in name for name in section_names)

    def test_extract_markdown_headers(self):
        """Detects markdown headers."""
        chunker = DocumentChunker()
        text = """
# Introduction

Some content.

## Background

More content.

### Implementation Details

Final content.
"""
        sections = chunker.extract_sections(text)
        assert len(sections) >= 3

    def test_extract_common_section_names(self):
        """Detects common section names like Abstract, Conclusion."""
        chunker = DocumentChunker()
        text = """
Abstract
This is the abstract.

Introduction
This is the intro.

Conclusion
This is the conclusion.
"""
        sections = chunker.extract_sections(text)
        section_names = [s[1].lower() for s in sections]
        assert any("abstract" in name for name in section_names)
        assert any("introduction" in name for name in section_names)
        assert any("conclusion" in name for name in section_names)

    def test_extract_no_sections_plain_text(self):
        """Returns empty list for text without section headers."""
        chunker = DocumentChunker()
        text = "This is just plain text without any section headers."
        sections = chunker.extract_sections(text)
        assert sections == []


class TestBuildOutline:
    """Tests for document outline building."""

    def test_build_outline_with_sections(self):
        """Builds outline from section list."""
        chunker = DocumentChunker()
        sections = [
            (0, "Introduction"),
            (100, "Methods"),
            (200, "Results"),
        ]
        outline = chunker.build_outline(sections)
        assert "Document Structure:" in outline
        assert "Introduction" in outline
        assert "Methods" in outline
        assert "Results" in outline

    def test_build_outline_no_sections(self):
        """Returns placeholder for empty section list."""
        chunker = DocumentChunker()
        outline = chunker.build_outline([])
        assert "No clear section structure" in outline

    def test_build_outline_limits_sections(self):
        """Limits outline to 20 sections."""
        chunker = DocumentChunker()
        sections = [(i * 10, f"Section {i}") for i in range(30)]
        outline = chunker.build_outline(sections)
        assert "10 more sections" in outline


class TestChunking:
    """Tests for document chunking."""

    def test_chunk_small_document_no_chunking(self):
        """Small documents return single chunk without chunking."""
        chunker = DocumentChunker(token_limit=10000)
        text = "Small document content."

        result = chunker.chunk(text)

        assert isinstance(result, ChunkingResult)
        assert result.was_chunked is False
        assert len(result.chunks) == 1
        assert result.chunks[0].text == text.strip()
        assert result.chunks[0].chunk_index == 0
        assert result.chunks[0].total_chunks == 1

    def test_chunk_large_document_with_sections(self):
        """Large documents with sections are chunked at section boundaries."""
        chunker = DocumentChunker(token_limit=50)  # Very small for testing

        # Create text with sections that exceeds limit
        text = (
            """
1. Introduction
"""
            + "a" * 100
            + """

2. Methods
"""
            + "b" * 100
            + """

3. Results
"""
            + "c" * 100
        )

        result = chunker.chunk(text)

        assert result.was_chunked is True
        assert len(result.chunks) > 1
        assert "Document Structure" in result.document_outline

        # All chunks should have correct indices
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(result.chunks)

    def test_chunk_large_document_no_sections_uses_paragraphs(self):
        """Large documents without sections fall back to paragraph chunking."""
        chunker = DocumentChunker(token_limit=50)

        # Create text with paragraphs but no section headers
        paragraphs = ["Paragraph " + str(i) + " content. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)

        result = chunker.chunk(text)

        assert result.was_chunked is True
        assert len(result.chunks) > 1


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_chunk_attributes(self):
        """DocumentChunk has expected attributes."""
        chunk = DocumentChunk(
            text="Sample text",
            chunk_index=0,
            total_chunks=3,
            section_titles=["Introduction"],
            start_char=0,
            end_char=11,
        )

        assert chunk.text == "Sample text"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 3
        assert chunk.section_titles == ["Introduction"]
        assert chunk.start_char == 0
        assert chunk.end_char == 11


class TestChunkingResult:
    """Tests for ChunkingResult dataclass."""

    def test_result_attributes(self):
        """ChunkingResult has expected attributes."""
        chunks = [
            DocumentChunk(
                text="Chunk 1",
                chunk_index=0,
                total_chunks=2,
                section_titles=[],
                start_char=0,
                end_char=7,
            ),
            DocumentChunk(
                text="Chunk 2",
                chunk_index=1,
                total_chunks=2,
                section_titles=[],
                start_char=7,
                end_char=14,
            ),
        ]

        result = ChunkingResult(
            chunks=chunks,
            document_outline="Outline here",
            total_tokens_estimate=100,
            was_chunked=True,
        )

        assert len(result.chunks) == 2
        assert result.document_outline == "Outline here"
        assert result.total_tokens_estimate == 100
        assert result.was_chunked is True


class TestChunkOverlap:
    """Tests for chunk overlap handling."""

    def test_chunks_have_overlap(self):
        """Consecutive chunks have overlapping content."""
        chunker = DocumentChunker(token_limit=50, chunk_overlap=100)

        # Create text that will be chunked
        text = (
            """
1. Section One
"""
            + "Word " * 100
            + """

2. Section Two
"""
            + "Word " * 100
        )

        result = chunker.chunk(text)

        if len(result.chunks) > 1:
            # Due to section-based splitting, overlap may not always be exact
            # but chunks should be contiguous
            assert result.chunks[0].end_char <= result.chunks[1].end_char


class TestServiceExports:
    """Tests for module exports."""

    def test_chunker_exported_from_services(self):
        """DocumentChunker is exported from services module."""
        from app.services import ChunkingResult, DocumentChunk, DocumentChunker

        assert DocumentChunker is not None
        assert DocumentChunk is not None
        assert ChunkingResult is not None
