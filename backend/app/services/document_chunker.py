"""Document chunking service for long document processing.

Splits documents into semantic chunks based on section boundaries,
ensuring each chunk has context about the overall document structure.
"""

import re
from dataclasses import dataclass

# Approximate token-to-character ratio (conservative estimate)
CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_LIMIT = 40000
DEFAULT_CHUNK_OVERLAP = 500  # Characters of overlap between chunks


@dataclass
class DocumentChunk:
    """A chunk of a document with context."""

    text: str
    chunk_index: int
    total_chunks: int
    section_titles: list[str]
    start_char: int
    end_char: int


@dataclass
class ChunkingResult:
    """Result of document chunking."""

    chunks: list[DocumentChunk]
    document_outline: str
    total_tokens_estimate: int
    was_chunked: bool


class DocumentChunker:
    """Splits long documents into semantic chunks for processing.

    Uses section headers to find natural break points, falling back
    to paragraph boundaries when sections aren't detected.
    """

    # Regex patterns for section detection
    SECTION_PATTERNS = [
        # Numbered sections (1., 1.1, etc.)
        r"^(?:\d+\.)+\s+[A-Z][^\n]+$",
        # All-caps headers
        r"^[A-Z][A-Z\s]{3,}[A-Z]$",
        # Title case with colon
        r"^[A-Z][a-z]+(?:\s+[A-Za-z]+)*:\s*$",
        # Markdown headers
        r"^#{1,4}\s+.+$",
        # Common section names
        r"^(?:Abstract|Introduction|Background|Methods?|Results?|Discussion|"
        r"Conclusion|References|Appendix|Related Work|Experiments?|"
        r"Implementation|Evaluation|Analysis|Summary)\s*$",
    ]

    def __init__(
        self,
        token_limit: int = DEFAULT_TOKEN_LIMIT,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """Initialize chunker with configuration.

        Args:
            token_limit: Maximum tokens per chunk (approximate).
            chunk_overlap: Characters of overlap between consecutive chunks.
        """
        self.token_limit = token_limit
        self.char_limit = token_limit * CHARS_PER_TOKEN
        self.chunk_overlap = chunk_overlap
        self._section_regex = re.compile(
            "|".join(f"({p})" for p in self.SECTION_PATTERNS),
            re.MULTILINE,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        return len(text) // CHARS_PER_TOKEN

    def needs_chunking(self, text: str) -> bool:
        """Check if document exceeds token limit and needs chunking.

        Args:
            text: Document text.

        Returns:
            True if document should be chunked.
        """
        return self.estimate_tokens(text) > self.token_limit

    def extract_sections(self, text: str) -> list[tuple[int, str]]:
        """Find section boundaries in document.

        Args:
            text: Document text.

        Returns:
            List of (position, section_title) tuples.
        """
        sections = []
        for match in self._section_regex.finditer(text):
            title = match.group().strip()
            sections.append((match.start(), title))
        return sections

    def build_outline(self, sections: list[tuple[int, str]]) -> str:
        """Build document outline from section list.

        Args:
            sections: List of (position, title) tuples.

        Returns:
            Formatted outline string.
        """
        if not sections:
            return "(No clear section structure detected)"

        lines = ["Document Structure:"]
        for _, title in sections[:20]:  # Limit to 20 sections
            lines.append(f"- {title}")
        if len(sections) > 20:
            lines.append(f"- ... ({len(sections) - 20} more sections)")
        return "\n".join(lines)

    def chunk_by_sections(
        self,
        text: str,
        sections: list[tuple[int, str]],
    ) -> list[DocumentChunk]:
        """Split document at section boundaries.

        Args:
            text: Full document text.
            sections: List of (position, title) tuples.

        Returns:
            List of DocumentChunk objects.
        """
        if not sections:
            return self._chunk_by_paragraphs(text)

        chunks = []
        current_start = 0
        current_sections: list[str] = []
        current_text_parts: list[str] = []

        # Add positions for start and end
        positions = [(0, "START")] + sections + [(len(text), "END")]

        for i in range(len(positions) - 1):
            pos, title = positions[i]
            next_pos = positions[i + 1][0]

            section_text = text[pos:next_pos]
            current_text_parts.append(section_text)
            if title != "START":
                current_sections.append(title)

            # Check if we've exceeded the limit
            combined_text = "".join(current_text_parts)
            if len(combined_text) > self.char_limit and len(current_text_parts) > 1:
                # Remove last section and create chunk
                current_text_parts.pop()
                if title != "START" and current_sections:
                    current_sections.pop()

                chunk_text = "".join(current_text_parts)
                chunks.append(
                    DocumentChunk(
                        text=chunk_text.strip(),
                        chunk_index=len(chunks),
                        total_chunks=0,  # Set later
                        section_titles=current_sections.copy(),
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                    )
                )

                # Start new chunk with overlap
                overlap_start = max(0, len(chunk_text) - self.chunk_overlap)
                current_start = current_start + overlap_start
                current_text_parts = [chunk_text[overlap_start:], section_text]
                current_sections = [title] if title != "START" else []

        # Final chunk
        if current_text_parts:
            chunk_text = "".join(current_text_parts)
            chunks.append(
                DocumentChunk(
                    text=chunk_text.strip(),
                    chunk_index=len(chunks),
                    total_chunks=0,
                    section_titles=current_sections.copy(),
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                )
            )

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _chunk_by_paragraphs(self, text: str) -> list[DocumentChunk]:
        """Fall back to paragraph-based chunking when no sections detected.

        Args:
            text: Document text.

        Returns:
            List of DocumentChunk objects.
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_parts: list[str] = []
        current_start = 0
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_with_sep = para + "\n\n"
            para_len = len(para_with_sep)

            if current_len + para_len > self.char_limit and current_parts:
                # Create chunk
                chunk_text = "\n\n".join(current_parts)
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        chunk_index=len(chunks),
                        total_chunks=0,
                        section_titles=[],
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                    )
                )

                # Start new chunk with overlap
                overlap_text = chunk_text[-self.chunk_overlap :] if chunks else ""
                current_start = current_start + len(chunk_text) - len(overlap_text)
                current_parts = [overlap_text, para] if overlap_text else [para]
                current_len = len(overlap_text) + para_len
            else:
                current_parts.append(para)
                current_len += para_len

        # Final chunk
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    section_titles=[],
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                )
            )

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def chunk(self, text: str) -> ChunkingResult:
        """Chunk a document if needed.

        Args:
            text: Full document text.

        Returns:
            ChunkingResult with chunks and metadata.
        """
        total_tokens = self.estimate_tokens(text)

        if not self.needs_chunking(text):
            # Single chunk for small documents
            return ChunkingResult(
                chunks=[
                    DocumentChunk(
                        text=text.strip(),
                        chunk_index=0,
                        total_chunks=1,
                        section_titles=[],
                        start_char=0,
                        end_char=len(text),
                    )
                ],
                document_outline="(Single document, no chunking needed)",
                total_tokens_estimate=total_tokens,
                was_chunked=False,
            )

        # Extract sections and build outline
        sections = self.extract_sections(text)
        outline = self.build_outline(sections)

        # Chunk the document
        chunks = self.chunk_by_sections(text, sections)

        return ChunkingResult(
            chunks=chunks,
            document_outline=outline,
            total_tokens_estimate=total_tokens,
            was_chunked=True,
        )
