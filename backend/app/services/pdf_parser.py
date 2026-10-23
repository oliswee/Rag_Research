"""PDF text extraction service using PyMuPDF."""

import hashlib
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.services.pdf_structure import (
    PARSER_VERSION,
    PDFBlock,
    block_id,
    structure_page,
)


class PDFParseError(Exception):
    """Raised when PDF parsing fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "parse_failed",
        document_version: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.document_version = document_version


@dataclass
class PDFParseResult:
    """Result of PDF parsing."""

    text: str
    page_count: int
    metadata: dict  # PDF internal metadata (title, author, etc.)
    pages: list["PDFPage"] = field(default_factory=list)
    document_version: str = ""
    source_id: str = ""
    parser_version: str = PARSER_VERSION
    content_scope: str = "full_text"


@dataclass
class PDFPage:
    """Extracted text with its one-based source page number."""

    number: int | None
    text: str
    structured_text: str = ""
    blocks: list[PDFBlock] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    status: str = "extracted"
    width: float | None = None
    height: float | None = None


class PDFParserService:
    """Service for extracting text from PDF files."""

    def __init__(self, max_pages: int = 200):
        """Initialize parser with page limit.

        Args:
            max_pages: Maximum pages to process. Raises error if exceeded.
        """
        self.max_pages = max_pages

    def parse(self, pdf_bytes: bytes, *, source_origin: str = "pdf") -> PDFParseResult:
        """Extract full text from PDF.

        Args:
            pdf_bytes: Raw PDF file content.

        Returns:
            PDFParseResult with extracted text and metadata.

        Raises:
            PDFParseError: If PDF is invalid, encrypted, or exceeds page limit.
        """
        try:
            document_version = "sha256:" + hashlib.sha256(pdf_bytes).hexdigest()
            source_id = "pdf:" + document_version
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise PDFParseError(f"Failed to open PDF: {e}") from e

        try:
            # Check page limit
            if doc.page_count > self.max_pages:
                raise PDFParseError(
                    f"PDF has {doc.page_count} pages, exceeds limit of {self.max_pages}"
                )

            # Check for encryption
            if doc.is_encrypted:
                raise PDFParseError("PDF is encrypted and cannot be processed")

            # Extract text from all pages
            text_parts = []
            pages = []
            section_id = None
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text("text")  # Plain text extraction
                if text.strip():
                    text_parts.append(text)
                    try:
                        structured, blocks, flags, section_id = structure_page(
                            page, document_version, source_id, source_origin, section_id
                        )
                    except Exception:
                        # Layout is optional; never erase successfully extracted text.
                        structured = text
                        flags = ["structure_extraction_failed"]
                        section_id = None
                        blocks = [
                            PDFBlock(
                                id=block_id(
                                    document_version, page_num + 1, 0, len(text)
                                ),
                                document_version=document_version,
                                source_id=source_id,
                                page=page_num + 1,
                                text=text,
                                raw_start=0,
                                raw_end=len(text),
                                bbox=None,
                                kind="unstructured",
                                confidence="low",
                                source_origin=source_origin,
                            )
                        ]
                    pages.append(
                        PDFPage(
                            number=page_num + 1,
                            text=text,
                            structured_text=structured,
                            blocks=blocks,
                            quality_flags=flags,
                            status="low_confidence" if flags else "extracted",
                            width=page.rect.width,
                            height=page.rect.height,
                        )
                    )
                else:
                    pages.append(
                        PDFPage(
                            number=page_num + 1,
                            text=text,
                            status="unsupported_no_text",
                            quality_flags=["no_extractable_text"],
                            width=page.rect.width,
                            height=page.rect.height,
                        )
                    )

            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                raise PDFParseError(
                    "PDF contains no extractable text (may be image-only)",
                    code="no_extractable_text",
                    document_version=document_version,
                )

            # Extract PDF metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
            }

            return PDFParseResult(
                text=full_text,
                page_count=doc.page_count,
                metadata=metadata,
                pages=pages,
                document_version=document_version,
                source_id=source_id,
            )
        finally:
            doc.close()

    def parse_abstract(self, text: str) -> PDFParseResult:
        """Represent a legitimate abstract without fabricating a PDF page."""
        if not text.strip():
            raise PDFParseError("Abstract is empty", code="empty_abstract")
        version = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_id = "abstract:" + version
        blocks = []
        for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, re.DOTALL):
            end = match.start() + len(match.group().rstrip())
            blocks.append(
                PDFBlock(
                    id=block_id(version, None, match.start(), end),
                    document_version=version,
                    source_id=source_id,
                    page=None,
                    text=text[match.start() : end],
                    raw_start=match.start(),
                    raw_end=end,
                    bbox=None,
                    kind="abstract",
                    source_origin="abstract",
                )
            )
        return PDFParseResult(
            text=text,
            page_count=0,
            metadata={},
            document_version=version,
            source_id=source_id,
            content_scope="abstract",
            pages=[
                PDFPage(number=None, text=text, structured_text=text, blocks=blocks)
            ],
        )

    def validate_pdf(self, pdf_bytes: bytes) -> tuple[bool, str]:
        """Quick validation without full parsing.

        Args:
            pdf_bytes: Raw PDF file content.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = doc.page_count
            is_encrypted = doc.is_encrypted
            doc.close()

            if is_encrypted:
                return False, "PDF is encrypted"
            if page_count > self.max_pages:
                return False, f"PDF has {page_count} pages (limit: {self.max_pages})"
            if page_count == 0:
                return False, "PDF has no pages"

            return True, ""
        except Exception as e:
            return False, f"Invalid PDF: {e}"


def parser_warnings(result: PDFParseResult) -> list[str]:
    """Expose safe uncertainty codes to upload/import callers without excerpts."""
    return [
        f"PDF page {page.number}: {flag}"
        for page in result.pages
        for flag in getattr(page, "quality_flags", [])
        if isinstance(flag, str)
    ]
