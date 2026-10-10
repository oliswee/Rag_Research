"""PDF text extraction service using PyMuPDF."""

from dataclasses import dataclass

import fitz  # PyMuPDF


class PDFParseError(Exception):
    """Raised when PDF parsing fails."""

    pass


@dataclass
class PDFParseResult:
    """Result of PDF parsing."""

    text: str
    page_count: int
    metadata: dict  # PDF internal metadata (title, author, etc.)


class PDFParserService:
    """Service for extracting text from PDF files."""

    def __init__(self, max_pages: int = 200):
        """Initialize parser with page limit.

        Args:
            max_pages: Maximum pages to process. Raises error if exceeded.
        """
        self.max_pages = max_pages

    def parse(self, pdf_bytes: bytes) -> PDFParseResult:
        """Extract full text from PDF.

        Args:
            pdf_bytes: Raw PDF file content.

        Returns:
            PDFParseResult with extracted text and metadata.

        Raises:
            PDFParseError: If PDF is invalid, encrypted, or exceeds page limit.
        """
        try:
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
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text("text")  # Plain text extraction
                if text.strip():
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                raise PDFParseError(
                    "PDF contains no extractable text (may be image-only)"
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
            )
        finally:
            doc.close()

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
