"""Tests for PDF parser service."""

import pytest
import fitz

from app.services.pdf_parser import PDFParseError, PDFParseResult, PDFParserService


def test_parse_preserves_one_based_page_text():
    document = fitz.open()
    for content in ("First page evidence", "Second page evidence"):
        page = document.new_page()
        page.insert_text((72, 72), content)
    pdf_bytes = document.tobytes()
    document.close()

    result = PDFParserService().parse(pdf_bytes)

    assert [(page.number, page.text.strip()) for page in result.pages] == [
        (1, "First page evidence"),
        (2, "Second page evidence"),
    ]
    assert "First page evidence" in result.text
    assert "Second page evidence" in result.text

# --- Fixtures ---


@pytest.fixture
def sample_pdf_bytes():
    """Create minimal valid PDF bytes for testing.

    This creates a simple PDF with one page containing text.
    """
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""
    return pdf_content


@pytest.fixture
def multi_page_pdf_bytes():
    """Create a PDF with multiple pages for testing page limits."""
    # This creates a 3-page PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 6 0 R 9 0 R] /Count 3 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Page 1 Content) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
6 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 7 0 R /Resources << /Font << /F1 8 0 R >> >> >>
endobj
7 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Page 2 Content) Tj ET
endstream
endobj
8 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
9 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 10 0 R /Resources << /Font << /F1 11 0 R >> >> >>
endobj
10 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Page 3 Content) Tj ET
endstream
endobj
11 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 12
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000123 00000 n
0000000274 00000 n
0000000367 00000 n
0000000432 00000 n
0000000583 00000 n
0000000676 00000 n
0000000741 00000 n
0000000893 00000 n
0000000987 00000 n
trailer
<< /Size 12 /Root 1 0 R >>
startxref
1052
%%EOF"""
    return pdf_content


# --- PDFParserService Initialization Tests ---


class TestPDFParserServiceInit:
    """Tests for PDFParserService initialization."""

    def test_init_default_max_pages(self):
        """Parser initializes with default max pages of 200."""
        parser = PDFParserService()
        assert parser.max_pages == 200

    def test_init_custom_max_pages(self):
        """Parser accepts custom max pages."""
        parser = PDFParserService(max_pages=10)
        assert parser.max_pages == 10

    def test_init_zero_max_pages(self):
        """Parser accepts zero max pages (blocks all PDFs)."""
        parser = PDFParserService(max_pages=0)
        assert parser.max_pages == 0

    def test_init_large_max_pages(self):
        """Parser accepts large max pages value."""
        parser = PDFParserService(max_pages=1000)
        assert parser.max_pages == 1000


# --- PDFParserService.parse() Tests ---


class TestPDFParserServiceParse:
    """Tests for PDFParserService.parse() method."""

    def test_parse_valid_pdf(self, sample_pdf_bytes):
        """Parser extracts text from valid PDF."""
        parser = PDFParserService()
        result = parser.parse(sample_pdf_bytes)

        assert result.text is not None
        assert len(result.text) > 0
        assert result.page_count == 1
        assert isinstance(result.metadata, dict)

    def test_parse_returns_pdf_parse_result(self, sample_pdf_bytes):
        """Parser returns PDFParseResult dataclass."""
        parser = PDFParserService()
        result = parser.parse(sample_pdf_bytes)

        assert isinstance(result, PDFParseResult)
        assert hasattr(result, "text")
        assert hasattr(result, "page_count")
        assert hasattr(result, "metadata")

    def test_parse_extracts_text_content(self, sample_pdf_bytes):
        """Parser extracts actual text content from PDF."""
        parser = PDFParserService()
        result = parser.parse(sample_pdf_bytes)

        # Our sample PDF contains "Test PDF Content"
        assert "Test" in result.text or len(result.text) > 0

    def test_parse_extracts_metadata_fields(self, sample_pdf_bytes):
        """Parser extracts PDF metadata fields."""
        parser = PDFParserService()
        result = parser.parse(sample_pdf_bytes)

        # Metadata should have expected keys (may be empty strings)
        assert "title" in result.metadata
        assert "author" in result.metadata
        assert "subject" in result.metadata
        assert "creator" in result.metadata

    def test_parse_multi_page_pdf(self, multi_page_pdf_bytes):
        """Parser handles multi-page PDFs."""
        parser = PDFParserService(max_pages=10)
        result = parser.parse(multi_page_pdf_bytes)

        assert result.page_count == 3

    def test_parse_joins_pages_with_double_newline(self, multi_page_pdf_bytes):
        """Parser joins page content with double newlines."""
        parser = PDFParserService(max_pages=10)
        result = parser.parse(multi_page_pdf_bytes)

        # Pages should be joined with \n\n
        assert "\n\n" in result.text or result.page_count == 1


# --- PDFParserService.parse() Error Tests ---


class TestPDFParserServiceParseErrors:
    """Tests for PDFParserService.parse() error handling."""

    def test_parse_invalid_bytes_raises_error(self):
        """Parser raises PDFParseError for invalid PDF bytes."""
        parser = PDFParserService()

        with pytest.raises(PDFParseError) as exc_info:
            parser.parse(b"not a pdf")

        assert "Failed to open PDF" in str(exc_info.value)

    def test_parse_empty_bytes_raises_error(self):
        """Parser raises PDFParseError for empty bytes."""
        parser = PDFParserService()

        with pytest.raises(PDFParseError):
            parser.parse(b"")

    def test_parse_exceeds_page_limit_raises_error(self, multi_page_pdf_bytes):
        """Parser raises PDFParseError when page limit exceeded."""
        parser = PDFParserService(max_pages=2)  # PDF has 3 pages

        with pytest.raises(PDFParseError) as exc_info:
            parser.parse(multi_page_pdf_bytes)

        assert "exceeds limit" in str(exc_info.value)
        assert "3 pages" in str(exc_info.value)

    def test_parse_corrupted_pdf_raises_error(self):
        """Parser raises PDFParseError for corrupted PDF."""
        parser = PDFParserService()
        # Truncated PDF header
        corrupted = b"%PDF-1.4\n1 0 obj\n<<"

        with pytest.raises(PDFParseError):
            parser.parse(corrupted)

    def test_parse_non_bytes_raises_error(self):
        """Parser raises error for non-bytes input."""
        parser = PDFParserService()

        with pytest.raises((TypeError, PDFParseError)):
            parser.parse("not bytes")  # type: ignore


# --- PDFParserService.validate_pdf() Tests ---


class TestPDFParserServiceValidate:
    """Tests for PDFParserService.validate_pdf() method."""

    def test_validate_valid_pdf_returns_true(self, sample_pdf_bytes):
        """Validation returns True for valid PDF."""
        parser = PDFParserService()
        is_valid, error = parser.validate_pdf(sample_pdf_bytes)

        assert is_valid is True
        assert error == ""

    def test_validate_invalid_pdf_returns_false(self):
        """Validation returns False for invalid PDF."""
        parser = PDFParserService()
        is_valid, error = parser.validate_pdf(b"not a pdf")

        assert is_valid is False
        assert "Invalid PDF" in error

    def test_validate_empty_bytes_returns_false(self):
        """Validation returns False for empty bytes."""
        parser = PDFParserService()
        is_valid, error = parser.validate_pdf(b"")

        assert is_valid is False
        assert len(error) > 0

    def test_validate_respects_page_limit(self, multi_page_pdf_bytes):
        """Validation respects page limit setting."""
        parser = PDFParserService(max_pages=2)  # PDF has 3 pages
        is_valid, error = parser.validate_pdf(multi_page_pdf_bytes)

        assert is_valid is False
        assert "pages" in error.lower()
        assert "limit" in error.lower()

    def test_validate_zero_page_limit_rejects_all(self, sample_pdf_bytes):
        """Validation with max_pages=0 rejects all PDFs."""
        parser = PDFParserService(max_pages=0)
        is_valid, error = parser.validate_pdf(sample_pdf_bytes)

        assert is_valid is False

    def test_validate_returns_tuple(self, sample_pdf_bytes):
        """Validation returns tuple of (bool, str)."""
        parser = PDFParserService()
        result = parser.validate_pdf(sample_pdf_bytes)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# --- PDFParseResult Tests ---


class TestPDFParseResult:
    """Tests for PDFParseResult dataclass."""

    def test_result_stores_text(self):
        """PDFParseResult stores text field."""
        result = PDFParseResult(
            text="sample text",
            page_count=1,
            metadata={},
        )
        assert result.text == "sample text"

    def test_result_stores_page_count(self):
        """PDFParseResult stores page_count field."""
        result = PDFParseResult(
            text="text",
            page_count=5,
            metadata={},
        )
        assert result.page_count == 5

    def test_result_stores_metadata(self):
        """PDFParseResult stores metadata field."""
        metadata = {"title": "Test", "author": "Author"}
        result = PDFParseResult(
            text="text",
            page_count=1,
            metadata=metadata,
        )
        assert result.metadata == metadata


# --- PDFParseError Tests ---


class TestPDFParseError:
    """Tests for PDFParseError exception."""

    def test_error_message(self):
        """PDFParseError includes message."""
        error = PDFParseError("Test error message")
        assert str(error) == "Test error message"

    def test_error_is_exception(self):
        """PDFParseError is a proper exception."""
        error = PDFParseError("test")
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self):
        """PDFParseError can be raised and caught."""
        with pytest.raises(PDFParseError):
            raise PDFParseError("test error")

    def test_error_preserves_cause(self):
        """PDFParseError preserves exception chain."""
        original = ValueError("original error")
        try:
            raise PDFParseError("wrapper") from original
        except PDFParseError as e:
            assert e.__cause__ is original
