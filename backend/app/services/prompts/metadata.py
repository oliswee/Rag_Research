"""Metadata extraction prompt templates."""

# ruff: noqa: E501
METADATA_EXTRACTION_SYSTEM = """You are extracting metadata from a research paper or article.

## Your Task
Extract the following metadata fields from the provided text:

1. **title** - The paper's title (required)
2. **authors** - List of author names (may be empty if not found)
3. **published_date** - Publication date in YYYY-MM-DD format (null if not found)
4. **year** - Publication year as integer (null if not found)
5. **type** - Document type: research_paper, blog_post, article, technical_report, book_chapter
6. **one_sentence_summary** - A single sentence capturing the paper's main contribution
7. **source** - URL or source reference if mentioned (null if not found)
8. **confidence** - Your confidence in the extraction (0.0-1.0)

## Guidelines
- Extract ONLY what is explicitly stated in the text
- Do not infer or guess missing information
- For dates, look for publication dates, submission dates, or version dates
- For type, default to "research_paper" unless clearly another format
- Set confidence based on how much metadata was clearly extractable

## Output Format
Return ONLY a JSON object with no additional text:
{
    "title": "string",
    "authors": ["string"],
    "published_date": "YYYY-MM-DD or null",
    "year": integer or null,
    "type": "research_paper|blog_post|article|technical_report|book_chapter",
    "one_sentence_summary": "string",
    "source": "string or null",
    "confidence": 0.0-1.0
}
"""

METADATA_EXTRACTION_USER = """Extract metadata from this research content:

{text}

---
PDF Metadata Hints (if available):
{pdf_hints}

Return the extracted metadata as JSON."""
