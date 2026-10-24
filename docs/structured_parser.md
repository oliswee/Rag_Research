# Structured parser and conservative cleaning (M2 foundation)

This implements the K3/K3b engineering foundation. Upload and discovery import
now stage versioned parent/child sources and separately publish verified vectors;
see [parent_child_schema.md](parent_child_schema.md) for the K7 contract.

The parser retains `PDFParseResult.text` and `PDFPage.text`. Each page also has a
deterministic `structured_text` transcript and blocks with source/document IDs,
one-based PDF page, bounding box, section ID, type and confidence. Block offsets
address **that page's structured transcript**, rather than PDF byte offsets or
the separate legacy text. The input bytes determine `document_version`; the
parser version includes PyMuPDF and the structured-extraction algorithm version.
The same input and parser produce the same block IDs. Equal paragraphs on
different pages have separate anchors.

Column ordering, paragraph grouping and ruled-table detection are heuristics.
They keep uncertainty flags and do not establish real-PDF parsing accuracy.
Column-local headings keep separate section state; ambiguous right-column
content is not assigned to a left-column section. Lines overlapping a detected
table boundary are retained in full with a review flag, even if that duplicates
the portion inside the table.
Formula text is retained with a review flag. Pages without extractable text keep
their original page numbers and `unsupported_no_text`; an entirely unreadable
PDF raises a typed `no_extractable_text` error. No OCR text is invented. Legitimate
abstracts have source IDs and null PDF pages.

The cleaner keeps raw block text and creates separate search text. Conservative
rules normalize CRLF, remove an explicit soft hyphen between letters, collapse
horizontal whitespace, and join a lower-case continuation within a paragraph.
Visible hyphens, numbers at a line boundary, paragraph boundaries, table/formula
text and uncertain layouts are retained. Repeated margin text is flagged after
agreement across three pages; it is not removed. Reference sections are marked
and retained. These choices favor preserving evidence until M8 review.
Numbered reference headings such as `7 References` are supported.

Every transformation records original page-transcript offsets and final cleaned
offsets. `CleanedBlock.raw_span(start, end)` returns a covering raw range for a
nonempty cleaned range, including characters removed within that range. Citation
evidence must use the raw text; cleaned text is for search. Rechunking must use
document/page/raw anchors as the source of truth, independently of chunk IDs.

Run the local inspection exporter from `backend/`:

```powershell
uv run python -m evaluation.inspect_parser --input paper.pdf --output data/parsing/paper.json --review-queue data/parsing/parser_review_queue.jsonl
```

The artifact includes raw/structured/cleaned text, source mappings, transformation
logs and uncertainty flags. Keep original PDFs alongside their hash-identified
artifacts for later review. The queue contains only version/anchor/flag metadata,
not extracted excerpts, and remains `unreviewed`. Repeated inspection deduplicates
identical queue records. Entirely unreadable PDFs produce an unsupported artifact
and a document-level review record with a null page. Input, artifact and queue
paths must be distinct. Default outputs stay under Git-ignored local `data/`.

Validation uses synthetic layouts and automatic source-span checks. M2 remains
open until its required human checks exist: six representative sample types and
at least 30 suspicious plus 20 ordinary cleaning fragments. M8 Gold labeling is
a separate, later requirement. Automated integrity checks do not satisfy either
human review gate.
