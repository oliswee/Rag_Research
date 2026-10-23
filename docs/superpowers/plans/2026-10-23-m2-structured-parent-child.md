# M2 implementation plan

Source: user-approved FolioTrace M1–M8 v6 checklist, M2 K3/K3b/K7.
Depends on M1 branch; merge remains a separate authorized operation.

## Builder decisions

Keep Python/FastAPI, PyMuPDF, SQLite/FTS5 and local BGE/Qdrant. Add structured
output alongside existing PDF text/pages so import APIs remain usable throughout
the work. Treat layout and formula classification as heuristics with explicit
confidence. Preserve original extracted text and page numbers. No OCR accuracy,
real-PDF quality or human-Gold claim is made from synthetic tests.

## Task 1: Structured parser foundation (K3)

Add block records containing document/source identity, source page, bounding box,
block type, section information, raw offsets and confidence flags. Preserve
legacy result.text/page.text. Detect layout uncertainty and table/formula
limitations rather than silently flattening them. Whole scanned PDFs retain a
typed unsupported parse error; mixed scanned pages are preserved as unsupported
page records. Add abstract parsing with page=null. Use SHA-256 document identity
and deterministic block IDs, not runtime UUIDs.

RED/GREEN: synthetic two-column, table, caption, formula, image-only/mixed scan,
abstract and repeated-page text fixtures. Assert actual extracted text, spans,
source identities and confidence behavior, including failure paths. Run existing
parser/import tests to preserve compatibility. Commit only after verification.

## Task 2: Conservative cleaning (K3b)

Clean only supported transformations (bounded whitespace/line joins and explicit
soft hyphens); preserve ambiguous visible hyphens and repeated results. Store
raw/cleaned offsets for each change. Repeated headers/footers require positional
and multi-page agreement and are flagged rather than erased by default. Keep
references and tables, marking boundaries. Produce an ignored review queue with
document/span/flag metadata for M8. Test both modify/retain examples for each rule.

## Task 3: Versioned schema and original anchors (K7)

Add immutable document versions and parent records. Extend children with parent,
document/index/parser/chunker/model versions and original spans. Raw child text
is citation evidence; cleaned child text is for search. Stable original anchors
include paper/document/page/raw offsets independently of child chunk size. Add a
migration that preserves existing source rows as legacy and makes missing
structured provenance explicit rather than fabricating it.

## Task 4: Version-consistent index build and retrieval

Build new parents/children deterministically and activate one per-paper index
version atomically in SQLite. Only query matching active versions in both FTS5
and Qdrant; incomplete vector publication must permit safe BM25 fallback. Retain
old source versions/anchors for old citations or return an explicit stale result.
Never reassign an old ID to changed text. Implement per-paper reindex tooling,
index repair fixtures and physical revision manifests. Wire both upload and
open-access import paths. Update M1 Trace with the now-available parent/spans and
verified version metadata.

## Task 5: Verification and delivery

Run migration upgrades against a copied local database first. Test cross-page,
duplicate paragraphs, table/abstract/no-text boundaries, rechunking, pending
vectors, interrupted publication and old citation behavior. Produce
parent_child_schema.md and source-jump examples. Full backend tests, formatting,
privacy inspection and one fresh code review before a dependent draft PR. Check
only requirements with real evidence; K3/K3b can land incrementally while M2
stays open until K7 acceptance is complete.

Pre-flight: parser block spans feed cleaner mappings; both feed immutable
document/parent/child records. Index version is shared across SQLite/FTS/Qdrant;
retrieval must filter the active version before ranking/resolution. This makes
index publication order and pending-vector fallback central review risks.
