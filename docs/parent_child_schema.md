# M2 K7: versioned sources, parent/child indexing and citation resolution

Implementation status (2026-10-24): schema, source builder, active-only retrieval,
publication verification, citation API and serial reindex tools implemented.
Full regression and independent review are recorded separately; this document
does not mark the entire M2 milestone complete. K3/K3b human audits are pending.

## Stored records

- SourceDocument: immutable per-paper extracted snapshot. IDs bind paper ID,
  source bytes/text version, parser version and canonical page/block snapshot
  digest. Store raw page transcripts, legacy text, block geometry/source flags,
  content scope and provenance. Rechunking reuses this snapshot.
- SourceIndexGeneration: immutable index_version key over document snapshot,
  cleaner/chunker settings, cleaned block/mapping manifest and embedding model.
  One active generation per paper, guarded by a partial unique index. Separate
  vector status (pending/ready/error) and physical publication manifest/revision.
- SourceParent: belongs to one generation/document. Reading unit composed of
  whole blocks grouped by section and bounded character budget. Tables/formulas
  remain intact as parents even if they exceed the target. Store original spans
  and raw/cleaned parent text.
- PaperChunk: child of a parent/generation/document. Keep raw citation text;
  cleaned_text is separate and search_text is computed as COALESCE(cleaned_text,
  text), preserving legacy rows. Source spans refer to immutable document page
  transcripts and include block geometry. Existing offsets are populated only
  for a single contiguous source span; abstract PDF pages stay null.
- Retain retired generation children and documents for old citation resolution.
  Replace unique(paper_id,chunk_index) with
  unique(paper_id,index_version,chunk_index); legacy rows get index_version=legacy
  and null structured foreign keys. New composite foreign keys prevent a child
  from pointing to another paper/document/generation's parent.

## Source anchors and IDs

Original anchor identity = paper/document snapshot + source document version +
parser version + page + raw offsets (and source block geometry). It does not use
the changing child ID. Child IDs bind generation + parent + cleaned slice; new
rechunking settings create new children and leave old ones resolvable. Validate
block text against its canonical transcript before persisting any snapshot.

Initial budgets: 1,200 characters per child and 4,800 per parent, explicitly
character heuristics rather than measured model token limits. Cleaner mappings
convert each child cleaned slice into one or more exact raw source spans. Returned
excerpts use only original source pieces; no invented page or normalized quote.

## Consistent retrieval and publication

SQLite activates a new generation atomically after all source/parent/child rows
are present. Active source content is searchable even when vectors are pending.
FTS uses an active-only content view, computed search text and a rebuild within
the same activation transaction, so retired rows do not distort BM25 statistics.

Qdrant payloads add generation/document/parent/model IDs. Dense searches receive
the active per-paper generation map from SQLite. Versioned points match active
generation IDs; legacy points are accepted only for papers without an active
structured generation. Staged or retired vectors never become active through a
collection-wide unfiltered search. Empty/stale dense resolution must allow BM25
fallback.

Publication upserts new generation vectors, retrieves those points including
vectors, validates scope/version/IDs/dimensions and stores a manifest hash of the
actual returned vectors. This is a physical revision verified at publication,
not a perpetual certificate against external changes. A failure leaves source
content and BM25 usable, vector_status pending/error, and no physical revision.
Cleanup of old vectors is optional because query version filters fence them.
Physical hashes cover returned float32 values exactly. Insert/update cosine
normalization may differ by one ULP, so a republish can change the physical hash
without changing the logical generation. Publication jobs must be serialized;
the collision preflight does not lock external Qdrant writers.
The publisher durably commits `pending` and clears old verification metadata
before any Qdrant write, including republication of an unchanged active version.
Its caller commits the final ready/error result. An interrupted publication thus
leaves pending vectors excluded, even if the process loses its final transaction.

Both upload and open-access/abstract import feed the same source builder. Add
per-paper reindex tooling. Legacy sources are migrated without invented
provenance; a text-only reindex can preserve null PDF pages when original bytes
are unavailable. Back up real SQLite first; verify migration on a copy and check
foreign keys/counts before applying it to the working library.

## Retrieval, citation and Trace

Returned evidence includes generation/document/parent IDs and original spans.
A source endpoint resolves current or retired child IDs to immutable raw source
and parent data, reporting active/retired status. Trace records span anchors and
parent IDs without source excerpts; physical revision fields state publication
verification time/scope. Add current/retired generation metadata to corpus
fingerprints while retaining secret-free config snapshots.
Only geometry and version/anchor IDs enter span Trace records. Raw source text,
source URLs and raw paper scope identifiers are excluded. A physical Trace hash
summarizes selected generation publications and explicitly carries publication
verification time; it does not attest to external changes after that time.

Pure bibliography children are retained and tagged `evidence_role=bibliography`;
QA and comparison exclude them from result evidence. Parser/cleaner quality flags
remain on resolved sources. Ambiguous mixed body/reference chunks require audit.

## Source jump and repair examples

Run these commands from `backend/` after `uv run alembic upgrade head`:

```powershell
uv run python -m evaluation.reindex_paper --paper-id PAPER_UUID --no-vectors
uv run python -m evaluation.reindex_paper --paper-id PAPER_UUID
uv run python -m evaluation.reindex_paper --paper-id PAPER_UUID --pdf original.pdf --child-size 1200 --parent-size 4800
```

`--no-vectors` commits source/FTS repair only. A vector failure leaves committed
sources and reports an exception type with a nonzero CLI exit. The all-paper
`evaluation.reindex` entrypoint uses the same generation builder, skips metadata
records without source text, preserves old child IDs and never deletes the
collection. Rebuilding a changed immutable row raises instead of silently
reusing its ID. Sources with no original PDF bytes use explicit
`legacy_extraction`, null PDF pages and no invented bounding boxes.

Each returned source has `citation_url=/api/sources/chunks/CHILD_UUID`:

```text
GET /api/sources/chunks/CHILD_UUID
  -> excerpt (unchanged original pieces), source_spans, parent.raw_text,
     parent.source_spans, document/parser/chunker/model/index versions,
     status=active|retired|legacy, quality_flags, original_pdf_url
GET /api/sources/documents/DOCUMENT_UUID/pdf
  -> preserved PDF bytes, or explicit 404 if no original PDF was stored
```

After changing chunk size, the first URL continues returning the original
child/parent/document and `status=retired`. Invalid stored text, checksums or
anchors produce 409; missing IDs produce 404. `raw_start/raw_end` always refer
to the immutable page's structured transcript, never PDF byte offsets. PDF
page numbers are one-based. Legacy citations explicitly have no structured
parent/spans or stored-PDF link.

## Required automatic checks

Same input rerun -> stable IDs/no duplicate rows. Rechunk -> new child IDs and
retained original anchors. Repeated text/pages and cross-page parents -> correct
distinct spans. Active-only BM25 and generation-filtered in-memory Qdrant -> no
retired/staged results. Pending/failed publication -> BM25 works. Old citations
resolve original text. Invalid cross-paper parent associations fail. Migration
preserves legacy IDs/evidence, and real-library backup/copy checks pass. These
are engineering tests. Six-type parser review and 30 suspicious/20 ordinary
cleaner fragment checks remain M2 human requirements. M8 Gold is separate.
