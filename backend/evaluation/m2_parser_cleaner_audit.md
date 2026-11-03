# M2 development parser and cleaner audit

Date: 2026-10-24. Code: `72f9cf3` plus the uncommitted audit fix set at execution.
Versions: `pymupdf-1.26.7:structured-v2`, `conservative-v2`.

## Scope and acceptance correction

The approved v6 checklist, sections B and M2 K3/K3b, requires automatic fixtures
and index consistency in M2. Formal human PDF audit and independent Gold belong
to M8. Earlier project notes incorrectly treated six-type / 30+20 human review
as a M2 gate. Those statements are superseded, not additional user obligations.

The user delegated remaining development inspection to Codex and requested no
further per-sample prompts. Codex inspected the original PDF crops for all 50
prepared rows alongside raw/cleaned text, and representative full pages. These
are model-assisted debugging observations, not human truth labels, a held-out
test set, or a real-PDF parsing accuracy estimate. The user's one sample-3
line-join confirmation is retained separately; it confirms no other criteria.

## Frozen development sample

16 public PDF files, 356 pages; 30 suspicious and 20 ordinary blocks, seed
`20261024`, stratified by paper. Ordinary controls include many short math,
table and metadata fragments; they do not represent ordinary prose generally.
The set has been used to diagnose and tune the parser.

- Frozen CSV SHA-256: `972dbebbe0b5aa5032bb5f538f7cb1d46f85f4d716c5d6d36432445810b0aa6b`.
- Pre-fix Codex decisions SHA-256: `c9c382ad9dc492ecd55cdf9be9ba9606c8c08aab62e0a5011c618da635a1286f`.
- Per-PDF byte hashes, raw excerpts, decisions, images and mappings remain in
  Git-ignored `data/parsing/m2_human_review_20261024/`.

## Observations and fixes

| Development check | Before | After / interpretation |
| --- | --- | --- |
| Visible source fragment and order | 49/50 matched in Codex inspection | Sample13 colored labels reassembled in place; no full second visual accuracy study |
| Cleaning transforms | 15/15 transformed rows judged text-preserving | Applies to the extracted raw text, not proof of parser fidelity |
| Reference flags against Codex debugging decisions | TP11 / FP1 / FN6 / TN32 | TP17 / FP1 / FN0 / TN32; one different false positive remains, described below |
| Frozen-span correspondence after reparse | Not applicable | All50 spatial counterparts found;49 old normalized texts remain contained. Sample13 changes because missing inline fragments were reinserted |
| Transcript anchors | Not a visual metric | All15,591 blocks across356 pages match their own canonical page slices, zero integrity violations |

No human precision/recall, OCR accuracy or quality improvement percentage is
calculated from these observations. The intermediate recheck before fixing
cross-page section carry had TP8 / FP0 / FN9 / TN33; that failed development
result prompted the final carry regression and is not presented as acceptance.

Fixes with failing-before / passing-after regressions:

1. Bibliography years and page ranges no longer become numeric section headings.
2. The latest actual heading carries to the next page; an ambiguous continuation
   in the other column cannot restore an older section.
3. Near-touching colored/font runs join before column detection; positive word
   gaps are preserved. In sample13 (`arxiv-2310.11511.pdf`, PDF p.22), labels had
   been placed later on the page, not globally deleted by the cleaner.
4. Narrow real column gutters are recognized. In `arxiv-2307.03172.pdf`, PDF p.6,
   the left column now finishes before the right column begins.
5. Isolated centered physical page numbers are retained but flagged as markers.
   Numeric tables are exempt; QA/compare exclude pure markers, while citation
   endpoints still resolve their raw source.

A fresh independent code review reproduced two P2 defects in the initial fixes:
word-gap concatenation and numeric table suppression. Exact-text and lower-page
ruled/unruled number controls failed before correction and now pass. Column
carry also has a two-page regression. Final verification: **848 backend tests
passed**, 16 existing warnings, 81.43s. Ruff check and format check passed for
141 files. Targeted final regression set: 47 passed, 0.62s.

## Six input types

| Type | Source / check | Result and limit |
| --- | --- | --- |
| Two columns | `arxiv-2307.03172.pdf`, PDF p.6 | Full-width figure above two columns; corrected order verified against page and automatic sequence check; heuristic confidence retained |
| Table | `arxiv-1106.1265.pdf`, PDF p.2 | Simple3×3 ruled table cells and multiline cell retained. Complex side-by-side tables in `arxiv-2005.11401.pdf`, PDF p.6 remain low-confidence; cell alignment not certified |
| Caption | `arxiv-1106.1265.pdf`, PDF p.2 | Table1 caption retained with source span and caption type |
| Formula | `arxiv-2212.02407.pdf`, PDF p.5 | Equations2.1/2.2 retain symbol fragments with review flags; fractions flatten, no LaTeX/layout fidelity claim |
| Scan | Real table page rendered to an image-only in-memory PDF | Typed `no_extractable_text`, no invented OCR. Mixed-page fixture preserves unsupported page2 and page3 numbering; not a real-scanner corpus audit |
| Abstract | QASPER dev0.3, paper1503.00841 | Actual public abstract text preserved; abstract scope, null PDF pages/geometry; source hash saved |

The initially selected “formula” page was a table/bullet example; it was
replaced with the genuine equation page above. A symbol-font flag alone is
insufficient to establish the presence of a formula.

## Remaining cases for M8

Sample12 (`arxiv-2309.15217.pdf`, PDF p.8) is an appendix example, incorrectly
inheriting a bibliography section because the preceding appendix heading is
split into a letter and title and is not recognized. It is a known false
positive in the final debugging count; raw text remains available, but pure
misclassified children can be excluded from QA evidence. Same-page ambiguous
column sections, complex tables and fraction layout also need independent M8
audit. Offset printed page numbers are conservatively left unclassified.

Formal real-PDF accuracy, semantic citation support, parent/child retrieval
quality and complete DeepEval generation quality remain unmeasured here.

## Saved artifacts and local index

All paths below are relative to `backend/` and ignored by Git:

- `data/parsing/m2_human_review_20261024/codex_review/summary.json`:
  source/sample hashes, pre/post debugging counts and transcript-integrity checks.
- Same directory: `decisions.jsonl`, `assistant_review_50.csv`, `sample-*.png`,
  `six_type_review.json`, `scan_derived_check.json`, `abstract_actual.json`.
- `postfix/`: reparsed documents, cleaned mappings and frozen-span rechecks.
- `data/evaluation/m2_v2_live_20261024/report.json`: backed-up local reindex;
  25 parents /93 current children, all93 source spans verified;181 historical
  citations retain raw hashes (88 legacy,93 retired); other datasets remain
  at1,910 Qdrant points. New library scope has274 points, with active filters.

The library lacks original PDF bytes for its text-bearing legacy paper. Its
immutable parser-v1 snapshot and null PDF pages are retained honestly; only the
cleaner-v2 index generation changes. New PDF imports use parser-v2. No paid
provider calls, generation calls or Judge calls were made for this audit.
