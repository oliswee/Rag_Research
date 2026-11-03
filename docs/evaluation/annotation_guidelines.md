# Evidence annotation and independent review

The M8 preparation command freezes40 new QASPER train questions on40 papers
excluded from the earlier dev40. Public dataset answers are **reference
development labels**, not independently verified FolioTrace Gold. Automated
anchor checks and Codex visual/source review retain their own reviewer types.
No AI decision may be relabelled as a human decision.

Run from `backend/` with the downloaded QASPER v0.3 source files:

```powershell
uv run python -m evaluation.prepare_final --output data/evaluation/m8_final_new
```

The output preserves papers, document SHA-256, exact original text offsets,
questions, dataset-reference answer points and pending annotation rows. It also
contains the30 suspicious/20 ordinary PDF review forms and a checksummed freeze
manifest. Raw source text, questions, answers and screenshots remain under
ignored `data/`; do not commit a private library or credentials.

For every question, an independent reviewer must read the original source,
then record `question`, `question_type`, `paper_id`, `document_version`,
`raw_quote`, `source_offset`, `pdf_page_or_null`, `answerable_local`,
`required_evidence`, `expected_answer_points` and `expected_web_route`.
Offsets refer to the frozen structured transcript, never PDF byte offsets.
PDF page means one-based file page, not printed page numbering. Provided text
and abstracts have null PDF pages. Questions with no answer have no invented
quote, offset or answer points. Runtime chunk IDs are mapped separately and
must not replace stable original-source anchors.

Use `review_status=independent_human_reviewed`, `reviewer_type=human`, an actual
reviewer identifier/date and `source_checked=true` only after that work occurs.
The validator rejects pending/dataset/model provenance when human Gold is
required. Metadata alone cannot independently prove who performed a review;
retain the actual review record and uncertainty/disagreement notes. A single
human reviewer is acceptable if disclosed; do not invent a second reviewer.

Split reviewed questions into development and untouched final test subsets
before tuning. This M8 run is explicitly reference development: its public
labels are visible to diagnostics. It is not an untouched final human test.
Copy completed annotations to a new directory and freeze a new manifest;
never edit an existing frozen file while retaining its old hash. Changes to
parser/chunker/model/corpus require refreeze and re-evaluation.

For PDF review, compare the original rendered page with the current parser's
reading order, raw text and cleaned text. Review double columns, tables,
captions, formulas, page boundaries and abstract/scan fallback separately.
Mark flag correctness, transformation correctness, missed errors, original
anchor correctness and reading order. Suspicious is a sampling stratum, not
an error label. Old-parser offsets that moved after a known fix are review
candidates; they do not automatically establish a new parser defect. Serious
errors require a parser fix, reindex, new freeze and rerun.

Human claim-support review must check whether each atomic conclusion follows
from its quoted source, including population, study design, units, effect
measure, direction and uncertainty. An exact quote or a high DeepEval score
does not establish scientific entailment. Independently review at least10
failure attributions and the six local/Web/OA/abstract/metadata/fault/conflict
scenario categories before formal M8 acceptance.

Current delivery includes review-ready materials and separately labelled
automatic/AI checks. Independent human Gold, human PDF decisions, calibrated
claim support and real user-study observations remain pending until performed.
