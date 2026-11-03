# Reproducible walkthrough and recording script

Use a separate demo database with public or synthetic papers. Keep credentials,
private questions and user-library titles out of recordings. Record the actual
model/config/source revision and whether Qdrant/Ollama are healthy.

1. Open Discover. Search a public research topic. Explain that provider failure
   can leave partial results, that citation/freshness labels are source signals,
   and that bibliographic metadata is not scientific result evidence.
2. Import an allowed public result. Check its actual full-text/abstract/metadata
   scope. Show a PDF page only if the original PDF was acquired and parsed.
3. Open Library. Inspect content scope and cached publication status. An unknown
   status is shown explicitly. A public DOI refresh is a separate user action.
4. Ask a question with a paper selected and Local-only enabled. Show the actual
   routing and any rewrite fallback; do not edit the response for recording.
5. Open the returned original citation. Match source ID, exact quote and PDF
   file page, or null page for abstracts/provided text. Explain an evidence-only
   response if generation/support verification is unavailable.
6. Compare two papers. Show independent source coverage and unknown cells.
   Different numbers alone do not establish comparable scientific conflict.
7. Report one answer issue using a source and an error reason. Show the local
   acknowledgement; explain that the feedback is pending review and does not
   automatically train a model or become Gold.
8. Demonstrate permitted Web fallback on public questions separately from
   Local-only. Identify temporary licensed/OA evidence, abstract-only results,
   metadata-only records and service failures accurately. Never imply a fixture
   response was a live provider result.

Suggested recording length is3–5 minutes. Keep failures and unknown information
visible when they occur; attach a separate run log rather than splicing a
false successful flow.

For a future qualitative study, invite3–5 actual participants when available.
Use these tasks, record time/clicks/misunderstandings and severity, then open
issues for observed serious problems. Current execution has no participant
observations; its developer walkthrough is not a user study. Completion rates
from a small qualitative sample must not be presented as population estimates.
