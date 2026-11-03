# Local reliability and maintenance

M7 retains SQLite originals as the source of truth. Qdrant is a derived index;
versioned source rows and citations remain resolvable after index repair.
Run the following commands from `backend/` with the application stopped for
maintenance. Back up the SQLite database using SQLite's backup API first.

```powershell
uv run alembic upgrade head
uv run python -m evaluation.check_index
uv run python -m evaluation.check_index --repair
```

The default command inspects actual FTS postings and this configured dataset's
Qdrant points. A nonzero exit means inconsistency or an unavailable dependency.
Repair serially rebuilds derived FTS postings, republishes invalid native
generations, migrates missing legacy vectors when needed and retries durable
scoped deletion jobs. It does not delete other datasets or original source
history. A second successful repair should have no remaining work. Keep the
reported error type and rerun after restoring the failed dependency; do not
mark an unavailable index consistent.

Embedding reuse requires the same model, dimensions and exact normalized
input with validated vectors. Unchanged physically valid generations skip
embedding; changed inputs are embedded and complete publication is verified
by readback. Arbitrary rechunking may rebuild the whole paper. Local BGE
weights live in `data/models/embedding`, configurable via
`FASTEMBED_CACHE_DIR`; the reranker uses `data/models/reranker`. Prepare these
public model caches before Local-only requests, which cannot download weights.
Temporary-directory caches are insufficient for reproducible offline use.

The Library publication panel reads cached state on demand. Its explicit
public-DOI check makes at most two bounded Crossref metadata requests.
Unknown means no verified event is available. It does not establish that a
paper is unretracted. DOI changes invalidate old event provenance; failed
refreshes retain known events with a stale warning. Verified related versions
are displayed as relations, not independent studies or authority rankings.

QA and comparison results offer **Report an issue** when their local run
manifest was saved. Six reasons refer to the whole answer or an admitted source.
Only minimum source identity, DOI/safe URL/granularity, reason, config and time
are retained. External full text and private question text are not copied to
feedback. Real feedback stays `needs_review`; synthetic guard results are not
semantic fixes, training data or human Gold. Fix references are opaque user
labels, not verified commits. The feedback API exposes local regression creation,
guard execution and fix-reference recording for development diagnostics.

QA, follow-up and comparison share a request-local resource budget:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `REQUEST_DEADLINE_SECONDS` | 240 | Overall async deadline, maximum 300 |
| `REQUEST_MAX_MODEL_ATTEMPTS` | 12 | LLM and remote embedding attempts, including retries |
| `REQUEST_MAX_INPUT_BYTES` | 262144 | Conservative UTF-8 input reservation |
| `REQUEST_MAX_OUTPUT_TOKENS` | 16384 | Reserved output ceilings across attempts |
| `REQUEST_MAX_EXTERNAL_ATTEMPTS` | 12 | Academic HTTP and remote model attempts |
| `REQUEST_MAX_CURRENCY` | unset | Optional ceiling using explicitly configured unit prices |

Input/output prices use `REQUEST_INPUT_PRICE_PER_MILLION` and
`REQUEST_OUTPUT_PRICE_PER_MILLION`; remote embedding has its separate
`REQUEST_EMBEDDING_PRICE_PER_MILLION`. Unknown prices fail closed when a
currency ceiling is set. Reservations are not invoices or observed token
usage. Local Ollama has zero provider API currency in this accounting, while
CPU/GPU time, memory and energy remain costs. Local BGE work is bounded by
retrieval's existing variant/batch/stage limits, not counted as a remote model
attempt. Native worker threads can finish after async cancellation; this is
not process isolation or a hard CPU interrupt.

Traces retain observed calls/tokens separately from reservations, generation
and verification stage timings/usage, external attempts, failures and total
elapsed time. Optional stage p50/p95 values describe observed samples only.
Missing stages are not assigned fabricated zero latency. No unverified answer
streaming is enabled. Credential-bearing generated claims and evidence are
rejected; a final response guard withholds any residual configured credential.

Actual local runtime diagnostics use a disposable synthetic corpus and real
SQLite FTS, BGE, Qdrant local engine, reranker and Ollama:

```powershell
uv run python -m evaluation.reliability_benchmark --output data/evaluation/m7_run
```

Use a fresh output directory. The command makes at least 30 serial requests;
it adapts storage to an isolated in-memory Qdrant engine, so it does not test
Docker/server transport, concurrent capacity or independent scientific accuracy.
Raw traces stay under ignored `data/`; share the sanitized development report
and separate metrics history instead of private library data.
