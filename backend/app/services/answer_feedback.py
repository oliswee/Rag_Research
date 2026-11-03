"""Source-aware feedback and labelled synthetic guards, never automatic Gold."""

import hashlib
import re
from datetime import UTC, datetime

from sqlalchemy import select

from app.models import AnswerFeedback, AnswerRun, FeedbackRegression, Paper, PaperChunk
from app.services.academic_search import normalize_doi, safe_url
from app.services.evidence_registry import build_registry, seal_web_sources
from app.services.retrieval_trace import _safe_query
from app.services.source_anchors import resolve_child
from app.services.trace_versions import digest

REASONS = {
    "wrong_paper",
    "missed_evidence",
    "unsupported_citation",
    "incorrect_interpretation",
    "outdated_conflicting",
    "web_result_irrelevant",
}
GUARDS = {"wrong_paper", "unsupported_citation", "web_result_irrelevant"}
LEVELS = {"full_text", "abstract_only", "metadata_only"}


def iso(value):
    return value.replace(tzinfo=UTC).isoformat() if value else None


def diagnostic_url(value):
    value = safe_url(value)
    if not value or _safe_query(value) != value:
        return None
    return value


async def source_ref(db, source):
    source_id = source.get("source_id") or source.get("chunk_id")
    paper_id = source.get("paper_id")
    if (
        not isinstance(source_id, str)
        or len(source_id) > 256
        or not isinstance(paper_id, str)
    ):
        return None
    origin = "web" if source.get("source_origin") == "web" else "local"
    level = (
        source.get("evidence_type")
        or source.get("evidence_level")
        or (
            "abstract_only"
            if source.get("content_scope") == "abstract"
            else "full_text"
        )
    )
    if level not in LEVELS:
        return None
    if origin == "local":
        chunk = await db.get(PaperChunk, source_id)
        if chunk is None or chunk.paper_id != paper_id:
            return None
        url = "/api/sources/chunks/" + source_id
    else:
        url = diagnostic_url(source.get("source_url"))
        if not url:
            return None
    original = source.get("raw_quote") or source.get("excerpt")
    return {
        "source_id": source_id,
        "paper_id": paper_id,
        "source_origin": origin,
        "evidence_type": level,
        "doi": normalize_doi(source.get("doi")),
        "source_url": url,
        "version": source.get("version") or source.get("document_version"),
        "granularity": "bibliographic_record"
        if level == "metadata_only"
        else "abstract"
        if level == "abstract_only"
        else "child_excerpt",
        "quote_hash": hashlib.sha256(original.encode()).hexdigest()
        if isinstance(original, str)
        else None,
    }


async def record_run(db, trace, response, *, paper_ids=None):
    if response.get("run_id") != trace.run_id:
        raise ValueError("Response and trace run identities differ")
    rows = [*response.get("citations", []), *response.get("evidence", [])]
    for row in response.get("rows", []):
        for cell in row.get("cells", []):
            rows.extend([*cell.get("citations", []), *cell.get("evidence", [])])
    selected = None
    if paper_ids is not None:
        selected = list(
            (
                await db.scalars(
                    select(Paper.id).where(Paper.id.in_(list(paper_ids)[:50]))
                )
            ).all()
        )
    refs = {}
    for source in rows[:500]:
        reference = await source_ref(db, source)
        if reference and (selected is None or reference["paper_id"] in selected):
            refs.setdefault(reference["source_id"], reference)
    for record in response.get("web_records", [])[:50]:
        url = diagnostic_url(record.get("source_url"))
        if not url or not isinstance(record.get("paper_id"), str):
            continue
        if selected is not None and record["paper_id"] not in selected:
            continue
        key = "web_record:" + digest([record["paper_id"], url])[:32]
        record["feedback_source_id"] = key
        refs.setdefault(
            key,
            {
                "source_id": key,
                "paper_id": record["paper_id"],
                "source_origin": "web",
                "source_url": url,
                "doi": normalize_doi(record.get("doi")),
                "evidence_type": "metadata_only",
                "granularity": "bibliographic_record",
                "version": None,
                "quote_hash": None,
            },
        )
    if len(refs) > 250:
        raise ValueError("Run feedback source limit exceeded")
    scope = {
        "paper_ids": selected,
        "local_only": trace.record["scope"]["local_only"],
        "only_uploaded": trace.record["scope"]["only_uploaded"],
    }
    minimal = {
        key: trace.record.get(key)
        for key in (
            "original_query_sha256",
            "support_status",
            "retrieval_mode",
            "web_decision",
            "stage_ms",
            "llm_calls",
            "token_usage",
        )
    }
    data = dict(
        config_version=trace.record["config_version"],
        scope=scope,
        sources=list(refs.values()),
        minimal_trace=minimal,
    )
    existing = await db.get(AnswerRun, trace.run_id)
    if existing is None:
        db.add(AnswerRun(run_id=trace.run_id, **data))
    elif any(getattr(existing, key) != value for key, value in data.items()):
        raise ValueError("Stored run identity cannot be replaced")
    await db.commit()
    response["feedback_available"] = True


async def get_run(db, run_id):
    stored = await db.get(AnswerRun, run_id)
    if stored is None:
        raise LookupError("Run not found")
    return {
        "run_id": stored.run_id,
        "config_version": stored.config_version,
        "scope": stored.scope,
        "sources": stored.sources,
        "trace": stored.minimal_trace,
        "created_at": iso(stored.created_at),
    }


async def create_feedback(
    db, *, run_id, reason, source_id=None, expected_paper_id=None
):
    if reason not in REASONS:
        raise ValueError("Unknown feedback category")
    run = await get_run(db, run_id)
    source = next((s for s in run["sources"] if s["source_id"] == source_id), None)
    if source_id is not None and source is None:
        raise ValueError("Source is not part of this run")
    if expected_paper_id is not None and (
        await db.get(Paper, expected_paper_id) is None
        or run["scope"]["paper_ids"] is not None
        and expected_paper_id not in run["scope"]["paper_ids"]
    ):
        raise ValueError("Expected paper is outside the stored scope")
    saved = AnswerFeedback(
        run_id=run_id, reason=reason, source=source, expected_paper_id=expected_paper_id
    )
    db.add(saved)
    await db.commit()
    return {
        "id": saved.id,
        "run_id": run_id,
        "reason": reason,
        "source": source,
        "config_version": run["config_version"],
        "created_at": iso(saved.created_at),
        "review_status": "needs_review",
    }


def regression_dict(row):
    return {
        "id": row.id,
        "feedback_id": row.feedback_id,
        "fixture": row.fixture,
        "status": row.status,
        "result": row.result,
        "fix_reference": row.fix_reference,
        "created_at": iso(row.created_at),
        "updated_at": iso(row.updated_at),
    }


async def create_regression(db, feedback_id, *, synthetic=False):
    feedback = await db.get(AnswerFeedback, feedback_id)
    if feedback is None:
        raise LookupError("Feedback not found")
    if synthetic and (feedback.reason not in GUARDS or feedback.source is None):
        raise ValueError("No synthetic source guard for this feedback")
    run = await get_run(db, feedback.run_id)
    row = FeedbackRegression(
        feedback_id=feedback_id,
        fixture={
            "synthetic": synthetic,
            "kind": feedback.reason,
            "source": feedback.source,
            "run_id": feedback.run_id,
            "config_version": run["config_version"],
            "scope": run["scope"],
            "purpose": "development_guard_not_semantic_reproduction",
        },
        status="pending" if synthetic else "needs_review",
    )
    db.add(row)
    await db.commit()
    return regression_dict(row)


async def run_regression(db, regression_id):
    row = await db.get(FeedbackRegression, regression_id)
    if row is None:
        raise LookupError("Regression not found")
    if not row.fixture.get("synthetic"):
        return regression_dict(row)
    source, kind = row.fixture["source"], row.fixture["kind"]
    passed = False
    if kind == "web_result_irrelevant":
        # A fixed irrelevant bibliographic record tests metadata exclusion only.
        # This is not a replay or judgement of the original Web result's topic.
        original = "Public synthetic irrelevant bibliographic record."
        checksum = hashlib.sha256(original.encode()).hexdigest()
        example = {
            "chunk_id": "web:feedback-fixture",
            "paper_id": source["paper_id"],
            "source_origin": "web",
            "title": "Synthetic bibliography",
            "evidence_level": "metadata_only",
            "content_scope": "abstract",
            "excerpt": original,
            "source_url": "https://example.org/fixture",
            "document_version": "sha256:" + checksum,
            "source_spans": [
                {
                    "paper_id": source["paper_id"],
                    "document_id": "feedback-fixture",
                    "block_id": "metadata",
                    "document_version": "sha256:" + checksum,
                    "page": None,
                    "raw_start": 0,
                    "raw_end": len(original),
                    "excerpt_sha256": checksum,
                }
            ],
        }
        passed = not (
            await build_registry(
                db, [example], web_snapshots=seal_web_sources([example])
            )
        ).sources
    else:
        child = await db.get(PaperChunk, source["source_id"])
        if child is None or child.paper_id != source["paper_id"]:
            row.status, row.result = "needs_review", {"error_type": "SourceUnavailable"}
            await db.commit()
            return regression_dict(row)
        original = await resolve_child(db, child, allow_parent_failure=True)
        if kind == "wrong_paper":
            passed = not (
                await build_registry(
                    db, [original], paper_ids=["synthetic-disallowed-paper"]
                )
            ).sources
        elif kind == "unsupported_citation":
            registry = await build_registry(db, [original])
            try:
                registry.cite(
                    child.id, original["excerpt"] + "\nSynthetic unsupported suffix."
                )
            except ValueError:
                passed = True
    row.status = "passed" if passed else "failed"
    row.result = {
        "guard_passed": passed,
        "semantic_fix_verified": False,
        "model_calls": 0,
        "external_requests": 0,
    }
    row.updated_at = datetime.now(UTC)
    await db.commit()
    return regression_dict(row)


async def record_fix(db, regression_id, reference):
    row = await db.get(FeedbackRegression, regression_id)
    if row is None:
        raise LookupError("Regression not found")
    if row.status != "passed":
        raise ValueError("Regression has not passed")
    if (
        not re.fullmatch(r"[A-Za-z0-9:._/\-]{1,160}", reference)
        or _safe_query(reference) != reference
    ):
        raise ValueError("Unsafe fix reference")
    row.fix_reference, row.updated_at = reference, datetime.now(UTC)
    await db.commit()
    return regression_dict(row)
