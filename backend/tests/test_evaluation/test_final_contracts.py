import hashlib

import pytest

from evaluation.final_contracts import (
    freeze_manifest,
    validate_annotation,
    verify_freeze,
    verify_runtime,
)

TEXT = "The reported accuracy is ninety percent."


def test_fresh_packets_do_not_collide_in_shared_qdrant_collection():
    from evaluation.prepare_final import runtime_paper_id

    first = "00000000-0000-0000-0000-000000000001"
    second = "00000000-0000-0000-0000-000000000002"
    assert runtime_paper_id(first, "paper") == runtime_paper_id(first, "paper")
    assert runtime_paper_id(first, "paper") != runtime_paper_id(second, "paper")


def test_frozen_runtime_cannot_silently_change_model_or_budget(monkeypatch):
    from app.config import settings
    from app.services.trace_versions import configuration_snapshot

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    _, runtime = configuration_snapshot()
    manifest = {"config": {"runtime": runtime}}
    verify_runtime(manifest)
    monkeypatch.setattr(settings, "ollama_model", "different-local-model")
    with pytest.raises(ValueError):
        verify_runtime(manifest)


def candidate():
    return {
        "case_id": "one",
        "paper_id": "p",
        "question": "What accuracy?",
        "document_version": "sha256:" + hashlib.sha256(TEXT.encode()).hexdigest(),
        "source_kind": "provided_text",
        "raw_quote": TEXT,
        "source_offset": [0, len(TEXT)],
        "pdf_page_or_null": None,
        "answerable_local": True,
        "required_evidence": ["one"],
        "expected_answer_points": ["ninety percent"],
        "expected_web_route": "not_required",
        "review_status": "pending_review",
        "reviewer_type": "dataset_reference",
        "reviewer_id": None,
    }


def test_reference_anchor_is_valid_but_not_independent_human_gold():
    row = candidate()
    validate_annotation(row, {"p": TEXT})
    with pytest.raises(ValueError):
        validate_annotation(row, {"p": TEXT}, require_human=True)
    row.update(
        review_status="independent_human_reviewed",
        reviewer_type="model_assisted",
        reviewer_id="Codex",
    )
    with pytest.raises(ValueError):
        validate_annotation(row, {"p": TEXT}, require_human=True)


@pytest.mark.parametrize(
    "mutation",
    [
        {"raw_quote": "fabricated"},
        {"source_offset": [-1, 4]},
        {"pdf_page_or_null": 1},
        {"document_version": "sha256:wrong"},
        {"paper_id": "foreign"},
        {"answerable_local": False},
    ],
)
def test_source_anchor_corruption_rejected(mutation):
    row = {**candidate(), **mutation}
    with pytest.raises(ValueError):
        validate_annotation(row, {"p": TEXT})


def test_freeze_rejects_mutation_and_paths_outside_root(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text(TEXT)
    manifest = freeze_manifest(tmp_path, [path], {"model": "local"})
    verify_freeze(tmp_path, manifest)
    path.write_text("changed")
    with pytest.raises(ValueError):
        verify_freeze(tmp_path, manifest)
    manifest["files"] = {"../escape.txt": "wrong"}
    with pytest.raises(ValueError):
        verify_freeze(tmp_path, manifest)


def test_pdf_anchors_require_actual_pdf_hash_and_existing_page():
    import fitz

    with fitz.open() as document:
        document.new_page()
        document.new_page().insert_text((50, 50), TEXT)
        pdf = document.tobytes()
    row = {
        **candidate(),
        "source_kind": "pdf",
        "pdf_page_or_null": 2,
        "document_version": "sha256:" + hashlib.sha256(pdf).hexdigest(),
    }
    original = {"pdf_bytes": pdf, "pages": {2: TEXT}}
    validate_annotation(row, {"p": original})
    row["pdf_page_or_null"] = 1
    with pytest.raises(ValueError):
        validate_annotation(row, {"p": original})


def test_forged_pdf_page_map_cannot_make_invalid_pdf_a_valid_source():
    pdf = b"%PDF-invalid-synthetic-bytes"
    row = {
        **candidate(),
        "source_kind": "pdf",
        "pdf_page_or_null": 2,
        "document_version": "sha256:" + hashlib.sha256(pdf).hexdigest(),
    }
    with pytest.raises(ValueError):
        validate_annotation(row, {"p": {"pdf_bytes": pdf, "pages": {2: TEXT}}})
