import json

import pytest

from evaluation.final_contracts import freeze_manifest, verify_models
from evaluation.final_generation import valid_score
from evaluation.final_pdf_audit import run as audit_pdf


def test_pdf_audit_rerun_preserves_completed_private_packet(tmp_path, monkeypatch):
    output = tmp_path / "pdf_audit"
    output.mkdir()
    target = output / "current_parser_human_audit.csv"
    target.write_text("sample_id,reviewer_id,verdict\n1,reviewer-a,error\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Preserve"):
        audit_pdf(tmp_path)
    assert "reviewer-a" in target.read_text()


@pytest.mark.parametrize("changed", ["weights", "tag"])
def test_same_name_model_drift_rejected(tmp_path, monkeypatch, changed):
    expected = {
        "embedding": {"model": "same", "files": {"model.onnx": "old-hash"}},
        "reranker": {},
        "generation_and_judge": {"model": "same", "manifest_digest": "old-digest"},
        "judge_limits": {},
    }
    path = tmp_path / "model_freeze.json"
    path.write_text(json.dumps(expected))
    manifest = freeze_manifest(tmp_path, [path], {})
    monkeypatch.setattr(
        "evaluation.final_model_freeze.capture_models", lambda: expected
    )
    verify_models(tmp_path, manifest)
    actual = json.loads(json.dumps(expected))
    if changed == "weights":
        actual["embedding"]["files"]["model.onnx"] = "new-hash"
    else:
        actual["generation_and_judge"]["manifest_digest"] = "new-digest"
    monkeypatch.setattr("evaluation.final_model_freeze.capture_models", lambda: actual)
    with pytest.raises(ValueError):
        verify_models(tmp_path, manifest)


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -0.01, 1.01, True])
def test_invalid_judge_score_is_not_evaluated(score):
    with pytest.raises(ValueError):
        valid_score(score)


@pytest.mark.parametrize("score", [0, 0.5, 1])
def test_valid_judge_score_preserves_range(score):
    assert valid_score(score) == score
