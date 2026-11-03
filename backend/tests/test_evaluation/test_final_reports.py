import pytest

from evaluation.final_reports import protect_human_artifacts


def test_report_cannot_overwrite_completed_gold(tmp_path):
    target = tmp_path / "gold_annotations.jsonl"
    target.write_text('{"reviewer_type":"human"}\n')
    with pytest.raises(ValueError):
        protect_human_artifacts(tmp_path)
    assert "human" in target.read_text()


def test_report_cannot_clear_completed_pdf_verdict(tmp_path):
    target = tmp_path / "pdf_parser_human_audit.csv"
    target.write_text("sample_id,reviewer_id,verdict\n1,reviewer-a,error\n")
    with pytest.raises(ValueError):
        protect_human_artifacts(tmp_path)
    assert "reviewer-a" in target.read_text()
