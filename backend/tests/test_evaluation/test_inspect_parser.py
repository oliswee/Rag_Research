"""Inspection exports must not overwrite their source PDF or combine outputs."""

import json

import fitz
import pytest

from evaluation.inspect_parser import inspect_document


@pytest.mark.parametrize("collision", ["input_output", "input_queue", "output_queue"])
def test_inspection_rejects_colliding_paths_before_writing(tmp_path, collision):
    document = fitz.open()
    document.new_page().insert_text((50, 80), "Original evidence.")
    original = document.tobytes()
    document.close()
    source, output, queue = (
        tmp_path / "source.pdf",
        tmp_path / "artifact.json",
        tmp_path / "queue.jsonl",
    )
    source.write_bytes(original)
    if collision == "input_output":
        output = source
    elif collision == "input_queue":
        queue = source
    else:
        queue = output
    with pytest.raises(ValueError, match="distinct"):
        inspect_document(source, output, queue)
    assert source.read_bytes() == original


def test_image_only_inspection_records_unsupported_in_review_queue(tmp_path):
    document = fitz.open()
    document.new_page().draw_rect(fitz.Rect(10, 10, 100, 100), fill=(0, 0, 0))
    source = tmp_path / "scan.pdf"
    source.write_bytes(document.tobytes())
    document.close()
    output, queue = tmp_path / "artifact.json", tmp_path / "queue.jsonl"
    result = inspect_document(source, output, queue)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    item = json.loads(queue.read_text(encoding="utf-8").strip())
    assert result["status"] == artifact["status"] == "unsupported"
    assert artifact["parsed"] is None
    assert item["flags"] == ["no_extractable_text"]
    assert item["page"] is None and item["review_status"] == "unreviewed"
    assert item["document_version"].startswith("sha256:")
