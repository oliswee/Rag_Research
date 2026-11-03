import json

import pytest

from evaluation.final_contracts import sha256
from evaluation.final_long_context import direct_context


def test_direct_context_keeps_whole_original_and_bound_child_ranges():
    text = "Heading\n\nOriginal body.\n\nReferences"
    paper = {"paper_id": "p", "text": text}
    source = {
        "chunk_id": "c",
        "paper_id": "p",
        "document_version": "sha256:" + sha256(text.encode()),
        "source_spans": [{"raw_start": 9, "raw_end": 23}],
    }
    result = json.loads(direct_context(paper, [source], 8192))
    assert result["full_paper"] == text
    assert result["citable_ranges"] == {"c": [[9, 23]]}


def test_direct_context_refuses_truncation_and_foreign_sources():
    paper = {"paper_id": "p", "text": "Complete original"}
    with pytest.raises(ValueError):
        direct_context(paper, [], 4)
    with pytest.raises(ValueError):
        direct_context(paper, [{"paper_id": "foreign"}], 8192)
