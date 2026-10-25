"""Hand-anchored windows catch quotation drift and cross-paper starvation."""

import json

import pytest

from app.services.retrieval_trace import RetrievalTrace


def span(text, start=0, page=1, block="b", paper="p", doc="d", version="v"):
    return {
        "paper_id": paper,
        "document_id": doc,
        "document_version": version,
        "page": page,
        "block_id": block,
        "raw_start": start,
        "raw_end": start + len(text),
        "bbox": None,
    }


def source(identifier="c1", text="Latency is 1.5 ms.", **overrides):
    result = {
        "chunk_id": identifier,
        "paper_id": "p",
        "document_id": "d",
        "document_version": "v",
        "index_version": "g",
        "parent_id": "parent",
        "excerpt": text,
        "title": "Paper",
        "content_scope": "full_text",
        "page_start": 1,
        "page_end": 1,
        "source_origin": "pdf",
        "source_spans": [span(text)],
        "parent": None,
    }
    result.update(overrides)
    return result


def parent(text, spans=None):
    return {
        "id": "parent",
        "raw_text": text,
        "source_spans": spans or [span(text)],
        "page_start": 1,
        "page_end": 1,
    }


def test_same_parent_merges_context_but_preserves_each_child_quote():
    from app.services.context_packing import pack_context

    full = "Latency is 1.5 ms. Memory is 20 MB."
    a = source(parent=parent(full))
    b = source(
        "c2",
        "Memory is 20 MB.",
        parent=parent(full),
        source_spans=[span("Memory is 20 MB.", 19)],
    )
    packed = pack_context([a, b], budget=1800, parent_budget=800)
    assert len(packed.blocks) == 1
    assert packed.blocks[0]["child_ids"] == ["c1", "c2"]
    assert full in packed.text
    assert [s["excerpt"] for s in packed.sources] == [
        "Latency is 1.5 ms.",
        "Memory is 20 MB.",
    ]
    assert packed.token_units == len(packed.text.encode()) <= 1800


@pytest.mark.parametrize(
    "bad_parent",
    [
        None,
        parent("Foreign parent."),
        {**parent("Latency is 1.5 ms."), "id": "foreign"},
    ],
)
def test_missing_or_corrupt_parent_keeps_original_child(bad_parent):
    from app.services.context_packing import pack_context

    packed = pack_context([source(parent=bad_parent)])
    assert len(packed.sources) == 1
    assert packed.sources[0]["excerpt"] == "Latency is 1.5 ms."
    assert "Foreign parent" not in packed.text
    assert packed.blocks[0]["expansion_status"] == "child_fallback"


def test_long_parent_window_keeps_decimal_unit_and_cross_page_anchors():
    from app.services.context_packing import pack_context

    lead = "Unrelated background sentence. " * 80
    result1, result2 = "Latency is 1.5 ms.", "Memory is 20 MB."
    full1 = lead + result1
    full = full1 + "\n\n" + result2 + " More background. " * 80
    spans = [span(full1), span(result2 + " More background. " * 80, page=2, block="b2")]
    item = source(
        text=result1 + "\n\n" + result2,
        page_end=2,
        source_spans=[span(result1, len(lead)), span(result2, page=2, block="b2")],
        parent=parent(full, spans),
    )
    packed = pack_context([item], budget=1400, parent_budget=200)
    assert result1 in packed.text and result2 in packed.text
    assert len(packed.blocks[0]["text"]) < 500
    assert {s["page"] for s in packed.blocks[0]["source_spans"]} == {1, 2}
    assert "1.5 ms" in packed.sources[0]["excerpt"]


def test_overlapping_parent_windows_do_not_duplicate_original_characters():
    from app.services.context_packing import pack_context

    full = "First result. Shared result. Last result."
    a = source(
        text="Shared result.",
        parent=parent(full),
        source_spans=[span("Shared result.", 14)],
    )
    b = source(
        "c2",
        "Shared result.",
        parent_id="parent2",
        parent={**parent(full), "id": "parent2"},
        source_spans=[span("Shared result.", 14)],
    )
    packed = pack_context([a, b], budget=1600)
    assert len(packed.blocks) == 1
    assert packed.blocks[0]["child_ids"] == ["c1", "c2"]
    assert packed.blocks[0]["text"].count("Shared result.") == 1


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_child_first_budget_reserves_all_explicit_papers(count):
    from app.services.context_packing import pack_context

    items = []
    for i in range(count):
        text = f"Paper {i} result."
        items.append(
            source(
                f"c{i}",
                text,
                paper_id=f"p{i}",
                parent_id=None,
                parent=None,
                source_spans=[],
            )
        )
    items[0]["parent"] = parent("Paper 0 result. " + "Large background. " * 1000)
    packed = pack_context(items, paper_ids=[f"p{i}" for i in range(count)], budget=2400)
    assert {s["paper_id"] for s in packed.sources} == {f"p{i}" for i in range(count)}
    assert all(v["context_status"] == "included" for v in packed.coverage.values())
    assert packed.token_units <= 2400


def test_whole_child_over_budget_is_unknown_never_partial_numeric_quote():
    from app.services.context_packing import pack_context

    recorder = RetrievalTrace("q", {}, corpus_version="test")
    packed = pack_context(
        [source(text="Result 1.5 ms. " * 300)],
        paper_ids=["p", "missing"],
        budget=250,
        trace=recorder,
    )
    assert not packed.sources and packed.text == ""
    assert packed.coverage["p"]["context_status"] == "unknown"
    assert packed.coverage["missing"]["candidate_status"] == "unknown"
    assert (
        recorder.record["context_packing"]["decisions"][0]["reason"]
        == "child_exceeds_budget"
    )
    assert "Result 1.5" not in json.dumps(recorder.record)


def test_same_doi_same_version_dedups_but_different_versions_survive():
    from app.services.context_packing import pack_context

    a = source(doi="10.1000/example", parent=None)
    duplicate = source(
        "web", doi="https://doi.org/10.1000/EXAMPLE", source_origin="web", parent=None
    )
    revised = source(
        "revised", doi="10.1000/example", document_version="v2", parent=None
    )
    packed = pack_context([a, duplicate, revised], budget=2400)
    assert {s["chunk_id"] for s in packed.sources} == {"c1", "web", "revised"}
    assert len(packed.blocks) == 2
    assert packed.blocks[0]["child_ids"] == ["c1", "web"]
    assert packed.blocks[1]["child_ids"] == ["revised"]


def test_mixed_abstract_web_local_sources_use_one_budget_no_invented_pages():
    from app.services.context_packing import pack_context

    abstract = source(
        "abstract",
        "Abstract reports a result.",
        parent_id=None,
        parent=None,
        content_scope="abstract",
        source_origin="web",
        page_start=None,
        page_end=None,
        source_spans=[],
    )
    packed = pack_context([source(), abstract], budget=1800)
    assert len(packed.sources) == 2
    assert packed.sources[1]["page_start"] is None
    assert '"content_scope":"abstract"' in packed.text
    assert packed.token_units <= 1800


def test_distinct_parents_with_overlapping_background_share_one_window():
    from app.services.context_packing import pack_context

    full = "First result. Shared background. Last result."
    a = source(
        text="First result.", parent=parent(full), source_spans=[span("First result.")]
    )
    b = source(
        "c2",
        "Last result.",
        parent_id="p2",
        parent={**parent(full), "id": "p2"},
        source_spans=[span("Last result.", 33)],
    )
    packed = pack_context([a, b], budget=1800)
    assert len(packed.blocks) == 1
    assert packed.text.count("Shared background.") == 1


def test_disallowed_bibliography_is_explained_and_not_packed():
    from app.services.context_packing import pack_context

    recorder = RetrievalTrace("q", {}, corpus_version="test")
    packed = pack_context([source(evidence_role="bibliography")], trace=recorder)
    assert not packed.sources
    assert (
        recorder.record["context_packing"]["decisions"][0]["reason"]
        == "non_body_evidence"
    )


def test_background_fraction_is_structural_not_an_invented_relevance_score():
    from app.services.context_packing import pack_context

    recorder = RetrievalTrace("q", {}, corpus_version="test")
    full = "Latency is 1.5 ms. Extra background."
    pack_context([source(parent=parent(full))], trace=recorder)
    assert recorder.record["context_packing"][
        "background_context_fraction"
    ] == pytest.approx(1 - 18 / 36)
