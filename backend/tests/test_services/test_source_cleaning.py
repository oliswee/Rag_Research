"""Cleaner fixtures preserve raw evidence while exercising conservative rules."""

import importlib.util

import fitz
import pytest

from app.services.pdf_parser import PDFParserService


def cleaner():
    assert importlib.util.find_spec("app.services.source_cleaning"), (
        "Conservative cleaner is missing"
    )
    from app.services.source_cleaning import clean_block

    return clean_block


def abstract_block(text):
    return PDFParserService().parse_abstract(text).pages[0].blocks[0]


def test_soft_hyphen_and_line_join_map_back_to_exact_original_span():
    raw = "The re\u00ad\ntrieval method\nworks well."
    result = cleaner()(abstract_block(raw))
    assert result.raw_text == raw
    assert result.cleaned_text == "The retrieval method works well."
    start = result.cleaned_text.index("retrieval")
    left, right = result.raw_span(start, start + len("retrieval"))
    assert raw[left:right] == "re\u00ad\ntrieval"
    assert {change.rule for change in result.transforms} == {
        "soft_hyphen_join",
        "wrapped_line_join",
    }
    for mapping in result.mapping:
        assert 0 <= mapping.raw_start < mapping.raw_end <= len(raw)
        assert 0 <= mapping.clean_start < mapping.clean_end <= len(result.cleaned_text)


@pytest.mark.parametrize(
    "raw",
    [
        "state-\nof-the-art",
        "Result 12\n13 failures",
        "Introduction\nMethods",
        "A first paragraph.\n\nA second paragraph.",
    ],
)
def test_ambiguous_hyphens_numbers_and_paragraphs_are_retained(raw):
    # A single raw block tests the rule independently of parser segmentation.
    block = abstract_block("placeholder")
    block.text = raw
    block.raw_start = 20
    block.raw_end = 20 + len(raw)
    result = cleaner()(block)
    assert result.cleaned_text == raw
    assert result.raw_text == raw
    assert result.raw_span(0, len(result.cleaned_text)) == (20, 20 + len(raw))


@pytest.mark.parametrize("kind", ["table", "formula", "unstructured"])
def test_table_formula_and_uncertain_layout_are_not_rewritten(kind):
    block = abstract_block("alpha  beta\ngamma")
    block.kind = kind
    result = cleaner()(block)
    assert result.cleaned_text == block.text
    assert not result.transforms


def test_whitespace_normalization_retains_raw_coverage_and_rejects_invalid_spans():
    raw = "A  useful\r\nmethod improves\nrecall."
    block = abstract_block("placeholder")
    block.text, block.raw_start, block.raw_end = raw, 10, 10 + len(raw)
    result = cleaner()(block)
    assert result.cleaned_text == "A useful method improves recall."
    assert result.raw_span(0, len(result.cleaned_text)) == (10, 10 + len(raw))
    with pytest.raises(ValueError):
        result.raw_span(-1, 1)
    with pytest.raises(ValueError):
        result.raw_span(1, len(result.cleaned_text) + 1)


def test_repeated_margin_text_is_flagged_without_erasing_legitimate_repeats():
    cleaner()
    from app.services.source_cleaning import clean_document

    document = fitz.open()
    for _ in range(3):
        page = document.new_page()
        page.insert_text((50, 25), "Study Report", fontsize=9)
        page.insert_text((50, 200), "Repeated experimental result.", fontsize=11)
    data = document.tobytes()
    document.close()
    parsed = PDFParserService().parse(data)
    cleaned = clean_document(parsed)
    headers = [
        item
        for item in cleaned.review_items
        if "repeated_header_candidate" in item["flags"]
    ]
    assert len(headers) == 3
    for page in parsed.pages:
        assert "Study Report" in page.structured_text
        assert "Repeated experimental result." in page.structured_text
    assert (
        sum(
            "Repeated experimental result." in block.cleaned_text
            for block in cleaned.blocks
        )
        == 3
    )
    assert "Repeated experimental result." not in str(cleaned.review_items)


@pytest.mark.parametrize("heading", ["References", "7 References", "7.2 Bibliography"])
def test_reference_boundary_is_marked_and_reference_text_is_retained(heading):
    cleaner()
    from app.services.source_cleaning import clean_document

    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 60), "Results", fontsize=16)
    page.insert_text((50, 100), "The body mentions references but is still a result.")
    page.insert_text((50, 160), heading, fontsize=16)
    page.insert_text((50, 200), "Author A. 2024. A published study.")
    data = document.tobytes()
    document.close()
    parsed = PDFParserService().parse(data)
    cleaned = clean_document(parsed)
    body = next(block for block in cleaned.blocks if "body mentions" in block.raw_text)
    citation = next(block for block in cleaned.blocks if "Author A." in block.raw_text)
    assert "reference_section" not in body.flags
    assert "reference_section" in citation.flags
    assert (
        citation.raw_text
        == citation.cleaned_text
        == "Author A. 2024. A published study."
    )
