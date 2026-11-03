"""Known synthetic layouts exercise provenance, not real-PDF accuracy claims."""

import fitz
import pytest

from app.services.pdf_parser import PDFParseError, PDFParserService


def pdf_bytes(draw):
    document = fitz.open()
    draw(document)
    data = document.tobytes()
    document.close()
    return data


def assert_anchors(result):
    assert result.document_version.startswith("sha256:")
    for page in result.pages:
        for block in page.blocks:
            assert block.text == page.structured_text[block.raw_start : block.raw_end]
            assert block.document_version == result.document_version
            assert block.source_id == result.source_id
            assert block.page == page.number
            assert block.id


def test_two_columns_and_heading_keep_reading_order_and_stable_source_anchors():
    def draw(document):
        page = document.new_page(width=600, height=800)
        # Deliberately insert the right column before the left.
        page.insert_text((330, 150), "Right column evidence.", fontsize=11)
        page.insert_text((50, 150), "Left column evidence.", fontsize=11)
        page.insert_text((50, 70), "1 Introduction", fontsize=18)

    data = pdf_bytes(draw)
    result = PDFParserService().parse(data)
    assert hasattr(result.pages[0], "blocks"), "Structured blocks are missing"
    blocks = result.pages[0].blocks
    assert [block.text for block in blocks] == [
        "1 Introduction",
        "Left column evidence.",
        "Right column evidence.",
    ]
    assert blocks[0].kind == "heading"
    assert blocks[1].section_id == blocks[2].section_id == blocks[0].id
    assert "multi_column_layout_heuristic" in result.pages[0].quality_flags
    assert_anchors(result)
    again = PDFParserService().parse(data)
    assert [block.id for block in again.pages[0].blocks] == [
        block.id for block in blocks
    ]


def test_ruled_table_keeps_cells_and_marks_structure_as_heuristic():
    def draw(document):
        page = document.new_page(width=600, height=800)
        for x in (50, 200, 350):
            page.draw_line((x, 100), (x, 190))
        for y in (100, 130, 160, 190):
            page.draw_line((50, y), (350, y))
        for row, values in enumerate(
            (("Method", "Recall"), ("Alpha", "0.7"), ("Beta", "0.8"))
        ):
            for column, value in enumerate(values):
                page.insert_text(
                    (60 + column * 150, 120 + row * 30), value, fontsize=11
                )

    result = PDFParserService().parse(pdf_bytes(draw))
    assert hasattr(result.pages[0], "blocks"), "Structured blocks are missing"
    tables = [block for block in result.pages[0].blocks if block.kind == "table"]
    assert len(tables) == 1
    assert tables[0].cells == [["Method", "Recall"], ["Alpha", "0.7"], ["Beta", "0.8"]]
    assert "Beta" in tables[0].text
    assert tables[0].confidence == "low"
    assert "table_structure_heuristic" in result.pages[0].quality_flags
    assert_anchors(result)


def test_inline_colored_labels_remain_between_their_surrounding_words():
    import re

    def draw(document):
        page = document.new_page(width=600, height=800)
        prefix = "The text describes memory is<p>["
        page.insert_text((50, 100), prefix, fontsize=11)
        x = 50 + fitz.get_text_length(prefix, fontsize=11)
        label = "REL=Relevant]Primary"
        page.insert_text((x, 100.4), label, fontsize=10, color=(0.1, 0.1, 0.8))
        page.insert_text((50, 116), "memory supports this answer.", fontsize=11)

    parsed = PDFParserService().parse(pdf_bytes(draw))
    assert any(
        "memoryis<p>[REL=Relevant]Primarymemory" in re.sub(r"\s+", "", block.text)
        for block in parsed.pages[0].blocks
    )


def test_inline_reassembly_preserves_positive_word_gaps_in_reverse_drawing_order():
    def draw(document):
        page = document.new_page(width=600, height=800)
        x = 50 + fitz.get_text_length("The first", fontsize=11) + 1.7
        page.insert_text((x, 100.4), "word matters.", fontsize=10)
        page.insert_text((50, 100), "The first", fontsize=11)

    parsed = PDFParserService().parse(pdf_bytes(draw))
    assert parsed.pages[0].structured_text == "The first word matters."


def test_narrow_gutter_columns_below_a_full_width_figure_read_down_each_column():
    def draw(document):
        page = document.new_page(width=600, height=800)
        page.insert_text(
            (50, 70), "Figure 1: A full width figure and caption", fontsize=11
        )
        for y, left, right in (
            (170, "Left column first evidence line.", "Right column first evidence."),
            (185, "Left column second evidence line.", "Right column second evidence."),
        ):
            # Text boxes leave a genuine but narrow gutter of about 12 points.
            while fitz.get_text_length(left, fontsize=11) < 240:
                left += "x"
            while fitz.get_text_length(right, fontsize=11) < 240:
                right += "x"
            page.insert_text((50, y), left, fontsize=11)
            page.insert_text((306, y), right, fontsize=11)

    parsed = PDFParserService().parse(pdf_bytes(draw))
    transcript = parsed.pages[0].structured_text
    assert transcript.index("Left column second") < transcript.index(
        "Right column first"
    )
    assert "multi_column_layout_heuristic" in parsed.pages[0].quality_flags


@pytest.mark.parametrize(
    "line",
    [
        "2020. URL https://arxiv.org/abs/2004.07202.",
        "2021 Conference on Empirical Methods in Natural Language Processing",
        "4749. Association for Computational Linguistics.",
    ],
)
def test_bibliography_years_and_page_numbers_do_not_create_sections(line):
    def draw(document):
        page = document.new_page()
        page.insert_text((50, 60), "References", fontsize=16)
        page.insert_text((50, 100), line, fontsize=11)
        page.insert_text(
            (50, 130), "Bibliography continuation remains in references.", fontsize=11
        )

    parsed = PDFParserService().parse(pdf_bytes(draw))
    blocks = parsed.pages[0].blocks
    assert blocks[1].kind != "heading"
    assert blocks[-1].section_id == blocks[0].id


def test_line_crossing_table_boundary_is_retained_in_full():
    def draw(document):
        page = document.new_page(width=600, height=800)
        for x in (50, 200, 350):
            page.draw_line((x, 100), (x, 190))
        for y in (100, 130, 160, 190):
            page.draw_line((50, y), (350, y))
        for row, values in enumerate(
            (("Method", "Recall"), ("Alpha", "0.7"), ("Beta", "0.8"))
        ):
            for column, value in enumerate(values):
                page.insert_text(
                    (60 + column * 150, 120 + row * 30), value, fontsize=11
                )
        page.insert_text(
            (240, 175), "Tail outside the table boundary is evidence", fontsize=11
        )

    result = PDFParserService().parse(pdf_bytes(draw))
    assert (
        "Tail outside the table boundary is evidence" in result.pages[0].structured_text
    )
    assert "table_boundary_overlap" in result.pages[0].quality_flags
    assert_anchors(result)


def test_column_local_headings_do_not_assign_right_column_to_left_sections():
    def draw(document):
        page = document.new_page(width=600, height=800)
        page.insert_text((50, 80), "1 Left section", fontsize=16)
        page.insert_text((50, 120), "Left body first.", fontsize=11)
        page.insert_text((50, 180), "2 Left section", fontsize=16)
        page.insert_text((50, 220), "Left body second.", fontsize=11)
        page.insert_text((330, 100), "Right body first.", fontsize=11)
        page.insert_text((330, 200), "Right body second.", fontsize=11)

    result = PDFParserService().parse(pdf_bytes(draw))
    page = result.pages[0]
    assert "multi_column_layout_heuristic" in page.quality_flags
    assert [block.text for block in page.blocks] == [
        "1 Left section",
        "Left body first.",
        "2 Left section",
        "Left body second.",
        "Right body first.",
        "Right body second.",
    ]
    assert all(
        block.section_id is None
        for block in page.blocks
        if block.text.startswith("Right")
    )
    assert all(block.confidence == "low" for block in page.blocks)
    assert_anchors(result)


def test_reference_heading_carries_to_next_page_after_other_column_continuation():
    from app.services.source_cleaning import clean_document

    def draw(document):
        page = document.new_page(width=600, height=800)
        page.insert_text((50, 80), "Conclusion", fontsize=16)
        page.insert_text((50, 120), "Left column body evidence.", fontsize=11)
        page.insert_text((50, 200), "References", fontsize=16)
        page.insert_text((50, 240), "Author. A cited article.", fontsize=11)
        page.insert_text((330, 120), "Other column continuation.", fontsize=11)
        page.insert_text((330, 240), "More continuation evidence.", fontsize=11)
        document.new_page(width=600, height=800).insert_text(
            (50, 80), "Author. Another cited article.", fontsize=11
        )

    parsed = PDFParserService().parse(pdf_bytes(draw))
    reference = next(b for b in parsed.pages[0].blocks if b.text == "References")
    assert parsed.pages[1].blocks[0].section_id == reference.id
    assert "reference_section" in clean_document(parsed).blocks[-1].flags


def test_caption_is_preserved_as_source_text():
    def draw(document):
        document.new_page().insert_text(
            (50, 80), "Figure 1: Measured recall by method."
        )

    result = PDFParserService().parse(pdf_bytes(draw))
    assert hasattr(result.pages[0], "blocks"), "Structured blocks are missing"
    assert result.pages[0].blocks[0].kind == "caption"
    assert result.pages[0].blocks[0].text == "Figure 1: Measured recall by method."
    assert_anchors(result)


def test_formula_is_retained_with_explicit_low_confidence():
    def draw(document):
        document.new_page().insert_text((50, 80), "E = m*c^2")

    result = PDFParserService().parse(pdf_bytes(draw))
    assert hasattr(result.pages[0], "blocks"), "Structured blocks are missing"
    formula = result.pages[0].blocks[0]
    assert formula.text == "E = m*c^2"
    assert formula.kind == "formula"
    assert formula.confidence == "low"
    assert "formula_requires_review" in result.pages[0].quality_flags
    assert_anchors(result)


def test_mixed_scan_retains_unsupported_page_without_renumbering():
    def draw(document):
        document.new_page().insert_text((50, 80), "First page evidence.")
        scan = document.new_page()
        raster = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
        raster.clear_with(255)
        scan.insert_image(fitz.Rect(30, 30, 200, 200), pixmap=raster)
        document.new_page().insert_text((50, 80), "Third page evidence.")

    result = PDFParserService().parse(pdf_bytes(draw))
    assert [page.number for page in result.pages] == [1, 2, 3]
    assert result.pages[1].status == "unsupported_no_text"
    assert result.pages[1].blocks == []
    assert result.pages[2].blocks[0].page == 3
    assert_anchors(result)


def test_all_scan_error_is_typed_for_abstract_fallback():
    def draw(document):
        document.new_page().draw_rect(fitz.Rect(30, 30, 200, 200), fill=(0, 0, 0))

    with pytest.raises(PDFParseError) as failure:
        PDFParserService().parse(pdf_bytes(draw))
    assert getattr(failure.value, "code", None) == "no_extractable_text"
    assert failure.value.document_version.startswith("sha256:")


def test_layout_failure_retains_raw_text_with_explicit_uncertainty(monkeypatch):
    def draw(document):
        document.new_page().insert_text((50, 80), "Original evidence remains.")

    def fail_layout(*args):
        raise RuntimeError("unreliable layout")

    monkeypatch.setattr("app.services.pdf_parser.structure_page", fail_layout)
    result = PDFParserService().parse(pdf_bytes(draw))
    assert "Original evidence remains." in result.pages[0].text
    assert result.pages[0].status == "low_confidence"
    assert "structure_extraction_failed" in result.pages[0].quality_flags
    assert result.pages[0].blocks[0].kind == "unstructured"
    assert_anchors(result)


def test_text_structure_does_not_materialize_embedded_image_payloads(monkeypatch):
    def draw(document):
        page = document.new_page()
        page.insert_text((50, 80), "Visible source text.")
        raster = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), 0)
        raster.clear_with(127)
        page.insert_image(fitz.Rect(50, 100, 150, 200), pixmap=raster)

    original = fitz.Page.get_text

    def text_only(page, option="text", *args, **kwargs):
        data = original(page, option, *args, **kwargs)
        if option == "dict":
            assert all(block["type"] == 0 for block in data["blocks"]), (
                "Image payload decoded during text-only structure extraction"
            )
        return data

    monkeypatch.setattr(fitz.Page, "get_text", text_only)
    result = PDFParserService().parse(pdf_bytes(draw))
    assert "structure_extraction_failed" not in result.pages[0].quality_flags
    assert "Visible source text." in result.pages[0].structured_text


def test_abstract_has_source_identity_and_null_pages():
    parser = PDFParserService()
    assert hasattr(parser, "parse_abstract"), "Abstract structure support is missing"
    result = parser.parse_abstract("An abstract result.\n\nA second paragraph.")
    assert result.content_scope == "abstract"
    assert result.page_count == 0
    assert result.pages[0].number is None
    assert len(result.pages[0].blocks) == 2
    assert all(block.page is None for block in result.pages[0].blocks)
    assert_anchors(result)
    with pytest.raises(PDFParseError) as failure:
        parser.parse_abstract("   ")
    assert failure.value.code == "empty_abstract"


@pytest.mark.parametrize("text", ["A result.\n", "  A result.  ", "\nA result.\n\n"])
def test_abstract_boundary_whitespace_does_not_drop_the_paragraph(text):
    result = PDFParserService().parse_abstract(text)
    assert [block.text for block in result.pages[0].blocks] == ["A result."]
    assert_anchors(result)


def test_repeated_paragraphs_on_different_pages_keep_distinct_anchors():
    def draw(document):
        for _ in range(2):
            document.new_page().insert_text(
                (50, 80), "Identical result, different page."
            )

    result = PDFParserService().parse(pdf_bytes(draw))
    assert hasattr(result.pages[0], "blocks"), "Structured blocks are missing"
    first, second = result.pages[0].blocks[0], result.pages[1].blocks[0]
    assert first.text == second.text
    assert first.id != second.id
    assert (first.page, second.page) == (1, 2)
    assert_anchors(result)
