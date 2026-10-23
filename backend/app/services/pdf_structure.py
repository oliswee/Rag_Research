"""Conservative layout hints and versioned anchors for extracted PDF text.

Offsets address page.structured_text, a deterministic reading-order transcript.
Legacy page.text is retained separately; neither is claimed to be PDF byte offsets.
"""

import re
import statistics
import uuid
from dataclasses import dataclass

import fitz

PARSER_VERSION = f"pymupdf-{fitz.VersionBind}:structured-v1"


@dataclass
class PDFBlock:
    id: str
    document_version: str
    source_id: str
    page: int | None
    text: str
    raw_start: int
    raw_end: int
    bbox: tuple[float, float, float, float] | None
    kind: str = "paragraph"
    section_id: str | None = None
    confidence: str = "normal"
    source_origin: str = "pdf"
    cells: list[list[str | None]] | None = None


@dataclass
class _Candidate:
    text: str
    bbox: tuple[float, float, float, float]
    size: float = 11
    kind: str = "paragraph"
    cells: list[list[str | None]] | None = None
    column: int | None = None


def block_id(version: str, page: int | None, start: int, end: int) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, f"{PARSER_VERSION}|{version}|{page}|{start}:{end}"
        )
    )


def _classify(text: str, size: float, body_size: float, fonts: list[str]) -> str:
    if re.match(r"^(?:Figure|Fig\.|Table)\s+\d+", text, re.IGNORECASE):
        return "caption"
    if (
        re.search(r"[∑∫√]", text)
        or ("=" in text and re.search(r"[+*/^]", text))
        or any(
            re.search(r"math|symbol|cmmi|cmsy", font, re.IGNORECASE) for font in fonts
        )
    ):
        return "formula"
    if len(text) <= 160 and (
        size > body_size * 1.2
        or re.match(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z]", text)
        or re.fullmatch(
            r"abstract|introduction|methods|results|discussion|conclusions?|references|bibliography",
            text,
            re.IGNORECASE,
        )
    ):
        return "heading"
    return "paragraph"


def _has_columns(candidates: list[_Candidate], width: float) -> bool:
    midpoint = width / 2
    body = [item for item in candidates if item.kind not in {"heading", "table"}]
    left = [item for item in body if item.bbox[2] < midpoint]
    right = [item for item in body if item.bbox[0] > midpoint]
    return bool(
        left
        and right
        and max(min(item.bbox[1] for item in left), min(item.bbox[1] for item in right))
        < min(max(item.bbox[3] for item in left), max(item.bbox[3] for item in right))
        and min(item.bbox[0] for item in right) - max(item.bbox[2] for item in left)
        > width * 0.03
    )


def _reading_order(
    candidates: list[_Candidate], width: float, flags: set[str]
) -> list[_Candidate]:
    if not _has_columns(candidates, width):
        return sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0]))
    flags.add("multi_column_layout_heuristic")
    midpoint = width / 2
    headings = [item for item in candidates if item.kind == "heading"]
    first_body = min(
        item.bbox[1] for item in candidates if item.kind not in {"heading", "table"}
    )
    only_top_headings = all(item.bbox[3] <= first_body for item in headings)
    separators = sorted(
        [
            item
            for item in candidates
            if item.kind in {"heading", "table"}
            and (
                item.bbox[0] < midpoint < item.bbox[2]
                or (item.kind == "heading" and only_top_headings)
            )
        ],
        key=lambda item: (item.bbox[1], item.bbox[0]),
    )
    remaining = [item for item in candidates if item not in separators]
    for item in remaining:
        item.column = int(item.bbox[0] >= midpoint)
    ordered = []

    def ordered_band(items):
        return sorted(items, key=lambda item: (item.column, item.bbox[1], item.bbox[0]))

    for separator in separators:
        before = [item for item in remaining if item.bbox[1] < separator.bbox[1]]
        ordered.extend(ordered_band(before))
        ordered.append(separator)
        remaining = [item for item in remaining if item.bbox[1] >= separator.bbox[1]]
    ordered.extend(ordered_band(remaining))
    return ordered


def structure_page(
    page: fitz.Page,
    version: str,
    source_id: str,
    source_origin: str,
    section_id: str | None,
) -> tuple[str, list[PDFBlock], list[str], str | None]:
    flags: set[str] = set()
    lines = [
        line
        for block in page.get_text(
            "dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
        )["blocks"]
        if block["type"] == 0
        for line in block["lines"]
    ]
    sizes = [
        span["size"] for line in lines for span in line["spans"] if span["text"].strip()
    ]
    body_size = statistics.median(sizes) if sizes else 11
    candidates = []
    table_rectangles = []
    # Grid detection is a layout hint; unruled/complex tables are not certified.
    if lines:
        try:
            tables = page.find_tables().tables if page.get_drawings() else []
            for table in tables:
                rectangle = fitz.Rect(table.bbox)
                text = page.get_textbox(rectangle).strip("\n")
                if text.strip():
                    candidates.append(
                        _Candidate(
                            text, tuple(rectangle), kind="table", cells=table.extract()
                        )
                    )
                    table_rectangles.append(rectangle)
                    flags.add("table_structure_heuristic")
        except Exception:
            flags.add("table_detection_failed")
    for line in lines:
        spans = line["spans"]
        text = "".join(span["text"] for span in spans)
        if not text.strip():
            continue
        rectangle = fitz.Rect(line["bbox"])
        if any(table.contains(rectangle) for table in table_rectangles):
            continue
        if any(table.intersects(rectangle) for table in table_rectangles):
            flags.add("table_boundary_overlap")
        size = max(span["size"] for span in spans)
        kind = _classify(text, size, body_size, [span["font"] for span in spans])
        if kind == "formula":
            flags.add("formula_requires_review")
        candidates.append(_Candidate(text, tuple(rectangle), size, kind))
    ordered = _reading_order(candidates, page.rect.width, flags)
    # Join only aligned consecutive body lines; never remove text or duplicates.
    paragraphs: list[_Candidate] = []
    for item in ordered:
        previous = paragraphs[-1] if paragraphs else None
        if (
            previous is not None
            and previous.kind == item.kind == "paragraph"
            and previous.column == item.column
            and abs(previous.bbox[0] - item.bbox[0]) < 8
            and abs(previous.size - item.size) < 1
            and 0 <= item.bbox[1] - previous.bbox[3] < item.size * 0.7
        ):
            previous.text += "\n" + item.text
            previous.bbox = tuple(fitz.Rect(previous.bbox) | fitz.Rect(item.bbox))
            flags.add("paragraph_grouping_heuristic")
        else:
            paragraphs.append(item)
    transcript = ""
    blocks = []
    column_sections = {0: section_id, 1: section_id}
    last_section = section_id
    for item in paragraphs:
        if transcript:
            transcript += "\n\n"
        start = len(transcript)
        transcript += item.text
        identifier = block_id(version, page.number + 1, start, len(transcript))
        if item.column is None:
            if item.kind == "heading":
                section_id = identifier
                column_sections = {0: identifier, 1: identifier}
            block_section = section_id
        else:
            if item.kind == "heading":
                column_sections[item.column] = identifier
            block_section = column_sections[item.column]
        last_section = block_section
        blocks.append(
            PDFBlock(
                id=identifier,
                document_version=version,
                source_id=source_id,
                page=page.number + 1,
                text=item.text,
                raw_start=start,
                raw_end=len(transcript),
                bbox=item.bbox,
                kind=item.kind,
                section_id=block_section,
                confidence="low" if flags else "normal",
                source_origin=source_origin,
                cells=item.cells,
            )
        )
    return transcript, blocks, sorted(flags), last_section
