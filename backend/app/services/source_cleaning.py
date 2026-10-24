"""Conservative search-text cleaning with mappings to unchanged source evidence."""

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass

from app.services.pdf_parser import PDFParseResult
from app.services.pdf_structure import PDFBlock

CLEANER_VERSION = "conservative-v2"


@dataclass
class MappingSpan:
    raw_start: int
    raw_end: int
    clean_start: int
    clean_end: int


@dataclass
class Transform(MappingSpan):
    rule: str


@dataclass
class CleanedBlock:
    block_id: str
    document_version: str
    page: int | None
    raw_text: str
    cleaned_text: str
    mapping: list[MappingSpan]
    transforms: list[Transform]
    flags: list[str]

    def raw_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a nonempty cleaned range to a covering page-transcript range."""
        if not 0 <= start < end <= len(self.cleaned_text):
            raise ValueError("Cleaned span is outside block text")
        covered = []
        for span in self.mapping:
            left, right = max(start, span.clean_start), min(end, span.clean_end)
            if left >= right:
                continue
            if span.raw_end - span.raw_start == span.clean_end - span.clean_start:
                covered.append(
                    (
                        span.raw_start + left - span.clean_start,
                        span.raw_start + right - span.clean_start,
                    )
                )
            else:
                covered.append((span.raw_start, span.raw_end))
        return min(left for left, _ in covered), max(right for _, right in covered)


@dataclass
class CleanedDocument:
    document_version: str
    source_id: str
    parser_version: str
    cleaner_version: str
    blocks: list[CleanedBlock]
    review_items: list[dict]


def clean_block(block: PDFBlock) -> CleanedBlock:
    # Units carry their original page-transcript range through each replacement.
    units = [
        (char, block.raw_start + i, block.raw_start + i + 1)
        for i, char in enumerate(block.text)
    ]
    changes = []
    flags = []
    if block.kind not in {"table", "formula", "unstructured", "heading", "caption"}:
        rules = (
            ("normalize_newlines", r"\r\n?", "\n"),
            ("soft_hyphen_join", r"(?<=[A-Za-z])\u00ad(?:\n[ \t]*)?(?=[a-z])", ""),
            ("horizontal_whitespace", r"(?<=\S)[ \t]{2,}(?=\S)", " "),
            ("wrapped_line_join", r"(?<=[A-Za-z,;])\n[ \t]*(?=[a-z])", " "),
        )
        for rule, pattern, replacement in rules:
            text = "".join(char for char, _, _ in units)
            updated = []
            previous = 0
            for match in re.finditer(pattern, text):
                updated.extend(units[previous : match.start()])
                raw_start, raw_end = units[match.start()][1], units[match.end() - 1][2]
                changes.append((rule, raw_start, raw_end))
                updated.extend((char, raw_start, raw_end) for char in replacement)
                previous = match.end()
            updated.extend(units[previous:])
            units = updated
        if re.search(r"-\s*\n", block.text):
            flags.append("visible_hyphen_break")
    mapping: list[MappingSpan] = []
    for position, (_, left, right) in enumerate(units):
        previous = mapping[-1] if mapping else None
        if (
            previous is not None
            and previous.raw_end == left
            and previous.raw_end - previous.raw_start
            == previous.clean_end - previous.clean_start
            and right - left == 1
        ):
            previous.raw_end = right
            previous.clean_end = position + 1
        else:
            mapping.append(MappingSpan(left, right, position, position + 1))
    transforms = []
    raw_starts = [start for _, start, _ in units]
    raw_ends = [end for _, _, end in units]
    for rule, left, right in changes:
        start = bisect_right(raw_ends, left)
        end = bisect_left(raw_starts, right)
        transforms.append(Transform(left, right, start, end, rule))
    return CleanedBlock(
        block.id,
        block.document_version,
        block.page,
        block.text,
        "".join(char for char, _, _ in units),
        mapping,
        transforms,
        flags,
    )


def clean_document(parsed: PDFParseResult) -> CleanedDocument:
    margins = defaultdict(set)
    candidates = {}
    reference_sections = {
        block.id
        for page in parsed.pages
        for block in page.blocks
        if block.kind == "heading"
        and re.fullmatch(
            r"(?:\d+(?:\.\d+)*[.)]?\s+)?(?:references|bibliography)",
            block.text.strip(),
            re.IGNORECASE,
        )
    }
    for page in parsed.pages:
        for block in page.blocks:
            if not block.bbox or not page.height or len(block.text) > 150:
                continue
            position = (
                "header"
                if block.bbox[1] < page.height * 0.08
                else "footer"
                if block.bbox[3] > page.height * 0.92
                else None
            )
            if position:
                key = (position, " ".join(block.text.casefold().split()))
                margins[key].add(page.number)
                candidates[block.id] = key
    blocks, review = [], []
    for page in parsed.pages:
        if not page.blocks:
            if page.quality_flags:
                review.append(
                    _review_item(parsed, page.number, None, page.quality_flags)
                )
            continue
        for block in page.blocks:
            cleaned = clean_block(block)
            flags = set(page.quality_flags) | set(cleaned.flags)
            key = candidates.get(block.id)
            if key and len(margins[key]) >= 3:
                flags.add(f"repeated_{key[0]}_candidate")
            page_marker = (
                block.kind == "paragraph"
                and block.bbox is not None
                and page.height is not None
                and page.width is not None
                and bool(re.fullmatch(r"\d{1,4}", block.text.strip()))
                # Prefer missing an offset printed page number to suppressing
                # an unrelated numeric cell or result near the lower margin.
                and block.text.strip() == str(page.number)
                and block.bbox[1] > page.height * 0.92
                and page.width * 0.35
                < (block.bbox[0] + block.bbox[2]) / 2
                < page.width * 0.65
            )
            if page_marker:
                flags.add("page_number_candidate")
            elif block.section_id in reference_sections:
                flags.add("reference_section")
            cleaned.flags = sorted(flags)
            blocks.append(cleaned)
            if flags:
                review.append(_review_item(parsed, page.number, block, sorted(flags)))
    return CleanedDocument(
        parsed.document_version,
        parsed.source_id,
        parsed.parser_version,
        CLEANER_VERSION,
        blocks,
        review,
    )


def _review_item(
    parsed: PDFParseResult, page: int | None, block: PDFBlock | None, flags: list[str]
) -> dict:
    # No extracted text goes into this queue; locate it through the saved document.
    item = {
        "document_version": parsed.document_version,
        "source_id": parsed.source_id,
        "parser_version": parsed.parser_version,
        "cleaner_version": CLEANER_VERSION,
        "page": page,
        "block_id": block.id if block else None,
        "raw_start": block.raw_start if block else None,
        "raw_end": block.raw_end if block else None,
        "bbox": block.bbox if block else None,
        "flags": flags,
        "review_status": "unreviewed",
    }
    item["review_id"] = hashlib.sha256(
        json.dumps(item, sort_keys=True).encode()
    ).hexdigest()
    return item
