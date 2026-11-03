"""Save local raw/cleaned parser artifacts and an unreviewed M8 review queue.

Run from backend/: uv run python -m evaluation.inspect_parser --input paper.pdf
Outputs may contain private source text; defaults are under ignored data/.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from app.services.pdf_parser import PDFParseError, PDFParserService
from app.services.pdf_structure import PARSER_VERSION
from app.services.source_cleaning import CLEANER_VERSION, clean_document


def inspect_document(input_path: Path, output: Path, review_queue: Path) -> dict:
    paths = [path.resolve() for path in (input_path, output, review_queue)]
    for index, path in enumerate(paths):
        if any(
            path == other or (path.exists() and other.exists() and path.samefile(other))
            for other in paths[index + 1 :]
        ):
            raise ValueError("Input, artifact and review queue paths must be distinct")
    try:
        parsed = PDFParserService().parse(input_path.read_bytes())
    except PDFParseError as exc:
        if exc.code != "no_extractable_text":
            raise
        item = {
            "document_version": exc.document_version,
            "source_id": "pdf:" + exc.document_version,
            "parser_version": PARSER_VERSION,
            "cleaner_version": CLEANER_VERSION,
            "page": None,
            "block_id": None,
            "raw_start": None,
            "raw_end": None,
            "bbox": None,
            "flags": [exc.code],
            "review_status": "unreviewed",
        }
        item["review_id"] = hashlib.sha256(
            json.dumps(item, sort_keys=True).encode()
        ).hexdigest()
        items = [item]
        artifact = {
            "schema_version": 1,
            "status": "unsupported",
            "parsed": None,
            "cleaned": None,
            "document_version": exc.document_version,
        }
        summary = {
            "status": "unsupported",
            "document_version": exc.document_version,
            "pages": None,
            "blocks": 0,
            "transform_count": 0,
            "review_items": 1,
        }
    else:
        cleaned = clean_document(parsed)
        items = cleaned.review_items
        artifact = {
            "schema_version": 1,
            "status": "parsed",
            "parsed": asdict(parsed),
            "cleaned": asdict(cleaned),
        }
        summary = {
            "status": "parsed",
            "document_version": parsed.document_version,
            "pages": parsed.page_count,
            "blocks": len(cleaned.blocks),
            "transform_count": sum(len(block.transforms) for block in cleaned.blocks),
            "review_items": len(items),
        }
    existing_ids = set()
    if review_queue.exists():
        existing_ids = {
            json.loads(line)["review_id"]
            for line in review_queue.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review_queue.parent.mkdir(parents=True, exist_ok=True)
    with review_queue.open("a", encoding="utf-8") as stream:
        for item in items:
            if item["review_id"] not in existing_ids:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                existing_ids.add(item["review_id"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/parsing/parser_artifact.json")
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=Path("data/parsing/parser_review_queue.jsonl"),
    )
    args = parser.parse_args()
    print(json.dumps(inspect_document(args.input, args.output, args.review_queue)))


if __name__ == "__main__":
    main()
