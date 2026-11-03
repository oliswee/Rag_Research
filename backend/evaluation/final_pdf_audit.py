"""Recheck retained parser samples against original PDFs; no human verdicts."""

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import fitz

from app.services.pdf_parser import PDFParserService
from app.services.source_cleaning import clean_document
from evaluation.final_contracts import freeze_manifest, sha256, write_jsonl


def run(root):
    output = root / "pdf_audit"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve existing PDF audit packet; use a fresh output root")
    samples = Path("data/parsing/m2_human_review_20261024/cleaning_review_50.csv")
    with samples.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output.mkdir(exist_ok=True)
    parsed, results = {}, []
    for number, row in enumerate(rows, 1):
        source = Path(row["source_file"])
        if source not in parsed:
            data = source.read_bytes()
            parsed[source] = ("sha256:" + sha256(data), PDFParserService().parse(data))
        version, current = parsed[source]
        page = next(
            (page for page in current.pages if page.number == int(row["page"])), None
        )
        left, right = int(row["raw_start"]), int(row["raw_end"])
        exact_offset = (
            page is not None and page.structured_text[left:right] == row["raw_text"]
        )
        same_text_block = page is not None and any(
            block.text == row["raw_text"] for block in page.blocks
        )
        results.append(
            {
                "sample_id": number,
                "cohort": row["cohort"],
                "public_source": source.name,
                "original_version_matches": version == row["document_version"],
                "current_parser_version": current.parser_version,
                "original_pdf_page": int(row["page"]),
                "old_offsets_match_current": exact_offset,
                "same_text_block_present": same_text_block,
                "reviewer_type": "automated_source_integrity_check",
                "human_verdict": None,
                "layout_quality": "not_inferred_from_text_match",
            }
        )
    write_jsonl(output / "anchor_checks.jsonl", results)
    pools = {"suspicious": [], "ordinary": []}
    for source, (_, current) in sorted(parsed.items()):
        cleaned = clean_document(current)
        originals = {b.id: b for page in current.pages for b in page.blocks}
        for block in cleaned.blocks:
            original = originals[block.block_id]
            cohort = "suspicious" if block.flags or block.transforms else "ordinary"
            pools[cohort].append(
                {
                    "cohort": cohort,
                    "source_file": source.name,
                    "document_version": current.document_version,
                    "parser_version": current.parser_version,
                    "cleaner_version": cleaned.cleaner_version,
                    "page": block.page,
                    "block_id": block.block_id,
                    "raw_start": original.raw_start,
                    "raw_end": original.raw_end,
                    "raw_text": block.raw_text,
                    "cleaned_text": block.cleaned_text,
                    "flags": block.flags,
                    "transforms": [asdict(t) for t in block.transforms],
                    "review_status": "pending_independent_human",
                    "reviewer_type": "",
                    "reviewer_id": "",
                    "verdict": "",
                }
            )
    rng = random.Random(61)
    current_samples = []
    for cohort, count in (("suspicious", 30), ("ordinary", 20)):
        rng.shuffle(pools[cohort])
        if len(pools[cohort]) < count:
            raise ValueError("Insufficient current-parser review strata")
        current_samples.extend(pools[cohort][:count])
    for number, sample in enumerate(current_samples, 1):
        sample["sample_id"] = number
        sample["raw_sha256"] = sha256(sample["raw_text"].encode())
    write_jsonl(output / "current_parser_samples.jsonl", current_samples)
    fields = [
        "sample_id",
        "cohort",
        "source_file",
        "document_version",
        "parser_version",
        "cleaner_version",
        "page",
        "block_id",
        "raw_start",
        "raw_end",
        "raw_sha256",
        "review_status",
        "reviewer_type",
        "reviewer_id",
        "verdict",
    ]
    with (output / "current_parser_human_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(current_samples)
    packet = freeze_manifest(
        output,
        [
            output / "current_parser_samples.jsonl",
            output / "current_parser_human_audit.csv",
        ],
        {
            "sampling_seed": 61,
            "human_reviewed": 0,
            "suspicious": 30,
            "ordinary": 20,
            "replaces_stale_offsets_for_future_review_only": True,
        },
    )
    (output / "current_review_freeze.json").write_text(json.dumps(packet, indent=2))
    visual = []
    for label, filename, page in (
        ("two_columns", "arxiv-2307.03172.pdf", 6),
        ("table_caption", "arxiv-1106.1265.pdf", 2),
        ("cross_page", "arxiv-1912.02920.pdf", 2),
        ("formula", "arxiv-2212.02407.pdf", 5),
        ("complex_table", "arxiv-2005.11401.pdf", 6),
    ):
        source = Path("data/evaluation/fulltext") / filename
        with fitz.open(source) as document:
            target = output / f"{label}.png"
            document[page - 1].get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(target)
            visual.append(
                {
                    "type": label,
                    "source": filename,
                    "pdf_page": page,
                    "image": target.name,
                    "review_status": "awaiting_ai_visual_review",
                    "independent_human": False,
                }
            )
    write_jsonl(output / "visual_materials.jsonl", visual)
    summary = {
        "samples": len(results),
        "suspicious": sum(r["cohort"] == "suspicious" for r in results),
        "ordinary": sum(r["cohort"] == "ordinary" for r in results),
        "document_hash_matches": sum(r["original_version_matches"] for r in results),
        "old_offsets_match_current": sum(
            r["old_offsets_match_current"] for r in results
        ),
        "same_text_block_present": sum(r["same_text_block_present"] for r in results),
        "independent_human_reviewed": 0,
        "current_parser_review_packet": len(current_samples),
        "notes": "Changed offsets require audit; not automatic error labels.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
