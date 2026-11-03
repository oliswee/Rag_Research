"""Prepare new public-reference development cases; no project-human Gold inferred."""

import argparse
import csv
import json
import random
import uuid
from pathlib import Path

from app.services.trace_versions import configuration_snapshot
from evaluation.final_contracts import (
    freeze_manifest,
    sha256,
    validate_annotation,
    write_jsonl,
)
from evaluation.final_model_freeze import capture_models
from evaluation.prepare_qasper import agreed_no_answer, single_evidence

TRAIN_HASH = "9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4"
SOURCE = Path("data/evaluation/qasper/qasper-train-v0.3.json")
PRIOR = Path("data/evaluation/qasper_benchmark/papers.json")


def paper_text(paper):
    blocks = ["Abstract", paper["abstract"]]
    for section in paper["full_text"]:
        blocks.extend([section["section_name"], *section["paragraphs"]])
    return "\n\n".join(str(block) for block in blocks if block)


def runtime_paper_id(namespace, identifier):
    return str(uuid.uuid5(uuid.UUID(namespace), "qasper-paper:" + identifier))


def prepare(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve prior freeze; choose an empty output directory")
    if sha256(SOURCE.read_bytes()) != TRAIN_HASH:
        raise ValueError("Unexpected QASPER source hash")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    excluded = {
        r["paper_id"].removeprefix("qasper:") for r in json.loads(PRIOR.read_text())
    }
    positive, negative = [], []
    for identifier, paper in sorted(source.items()):
        if identifier in excluded:
            continue
        paragraphs = [
            p for section in paper["full_text"] for p in section["paragraphs"]
        ]
        text = paper_text(paper)
        for qa in paper["qas"]:
            quote = single_evidence(qa, paragraphs)
            if quote and text.count(quote) == 1:
                positive.append((identifier, paper, qa, quote))
            elif agreed_no_answer(qa):
                negative.append((identifier, paper, qa, None))
    random.Random(61).shuffle(positive)
    random.Random(62).shuffle(negative)
    selected, used = [], set()
    namespace = str(uuid.uuid4())
    for pool, count in ((positive, 32), (negative, 8)):
        picked = 0
        for item in pool:
            if item[0] in used:
                continue
            selected.append(item)
            used.add(item[0])
            picked += 1
            if picked == count:
                break
        if picked != count:
            raise ValueError("Insufficient distinct reference papers")
    output.mkdir(parents=True, exist_ok=True)
    papers, references = [], []
    for identifier, paper, qa, quote in selected:
        text = paper_text(paper)
        paper_id = runtime_paper_id(namespace, identifier)
        papers.append(
            {
                "paper_id": paper_id,
                "dataset_paper_id": identifier,
                "title": paper["title"],
                "source_url": "https://arxiv.org/abs/" + identifier,
                "text": text,
            }
        )
        points = []
        if quote:
            answer = qa["answers"][0]["answer"]
            if answer.get("yes_no") is not None:
                points = ["Yes" if answer["yes_no"] else "No"]
            else:
                points = answer.get("extractive_spans") or (
                    [answer["free_form_answer"]]
                    if answer.get("free_form_answer")
                    else []
                )
            if not points:
                raise ValueError("Answerable reference lacks an answer")
        row = {
            "case_id": qa["question_id"],
            "question": qa["question"],
            "question_type": "public_dataset_answerable"
            if quote
            else "public_dataset_unanswerable",
            "paper_id": paper_id,
            "document_version": "sha256:" + sha256(text.encode()),
            "source_kind": "provided_text",
            "raw_quote": quote,
            "source_offset": [text.index(quote), text.index(quote) + len(quote)]
            if quote
            else None,
            "pdf_page_or_null": None,
            "answerable_local": quote is not None,
            "required_evidence": ["sha256:" + sha256(quote.encode())] if quote else [],
            "expected_answer_points": points,
            "expected_web_route": "not_required" if quote else "supplement_allowed",
            "label_provenance": "qasper_train_dataset_reference_not_project_gold",
            "review_status": "pending_review",
            "reviewer_type": "dataset_reference",
            "reviewer_id": None,
            "reviewed_at": None,
            "source_checked": False,
            "split": "frozen_reference_development",
        }
        validate_annotation(row, {paper_id: text})
        references.append(row)
    if len({row["case_id"] for row in references}) != 40 or used & excluded:
        raise ValueError("Duplicate/reused cases")
    write_jsonl(output / "papers.jsonl", papers)
    write_jsonl(output / "development_reference.jsonl", references)
    write_jsonl(output / "gold_annotations.jsonl", references)
    audit = Path("data/parsing/m2_human_review_20261024/cleaning_review_50.csv")
    with audit.open(encoding="utf-8-sig", newline="") as stream:
        sampled = list(csv.DictReader(stream))
    pdf_rows = [
        {
            "sample_id": index + 1,
            "cohort": row["cohort"],
            "source_file": row["source_file"],
            "document_version": row["document_version"],
            "page": row["page"],
            "block_id": row["block_id"],
            "raw_start": row["raw_start"],
            "raw_end": row["raw_end"],
            "raw_sha256": sha256(row["raw_text"].encode()),
            "review_status": "pending_independent_human",
            "reviewer_type": "",
            "reviewer_id": "",
            "verdict": "",
            "notes": "",
        }
        for index, row in enumerate(sampled)
    ]
    with (output / "pdf_parser_human_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pdf_rows[0]))
        writer.writeheader()
        writer.writerows(pdf_rows)
    config_version, config = configuration_snapshot()
    (output / "model_freeze.json").write_text(
        json.dumps(capture_models(), indent=2), encoding="utf-8"
    )
    manifest = freeze_manifest(
        output,
        [
            output / name
            for name in (
                "papers.jsonl",
                "development_reference.jsonl",
                "gold_annotations.jsonl",
                "pdf_parser_human_audit.csv",
                "model_freeze.json",
            )
        ],
        {
            "base_commit": "2b48206285f3237ce7590b3e5593c415bd199bda",
            "config_version": config_version,
            "runtime": config,
            "dataset_sha256": TRAIN_HASH,
            "sampling_seed": [61, 62],
            "runtime_namespace": namespace,
            "prior_dev_papers_excluded": len(excluded),
            "project_human_gold_count": 0,
            "reference_cases": 40,
            "pdf_review_pending": len(pdf_rows),
        },
    )
    (output / "freeze_config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(references),
                "papers": len(papers),
                "human_gold": 0,
                "pdf_samples": len(pdf_rows),
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
