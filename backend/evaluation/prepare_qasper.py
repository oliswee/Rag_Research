"""Prepare a private, deterministic QASPER v0.3 paper-scoped BM25 benchmark."""

import hashlib
import json
import random
import uuid
from pathlib import Path

from app.services.source_index import build_chunks
from evaluation.isolated_corpus import validate_corpus

ROOT = Path(__file__).resolve().parents[1] / "data" / "evaluation"
SOURCE = ROOT / "qasper" / "qasper-dev-v0.3.json"
OUTPUT = ROOT / "qasper_benchmark"
SOURCE_URL = "https://huggingface.co/datasets/allenai/qasper"
SOURCE_SHA256 = "2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93"


def single_evidence(qa: dict, paragraphs: list[str]) -> str | None:
    answers = [item["answer"] for item in qa["answers"]]
    if not answers or any(answer["unanswerable"] for answer in answers):
        return None
    evidence = {
        text
        for answer in answers
        for text in answer["evidence"]
        if isinstance(text, str) and text.strip()
    }
    if len(evidence) != 1:
        return None
    text = next(iter(evidence))
    if len(text) > 1200 or paragraphs.count(text) != 1:
        return None
    return text


def agreed_no_answer(qa: dict) -> bool:
    answers = [item["answer"] for item in qa["answers"]]
    return bool(answers) and all(
        answer["unanswerable"] and not answer["evidence"] for answer in answers
    )


def main() -> None:
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise ValueError(f"Unexpected QASPER dev source SHA-256: {source_digest}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    positive = []
    negative = []
    for paper_id, paper in sorted(source.items()):
        paragraphs = [
            paragraph
            for section in paper["full_text"]
            for paragraph in section["paragraphs"]
        ]
        for qa in paper["qas"]:
            evidence = single_evidence(qa, paragraphs)
            if evidence is not None:
                positive.append((paper_id, qa, evidence))
            elif agreed_no_answer(qa):
                negative.append((paper_id, qa, None))

    random.Random(44).shuffle(positive)
    random.Random(45).shuffle(negative)
    selected = []
    used_papers = set()
    for candidates, count in ((positive, 32), (negative, 8)):
        chosen = 0
        for candidate in candidates:
            if candidate[0] in used_papers:
                continue
            used_papers.add(candidate[0])
            selected.append(candidate)
            chosen += 1
            if chosen == count:
                break
        if chosen != count:
            raise RuntimeError(f"Only selected {chosen} of {count} distinct papers")

    manifest = []
    chunks = []
    labels = []
    for source_id, qa, evidence_text in selected:
        paper = source[source_id]
        corpus_id = f"qasper:{source_id}"
        manifest.append(
            {
                "paper_id": corpus_id,
                "provider": "qasper",
                "title": paper["title"],
                "source_url": SOURCE_URL,
                "status": "parsed",
            }
        )
        gold_ids = []
        blocks = [("Abstract", paper["abstract"], False)]
        blocks.extend(
            (section["section_name"], paragraph, True)
            for section in paper["full_text"]
            for paragraph in section["paragraphs"]
        )
        paper_chunk_index = 0
        for block_index, (section, paragraph, is_fulltext) in enumerate(blocks):
            for piece_index, chunk in enumerate(
                build_chunks([(paragraph, None)], "full_text", SOURCE_URL)
            ):
                chunk["id"] = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"qasper-v0.3:{source_id}:{block_index}:{piece_index}",
                    )
                )
                chunk["paper_id"] = corpus_id
                chunk["chunk_index"] = paper_chunk_index
                paper_chunk_index += 1
                chunk["section"] = section
                chunk["start_char"] = None
                chunk["end_char"] = None
                chunks.append(chunk)
                if is_fulltext and paragraph == evidence_text:
                    gold_ids.append(chunk["id"])
        if evidence_text is not None and len(gold_ids) != 1:
            raise RuntimeError(f"Expected one mapped gold chunk for {source_id}")
        labels.append(
            {
                "id": qa["question_id"],
                "query": qa["question"],
                "paper_ids": [corpus_id],
                "group": "paper_answerable" if gold_ids else "paper_no_answer",
                "split": "external_validation",
                "relevant_chunk_ids": gold_ids,
                "label_status": "published_human_annotations",
                "label_provenance": "QASPER v0.3 validation",
                "source_paper_id": source_id,
                "source_url": SOURCE_URL,
            }
        )

    validate_corpus(manifest, chunks, labels)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "papers.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for filename, records in (("chunks.jsonl", chunks), ("labels.jsonl", labels)):
        (OUTPUT / filename).write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "papers": len(manifest),
                "chunks": len(chunks),
                "labels": len(labels),
                "answerable": sum(bool(item["relevant_chunk_ids"]) for item in labels),
                "no_answer": sum(not item["relevant_chunk_ids"] for item in labels),
                "source_sha256": source_digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
