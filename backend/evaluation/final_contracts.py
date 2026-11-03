"""Strict source/freeze contracts; dataset or AI review never implies human Gold."""

import hashlib
import json
from pathlib import Path


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def validate_annotation(row, documents, *, require_human=False):
    if not row.get("case_id") or not row.get("question"):
        raise ValueError("Missing annotation identity")
    original = documents.get(row.get("paper_id"))
    if isinstance(original, dict):
        if row.get("source_kind") != "pdf" or not isinstance(
            original.get("pdf_bytes"), bytes
        ):
            raise ValueError("Native PDF bytes required")
        page = row.get("pdf_page_or_null")
        text = (
            original.get("pages", {}).get(page)
            if type(page) is int and page > 0
            else None
        )
        version = "sha256:" + sha256(original["pdf_bytes"])
    else:
        text = original
        version = "sha256:" + sha256(text.encode()) if isinstance(text, str) else None
        if row.get("source_kind") == "pdf":
            raise ValueError("PDF label needs original bytes/pages")
    if text is None or row.get("document_version") != version:
        raise ValueError("Source identity/version mismatch")
    if row.get("source_kind") != "pdf" and row.get("pdf_page_or_null") is not None:
        raise ValueError("Text references cannot claim a PDF page")
    if type(row.get("answerable_local")) is not bool:
        raise ValueError("Answerability needs a reviewed boolean")
    if row["answerable_local"]:
        span, quote = row.get("source_offset"), row.get("raw_quote")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(n) is not int for n in span)
            or not 0 <= span[0] < span[1] <= len(text)
            or not isinstance(quote, str)
            or not quote
            or text[span[0] : span[1]] != quote
            or not row.get("required_evidence")
            or not row.get("expected_answer_points")
        ):
            raise ValueError("Invalid original evidence anchor")
    elif any(
        row.get(key)
        for key in (
            "raw_quote",
            "source_offset",
            "required_evidence",
            "expected_answer_points",
        )
    ):
        raise ValueError("Unanswerable reference cannot invent evidence/answer")
    if require_human and (
        row.get("review_status") != "independent_human_reviewed"
        or row.get("reviewer_type") != "human"
        or not isinstance(row.get("reviewer_id"), str)
        or not row["reviewer_id"].strip()
        or row["reviewer_id"].strip().lower() in {"codex", "chatgpt", "ai", "model"}
        or not row.get("reviewed_at")
        or row.get("source_checked") is not True
    ):
        raise ValueError("Independent human provenance is required")
    return row


def freeze_manifest(root, paths, config):
    root = Path(root).resolve()
    files = {}
    for path in paths:
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Freeze path escaped root")
        files[path.relative_to(root).as_posix()] = sha256(path.read_bytes())
    body = {"files": files, "config": config}
    return {
        **body,
        "manifest_sha256": sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ),
    }


def verify_freeze(root, manifest):
    root = Path(root).resolve()
    paths = []
    for name in manifest["files"]:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Freeze path escaped root")
        paths.append(path)
    if freeze_manifest(root, paths, manifest["config"]) != manifest:
        raise ValueError("Frozen inputs changed")


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
