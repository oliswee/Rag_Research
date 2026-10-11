"""Validate human-labelled retrieval queries against the current local corpus."""

import json
from pathlib import Path

GROUPS = {"exact_terms", "semantic_rephrase", "cross_paper", "no_answer"}


def load_cases(path: Path, valid_chunk_ids: set[str]) -> list[dict]:
    """Load JSONL labels, rejecting ambiguous or stale benchmark inputs."""
    cases = []
    seen_queries = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON") from exc
        if not isinstance(case, dict):
            raise ValueError(f"Line {line_number}: expected an object")
        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Line {line_number}: query must be non-empty")
        normalized = " ".join(query.split()).casefold()
        if normalized in seen_queries:
            raise ValueError(f"Line {line_number}: Duplicate query")
        seen_queries.add(normalized)
        group = case.get("group")
        if group not in GROUPS:
            raise ValueError(f"Line {line_number}: invalid group {group!r}")
        ids = case.get("relevant_chunk_ids")
        if (
            not isinstance(ids, list)
            or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(f"Line {line_number}: invalid relevant_chunk_ids")
        if group == "no_answer" and ids:
            raise ValueError(
                f"Line {line_number}: No-answer query must have no relevant chunks"
            )
        if group != "no_answer" and not ids:
            raise ValueError(
                f"Line {line_number}: answerable query needs relevant chunks"
            )
        unknown = set(ids) - valid_chunk_ids
        if unknown:
            raise ValueError(
                f"Line {line_number}: Unknown chunk ID {sorted(unknown)[0]}"
            )
        cases.append(
            {"query": query.strip(), "group": group, "relevant_chunk_ids": ids}
        )
    if not cases:
        raise ValueError("No labelled queries provided")
    return cases
