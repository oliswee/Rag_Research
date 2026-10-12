"""Evaluate saved discovery responses against human-labeled relevant DOIs.

Run ``uv run python -m evaluation.discovery_metrics judgments.json``.
The file contains ``{"cases": [{"query": ..., "relevant_dois": [...],
"result": <GET /api/discovery/search response>}]}``. Add ``content_scope`` to
result candidates after import to measure actual full-text availability.
"""

import argparse
import json
import statistics
from pathlib import Path

from app.services.discovery import normalize_doi


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def evaluate_cases(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        result = case["result"]
        relevant = {
            doi for value in case["relevant_dois"] if (doi := normalize_doi(value))
        }
        seen: set[str] = set()
        matched: set[str] = set()
        scope_by_identity: dict[str, str] = {}
        for paper in result["candidates"][:50]:
            doi = normalize_doi(paper.get("doi"))
            identity = doi or f"{paper['provider']}:{paper['provider_id']}"
            seen.add(identity)
            if doi in relevant:
                matched.add(doi)
            scope = paper.get("content_scope")
            if scope in {"full_text", "abstract", "metadata"}:
                previous = scope_by_identity.get(identity)
                if previous is None or scope == "full_text":
                    scope_by_identity[identity] = scope
        sources = set(result.get("searched_sources") or [])
        failed = set(result.get("failed_sources") or []) & sources
        full_text_scopes = list(scope_by_identity.values())
        rows.append(
            {
                "query": case["query"],
                "unique_papers_at_50": len(seen),
                "unique_relevant_papers_at_50": len(matched),
                "recall_at_50": round(len(matched) / len(relevant), 3)
                if relevant
                else None,
                "full_text_availability": (
                    round(
                        full_text_scopes.count("full_text") / len(full_text_scopes), 3
                    )
                    if full_text_scopes
                    else None
                ),
                "full_text_samples": len(full_text_scopes),
                "search_latency_ms": result.get("search_latency_ms"),
                "api_failure_rate": round(len(failed) / len(sources), 3)
                if sources
                else None,
            }
        )
    return {
        "cases": rows,
        "summary": {
            "queries": len(rows),
            "mean_unique_relevant_papers_at_50": _mean(
                [row["unique_relevant_papers_at_50"] for row in rows]
            ),
            "mean_recall_at_50": _mean(
                [row["recall_at_50"] for row in rows if row["recall_at_50"] is not None]
            ),
            "mean_full_text_availability": _mean(
                [
                    row["full_text_availability"]
                    for row in rows
                    if row["full_text_availability"] is not None
                ]
            ),
            "mean_search_latency_ms": _mean(
                [
                    row["search_latency_ms"]
                    for row in rows
                    if isinstance(row["search_latency_ms"], (int, float))
                ]
            ),
            "mean_api_failure_rate": _mean(
                [
                    row["api_failure_rate"]
                    for row in rows
                    if row["api_failure_rate"] is not None
                ]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judgments", type=Path, help="JSON file with labeled cases")
    arguments = parser.parse_args()
    cases = json.loads(arguments.judgments.read_text(encoding="utf-8"))["cases"]
    print(json.dumps(evaluate_cases(cases), indent=2))


if __name__ == "__main__":
    main()
