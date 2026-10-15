"""Small privacy-conscious local trace for retrieval and evidence routing."""

import json
import math
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings

_KEY_PATTERN = re.compile(
    r"(?i)\b(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|api_key)\s*[:=]\s*\S+"
)
_TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+\S+")


def _safe_query(query: str) -> str:
    query = _KEY_PATTERN.sub("[redacted-key]", query)
    query = _TOKEN_PATTERN.sub("[redacted-key]", query)
    query = _BEARER_PATTERN.sub("Bearer [redacted-key]", query)
    for field in (
        "anthropic_api_key",
        "openai_api_key",
        "semantic_scholar_api_key",
        "ncbi_api_key",
        "langchain_api_key",
    ):
        secret = getattr(settings, field)
        if len(secret) >= 8:
            query = query.replace(secret, "[redacted-key]")
    return query[:500]


class RetrievalTrace:
    """Collect IDs and timings; never accept source excerpts or error messages."""

    def __init__(
        self,
        original_query: str,
        scope: dict,
        *,
        corpus_version: str,
        corpus_public: bool = False,
    ) -> None:
        self._started = time.perf_counter()
        self.record = {
            "schema_version": 1,
            "run_id": str(uuid.uuid4()),
            "started_at": datetime.now(UTC).isoformat(),
            "corpus_version": corpus_version,
            "index_version": "unversioned",
            "retrieval_schema": "legacy_flat_v1",
            "config_version": "m1_v1",
            "corpus_public": corpus_public,
            "original_query": _safe_query(original_query),
            "retrieval_query": None,
            "rewrite_status": None,
            "scope": scope,
            "retrieval_mode": None,
            "rankings": {},
            "stage_ms": {},
            "failures": [],
            "prompt_evidence_ids": [],
            "returned_evidence_ids": [],
            "local_retry_count": 0,
            "web_decision": "not_implemented",
            "web_events": [],
            "support_status": "pending",
            "total_ms": None,
        }

    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    def record_query(self, retrieval_query: str, status: str) -> None:
        self.record["retrieval_query"] = _safe_query(retrieval_query)
        self.record["rewrite_status"] = status

    def set_mode(self, mode: str) -> None:
        self.record["retrieval_mode"] = mode

    def add_ranked(
        self,
        stage: str,
        ranked: list[tuple[str, float | None]],
        *,
        duration_ms: float,
    ) -> None:
        self.record["rankings"][stage] = [
            {
                "child_id": child_id,
                "rank": rank,
                "score": float(score)
                if score is not None and math.isfinite(float(score))
                else None,
            }
            for rank, (child_id, score) in enumerate(ranked, start=1)
        ]
        self.record["stage_ms"][stage] = round(duration_ms, 3)

    def record_failure(self, stage: str, error: BaseException) -> None:
        self.record["failures"].append(
            {"stage": stage, "error_type": type(error).__name__}
        )

    def finish(
        self,
        status: str,
        prompt_evidence_ids: list[str],
        returned_evidence_ids: list[str] | None = None,
    ) -> None:
        self.record["support_status"] = status
        self.record["prompt_evidence_ids"] = list(dict.fromkeys(prompt_evidence_ids))
        self.record["returned_evidence_ids"] = list(
            dict.fromkeys(returned_evidence_ids or [])
        )
        self.record["total_ms"] = round((time.perf_counter() - self._started) * 1000, 3)

    def as_dict(self) -> dict:
        return self.record.copy()

    def append(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.record, ensure_ascii=False) + "\n")
