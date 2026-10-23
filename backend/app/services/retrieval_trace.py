"""Small privacy-conscious local trace for retrieval and evidence routing."""

import hashlib
import json
import math
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.services.trace_versions import configuration_snapshot, digest

_KEY_PATTERN = re.compile(
    r"(?i)\b[A-Za-z0-9_.-]*(?:api[_-]?key|secret|access[_-]?token|"
    r"auth[_-]?token|password)[A-Za-z0-9_.-]*[\"']?\s*[:=]\s*[\"']?[^\s,\"'}]+"
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
        "deepseek_api_key",
        "semantic_scholar_api_key",
        "ncbi_api_key",
        "langchain_api_key",
    ):
        secret = getattr(settings, field)
        if len(secret) >= 8:
            query = query.replace(secret, "[redacted-key]")
    return query[:500]


def _query_digest(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _safe_scope(scope: dict) -> dict:
    """Keep routing controls while avoiding raw request-supplied identifiers."""
    paper_ids = scope.get("paper_ids")
    mode = scope.get("requested_mode")
    return {
        "paper_id_count": len(paper_ids) if isinstance(paper_ids, list) else 0,
        "paper_id_hashes": [
            hashlib.sha256(item.encode("utf-8")).hexdigest()
            for item in paper_ids
            if isinstance(item, str)
        ]
        if isinstance(paper_ids, list)
        else [],
        "provider_hash": hashlib.sha256(scope["provider"].encode("utf-8")).hexdigest()
        if isinstance(scope.get("provider"), str)
        else None,
        "year_from": scope.get("year_from")
        if type(scope.get("year_from")) is int
        else None,
        "year_to": scope.get("year_to") if type(scope.get("year_to")) is int else None,
        "requested_mode": mode
        if mode in {"bm25", "dense", "hybrid", "legacy_pattern"}
        else None,
        "local_only": scope.get("local_only") is True,
        "latest": scope.get("latest") is True,
    }


class RetrievalTrace:
    """Collect IDs and timings; never accept source excerpts or error messages."""

    def __init__(
        self,
        original_query: str,
        scope: dict,
        *,
        corpus_version: str,
        corpus_public: bool = False,
        index_version: str = "unversioned",
    ) -> None:
        self._started = time.perf_counter()
        config_version, config_manifest = configuration_snapshot()
        self.record = {
            "schema_version": 2,
            "run_id": str(uuid.uuid4()),
            "started_at": datetime.now(UTC).isoformat(),
            "corpus_version": corpus_version,
            "index_version": index_version,
            "index_locator": index_version,
            "attempted_index_locators": [],
            "physical_index_revision": None,
            "retrieval_schema": "legacy_flat_v1",
            "config_version": config_version,
            "config_manifest": config_manifest,
            "corpus_public": corpus_public,
            "original_query": _safe_query(original_query)
            if corpus_public
            else "[private-query]",
            "original_query_sha256": _query_digest(original_query),
            "retrieval_query": None,
            "retrieval_query_sha256": None,
            "rewrite_status": None,
            "scope": _safe_scope(scope),
            "retrieval_mode": None,
            "rankings": {},
            "stage_ms": {},
            "failures": [],
            "prompt_evidence_ids": [],
            "returned_evidence_ids": [],
            "local_retry_count": 0,
            "llm_retry_count": 0,
            "llm_calls": 0,
            "llm_models": [],
            "token_usage": {"input_tokens": None, "output_tokens": None},
            "web_decision": "not_implemented",
            "web_events": [],
            "support_status": "pending",
            "total_ms": None,
        }

    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    def record_query(self, retrieval_query: str, status: str) -> None:
        self.record["retrieval_query"] = (
            _safe_query(retrieval_query)
            if self.record["corpus_public"]
            else "[private-query]"
        )
        self.record["retrieval_query_sha256"] = _query_digest(retrieval_query)
        self.record["rewrite_status"] = status

    def set_mode(self, mode: str) -> None:
        self.record["retrieval_mode"] = mode

    def record_index(self, locator: str) -> None:
        """Name the logical index; do not imply a physical consistency check."""
        self.record["index_locator"] = locator
        if locator not in self.record["attempted_index_locators"]:
            self.record["attempted_index_locators"].append(locator)
        self.record["index_version"] = "logical:sha256:" + digest(
            {
                "corpus_version": self.record["corpus_version"],
                "config_version": self.record["config_version"],
                "locator": locator,
            }
        )

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

    def record_llm_usage(
        self, model: str, input_tokens: int | None, output_tokens: int | None
    ) -> None:
        self.record["llm_calls"] += 1
        if model not in self.record["llm_models"]:
            self.record["llm_models"].append(model)
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
        ):
            current = self.record["token_usage"][name]
            if self.record["llm_calls"] == 1:
                self.record["token_usage"][name] = value
            elif value is None or current is None:
                self.record["token_usage"][name] = None
            else:
                self.record["token_usage"][name] = current + value

    def record_llm_retry(self, reason: str) -> None:
        self.record["llm_retry_count"] += 1
        self.record["failures"].append({"stage": "llm_retry", "error_type": reason})

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
