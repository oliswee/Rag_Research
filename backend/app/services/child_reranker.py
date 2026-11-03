"""Bounded local child cross-encoding; native jobs keep their slot after timeout."""

import asyncio
import hashlib
import math
import threading
import time
from pathlib import Path

from app.config import settings

_worker_slot = threading.BoundedSemaphore(1)
_models = {}


class LocalChildScorer:
    def __init__(self, cache_dir: str | None = None):
        self.model_id = settings.reranker_model
        self.cache_dir = cache_dir or settings.reranker_cache_dir
        self.revision = None
        self._encoder = None

    def load(self, *, local_files_only: bool = True):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        if self._encoder is None:
            self._encoder = TextCrossEncoder(
                model_name=self.model_id,
                cache_dir=self.cache_dir,
                threads=2,
                providers=["CPUExecutionProvider"],
                local_files_only=local_files_only,
            )
            model_dir = getattr(
                getattr(self._encoder, "model", None), "_model_dir", None
            )
            if model_dir is not None:
                fingerprint = hashlib.sha256()
                for path in sorted(Path(model_dir).rglob("*")):
                    if path.is_file() and path.suffix in {".onnx", ".json", ".txt"}:
                        fingerprint.update(
                            path.relative_to(model_dir).as_posix().encode()
                        )
                        with path.open("rb") as handle:
                            for block in iter(lambda: handle.read(1024 * 1024), b""):
                                fingerprint.update(block)
                self.revision = "sha256:" + fingerprint.hexdigest()
        return self._encoder

    def score(self, question: str, texts: list[str]) -> list[float]:
        return list(self.load().rerank(question, texts, batch_size=8))


def _default_scorer():
    key = (settings.reranker_model, settings.reranker_cache_dir)
    if key not in _models:
        _models[key] = LocalChildScorer()
    return _models[key]


def select_children(sources, limit, paper_ids=None):
    """Reserve explicit paper coverage, then fill; retain score order in output."""
    if not 1 <= limit <= 10:
        raise ValueError("Invalid child selection budget")
    unique = list({source["chunk_id"]: source for source in reversed(sources)}.values())
    unique.reverse()
    selected = set()
    required = set(paper_ids or [])
    for source in unique:
        if source["paper_id"] in required and len(selected) < limit:
            selected.add(source["chunk_id"])
            required.remove(source["paper_id"])
    for source in unique:
        if len(selected) >= limit:
            break
        selected.add(source["chunk_id"])
    return [source for source in unique if source["chunk_id"] in selected]


async def rerank_children(
    question: str, sources: list[dict], *, trace=None, scorer=None, timeout=None
) -> list[dict]:
    if len(sources) > 50:
        raise ValueError("Too many rerank candidates")
    sources = (
        select_children(sources, 10)
        if not sources
        else list({s["chunk_id"]: s for s in reversed(sources)}.values())[::-1]
    )
    start = time.perf_counter()
    status, error_type, ranked, scores = "empty", None, sources, None
    model = scorer
    timeout = settings.reranker_timeout_seconds if timeout is None else timeout
    if not 0 < timeout <= 15:
        raise ValueError("Invalid rerank timeout")
    if sources and not _worker_slot.acquire(blocking=False):
        status = "busy"
    elif sources:

        def work():
            nonlocal model
            try:
                model = model or _default_scorer()
                texts = [
                    (f"Section: {s['section']}\n" if s.get("section") else "")
                    + s["excerpt"]
                    for s in sources
                ]
                return [float(value) for value in model.score(question, texts)]
            finally:
                _worker_slot.release()

        task = asyncio.create_task(asyncio.to_thread(work))
        # Timed-out work never writes Trace; consume any eventual exception.
        task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        try:
            scores = await asyncio.wait_for(asyncio.shield(task), timeout)
            if len(scores) != len(sources) or not all(map(math.isfinite, scores)):
                raise ValueError("Invalid reranker score vector")
            ranked = [
                sources[i]
                for i in sorted(range(len(sources)), key=lambda i: -scores[i])
            ]
            status = "ok"
        except TimeoutError:
            status, error_type = "timeout", "TimeoutError"
        except Exception as exc:
            status, error_type = "failed", type(exc).__name__
    if trace is not None:
        duration = (time.perf_counter() - start) * 1000
        trace.record["reranking"] = {
            "status": status,
            "error_type": error_type,
            "model": getattr(model, "model_id", settings.reranker_model),
            "model_revision": getattr(model, "revision", None),
            "device": "cpu",
            "candidate_count": len(sources),
            "timeout_seconds": timeout,
            "duration_ms": round(duration, 3),
        }
        if error_type:
            trace.record["failures"].append(
                {"stage": "rerank", "error_type": error_type}
            )
        score_by_id = (
            {s["chunk_id"]: score for s, score in zip(sources, scores, strict=True)}
            if status == "ok"
            else {}
        )
        trace.add_ranked(
            "rerank",
            [(s["chunk_id"], score_by_id.get(s["chunk_id"])) for s in ranked],
            duration_ms=duration,
        )
    return ranked


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explicit public model setup")
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    scorer = LocalChildScorer()
    scorer.load(local_files_only=False)
    print({"model": scorer.model_id, "revision": scorer.revision, "device": "cpu"})
