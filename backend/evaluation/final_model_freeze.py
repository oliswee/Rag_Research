"""Record actual cached weight/tokenizer hashes without provider credentials."""

import argparse
import json
from pathlib import Path

import httpx

from app.config import settings
from app.services.child_reranker import LocalChildScorer
from app.services.embedding_service import EmbeddingService
from app.services.local_execution import local_only_request, loopback_url
from evaluation.final_contracts import (
    freeze_manifest,
    sha256,
    verify_freeze,
    verify_models,
)


def files(directory):
    directory = Path(directory)
    return {
        p.relative_to(directory).as_posix(): {
            "sha256": sha256(p.read_bytes()),
            "bytes": p.stat().st_size,
        }
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix in {".onnx", ".json", ".txt"}
    }


def capture_models():
    if not loopback_url(settings.ollama_base_url):
        raise ValueError("Only local frozen models are allowed")
    token = local_only_request.set(True)
    try:
        embedding = EmbeddingService()
        dimension = len(embedding._embed_local(["Offline model hash preparation."])[0])
        embed_dir = embedding._local_model.model._model_dir
        scorer = LocalChildScorer()
        scorer.load()
        rerank_dir = scorer._encoder.model._model_dir
    finally:
        local_only_request.reset(token)
    with httpx.Client(trust_env=False, timeout=10) as client:
        base = settings.ollama_base_url.rstrip("/").removesuffix("/v1")
        version = client.get(base + "/api/version").json()
        tags = client.get(base + "/api/tags").json()
    model = next(m for m in tags["models"] if m["name"] == settings.ollama_model)
    result = {
        "embedding": {
            "model": embedding.model,
            "dimensions": dimension,
            "files": files(embed_dir),
        },
        "reranker": {
            "model": scorer.model_id,
            "revision": scorer.revision,
            "files": files(rerank_dir),
        },
        "generation_and_judge": {
            "provider": "ollama",
            "model": settings.ollama_model,
            "manifest_digest": model["digest"],
            "model_bytes": model["size"],
            "runtime_version": version["version"],
            "temperature": 0,
        },
        "judge_limits": {
            "metric_timeout_seconds": 45,
            "attempts_per_metric": 6,
            "num_predict": 512,
            "num_ctx": 4096,
        },
        "preparation": "one_offline_embedding_for_cache_identification_only",
    }
    return result


def bind_existing(root):
    manifest = json.loads((root / "freeze_config.json").read_text())
    verify_freeze(root, manifest)
    target = root / "model_bound_freeze.json"
    if not target.exists():
        binding = freeze_manifest(
            root,
            [root / name for name in manifest["files"]]
            + [root / "model_freeze.json", root / "freeze_config.json"],
            {
                "previous_manifest_sha256": manifest["manifest_sha256"],
                "capture_timing": "after_initial_development_run_not_prospective",
            },
        )
        target.write_text(json.dumps(binding, indent=2))
    status = verify_models(root, manifest)
    print(json.dumps(status))


def run(root):
    if (root / "model_freeze.json").exists():
        manifest = json.loads((root / "freeze_config.json").read_text())
        print(json.dumps(verify_models(root, manifest)))
        return
    result = capture_models()
    (root / "model_freeze.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "embedding_dimensions": result["embedding"]["dimensions"],
                "generation_manifest_digest": result["generation_and_judge"][
                    "manifest_digest"
                ],
                "ollama_version": result["generation_and_judge"]["runtime_version"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bind-existing-development", action="store_true")
    args = parser.parse_args()
    bind_existing(args.root) if args.bind_existing_development else run(args.root)
