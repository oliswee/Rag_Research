"""Replay synthetic vectors on the real server; no relevance/capacity inference."""

import argparse
import json
import random
import subprocess
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.services.local_execution import loopback_host
from evaluation.final_statistics import quantile


def memory():
    try:
        import psutil

        rss = psutil.Process().memory_info().rss
    except ImportError:
        rss = None
    try:
        result = subprocess.run(
            [
                "docker",
                "stats",
                "research-vault-main-qdrant-1",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        container = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        container = None
    return {"python_rss_bytes": rss, "qdrant_container_memory": container}


def run(root):
    if not loopback_host(settings.qdrant_host):
        raise ValueError("Only local server diagnostics are authorized")
    output = root / "stress.json"
    if output.exists():
        raise ValueError("Preserve existing replay")
    manifest = json.loads((root / "freeze_config.json").read_text())
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=15,
        trust_env=False,
    )
    rows = []
    try:
        for count in (1000, 10000):
            name = "m8_stress_" + manifest["manifest_sha256"][:8] + "_" + str(count)
            if client.collection_exists(name):
                raise ValueError("Refusing to replace an existing collection")
            client.create_collection(
                name, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            before = memory()
            started = time.perf_counter()
            rng = random.Random(61)
            for start in range(0, count, 256):
                client.upsert(
                    name,
                    [
                        PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, name + ":" + str(i))),
                            vector=[rng.uniform(-1, 1) for _ in range(384)],
                            payload={
                                "diagnostic": "synthetic_m8_stress",
                                "position": i,
                            },
                        )
                        for i in range(start, min(count, start + 256))
                    ],
                    wait=True,
                )
            write_ms = (time.perf_counter() - started) * 1000
            query, elapsed = [rng.uniform(-1, 1) for _ in range(384)], []
            for _ in range(32):
                started = time.perf_counter()
                points = client.query_points(name, query=query, limit=10).points
                elapsed.append((time.perf_counter() - started) * 1000)
                if len(points) != 10:
                    raise ValueError("Incomplete synthetic query")
            info = client.get_collection(name)
            rows.append(
                {
                    "collection": name,
                    "synthetic_points": count,
                    "dimensions": 384,
                    "queries": 32,
                    "upsert_ms": write_ms,
                    "query_p50_ms": quantile(elapsed, 0.5),
                    "query_p95_ms": quantile(elapsed, 0.95),
                    "memory_before": before,
                    "memory_after": memory(),
                    "indexed_vectors": info.indexed_vectors_count,
                    "collection_status": str(info.status),
                    "quality_metric": None,
                    "multiuser_capacity": None,
                    "embedding_inference": "not_run_random_vector_replay",
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    finally:
        client.close()
    output.write_text(
        json.dumps(
            {
                "rows": rows,
                "scope": "single_local_server_synthetic_replay_only",
                "user_collections_modified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
