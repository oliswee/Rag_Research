import pytest

from app.workflows.source_query import answer_from_evidence, compare_evidence


@pytest.mark.asyncio
async def test_rejects_fabricated_source_id(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test")

    class FakeLLM:
        async def complete_json(self, **kwargs):
            return {
                "claims": [
                    {"text": "Unsupported", "source_ids": ["invented"]},
                    {"text": "Supported", "source_ids": ["real"]},
                ]
            }

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    answer, used = await answer_from_evidence(
        "question",
        [
            {
                "chunk_id": "real",
                "title": "Paper",
                "content_scope": "abstract",
                "page_start": None,
                "excerpt": "Supported",
                "paper_id": "p",
                "source_url": None,
                "page_end": None,
            }
        ],
    )
    assert "Unsupported" not in answer
    assert "Supported [C:real]" in answer
    assert [source["chunk_id"] for source in used] == ["real"]


@pytest.mark.asyncio
async def test_comparison_rejects_cross_paper_citation(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test")

    class FakeLLM:
        async def complete_json(self, **kwargs):
            return {
                "cells": [
                    {
                        "paper_id": "one",
                        "text": "Wrong source",
                        "source_ids": ["two-chunk"],
                    }
                ]
            }

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)

    def source(paper, chunk):
        return {
            "paper_id": paper,
            "chunk_id": chunk,
            "title": paper,
            "excerpt": "source text",
            "content_scope": "abstract",
            "source_url": None,
            "page_start": None,
            "page_end": None,
        }

    cells = await compare_evidence(
        "Common findings",
        None,
        {"one": [source("one", "one-chunk")], "two": [source("two", "two-chunk")]},
    )
    assert cells[0]["text"] == "Insufficient evidence found"
    assert cells[0]["evidence"] == []
