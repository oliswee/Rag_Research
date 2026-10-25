import pytest

from app.workflows.source_query import answer_from_evidence, compare_evidence

EVIDENCE_QUOTE = "Active retrieval reduced latency by 20 percent."


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["bibliography", "page_marker"])
async def test_bibliography_entries_are_not_used_as_result_evidence(monkeypatch, role):
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "")
    answer, used = await answer_from_evidence(
        "What was found?",
        [
            {
                "chunk_id": "reference-entry",
                "title": "Paper",
                "content_scope": "full_text",
                "excerpt": "Smith et al. Retrieval improves results. 2024.",
                "page_start": 5,
                "evidence_role": role,
            }
        ],
    )
    assert used == []
    assert "Insufficient evidence" in answer


@pytest.mark.asyncio
async def test_rejects_fabricated_source_id(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test")

    class FakeLLM:
        calls = 0

        async def complete_json(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "claims": [
                        {"text": "Unsupported", "source_ids": ["invented"]},
                        {
                            "text": "Active retrieval reduced latency.",
                            "source_ids": ["real"],
                            "supporting_quotes": {"real": EVIDENCE_QUOTE},
                        },
                    ]
                }
            return {"supported_indices": [0]}

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    answer, used = await answer_from_evidence(
        "question",
        [
            {
                "chunk_id": "real",
                "title": "Paper",
                "content_scope": "abstract",
                "page_start": None,
                "excerpt": EVIDENCE_QUOTE,
                "paper_id": "p",
                "source_url": None,
                "page_end": None,
            }
        ],
    )
    assert "Unsupported" not in answer
    assert "Active retrieval reduced latency. [C:real]" in answer
    assert [source["chunk_id"] for source in used] == ["real"]


@pytest.mark.asyncio
async def test_rejects_unsupported_claim_with_real_citation(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test")

    class FakeLLM:
        calls = 0

        async def complete_json(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "claims": [
                        {
                            "text": "The paper proves all answers are correct.",
                            "source_ids": ["real"],
                            "supporting_quotes": {"real": EVIDENCE_QUOTE},
                        }
                    ]
                }
            return {"supported_indices": []}

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    answer, used = await answer_from_evidence(
        "Does retrieval improve accuracy?",
        [
            {
                "chunk_id": "real",
                "title": "Paper",
                "content_scope": "abstract",
                "page_start": None,
                "excerpt": EVIDENCE_QUOTE,
                "paper_id": "p",
                "source_url": None,
                "page_end": None,
            }
        ],
    )
    assert "proves all answers" not in answer
    assert used == []


@pytest.mark.asyncio
async def test_support_check_failure_does_not_publish_unchecked_claim(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test")

    class FakeLLM:
        calls = 0

        async def complete_json(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "claims": [
                        {
                            "text": "Active retrieval reduced latency.",
                            "source_ids": ["real"],
                            "supporting_quotes": {"real": EVIDENCE_QUOTE},
                        }
                    ]
                }
            raise TimeoutError("verification unavailable")

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    answer, used = await answer_from_evidence(
        "What happened to latency?",
        [
            {
                "chunk_id": "real",
                "title": "Paper",
                "content_scope": "abstract",
                "page_start": None,
                "excerpt": EVIDENCE_QUOTE,
                "paper_id": "p",
                "source_url": None,
                "page_end": None,
            }
        ],
    )
    assert answer == "Unable to verify answer support from the retrieved excerpts."
    assert used == []


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim",
    ["The accuracy is 99 percent.", "The full text proves accuracy is 95 percent."],
)
async def test_web_abstract_cannot_upgrade_access_or_invent_numbers(monkeypatch, claim):
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test")

    class FakeLLM:
        async def complete_json(self, **kwargs):
            if "supported_indices" in kwargs["system_prompt"]:
                return {"supported_indices": [0]}
            return {
                "claims": [
                    {
                        "text": claim,
                        "source_ids": ["web:1"],
                        "supporting_quotes": {
                            "web:1": "Memory accuracy is 95 percent."
                        },
                    }
                ]
            }

    monkeypatch.setattr("app.services.llm_client.LLMClient", FakeLLM)
    answer, used = await answer_from_evidence(
        "accuracy?",
        [
            {
                "chunk_id": "web:1",
                "paper_id": "p",
                "title": "Memory",
                "content_scope": "abstract",
                "evidence_level": "abstract_only",
                "page_start": None,
                "excerpt": "Memory accuracy is 95 percent.",
            }
        ],
    )
    assert used == [] and claim not in answer


@pytest.mark.asyncio
async def test_web_metadata_never_triggers_factual_model(monkeypatch):
    def forbidden():
        pytest.fail("Metadata reached generation")

    monkeypatch.setattr("app.services.llm_client.LLMClient", forbidden)
    answer, used = await answer_from_evidence(
        "experiments?", [{"evidence_level": "metadata_only"}]
    )
    assert used == []
