from types import MappingProxyType

import pytest

from app.schemas.evidence import GroundedAnswer
from app.services.evidence_registry import EvidenceRegistry
from evaluation.final_generation import citation_truth, metric_names


def test_reference_metrics_only_when_reference_answer_exists():
    assert len(metric_names(None)) == 3
    assert len(metric_names("Known reference")) == 5
    assert len(metric_names("")) == 3


def test_supported_answer_without_original_citation_is_not_true_by_vacuity():
    registry = EvidenceRegistry(MappingProxyType({}))
    result = citation_truth(
        GroundedAnswer(text="A fabricated result", support_status="supported"),
        registry,
        local_only=True,
    )
    assert result["citation_integrity"] is False
    assert result["human_claim_support"] is None


def test_abstention_with_no_citation_does_not_become_claim_support():
    registry = EvidenceRegistry(MappingProxyType({}))
    result = citation_truth(
        GroundedAnswer(text="Insufficient evidence"), registry, local_only=True
    )
    assert result["citation_integrity"] is True
    assert result["supported_claim_count"] == 0
    assert result["human_claim_support"] is None


@pytest.mark.asyncio
async def test_local_pdf_sufficiency_fixture_has_citable_body(async_db_session):
    from app.services.evidence_registry import build_registry
    from evaluation.final_fixture_generation import local

    sources = await local(
        async_db_session, "Memory accuracy is 95 percent.", full_text=True
    )
    registry = await build_registry(async_db_session, sources)
    assert registry.sources
    citation = registry.cite(
        next(iter(registry.sources)), "Memory accuracy is 95 percent."
    )
    assert citation.pdf_page == 1
