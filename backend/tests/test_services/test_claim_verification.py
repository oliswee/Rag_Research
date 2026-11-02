import importlib.util
from types import MappingProxyType

import pytest

from app.services.evidence_registry import EvidenceRegistry, seal_web_sources
from tests.test_services.test_evidence_registry import web_source


def check(proposal, registry, **kwargs):
    assert importlib.util.find_spec("app.services.claim_verification"), (
        "Atomic verifier is missing"
    )
    from app.services.claim_verification import check_proposal

    return check_proposal(proposal, registry, **kwargs)


def registry(quote, **fields):
    import hashlib

    raw = web_source()
    version = "sha256:" + hashlib.sha256(quote.encode()).hexdigest()
    raw.update(excerpt=quote, document_version=version, **fields)
    raw["source_spans"] = [
        {
            **raw["source_spans"][0],
            "document_version": version,
            "raw_end": len(quote),
            "excerpt_sha256": hashlib.sha256(quote.encode()).hexdigest(),
        }
    ]
    snapshots = seal_web_sources([raw])
    return EvidenceRegistry(MappingProxyType({s.source_id: s for s in snapshots}))


def proposal(text, quote, **fields):
    return {
        "claim_id": "c1",
        "paper_id": "p",
        "text": text,
        "source_ids": ["web:one"],
        "supporting_quotes": {"web:one": quote},
        **fields,
    }


def test_exact_candidate_requires_semantic_support_and_has_original_citation():
    quote = "Memory accuracy is 95 percent."
    result = check(proposal(quote, quote), registry(quote), paper_id="p")
    assert result.text == quote and result.support_status == "unknown"
    assert result.citations[0].raw_quote == quote


@pytest.mark.parametrize(
    "quote,claim,reason",
    [
        (
            "Memory accuracy is 95 percent.",
            "Memory accuracy is 99 percent.",
            "numeric_support_missing",
        ),
        (
            "The OR was 0.8 in the study.",
            "The RR was 0.8 in the study.",
            "effect_measure_mismatch",
        ),
        (
            "Retrieval latency was 20 ms.",
            "Retrieval latency was 20 s.",
            "unit_support_missing",
        ),
        (
            "The sample size was 200 and accuracy was 95 percent.",
            "The sample size was 95 and accuracy was 200 percent.",
            "numeric_role_mismatch",
        ),
        (
            "Treatment was associated with lower risk.",
            "Treatment caused lower risk.",
            "causal_escalation",
        ),
        (
            "Risk decreased in the study.",
            "Risk increased in the study.",
            "direction_mismatch",
        ),
        (
            "Memory accuracy did not improve.",
            "Memory accuracy improved.",
            "direction_mismatch",
        ),
        (
            "The sample included 200 patients.",
            "The sample included 500 patients.",
            "numeric_support_missing",
        ),
        (
            "Memory accuracy is 95 percent.",
            "The full text proves accuracy is 95 percent.",
            "abstract_access_upgrade",
        ),
        (
            "Memory accuracy is 95 percent.",
            "Memory accuracy is 95 percent. Survival is improved.",
            "non_atomic_claim",
        ),
    ],
)
def test_scientific_counterexamples_fail_before_semantic_judge(quote, claim, reason):
    result = check(proposal(claim, quote), registry(quote), paper_id="p")
    assert result.support_status == "unsupported" and result.text == ""
    assert reason in result.uncertainty


@pytest.mark.parametrize(
    "fields",
    [
        {"source_ids": ["web:invented"]},
        {"paper_id": "other"},
        {"supporting_quotes": {"web:one": "A result outside the cited excerpt."}},
        {"source_ids": []},
        {"supporting_quotes": {}},
    ],
)
def test_cross_paper_or_invalid_original_citation_fails_closed(fields):
    quote = "Memory accuracy is 95 percent."
    result = check(proposal(quote, quote, **fields), registry(quote), paper_id="p")
    assert result.support_status == "unsupported" and result.text == ""


def test_characteristics_need_their_own_original_quote_and_value():
    quote = "The adult population was studied in an observational design."
    result = check(
        proposal(
            quote,
            quote,
            characteristics={
                "population": {
                    "value": "children",
                    "source_id": "web:one",
                    "raw_quote": quote,
                },
                "study_design": {
                    "value": "observational",
                    "source_id": "web:one",
                    "raw_quote": quote,
                },
            },
        ),
        registry(quote),
        paper_id="p",
    )
    assert result.characteristics.population.value is None
    assert result.characteristics.study_design.value == "observational"
    assert result.characteristics.study_design.support_status == "unknown"
