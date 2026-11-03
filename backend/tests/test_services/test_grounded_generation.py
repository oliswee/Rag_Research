import importlib.util
import json

import pytest

from app.services.context_packing import pack_context
from app.services.retrieval_trace import RetrievalTrace
from tests.test_services.test_claim_verification import proposal, registry


def api():
    assert importlib.util.find_spec("app.services.grounded_generation"), (
        "Bounded grounded generator is missing"
    )
    from app.services.grounded_generation import VerificationBudget, generate_grounded

    return VerificationBudget, generate_grounded


class Model:
    def __init__(self, quote, *, status="supported", claimed=None, failure=None):
        self.quote, self.status, self.claimed, self.failure = (
            quote,
            status,
            claimed,
            failure,
        )
        self.calls = []

    async def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {"claims": [proposal(self.claimed or self.quote, self.quote)]}
        if self.failure:
            raise self.failure
        data = json.loads(kwargs["user_prompt"])
        return {
            "verdicts": [
                {
                    "claim_id": row["claim_id"],
                    "status": self.status,
                    "characteristics": {},
                }
                for row in data["claims"]
            ]
        }


@pytest.mark.asyncio
async def test_generated_claim_is_published_only_after_atomic_semantic_support():
    Budget, generate = api()
    quote = "Memory accuracy is 95 percent."
    evidence = registry(quote)
    trace = RetrievalTrace("accuracy?", {}, corpus_version="fixture")
    model = Model(quote)
    result = await generate(
        "accuracy?",
        evidence,
        pack_context(evidence.source_dicts()),
        trace,
        client=model,
    )
    assert result.support_status == "supported" and len(result.claims) == 1
    assert (
        result.claims[0].support_status == "supported"
        and result.citations[0].raw_quote == quote
    )
    assert quote in result.text and len(model.calls) == 2
    assert trace.record["prompt_evidence_ids"] == ["web:one"]
    assert quote not in str(trace.record)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["contradicted", "insufficient_evidence", "unknown"])
async def test_non_supported_semantic_verdict_never_publishes_factual_candidate(status):
    _, generate = api()
    quote = "Memory accuracy is 95 percent."
    evidence = registry(quote)
    result = await generate(
        "accuracy?",
        evidence,
        pack_context(evidence.source_dicts()),
        None,
        client=Model(quote, status=status),
    )
    assert quote not in result.text and result.claims[0].text == ""
    assert result.claims[0].support_status == status
    assert all(c.support_status == status for c in result.claims[0].citations)


@pytest.mark.asyncio
async def test_even_approving_judge_cannot_publish_wrong_number():
    _, generate = api()
    quote = "Memory accuracy is 95 percent."
    evidence, model = (
        registry(quote),
        Model(quote, claimed="Memory accuracy is 99 percent."),
    )
    result = await generate(
        "accuracy?", evidence, pack_context(evidence.source_dicts()), None, client=model
    )
    assert "99" not in result.text and result.claims[0].support_status == "unsupported"
    assert len(model.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quote,claimed",
    [
        ("The treatment did not reduce mortality.", "The treatment reduced mortality."),
        ("Risk did not increase with treatment.", "Risk increased with treatment."),
        ("本研究发现治疗未降低患者死亡率。", "本研究发现治疗降低患者死亡率。"),
    ],
)
async def test_negated_change_cannot_be_approved_by_a_permissive_judge(quote, claimed):
    _, generate = api()
    evidence, model = registry(quote), Model(quote, claimed=claimed)
    result = await generate(
        "What changed?",
        evidence,
        pack_context(evidence.source_dicts()),
        None,
        client=model,
    )
    assert result.support_status == "evidence_only"
    assert result.claims[0].support_status == "unsupported"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_verification_timeout_preserves_original_excerpts_not_unchecked_answer():
    _, generate = api()
    quote = "Memory accuracy is 95 percent."
    evidence = registry(quote)
    trace = RetrievalTrace("accuracy?", {}, corpus_version="fixture")
    result = await generate(
        "accuracy?",
        evidence,
        pack_context(evidence.source_dicts()),
        trace,
        client=Model(quote, failure=TimeoutError("private details")),
    )
    assert quote not in result.text and result.evidence[0]["excerpt"] == quote
    assert result.claims[0].support_status == "unknown"
    assert "private details" not in str(trace.record)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits", [{"max_calls": 1}, {"max_output_tokens": 1200}, {"max_input_bytes": 128}]
)
async def test_budget_exhaustion_never_makes_extra_calls(limits):
    Budget, generate = api()
    quote = "Memory accuracy is 95 percent."
    evidence, model = registry(quote), Model(quote)
    budget = Budget(**limits)
    result = await generate(
        "accuracy?",
        evidence,
        pack_context(evidence.source_dicts()),
        None,
        client=model,
        budget=budget,
    )
    assert len(model.calls) <= 1 and result.support_status != "supported"
    assert (
        budget.calls <= budget.max_calls
        and budget.reserved_output_tokens <= budget.max_output_tokens
    )


@pytest.mark.asyncio
async def test_owned_model_is_closed_after_generation_error(monkeypatch):
    _, generate = api()
    from app.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test")

    class Broken:
        closed = False

        async def complete_json(self, **kwargs):
            raise ValueError("secret provider body")

        async def close(self):
            self.closed = True

    model = Broken()
    monkeypatch.setattr("app.services.llm_client.LLMClient", lambda **kwargs: model)
    evidence = registry("Memory accuracy is 95 percent.")
    result = await generate(
        "accuracy?", evidence, pack_context(evidence.source_dicts()), None
    )
    assert model.closed and result.support_status != "supported"
