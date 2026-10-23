"""QASPER reference extraction must preserve the published answer type."""

import hashlib

import pytest

from evaluation.qasper_generation import reference_answer, validate_generation_inputs


@pytest.mark.asyncio
async def test_generation_records_own_calls_and_links_retrieval(monkeypatch):
    from app.config import settings
    from app.services.local_execution import local_only_request
    from evaluation.qasper_generation import generate_answer

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1/")

    async def answer(query, evidence, trace):
        assert local_only_request.get()
        trace.record_llm_usage("local", 5, 2)
        return "answer", evidence

    monkeypatch.setattr("evaluation.qasper_generation.answer_from_evidence", answer)
    retrieval = {
        "run_id": "retrieval-1",
        "corpus_version": "test",
        "index_version": "index",
        "index_locator": "source_collection:scope",
        "config_version": "retrieval-config",
        "total_ms": 25,
        "rankings": {},
    }
    _, trace = await generate_answer(
        {"query": "q", "paper_ids": ["p"]}, [{"chunk_id": "c"}], retrieval
    )
    assert trace["run_id"] != "retrieval-1"
    assert trace["retrieval_run_id"] == "retrieval-1"
    assert trace["index_locator"] == "source_collection:scope"
    assert trace["retrieval_config_version"] == "retrieval-config"
    assert trace["llm_calls"] == 1
    assert trace["token_usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert trace["returned_evidence_ids"] == ["c"]
    assert "llm_calls" not in retrieval
    assert not local_only_request.get()


@pytest.mark.asyncio
async def test_generation_rejects_cloud_before_model_call(monkeypatch):
    from app.config import settings
    from evaluation.qasper_generation import generate_answer

    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    with pytest.raises(ValueError, match="loopback Ollama"):
        await generate_answer({}, [], {})


def generation_fixture():
    labels = [
        {
            "id": "q1",
            "query": "Question?",
            "paper_ids": ["qasper:p1"],
            "source_paper_id": "p1",
            "group": "paper_answerable",
        }
    ]
    report = {
        "results": [
            {
                "query": "Question?",
                "paper_ids": ["qasper:p1"],
                "actual_mode": "hybrid",
                "ranked_chunk_ids": ["c1"],
            }
        ]
    }
    traces = [
        {
            "original_query_sha256": hashlib.sha256(b"Question?").hexdigest(),
            "scope": {"paper_id_hashes": [hashlib.sha256(b"qasper:p1").hexdigest()]},
            "retrieval_mode": "hybrid",
            "rankings": {"sqlite_resolved": [{"child_id": "c1"}]},
        }
    ]
    return dict(
        report=report,
        traces=traces,
        labels=labels,
        chunks={"c1": {"paper_id": "qasper:p1"}},
        titles={"qasper:p1": "Paper"},
        source={"p1": {"qas": [{"question_id": "q1", "question": "Question?"}]}},
        limits={"paper_answerable": 1, "paper_no_answer": 0},
        evidence_limit=1,
    )


def test_aligned_generation_inputs_are_accepted():
    assert "q1" in validate_generation_inputs(**generation_fixture())


@pytest.mark.parametrize(
    "mismatch", ["query", "scope", "rank", "source", "count", "limit"]
)
def test_generation_rejects_mixed_inputs_before_model_calls(mismatch):
    inputs = generation_fixture()
    if mismatch == "query":
        inputs["traces"][0]["original_query_sha256"] = "another-query"
    elif mismatch == "scope":
        inputs["report"]["results"][0]["paper_ids"] = ["qasper:p2"]
    elif mismatch == "rank":
        inputs["traces"][0]["rankings"]["sqlite_resolved"] = []
    elif mismatch == "source":
        inputs["source"]["p1"]["qas"][0]["question"] = "Another question?"
    elif mismatch == "count":
        inputs["limits"]["paper_answerable"] = 2
    else:
        inputs["evidence_limit"] = 0
    with pytest.raises(ValueError):
        validate_generation_inputs(**inputs)


def test_reference_answer_preserves_free_form_yes_no_and_unanswerable():
    def qa(**answer):
        return {
            "answers": [
                {
                    "answer": {
                        "unanswerable": False,
                        "free_form_answer": "",
                        "yes_no": None,
                        "extractive_spans": [],
                        **answer,
                    }
                }
            ]
        }

    assert reference_answer(qa(free_form_answer="Observed gain.")) == "Observed gain."
    assert reference_answer(qa(yes_no=False)) == "No"
    assert reference_answer(qa(extractive_spans=["one", "two"])) == "one; two"
    assert reference_answer(qa(unanswerable=True)) is None
