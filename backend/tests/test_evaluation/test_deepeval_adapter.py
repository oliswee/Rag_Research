"""The adapter must never turn a missing or failed judge into a quality score."""

import json

import pytest

from evaluation.deepeval_adapter import JudgeConfig, evaluate_trace, run_batch

TRACE = {
    "run_id": "run-1",
    "original_query": "What did the paper find?",
    "prompt_evidence_ids": ["chunk-1"],
    "corpus_public": True,
    "corpus_version": "qasper-v0.3-dev",
}


def test_answerable_public_case_uses_reference_metrics_and_resolved_context():
    seen = []

    def measure(name, case, _model):
        seen.append((name, case.input, case.retrieval_context, case.expected_output))
        return {"score": 0.75, "cost": None}

    result = evaluate_trace(
        TRACE,
        actual_output="The method improved recall. [C:chunk-1]",
        evidence_by_id={"chunk-1": "The method improved recall."},
        expected_output="The method improved recall.",
        judge=JudgeConfig(provider="anthropic", model="claude-test", api_key="fake"),
        measure=measure,
    )

    assert result["status"] == "evaluated"
    assert set(result["metrics"]) == {
        "faithfulness",
        "answer_relevancy",
        "contextual_relevancy",
        "contextual_precision",
        "contextual_recall",
    }
    assert all(
        item[1:]
        == (
            "What did the paper find?",
            ["The method improved recall."],
            "The method improved recall.",
        )
        for item in seen
    )
    assert result["metrics"]["faithfulness"]["score"] == 0.75
    assert result["api_call_count"] is None
    assert result["token_usage"] is None
    assert result["total_cost"] is None
    assert result["latency_ms"] is not None
    assert "fake" not in str(result)


def test_unanswerable_case_does_not_invent_reference_or_score():
    names = []

    def measure(name, _case, _model):
        names.append(name)
        return {"score": 0.5, "cost": 0.0}

    result = evaluate_trace(
        TRACE,
        actual_output="Insufficient evidence.",
        evidence_by_id={"chunk-1": "Irrelevant passage."},
        expected_output=None,
        judge=JudgeConfig(provider="anthropic", model="claude-test", api_key="fake"),
        measure=measure,
    )

    assert result["status"] == "evaluated"
    assert names == ["faithfulness", "answer_relevancy", "contextual_relevancy"]


def test_missing_key_skips_without_importing_or_calling_judge():
    result = evaluate_trace(
        TRACE,
        actual_output="An answer.",
        evidence_by_id={"chunk-1": "Evidence."},
        expected_output=None,
        judge=JudgeConfig(provider="anthropic", model="claude-test", api_key=""),
        measure=lambda *_: (_ for _ in ()).throw(AssertionError("judge called")),
    )
    assert result["status"] == "not_evaluated"
    assert result["error"] == "missing_key"
    assert all(item["score"] is None for item in result["metrics"].values())


def test_judge_rate_limit_preserves_explicit_failure_status():
    def rate_limited(*_):
        raise TimeoutError("429 secret=do-not-log")

    result = evaluate_trace(
        TRACE,
        actual_output="An answer.",
        evidence_by_id={"chunk-1": "Evidence."},
        expected_output=None,
        judge=JudgeConfig(provider="anthropic", model="claude-test", api_key="fake"),
        measure=rate_limited,
    )
    assert result["status"] == "not_evaluated"
    assert result["error"] == "TimeoutError"
    assert all(item["score"] is None for item in result["metrics"].values())
    assert "do-not-log" not in str(result)


def test_private_context_requires_explicit_upload_permission():
    result = evaluate_trace(
        {**TRACE, "corpus_public": False},
        actual_output="An answer.",
        evidence_by_id={"chunk-1": "Private source."},
        expected_output=None,
        judge=JudgeConfig(provider="anthropic", model="claude-test", api_key="fake"),
        measure=lambda *_: (_ for _ in ()).throw(AssertionError("judge called")),
    )
    assert result["status"] == "not_evaluated"
    assert result["error"] == "private_corpus_not_allowed"


def test_judge_config_rejects_unknown_provider_and_invalid_threshold():
    with pytest.raises(ValueError, match="provider"):
        JudgeConfig(provider="unknown", model="a-model", api_key="fake")
    with pytest.raises(ValueError, match="threshold"):
        JudgeConfig(
            provider="anthropic", model="a-model", api_key="fake", threshold=1.2
        )


def test_default_metric_receives_configured_threshold(monkeypatch):
    from evaluation import deepeval_adapter

    seen = []

    def fake_metric(name, _case, _model, *, threshold):
        seen.append((name, threshold))
        return {"score": 0.6, "cost": None}

    monkeypatch.setattr(deepeval_adapter, "_judge_model", lambda _: object())
    monkeypatch.setattr(deepeval_adapter, "_measure", fake_metric)
    monkeypatch.setattr(deepeval_adapter, "_test_case", lambda *_, **__: object())
    result = evaluate_trace(
        TRACE,
        actual_output="An answer.",
        evidence_by_id={"chunk-1": "Evidence."},
        expected_output=None,
        judge=JudgeConfig(
            provider="anthropic", model="claude-test", api_key="fake", threshold=0.8
        ),
    )
    assert result["status"] == "evaluated"
    assert seen == [
        (name, 0.8)
        for name in ("faithfulness", "answer_relevancy", "contextual_relevancy")
    ]


def test_batch_output_has_scores_but_no_source_text_or_api_key(tmp_path):
    source = tmp_path / "inputs.jsonl"
    output = tmp_path / "eval_results.jsonl"
    source.write_text(
        json.dumps(
            {
                "trace": TRACE,
                "actual_output": "Answer from a public paper.",
                "evidence_by_id": {"chunk-1": "Full public paragraph."},
                "expected_output": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    count = run_batch(
        source,
        output,
        JudgeConfig(provider="anthropic", model="claude-test", api_key="fake-key"),
        measure=lambda *_: {"score": 0.9, "cost": None},
    )
    saved = output.read_text(encoding="utf-8")
    assert count == 1
    assert json.loads(saved)["metrics"]["faithfulness"]["score"] == 0.9
    assert "Full public paragraph" not in saved
    assert "fake-key" not in saved
