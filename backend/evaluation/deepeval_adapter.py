"""Opt-in DeepEval judge results, separate from deterministic retrieval scores."""

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

METRICS = ("faithfulness", "answer_relevancy", "contextual_relevancy")
REFERENCE_METRICS = ("contextual_precision", "contextual_recall")


@dataclass(frozen=True)
class JudgeConfig:
    provider: Literal["anthropic", "openai"]
    model: str
    api_key: str
    temperature: float = 0
    threshold: float = 0.5
    allow_private_context: bool = False

    def __post_init__(self) -> None:
        if self.provider not in {"anthropic", "openai"}:
            raise ValueError("Unsupported judge provider")
        if not self.model.strip():
            raise ValueError("Judge model is required")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Judge threshold must be between 0 and 1")


def _judge_model(config: JudgeConfig):
    from deepeval.models import AnthropicModel, OpenAIModel

    model_class = AnthropicModel if config.provider == "anthropic" else OpenAIModel
    return model_class(
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
    )


def _measure(name: str, case, model, *, threshold: float) -> dict:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    classes = {
        "faithfulness": FaithfulnessMetric,
        "answer_relevancy": AnswerRelevancyMetric,
        "contextual_relevancy": ContextualRelevancyMetric,
        "contextual_precision": ContextualPrecisionMetric,
        "contextual_recall": ContextualRecallMetric,
    }
    metric = classes[name](model=model, threshold=threshold, async_mode=False)
    metric.measure(case)
    return {
        "score": metric.score,
        "cost": getattr(metric, "evaluation_cost", None),
    }


def _test_case(
    question: str,
    actual_output: str,
    context: list[str],
    expected_output: str | None,
    *,
    use_deepeval: bool,
):
    fields = {
        "input": question,
        "actual_output": actual_output,
        "retrieval_context": context,
        "expected_output": expected_output,
    }
    if use_deepeval:
        from deepeval.test_case import LLMTestCase

        return LLMTestCase(**fields)
    return SimpleNamespace(**fields)


def evaluate_trace(
    trace: dict,
    *,
    actual_output: str,
    evidence_by_id: dict[str, str],
    expected_output: str | None,
    judge: JudgeConfig,
    measure: Callable[[str, object, object], dict] | None = None,
    question: str | None = None,
) -> dict:
    """Evaluate an in-memory answer; persisted traces contain only evidence IDs.

    A failed judge never replaces deterministic retrieval metrics with a score.
    Exception messages and source text are excluded from the returned result.
    """
    names = METRICS + (REFERENCE_METRICS if expected_output else ())
    try:
        deepeval_version = version("deepeval")
    except PackageNotFoundError:
        deepeval_version = None
    result = {
        "run_id": trace["run_id"],
        "evaluated_at": datetime.now(UTC).isoformat(),
        "corpus_version": trace.get("corpus_version"),
        "deepeval_version": deepeval_version,
        "judge": {
            "provider": judge.provider,
            "model": judge.model,
            "temperature": judge.temperature,
            "threshold": judge.threshold,
            "prompt_template": "deepeval_default",
        },
        "status": "not_evaluated",
        "error": None,
        "api_call_count": None,
        "token_usage": None,
        "total_cost": None,
        "latency_ms": None,
        "metrics": {
            name: {"status": "not_evaluated", "score": None, "cost": None}
            for name in names
        },
    }
    if not judge.api_key:
        result["error"] = "missing_key"
        return result
    if not trace.get("corpus_public") and not judge.allow_private_context:
        result["error"] = "private_corpus_not_allowed"
        return result
    if not trace.get("corpus_public") and not question:
        result["error"] = "missing_private_query"
        return result
    try:
        context = [evidence_by_id[item] for item in trace["prompt_evidence_ids"]]
    except (KeyError, TypeError):
        result["error"] = "missing_evidence"
        return result
    applicable_names = names if context else ("answer_relevancy",)
    if not context:
        for name in names:
            if name not in applicable_names:
                result["metrics"][name]["status"] = "not_applicable"
    try:
        started = time.perf_counter()
        real_measure = measure or partial(_measure, threshold=judge.threshold)
        model = None if measure else _judge_model(judge)
        case = _test_case(
            question or trace["original_query"],
            actual_output,
            context,
            expected_output,
            use_deepeval=measure is None,
        )
        for name in applicable_names:
            try:
                reading = real_measure(name, case, model)
                score = reading["score"]
                if score is None or not 0 <= float(score) <= 1:
                    raise ValueError("invalid judge score")
                result["metrics"][name] = {
                    "status": "evaluated",
                    "score": float(score),
                    "cost": reading.get("cost"),
                }
            except Exception as exc:
                result["error"] = type(exc).__name__
        if all(
            result["metrics"][name]["status"] == "evaluated"
            for name in applicable_names
        ) and len(applicable_names) == len(names):
            result["status"] = "evaluated"
        elif any(item["status"] == "evaluated" for item in result["metrics"].values()):
            result["status"] = "partially_evaluated"
    except Exception as exc:
        result["error"] = type(exc).__name__
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    costs = [item["cost"] for item in result["metrics"].values()]
    if costs and all(cost is not None for cost in costs):
        result["total_cost"] = sum(costs)
    return result


def run_batch(
    source: Path,
    output: Path,
    judge: JudgeConfig,
    *,
    measure: Callable[[str, object, object], dict] | None = None,
) -> int:
    """Evaluate local JSONL inputs and replace an old score file atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    try:
        with (
            source.open(encoding="utf-8") as inputs,
            temporary.open("w", encoding="utf-8") as results,
        ):
            for line in inputs:
                if not line.strip():
                    continue
                record = json.loads(line)
                result = evaluate_trace(
                    record["trace"],
                    actual_output=record["actual_output"],
                    evidence_by_id=record["evidence_by_id"],
                    expected_output=record.get("expected_output"),
                    judge=judge,
                    measure=measure,
                    question=record.get("question"),
                )
                results.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eval_results.jsonl"))
    parser.add_argument("--provider", choices=("anthropic", "openai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-private-context", action="store_true")
    args = parser.parse_args()
    from app.config import settings

    key = (
        settings.anthropic_api_key
        if args.provider == "anthropic"
        else settings.openai_api_key
    )
    config = JudgeConfig(
        provider=args.provider,
        model=args.model,
        api_key=key,
        threshold=args.threshold,
        allow_private_context=args.allow_private_context,
    )
    print(
        f"Wrote {run_batch(args.input, args.output, config)} results to {args.output}"
    )


if __name__ == "__main__":
    main()
