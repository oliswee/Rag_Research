"""Paired development statistics and evidence-based loss attribution."""

import math
import random


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_summary(baseline, candidate, metric, *, seed=61, draws=2000):
    def keyed(rows):
        result = {row["case_id"]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate paired case IDs")
        return result

    left, right = keyed(baseline), keyed(candidate)
    if left.keys() != right.keys():
        raise ValueError("Paired case sets differ")
    if draws < 1:
        raise ValueError("Bootstrap needs positive draws")
    deltas = [
        right[key][metric] - left[key][metric]
        for key in sorted(left)
        if left[key].get(metric) is not None and right[key].get(metric) is not None
    ]
    if any(not math.isfinite(value) for value in deltas):
        raise ValueError("Nonfinite paired measurement")
    rng = random.Random(seed)
    bootstrap = (
        [sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(draws)]
        if deltas
        else []
    )
    return {
        "metric": metric,
        "n": len(deltas),
        "paired_by": "case_id",
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "win_tie_loss": {
            "win": sum(d > 0 for d in deltas),
            "tie": sum(d == 0 for d in deltas),
            "loss": sum(d < 0 for d in deltas),
        },
        "bootstrap_95_percentile": [
            quantile(bootstrap, 0.025),
            quantile(bootstrap, 0.975),
        ],
        "bootstrap_draws": draws,
        "seed": seed,
        "interpretation": "reference_development_not_confirmatory_gold",
    }


def evidence_coverage(reference, spans):
    target = reference.get("source_offset")
    if not target:
        return None
    start, end = target
    intervals = sorted(
        (max(start, s["raw_start"]), min(end, s["raw_end"]))
        for s in spans
        if s.get("paper_id") == reference["paper_id"]
        and s.get("document_version") == reference["document_version"]
        and s.get("page") == reference.get("pdf_page_or_null")
        and s["raw_start"] < end
        and s["raw_end"] > start
    )
    total, cursor = 0, start
    for left, right in intervals:
        if right > cursor:
            total += right - max(left, cursor)
            cursor = right
    return total / (end - start)


def failure_stage(observed):
    for field, label in (
        ("paper_present", "1_corpus_missing"),
        ("parse_valid", "2_parse_or_chunk"),
        ("candidate_found", "3_candidate_recall"),
        ("ranked_found", "4_fusion_rerank"),
        ("packed_found", "5_context_packing"),
    ):
        if observed.get(field) is False:
            return label
        if observed.get(field) is not True:
            return "unknown_" + field
    if observed.get("web_failed") is True:
        return "7_web_fallback"
    if observed.get("generation_supported") is False:
        return "6_generation_or_verification"
    return (
        "no_observed_failure"
        if observed.get("generation_supported") is True
        else "generation_not_evaluated"
    )
