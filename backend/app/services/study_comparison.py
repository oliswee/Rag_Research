"""Conservative comparability of independently supported study characteristics."""

from app.schemas.evidence import ComparisonAssessment


def _normalize(value):
    return " ".join((value or "").casefold().split())


def _direction(value):
    value = _normalize(value)
    for names, direction in (
        ({"increase", "increased", "higher", "positive"}, "increase"),
        ({"decrease", "decreased", "lower", "negative", "reduced"}, "decrease"),
        ({"no effect", "no change", "unchanged", "null"}, "no_effect"),
    ):
        if value in names:
            return direction
    return None


def compare_studies(left, right, *, paper_ids=("left", "right")):
    def result(status, *reasons):
        return ComparisonAssessment(
            paper_ids=paper_ids, status=status, reasons=list(reasons)
        )

    if left.doi and _normalize(left.doi) == _normalize(right.doi):
        return result(
            "not_comparable",
            "same_study_different_version"
            if left.version != right.version
            else "same_study_not_independent",
        )
    required = (
        "population",
        "study_design",
        "outcome",
        "exposure",
        "effect_measure",
        "adjustment",
        "direction",
    )
    missing = [
        name
        for name in required
        if any(
            getattr(study, name).support_status != "supported"
            or not getattr(study, name).value
            or not getattr(study, name).citation_ids
            for study in (left, right)
        )
    ]
    if missing:
        return result("insufficient_evidence", *(name + "_unknown" for name in missing))
    mismatched = [
        name
        for name in required[:-1]
        if _normalize(getattr(left, name).value)
        != _normalize(getattr(right, name).value)
    ]
    if mismatched:
        return result("not_comparable", *(name + "_mismatch" for name in mismatched))
    if not left.doi or not right.doi:
        return result("insufficient_evidence", "study_independence_unknown")
    a, b = _direction(left.direction.value), _direction(right.direction.value)
    if not a or not b:
        return result("insufficient_evidence", "direction_unknown")
    return result(
        "comparable_agreement" if a == b else "comparable_conflict",
        "matched_supported_characteristics",
    )
