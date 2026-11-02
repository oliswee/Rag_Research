import importlib.util

import pytest

from app.schemas.evidence import Characteristic, StudyCharacteristics


def study(*, doi="10.1234/a", **changes):
    fields = {
        k: Characteristic(
            value=v,
            raw_quote=v,
            citation_ids=["cite:fixture"],
            support_status="supported",
        )
        for k, v in {
            "population": "adults",
            "study_design": "observational",
            "outcome": "mortality",
            "exposure": "smoking",
            "effect_measure": "OR",
            "adjustment": "adjusted for age and sex",
            "direction": "increase",
        }.items()
    }
    fields.update(changes)
    return StudyCharacteristics(**fields, doi=doi, year=2025, version="v1")


def assess(left, right):
    assert importlib.util.find_spec("app.services.study_comparison"), (
        "Study comparability service is missing"
    )
    from app.services.study_comparison import compare_studies

    return compare_studies(left, right, paper_ids=("a", "b"))


def test_matching_independent_studies_can_agree_or_conflict():
    assert assess(study(), study(doi="10.1234/b")).status == "comparable_agreement"
    negative = Characteristic(
        value="decrease",
        raw_quote="decrease",
        citation_ids=["cite:b"],
        support_status="supported",
    )
    assert (
        assess(study(), study(doi="10.1234/b", direction=negative)).status
        == "comparable_conflict"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("effect_measure", "RR"),
        ("population", "children"),
        ("study_design", "randomized trial"),
        ("outcome", "hospitalization"),
    ],
)
def test_incompatible_fields_are_not_scientific_conflicts(field, value):
    other = Characteristic(
        value=value,
        raw_quote=value,
        citation_ids=["cite:b"],
        support_status="supported",
    )
    result = assess(study(), study(doi="10.1234/b", **{field: other}))
    assert result.status == "not_comparable" and field + "_mismatch" in result.reasons


def test_missing_or_unverified_adjustment_is_unknown_not_agreement():
    assert (
        assess(study(), study(doi="10.1234/b", adjustment=Characteristic())).status
        == "insufficient_evidence"
    )
    assert (
        assess(
            study(),
            study(doi="10.1234/b", adjustment=Characteristic(value="unadjusted")),
        ).status
        == "insufficient_evidence"
    )


def test_same_doi_or_preprint_versions_do_not_count_as_two_independent_studies():
    left, right = study(), study()
    assert assess(left, right).status == "not_comparable"
    right.version = "v2"
    result = assess(left, right)
    assert (
        result.status == "not_comparable"
        and "same_study_different_version" in result.reasons
    )
