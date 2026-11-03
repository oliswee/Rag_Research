import pytest

from evaluation.final_statistics import evidence_coverage, failure_stage, paired_summary


def test_pairs_align_by_id_and_skip_unknown_values():
    first = [
        {"case_id": "a", "recall": 0},
        {"case_id": "b", "recall": 1},
        {"case_id": "c", "recall": None},
    ]
    second = [
        {"case_id": "c", "recall": None},
        {"case_id": "b", "recall": 0},
        {"case_id": "a", "recall": 1},
    ]
    result = paired_summary(first, second, "recall", draws=100)
    assert result["n"] == 2 and result["mean_delta"] == 0
    assert result["win_tie_loss"] == {"win": 1, "tie": 0, "loss": 1}
    assert result == paired_summary(first, second, "recall", draws=100)


@pytest.mark.parametrize(
    "candidate",
    [
        [{"case_id": "other", "score": 1}],
        [{"case_id": "one", "score": 1}, {"case_id": "one", "score": 1}],
    ],
)
def test_missing_or_duplicate_pairs_fail(candidate):
    with pytest.raises(ValueError):
        paired_summary([{"case_id": "one", "score": 0}], candidate, "score")


def test_evidence_coverage_requires_correct_paper_and_union_without_double_count():
    row = {"paper_id": "p", "document_version": "v", "source_offset": [10, 20]}
    spans = [
        {"paper_id": "p", "document_version": "v", "raw_start": 10, "raw_end": 16},
        {"paper_id": "p", "document_version": "v", "raw_start": 14, "raw_end": 18},
    ]
    assert evidence_coverage(row, spans) == 0.8
    assert evidence_coverage(row, [{**spans[0], "paper_id": "foreign"}]) == 0
    assert evidence_coverage({**row, "source_offset": None}, spans) is None


def test_funnel_never_calls_unsent_or_unmeasured_answer_correct():
    assert failure_stage({"paper_present": False}) == "1_corpus_missing"
    assert (
        failure_stage({"paper_present": True, "parse_valid": False})
        == "2_parse_or_chunk"
    )
    assert (
        failure_stage(
            {
                "paper_present": True,
                "parse_valid": True,
                "candidate_found": True,
                "ranked_found": True,
                "packed_found": False,
            }
        )
        == "5_context_packing"
    )
    assert (
        failure_stage(
            {
                "paper_present": True,
                "parse_valid": True,
                "candidate_found": True,
                "ranked_found": True,
                "packed_found": True,
            }
        )
        == "generation_not_evaluated"
    )
