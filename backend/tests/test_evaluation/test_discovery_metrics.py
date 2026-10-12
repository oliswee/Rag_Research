"""Offline discovery metrics require explicit relevance judgments."""

from evaluation.discovery_metrics import evaluate_cases


def test_evaluate_cases_deduplicates_dois_and_counts_known_full_text():
    report = evaluate_cases(
        [
            {
                "query": "clinical cancer",
                "relevant_dois": ["10.1000/ONE", "10.1000/two"],
                "result": {
                    "candidates": [
                        {
                            "provider": "pubmed",
                            "provider_id": "1",
                            "doi": "10.1000/one",
                            "content_scope": "full_text",
                        },
                        {
                            "provider": "openalex",
                            "provider_id": "W1",
                            "doi": "https://doi.org/10.1000/ONE",
                            "content_scope": "abstract",
                        },
                        {
                            "provider": "openalex",
                            "provider_id": "W2",
                            "doi": "10.1000/other",
                        },
                    ],
                    "searched_sources": ["openalex", "pubmed"],
                    "failed_sources": ["pubmed"],
                    "search_latency_ms": 1500,
                },
            }
        ]
    )
    case = report["cases"][0]
    assert case["unique_papers_at_50"] == 2
    assert case["unique_relevant_papers_at_50"] == 1
    assert case["recall_at_50"] == 0.5
    assert case["full_text_availability"] == 1.0
    assert case["full_text_samples"] == 1
    assert case["api_failure_rate"] == 0.5
    assert report["summary"]["mean_recall_at_50"] == 0.5


def test_evaluate_cases_marks_unmeasured_metrics_as_null():
    report = evaluate_cases(
        [
            {
                "query": "research",
                "relevant_dois": [],
                "result": {
                    "candidates": [],
                    "searched_sources": [],
                    "failed_sources": [],
                    "search_latency_ms": 2,
                },
            }
        ]
    )
    assert report["cases"][0]["recall_at_50"] is None
    assert report["cases"][0]["full_text_availability"] is None
    assert report["cases"][0]["api_failure_rate"] is None
