"""Australian researcher search is grounded in OpenAlex affiliations and works."""

from unittest.mock import AsyncMock

import pytest

from app.services.researchers import ResearcherService


@pytest.mark.asyncio
async def test_search_filters_au_authors_by_institution_and_topic():
    service = ResearcherService()
    service._get = AsyncMock(
        side_effect=[
            {
                "results": [
                    {
                        "id": "https://openalex.org/I1",
                        "display_name": "University of Sydney",
                    }
                ]
            },
            {
                "results": [
                    {
                        "id": "https://openalex.org/T1",
                        "display_name": "Machine Learning",
                    }
                ]
            },
            {
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/A1",
                        "display_name": "Ada Researcher",
                        "last_known_institutions": [
                            {
                                "id": "https://openalex.org/I1",
                                "display_name": "University of Sydney",
                                "country_code": "AU",
                            }
                        ],
                        "topics": [
                            {
                                "display_name": "Machine Learning",
                                "field": {"display_name": "Computer Science"},
                            }
                        ],
                        "orcid": "https://orcid.org/0000-0001-0000-0000",
                        "works_count": 12,
                        "cited_by_count": 40,
                    }
                ],
            },
        ]
    )
    try:
        result = await service.search(
            name="Ada", institution="Sydney", topic="machine learning"
        )
    finally:
        await service.close()
    assert result["researchers"][0]["name"] == "Ada Researcher"
    assert result["researchers"][0]["institutions"] == ["University of Sydney"]
    author_params = service._get.call_args_list[-1].args[1]
    assert "last_known_institutions.country_code:AU" in author_params["filter"]
    assert "last_known_institutions.id:I1" in author_params["filter"]
    assert "topics.id:T1" in author_params["filter"]


@pytest.mark.asyncio
async def test_profile_shows_publications_and_coauthors():
    service = ResearcherService()
    service._get = AsyncMock(
        side_effect=[
            {
                "id": "https://openalex.org/A1",
                "display_name": "Ada Researcher",
                "last_known_institutions": [
                    {"display_name": "University of Sydney", "country_code": "AU"}
                ],
                "topics": [],
                "works_count": 4,
                "cited_by_count": 10,
            },
            {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "A paper",
                        "publication_year": 2025,
                        "doi": "https://doi.org/10.1/abc",
                        "cited_by_count": 3,
                        "primary_topic": {"display_name": "AI"},
                        "authorships": [
                            {
                                "author": {
                                    "id": "https://openalex.org/A1",
                                    "display_name": "Ada Researcher",
                                }
                            },
                            {
                                "author": {
                                    "id": "https://openalex.org/A2",
                                    "display_name": "Bea Researcher",
                                }
                            },
                        ],
                    }
                ]
            },
        ]
    )
    try:
        result = await service.profile("A1")
    finally:
        await service.close()
    assert result["publications"][0]["title"] == "A paper"
    assert result["coauthors"] == [{"id": "A2", "name": "Bea Researcher"}]


@pytest.mark.asyncio
async def test_profile_rejects_non_australian_affiliation():
    service = ResearcherService()
    service._get = AsyncMock(
        return_value={
            "id": "https://openalex.org/A1",
            "display_name": "Other",
            "last_known_institutions": [
                {"display_name": "Outside", "country_code": "US"}
            ],
        }
    )
    try:
        result = await service.profile("A1")
    finally:
        await service.close()
    assert result is None
