"""Read-only OpenAlex directory of authors with a latest AU affiliation."""

import re

import httpx


def _id(url: str | None) -> str:
    return (url or "").rsplit("/", 1)[-1]


def _australian_institutions(author: dict) -> list[str]:
    return [
        item["display_name"]
        for item in author.get("last_known_institutions") or []
        if item.get("country_code") == "AU" and item.get("display_name")
    ]


def _author(author: dict) -> dict:
    topics = author.get("topics") or []
    return {
        "id": _id(author.get("id")),
        "name": author.get("display_name") or "Name unavailable",
        "institutions": _australian_institutions(author),
        "topics": [
            topic["display_name"] for topic in topics[:5] if topic.get("display_name")
        ],
        "fields": list(
            dict.fromkeys(
                topic["field"]["display_name"]
                for topic in topics
                if (topic.get("field") or {}).get("display_name")
            )
        )[:5],
        "works_count": author.get("works_count") or 0,
        "cited_by_count": author.get("cited_by_count") or 0,
        "orcid": author.get("orcid"),
        "source_url": author.get("id"),
    }


class ResearcherService:
    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(
            timeout=15,
            follow_redirects=False,
            headers={"User-Agent": "FolioTrace/0.1 (research directory)"},
        )
        self.owns_client = client is None

    async def close(self) -> None:
        if self.owns_client:
            await self.client.aclose()

    async def _get(self, url: str, params: dict | None = None) -> dict:
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def search(
        self,
        name: str = "",
        institution: str = "",
        topic: str = "",
        discipline: str = "",
        limit: int = 20,
    ) -> dict:
        if not any(
            (name.strip(), institution.strip(), topic.strip(), discipline.strip())
        ):
            return {"researchers": [], "total": 0, "resolved_filters": {}}
        filters = ["last_known_institutions.country_code:AU"]
        resolved = {}
        if institution.strip():
            institutions = await self._get(
                "https://api.openalex.org/institutions",
                {
                    "search": institution.strip(),
                    "filter": "country_code:AU",
                    "per_page": 1,
                    "select": "id,display_name",
                },
            )
            found = (institutions.get("results") or [])[:1]
            if not found:
                return {"researchers": [], "total": 0, "resolved_filters": {}}
            filters.append(f"last_known_institutions.id:{_id(found[0]['id'])}")
            resolved["institution"] = found[0]["display_name"]
        if topic.strip():
            topics = await self._get(
                "https://api.openalex.org/topics",
                {
                    "search": topic.strip(),
                    "per_page": 1,
                    "select": "id,display_name",
                },
            )
            found = (topics.get("results") or [])[:1]
            if not found:
                return {"researchers": [], "total": 0, "resolved_filters": resolved}
            filters.append(f"topics.id:{_id(found[0]['id'])}")
            resolved["topic"] = found[0]["display_name"]
        if discipline.strip():
            fields = await self._get(
                "https://api.openalex.org/fields",
                {
                    "search": discipline.strip(),
                    "per_page": 1,
                    "select": "id,display_name",
                },
            )
            found = (fields.get("results") or [])[:1]
            if not found:
                return {"researchers": [], "total": 0, "resolved_filters": resolved}
            filters.append(f"topics.field.id:{_id(found[0]['id'])}")
            resolved["discipline"] = found[0]["display_name"]
        params = {
            "filter": ",".join(filters),
            "per_page": max(1, min(limit, 20)),
            "select": (
                "id,display_name,last_known_institutions,topics,orcid,"
                "works_count,cited_by_count"
            ),
        }
        if name.strip():
            params["search"] = name.strip()
        else:
            params["sort"] = "works_count:desc"
        response = await self._get("https://api.openalex.org/authors", params)
        return {
            "researchers": [
                _author(author)
                for author in response.get("results", [])
                if _australian_institutions(author)
            ],
            "total": (response.get("meta") or {}).get("count", 0),
            "resolved_filters": resolved,
        }

    async def profile(self, author_id: str) -> dict | None:
        if not re.fullmatch(r"A\d{1,20}", author_id):
            raise ValueError("Invalid author ID")
        author = await self._get(f"https://api.openalex.org/authors/{author_id}")
        if not _australian_institutions(author):
            return None
        works = await self._get(
            "https://api.openalex.org/works",
            {
                "filter": f"author.id:{author_id}",
                "sort": "publication_date:desc",
                "per_page": 8,
                "select": (
                    "id,display_name,publication_year,doi,cited_by_count,"
                    "primary_topic,authorships"
                ),
            },
        )
        publications = []
        coauthors = {}
        for work in works.get("results", []):
            publications.append(
                {
                    "id": _id(work.get("id")),
                    "title": work.get("display_name") or "Untitled",
                    "year": work.get("publication_year"),
                    "citation_count": work.get("cited_by_count") or 0,
                    "topic": (work.get("primary_topic") or {}).get("display_name"),
                    "source_url": work.get("doi") or work.get("id"),
                }
            )
            for authorship in work.get("authorships") or []:
                item = authorship.get("author") or {}
                other_id = _id(item.get("id"))
                if other_id and other_id != author_id and item.get("display_name"):
                    coauthors[other_id] = {"id": other_id, "name": item["display_name"]}
        return {
            **_author(author),
            "publications": publications,
            "coauthors": list(coauthors.values())[:20],
        }
