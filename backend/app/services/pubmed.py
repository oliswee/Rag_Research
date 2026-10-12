"""PubMed E-utilities search and source-grounded record parsing."""

import re
from collections.abc import Awaitable, Callable
from xml.etree import ElementTree as ET


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def parse_pubmed_xml(payload: bytes) -> list[dict]:
    """Parse one bounded EFetch response; PubMed records do not imply PDF access."""
    root = ET.fromstring(payload)
    records = []
    for item in root.findall("./PubmedArticle"):
        pmid = item.findtext("./MedlineCitation/PMID")
        if not pmid or not pmid.isdigit():
            continue
        article = item.find("./MedlineCitation/Article")
        if article is None:
            continue
        identifiers = {
            element.get("IdType"): _text(element)
            for element in item.findall("./PubmedData/ArticleIdList/ArticleId")
        }
        abstract_nodes = article.findall("./Abstract/AbstractText")
        abstract = (
            " ".join(
                f"{node.get('Label')}: {_text(node)}"
                if node.get("Label")
                else _text(node)
                for node in abstract_nodes
                if _text(node)
            )
            or None
        )
        authors = []
        for author in article.findall("./AuthorList/Author"):
            name = " ".join(
                part
                for part in (author.findtext("ForeName"), author.findtext("LastName"))
                if part
            ) or author.findtext("CollectiveName")
            if name:
                authors.append(name)
        publication_date = (
            article.findtext("./Journal/JournalIssue/PubDate/Year")
            or article.findtext("./ArticleDate/Year")
            or article.findtext("./Journal/JournalIssue/PubDate/MedlineDate")
            or ""
        )
        year_match = re.search(r"\b(1[89]\d{2}|20\d{2}|2100)\b", publication_date)
        tags = [
            _text(node)
            for node in item.findall(
                "./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName"
            )
            + item.findall("./MedlineCitation/KeywordList/Keyword")
        ]
        doi = identifiers.get("doi") or None
        pmcid = identifiers.get("pmc") or ""
        records.append(
            {
                "provider": "pubmed",
                "provider_id": pmid,
                "title": _text(article.find("./ArticleTitle")) or "Untitled",
                "authors": authors,
                "year": int(year_match.group()) if year_match else None,
                "abstract": abstract,
                "source_url": (
                    f"https://doi.org/{doi}"
                    if doi
                    else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ),
                "pdf_url": None,
                "open_access": bool(re.fullmatch(r"PMC\d+", pmcid)),
                "doi": doi.lower() if doi else None,
                "arxiv_id": None,
                "citation_count": None,
                "tags": list(dict.fromkeys(tag for tag in tags if tag))[:5],
            }
        )
    return records


async def search_pubmed(
    fetch_json: Callable[[str, dict], Awaitable[dict]],
    fetch_xml: Callable[[str, dict], Awaitable[bytes]],
    query: str,
    limit: int,
    offset: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access: bool = False,
    api_key: str = "",
    email: str = "",
) -> dict:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    common = {"db": "pubmed", "tool": "FolioTrace"}
    if api_key:
        common["api_key"] = api_key
    if email:
        common["email"] = email
    params = {
        **common,
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "retstart": offset,
        "sort": "relevance",
    }
    if year_from or year_to:
        params.update(
            datetype="pdat",
            mindate=str(year_from or 1800),
            maxdate=str(year_to or 2100),
        )
    search = await fetch_json(base + "esearch.fcgi", params)
    result = search.get("esearchresult") or {}
    ids = [value for value in result.get("idlist") or [] if str(value).isdigit()]
    total = int(result.get("count") or 0)
    if not ids:
        return {"results": [], "total": total, "fetched_count": 0}
    payload = await fetch_xml(
        base + "efetch.fcgi",
        {**common, "id": ",".join(ids), "retmode": "xml"},
    )
    by_id = {record["provider_id"]: record for record in parse_pubmed_xml(payload)}
    records = [
        record if not open_access or (record and record["open_access"]) else None
        for record in (by_id.get(str(pmid)) for pmid in ids)
    ]
    return {"results": records, "total": total, "fetched_count": len(ids)}
