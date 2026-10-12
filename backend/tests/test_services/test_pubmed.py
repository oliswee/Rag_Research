"""PubMed discovery retains source labels and supports paged retrieval."""

from unittest.mock import AsyncMock

import pytest

from app.services.pubmed import parse_pubmed_xml, search_pubmed

PUBMED_XML = b"""<?xml version="1.0"?><PubmedArticleSet><PubmedArticle>
<MedlineCitation><PMID>12345</PMID><Article>
<ArticleTitle>Retrieval in medicine</ArticleTitle>
<Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
<Abstract><AbstractText Label="BACKGROUND">A question.</AbstractText>
<AbstractText Label="RESULTS">A grounded answer.</AbstractText></Abstract>
<AuthorList><Author><ForeName>Ada</ForeName><LastName>Lee</LastName></Author></AuthorList>
</Article><MeshHeadingList><MeshHeading>
<DescriptorName>Information Retrieval</DescriptorName>
</MeshHeading></MeshHeadingList></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1234/test</ArticleId>
<ArticleId IdType="pmc">PMC12345</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>"""


def test_parse_pubmed_abstract_authors_and_mesh():
    papers = parse_pubmed_xml(PUBMED_XML)
    assert len(papers) == 1
    paper = papers[0]
    assert paper["provider_id"] == "12345"
    assert paper["authors"] == ["Ada Lee"]
    assert paper["abstract"] == "BACKGROUND: A question. RESULTS: A grounded answer."
    assert paper["tags"] == ["Information Retrieval"]
    assert paper["doi"] == "10.1234/test"
    assert paper["source_url"] == "https://doi.org/10.1234/test"
    assert paper["open_access"] is True
    assert paper["pdf_url"] is None


@pytest.mark.asyncio
async def test_pubmed_search_uses_retstart_and_fetches_ids_in_one_batch():
    fetch_json = AsyncMock(
        return_value={"esearchresult": {"count": "42", "idlist": ["12345"]}}
    )
    fetch_xml = AsyncMock(return_value=PUBMED_XML)
    result = await search_pubmed(
        fetch_json, fetch_xml, "retrieval", limit=10, offset=20, year_from=2020
    )
    assert result["total"] == 42
    assert result["results"][0]["provider"] == "pubmed"
    assert fetch_json.call_args.args[1]["retstart"] == 20
    assert fetch_json.call_args.args[1]["retmax"] == 10
    assert fetch_xml.call_args.args[1]["id"] == "12345"
