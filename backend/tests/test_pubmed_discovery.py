"""Pure PubMed normalization tests; no live API calls."""
from xml.etree import ElementTree as ET

from app.services.discovery import pubmed_candidate, normalize_doi


def test_pubmed_candidate_normalizes_metadata():
    xml = """<PubmedArticle><MedlineCitation>
        <PMID>123456</PMID><Article>
        <ArticleTitle>Vitamin D and <i>kidney</i> health</ArticleTitle>
        <Abstract><AbstractText Label="BACKGROUND">First part.</AbstractText>
        <AbstractText Label="RESULTS">Second part.</AbstractText></Abstract>
        <AuthorList><Author><ForeName>Jane</ForeName><LastName>Doe</LastName></Author></AuthorList>
        <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
        <PublicationTypeList><PublicationType>Review</PublicationType></PublicationTypeList>
        </Article></MedlineCitation><PubmedData><ArticleIdList>
        <ArticleId IdType="doi">10.1000/EXAMPLE</ArticleId>
        <ArticleId IdType="pmc">PMC12345</ArticleId>
        </ArticleIdList></PubmedData></PubmedArticle>"""
    paper = pubmed_candidate(ET.fromstring(xml))
    assert paper["provider"] == "pubmed"
    assert paper["provider_id"] == "123456"
    assert paper["title"] == "Vitamin D and kidney health"
    assert paper["year"] == 2024
    assert paper["authors"] == ["Jane Doe"]
    assert "First part." in paper["abstract"]
    assert paper["doi"] == "10.1000/example"
    assert paper["open_access"] is True
    assert paper["tags"] == ["Review"]


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1/TEST") == "10.1/test"
