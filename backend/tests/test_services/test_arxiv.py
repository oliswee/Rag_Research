"""arXiv Atom records normalize to importable paper candidates."""

from app.services.arxiv import parse_arxiv_feed

FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'
      xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'
      xmlns:arxiv='http://arxiv.org/schemas/atom'>
  <opensearch:totalResults>3</opensearch:totalResults>
  <entry>
    <id>https://arxiv.org/abs/2401.12345v2</id>
    <title> Retrieval  methods </title>
    <summary>A study of retrieval. It compares two approaches.</summary>
    <published>2024-01-12T10:00:00Z</published>
    <author><name>A. Researcher</name></author>
    <category term='cs.IR'/><category term='cs.AI'/>
    <arxiv:doi>10.1234/example</arxiv:doi>
    <link rel='alternate' href='https://arxiv.org/abs/2401.12345v2'/>
    <link title='pdf' type='application/pdf' href='https://arxiv.org/pdf/2401.12345v2'/>
  </entry>
</feed>"""


def test_parse_arxiv_feed_uses_base_id_and_source_pdf():
    feed = parse_arxiv_feed(FEED)
    assert feed["total"] == 3
    assert feed["fetched_count"] == 1
    paper = feed["results"][0]
    assert paper["provider"] == "arxiv"
    assert paper["provider_id"] == "2401.12345"
    assert paper["arxiv_id"] == "2401.12345"
    assert paper["doi"] == "10.1234/example"
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2401.12345v2"
    assert paper["open_access"] is True
    assert paper["tags"] == ["cs.IR", "cs.AI"]
    assert paper["abstract"] == "A study of retrieval. It compares two approaches."


def test_parse_arxiv_feed_handles_legacy_ids_without_doi():
    payload = FEED.replace(b"2401.12345v2", b"cs/9901001v1").replace(
        b"<arxiv:doi>10.1234/example</arxiv:doi>", b""
    )
    paper = parse_arxiv_feed(payload)["results"][0]
    assert paper["provider_id"] == "cs/9901001"
    assert paper["doi"] is None
