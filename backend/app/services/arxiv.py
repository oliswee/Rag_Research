"""Normalize the official arXiv Atom API into discovery candidates."""

import re
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})")


def base_arxiv_id(value: str) -> str | None:
    """Strip a version suffix while retaining legacy category prefixes."""
    identifier = re.sub(r"v\d+$", "", value.strip(), flags=re.I)
    return identifier if ARXIV_ID.fullmatch(identifier) else None


def _entry_id(entry: ET.Element) -> str | None:
    url = entry.findtext(f"{ATOM}id") or ""
    parsed = urlsplit(url)
    if parsed.hostname not in {"arxiv.org", "www.arxiv.org"}:
        return None
    return base_arxiv_id(parsed.path.removeprefix("/abs/"))


def parse_arxiv_feed(payload: bytes) -> dict:
    root = ET.fromstring(payload)
    if root.tag != f"{ATOM}feed":
        raise ValueError("Invalid arXiv feed")
    entries = root.findall(f"{ATOM}entry")
    results = []
    for entry in entries:
        paper_id = _entry_id(entry)
        if not paper_id:
            results.append(None)
            continue
        abstract = " ".join((entry.findtext(f"{ATOM}summary") or "").split()) or None
        published = entry.findtext(f"{ATOM}published") or ""
        year = int(published[:4]) if re.match(r"^\d{4}-", published) else None
        doi = (entry.findtext(f"{ARXIV}doi") or "").strip().lower() or None
        pdf_url = None
        for link in entry.findall(f"{ATOM}link"):
            if link.get("type") == "application/pdf":
                url = link.get("href") or ""
                parsed = urlsplit(url)
                if parsed.scheme == "https" and parsed.hostname in {
                    "arxiv.org",
                    "www.arxiv.org",
                }:
                    pdf_url = url
                    break
        results.append(
            {
                "provider": "arxiv",
                "provider_id": paper_id,
                "title": " ".join(
                    (entry.findtext(f"{ATOM}title") or "Untitled").split()
                ),
                "authors": [
                    " ".join((author.findtext(f"{ATOM}name") or "").split())
                    for author in entry.findall(f"{ATOM}author")
                    if author.findtext(f"{ATOM}name")
                ],
                "year": year,
                "abstract": abstract,
                "source_url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": pdf_url,
                "open_access": bool(pdf_url),
                "doi": doi,
                "arxiv_id": paper_id,
                "citation_count": None,
                "tags": list(
                    dict.fromkeys(
                        category.get("term")
                        for category in entry.findall(f"{ATOM}category")
                        if category.get("term")
                    )
                )[:5],
            }
        )
    total_text = root.findtext(f"{OPENSEARCH}totalResults") or ""
    return {
        "results": results,
        "total": int(total_text) if total_text.isdigit() else len(entries),
        "fetched_count": len(entries),
    }
