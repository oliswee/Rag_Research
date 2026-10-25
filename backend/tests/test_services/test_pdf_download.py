import httpx
import pytest

from app.services.pdf_download import UnsafePDFURL, download_pdf, validate_public_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.pdf",
        "http://169.254.169.254/latest",
        "file:///etc/passwd",
        "http://localhost/x.pdf",
        "http://[::1]/x.pdf",
    ],
)
async def test_blocks_local_pdf_urls(url):
    with pytest.raises(UnsafePDFURL):
        await validate_public_url(url)


@pytest.mark.asyncio
async def test_redirect_cannot_reach_local_network():
    def redirect(_request):
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private.pdf"})

    transport = httpx.MockTransport(redirect)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UnsafePDFURL):
            await download_pdf("https://1.1.1.1/public.pdf", client)


@pytest.mark.asyncio
async def test_web_same_host_blocks_redirect_before_dns(monkeypatch):
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                302, headers={"location": "https://example.org/a.pdf"}
            )
        )
    ) as client:
        with pytest.raises(UnsafePDFURL, match="Cross-domain"):
            await download_pdf(
                "https://1.1.1.1/a.pdf",
                client,
                same_host=True,
                max_redirects=2,
                on_attempt=lambda: calls.append(1),
            )
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,headers",
    [
        (b"%PDF-123456", {}),
        (b"%PDF-1", {"content-length": "20"}),
        (b"<html>", {"content-type": "text/html"}),
    ],
)
async def test_web_download_enforces_explicit_bytes_and_type(body, headers):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=body, headers=headers)
        )
    ) as client:
        with pytest.raises(ValueError):
            await download_pdf("https://1.1.1.1/a.pdf", client, max_bytes=8)
