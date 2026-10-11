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
