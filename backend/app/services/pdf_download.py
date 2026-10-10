"""Bounded public PDF retrieval for provider supplied open access links."""

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx

from app.config import settings


class UnsafePDFURL(ValueError):
    pass


async def validate_public_url(
    url: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise UnsafePDFURL("Invalid PDF URL")
    host = parsed.hostname
    if host.lower() in {"localhost", "localhost.localdomain"} or host.lower().endswith(
        ".local"
    ):
        raise UnsafePDFURL("Private PDF host")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            records = await loop.getaddrinfo(
                host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise UnsafePDFURL("PDF host cannot be resolved") from exc
        addresses = [ipaddress.ip_address(record[4][0]) for record in records]
    if not addresses or any(not ip.is_global for ip in addresses):
        raise UnsafePDFURL("Private PDF address")
    return next((ip for ip in addresses if ip.version == 4), addresses[0])


async def download_pdf(url: str, client: httpx.AsyncClient | None = None) -> bytes:
    """Check each redirect and stream at most the configured upload limit."""
    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=25, follow_redirects=False, trust_env=False
    )
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    try:
        for _ in range(6):
            resolved = await validate_public_url(url)
            parsed = urlsplit(url)
            host = parsed.hostname
            pinned_host = f"[{resolved}]" if resolved.version == 6 else str(resolved)
            netloc = pinned_host + (f":{parsed.port}" if parsed.port else "")
            pinned_url = parsed._replace(netloc=netloc).geturl()
            async with client.stream(
                "GET",
                pinned_url,
                follow_redirects=False,
                headers={"Host": parsed.netloc},
                extensions={"sni_hostname": host},
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise UnsafePDFURL("Redirect without location")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if (
                    content_type
                    and "pdf" not in content_type
                    and "octet-stream" not in content_type
                ):
                    raise ValueError("Source did not return a PDF")
                if int(response.headers.get("content-length", "0")) > max_bytes:
                    raise ValueError("PDF exceeds configured size limit")
                payload = bytearray()
                async for part in response.aiter_bytes():
                    payload.extend(part)
                    if len(payload) > max_bytes:
                        raise ValueError("PDF exceeds configured size limit")
                if not payload.startswith(b"%PDF-"):
                    raise ValueError("Source did not return a PDF")
                return bytes(payload)
        raise UnsafePDFURL("Too many PDF redirects")
    finally:
        if owns_client:
            await client.aclose()
