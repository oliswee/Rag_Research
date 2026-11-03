"""Request-scoped offline policy, propagated through asyncio worker threads."""

from contextvars import ContextVar
from ipaddress import ip_address
from urllib.parse import urlsplit

local_only_request: ContextVar[bool] = ContextVar("local_only_request", default=False)


def loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def loopback_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https"}
            and not parsed.username
            and not parsed.password
            and loopback_host(parsed.hostname or "")
        )
    except ValueError:
        return False
