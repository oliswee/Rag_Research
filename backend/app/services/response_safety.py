"""Detect configured credentials without printing them or changing original text."""

import json
import re

from app.config import settings

_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b|(?i:Bearer)\s+\S+")


def contains_credential(value):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if _TOKEN.search(text):
        return True
    for field in (
        "anthropic_api_key",
        "openai_api_key",
        "deepseek_api_key",
        "semantic_scholar_api_key",
        "ncbi_api_key",
        "langchain_api_key",
    ):
        secret = getattr(settings, field)
        if len(secret) >= 8 and secret in text:
            return True
    return False
