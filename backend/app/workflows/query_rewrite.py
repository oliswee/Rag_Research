"""Opt-in, bounded rewriting of questions into standalone retrieval queries."""

import json
import logging
import re
from typing import Literal

from app.schemas.session import ConversationTurn
from app.services.retrieval_trace import RetrievalTrace

logger = logging.getLogger(__name__)

RewriteStatus = Literal["disabled", "applied", "unavailable", "failed"]


async def rewrite_for_retrieval(
    question: str,
    history: list[ConversationTurn],
    enabled: bool,
    trace: RetrievalTrace | None = None,
) -> tuple[str, RewriteStatus]:
    """Return a search query and its status, preserving context on fallback."""
    previous = next(
        (turn.content for turn in reversed(history) if turn.role == "user"), ""
    )
    fallback = f"{previous} {question}" if previous else question
    if not enabled:
        return fallback, "disabled"
    from app.services.llm_client import llm_available

    if not llm_available():
        return fallback, "unavailable"

    from app.services.llm_client import LLMClient

    recent_turns = [
        {"role": turn.role, "content": turn.content[:1000]} for turn in history[-6:]
    ]
    try:
        client = LLMClient()
        client.trace = trace
        result = await client.complete_json(
            system_prompt=(
                "Rewrite the current question as one standalone search query for "
                "academic source excerpts. Resolve references using the conversation "
                "history. Preserve names, exact terms and constraints. Add up to "
                "three closely related search keywords or short synonyms; do not "
                "add facts or new named entities. Do not answer the question. Return "
                'only JSON: {"query":"...","keywords":["..."]}.'
            ),
            user_prompt=json.dumps(
                {"history": recent_turns, "question": question[:1500]},
                ensure_ascii=False,
            ),
            max_tokens=240,
        )
    except Exception as exc:
        logger.warning("Query rewrite unavailable: %s", type(exc).__name__)
        return fallback, "failed"

    rewritten = result.get("query") if isinstance(result, dict) else None
    if not isinstance(rewritten, str):
        return fallback, "failed"
    rewritten = " ".join(rewritten.split())
    if not rewritten or len(rewritten) > 500:
        return fallback, "failed"
    keywords = result.get("keywords")
    if isinstance(keywords, list):
        accepted = 0
        for value in keywords:
            if not isinstance(value, str):
                continue
            term = " ".join(value.split())
            if (
                not term
                or len(term) > 60
                or not re.fullmatch(r"[\w-]+(?:\s+[\w-]+){0,2}", term)
                or term.casefold() in rewritten.casefold()
                or len(rewritten) + len(term) + 1 > 500
            ):
                continue
            rewritten += " " + term
            accepted += 1
            if accepted == 3:
                break
    return rewritten, "applied"
