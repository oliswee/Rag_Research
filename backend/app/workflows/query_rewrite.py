"""Opt-in, bounded rewriting of questions into standalone retrieval queries."""

import json
import logging
from typing import Literal

from app.config import settings
from app.schemas.session import ConversationTurn

logger = logging.getLogger(__name__)

RewriteStatus = Literal["disabled", "applied", "unavailable", "failed"]


async def rewrite_for_retrieval(
    question: str, history: list[ConversationTurn], enabled: bool
) -> tuple[str, RewriteStatus]:
    """Return a search query and its status, preserving context on fallback."""
    previous = next(
        (turn.content for turn in reversed(history) if turn.role == "user"), ""
    )
    fallback = f"{previous} {question}" if previous else question
    if not enabled:
        return fallback, "disabled"
    if not settings.anthropic_api_key:
        return fallback, "unavailable"

    from app.services.llm_client import LLMClient

    recent_turns = [
        {"role": turn.role, "content": turn.content[:1000]} for turn in history[-6:]
    ]
    try:
        result = await LLMClient().complete_json(
            system_prompt=(
                "Rewrite the current question as one standalone search query for "
                "academic source excerpts. Resolve references using the conversation "
                "history. Preserve names, exact terms and constraints. Do not answer "
                'or add facts. Return only JSON: {"query":"..."}.'
            ),
            user_prompt=json.dumps(
                {"history": recent_turns, "question": question[:1500]},
                ensure_ascii=False,
            ),
            max_tokens=180,
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
    return rewritten, "applied"
