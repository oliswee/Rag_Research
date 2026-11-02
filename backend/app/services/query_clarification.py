"""Necessary scope clarification and bounded assumptions, without model calls."""

import re
from collections import Counter

from pydantic import BaseModel, Field


class ScopeClarification(BaseModel):
    code: str | None = None
    question: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    discovery_url: str = "/discover"


def clarify_scope(question, papers, history, policy):
    result = ScopeClarification()
    if policy.intent == "latest":
        result.assumptions.append(
            "Local-only evidence cannot establish latest external coverage."
            if policy.web_prohibited
            else (
                "Latest means permitted providers' results at retrieval time; "
                "coverage is not exhaustive."
            )
        )
    if policy.paper_ids is not None:
        chosen = [p for p in papers if p.id in policy.paper_ids]
        if chosen and all(p.content_scope == "abstract" for p in chosen):
            result.assumptions.append(
                "Selected papers have abstract-only local evidence; "
                "full-text experiment details remain unknown."
            )
        return result
    titles = Counter(p.title.strip().casefold() for p in papers if p.title.strip())
    if any(
        count > 1 and title in question.casefold() for title, count in titles.items()
    ):
        return result.model_copy(
            update={
                "code": "ambiguous_paper_title",
                "question": "Which identity should I use for the same-title papers?",
            }
        )
    pronoun = re.search(
        r"\b(?:it|its|that paper|this paper|this study)\b|它|这篇|那篇|该文",
        question,
        re.I,
    )
    references = []
    for turn in reversed(history):
        if getattr(turn, "role", None) == "assistant":
            references = getattr(turn, "sources", None) or []
            break
    if pronoun and (not references or len(references) > 1):
        return result.model_copy(
            update={
                "code": "unbound_paper_reference",
                "question": "Which paper or study should I use for this question?",
            }
        )
    if pronoun:
        result.assumptions.append(
            "The pronoun refers to the current session's single referenced source."
        )
    return result
