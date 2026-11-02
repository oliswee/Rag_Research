"""Deterministic scientific/citation checks before semantic model verification.

Rules are conservative guardrails, not a substitute for scientific interpretation.
"""

import re
from decimal import Decimal, InvalidOperation

from app.schemas.evidence import AtomicClaim, Characteristic, StudyCharacteristics

NUMBER = r"[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"
_UNITS = re.compile(
    rf"({NUMBER})\s*(%|percent\b|milliseconds?\b|ms\b|seconds?\b|s\b|mg\b"
    r"|kg\b|ml\b|years?\b|months?\b|patients?\b|participants?\b)",
    re.I,
)
_ROLES = re.compile(
    r"\b(sample(?: size)?|n\s*=|accuracy|latency|OR|RR|HR|odds ratio|risk ratio"
    r"|relative risk|hazard ratio)\s*(?:was|is|of|=|:)?\s*$",
    re.I,
)
CHARACTERISTICS = (
    "population",
    "study_design",
    "outcome",
    "exposure",
    "effect_measure",
    "adjustment",
    "direction",
)


def _value(value):
    try:
        return str(Decimal(value.replace(",", "").replace("−", "-")).normalize())
    except InvalidOperation:
        return value


def _numbers(text):
    return {_value(m.group()) for m in re.finditer(NUMBER, text)}


def _units(text):
    aliases = {
        "percent": "%",
        "milliseconds": "ms",
        "millisecond": "ms",
        "seconds": "s",
        "second": "s",
        "patients": "patient",
        "participants": "participant",
        "years": "year",
        "months": "month",
    }
    return {
        (_value(m[1]), aliases.get(m[2].lower(), m[2].lower()))
        for m in _UNITS.finditer(text)
    }


def _roles(text):
    result = set()
    for number in re.finditer(NUMBER, text):
        role = _ROLES.search(text[max(0, number.start() - 32) : number.start()])
        if role:
            name = role[1].lower().strip()
            if name.startswith("sample") or name.startswith("n"):
                name = "sample"
            result.add((name, _value(number.group())))
    return result


def _measures(text):
    result = set(re.findall(r"\b(?:OR|RR|HR|SMD)\b", text))
    for pattern, label in (
        ("odds ratio", "OR"),
        ("relative risk|risk ratio", "RR"),
        ("hazard ratio", "HR"),
    ):
        if re.search(pattern, text, re.I):
            result.add(label)
    return result


def scientific_reasons(text, quotes, *, abstract_only=False):
    reasons = []
    if (
        len(re.split(r"(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff])|[;；]", text.strip()))
        > 1
    ):
        reasons.append("non_atomic_claim")
    if _numbers(text) - _numbers(quotes):
        reasons.append("numeric_support_missing")
    if _units(text) - _units(quotes):
        reasons.append("unit_support_missing")
    if _roles(text) - _roles(quotes):
        reasons.append("numeric_role_mismatch")
    if _measures(text) - _measures(quotes):
        reasons.append("effect_measure_mismatch")
    causal = r"\bcaus(?:e|es|ed|al)\b|\bleads? to\b|\bresults? in\b|导致|造成|因果"
    if re.search(causal, text, re.I) and not re.search(causal, quotes, re.I):
        reasons.append("causal_escalation")
    negative = r"\b(?:not|no|never)\b|没有|不改善|无改善"
    up = r"\b(?:increas\w*|higher|improv\w*)\b|提高|上升|改善"
    down = r"\b(?:decreas\w*|lower|reduc\w*)\b|降低|下降"
    quoted_negative, claimed_negative = (
        bool(re.search(negative, quotes, re.I)),
        bool(re.search(negative, text, re.I)),
    )
    if (
        (
            re.search(up, text, re.I)
            and re.search(down, quotes, re.I)
            and not re.search(up, quotes, re.I)
        )
        or (
            re.search(down, text, re.I)
            and re.search(up, quotes, re.I)
            and not re.search(down, quotes, re.I)
        )
        or quoted_negative != claimed_negative
        and re.search(r"improv\w*|改善", text, re.I)
    ):
        reasons.append("direction_mismatch")
    if abstract_only and re.search(
        r"full[ -]?text|全文|全篇|read the (?:entire|complete) paper", text, re.I
    ):
        reasons.append("abstract_access_upgrade")
    return reasons


def check_proposal(proposal, registry, *, paper_id=None):
    proposal = proposal if isinstance(proposal, dict) else {}
    result = AtomicClaim(
        claim_id=str(proposal.get("claim_id", "claim"))[:100],
        paper_id=paper_id or proposal.get("paper_id"),
        dimension=proposal.get("dimension"),
    )
    text, ids, quotes = (
        proposal.get("text"),
        proposal.get("source_ids"),
        proposal.get("supporting_quotes"),
    )
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > 1500
        or not isinstance(ids, list)
        or not ids
        or len(ids) > 5
        or not isinstance(quotes, dict)
    ):
        return result.model_copy(
            update={"support_status": "unsupported", "uncertainty": ["malformed_claim"]}
        )
    citations = []
    try:
        for source_id in dict.fromkeys(ids):
            if not isinstance(source_id, str):
                raise ValueError("Invalid source ID")
            citation = registry.cite(source_id, quotes.get(source_id))
            if (paper_id is not None and citation.paper_id != paper_id) or (
                proposal.get("paper_id") is not None
                and citation.paper_id != proposal["paper_id"]
            ):
                raise ValueError("Cross-paper citation")
            citations.append(citation)
    except (ValueError, TypeError):
        return result.model_copy(
            update={
                "support_status": "unsupported",
                "uncertainty": ["citation_or_paper_mismatch"],
            }
        )
    if len({c.paper_id for c in citations}) != 1:
        return result.model_copy(
            update={
                "support_status": "unsupported",
                "uncertainty": ["cross_paper_atomic_claim"],
            }
        )
    result.paper_id = citations[0].paper_id
    reasons = scientific_reasons(
        text,
        " ".join(c.raw_quote for c in citations),
        abstract_only=all(c.evidence_type == "abstract_only" for c in citations),
    )
    if reasons:
        return result.model_copy(
            update={"support_status": "unsupported", "uncertainty": reasons}
        )
    result.text, result.citations = text.strip(), citations
    snapshots = [registry.sources[c.source_id] for c in citations]
    characteristics = StudyCharacteristics(
        doi=snapshots[0].doi,
        year=snapshots[0].year,
        version=snapshots[0].version
        if len({s.version for s in snapshots}) == 1
        else None,
    )
    raw_fields = proposal.get("characteristics") or {}
    if isinstance(raw_fields, dict):
        for name in CHARACTERISTICS:
            field = raw_fields.get(name)
            if (
                not isinstance(field, dict)
                or not isinstance(field.get("value"), str)
                or not field["value"].strip()
            ):
                continue
            try:
                citation = registry.cite(field.get("source_id"), field.get("raw_quote"))
                if (
                    citation.paper_id != result.paper_id
                    or len(field["value"]) > 300
                    or field["value"].casefold() not in citation.raw_quote.casefold()
                ):
                    continue
                if scientific_reasons(
                    field["value"],
                    citation.raw_quote,
                    abstract_only=citation.evidence_type == "abstract_only",
                ):
                    continue
                setattr(
                    characteristics,
                    name,
                    Characteristic(
                        value=field["value"],
                        raw_quote=citation.raw_quote,
                        citation_ids=[citation.citation_id],
                    ),
                )
                if citation not in result.citations:
                    result.citations.append(citation)
            except (ValueError, TypeError):
                continue
    result.characteristics = characteristics
    return result
