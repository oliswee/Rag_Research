"""Pack original child anchors and bounded parent windows without quote rewriting."""

import hashlib
import json
import re
import time
from dataclasses import dataclass

from app.config import settings


@dataclass
class PackedContext:
    sources: list[dict]
    blocks: list[dict]
    text: str
    coverage: dict
    token_units: int


def _span_key(span):
    return tuple(
        span.get(k)
        for k in ("paper_id", "document_id", "document_version", "page", "block_id")
    )


def _doi(source):
    value = (source.get("doi") or "").strip().lower()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)


def _overlap(left, right):
    return any(
        _span_key(a) == _span_key(b)
        and a["raw_start"] < b["raw_end"]
        and b["raw_start"] < a["raw_end"]
        for a in left.get("source_spans", [])
        for b in right.get("source_spans", [])
    )


def _same_group(left, right):
    if (left.get("document_version"), left.get("content_scope")) != (
        right.get("document_version"),
        right.get("content_scope"),
    ):
        return False
    # Unknown versions cannot establish cross-origin DOI equivalence.
    if (
        _doi(left)
        and _doi(left) == _doi(right)
        and left.get("document_version")
        and left["excerpt"] == right["excerpt"]
    ):
        return True
    if (left["paper_id"], left.get("document_id")) != (
        right["paper_id"],
        right.get("document_id"),
    ):
        return False
    return bool(
        left.get("parent_id")
        and left.get("parent_id") == right.get("parent_id")
        or _overlap(left, right)
        or _overlap(left.get("parent") or {}, right.get("parent") or {})
    )


def _merge_ranges(ranges):
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _child_block(group):
    """Whole-excerpt fallback, exact duplicate excerpts share one context string."""
    text, positions, citations = "", {}, []
    spans = []
    for source in group:
        excerpt = source["excerpt"]
        if excerpt not in positions:
            start = len(text) + (2 if text else 0)
            text += ("\n\n" if text else "") + excerpt
            positions[excerpt] = [start, len(text)]
            spans.extend(source.get("source_spans", []))
        citations.append({"id": source["chunk_id"], "ranges": [positions[excerpt]]})
    return {
        "text": text,
        "child_ids": [s["chunk_id"] for s in group],
        "citations": citations,
        "source_spans": spans,
        "parent_ids": list(
            dict.fromkeys(s["parent_id"] for s in group if s.get("parent_id"))
        ),
        "paper_ids": list(dict.fromkeys(s["paper_id"] for s in group)),
        "document_version": group[0].get("document_version"),
        "content_scope": group[0].get("content_scope", "full_text"),
        "source_origins": list(
            dict.fromkeys(s.get("source_origin", "local") for s in group)
        ),
        "pages": sorted(
            {
                page
                for s in group
                for page in (s.get("page_start"), s.get("page_end"))
                if page is not None
            }
        ),
        "expansion_status": "child_fallback",
    }


def _parent_block(group, budget):
    """Map child anchors into one validated parent; expand at sentence boundaries."""
    fallback = _child_block(group)
    source = group[0]
    parent = source.get("parent")
    if not parent or parent.get("id") != source.get("parent_id"):
        return fallback
    raw = parent.get("raw_text", "")
    spans = parent.get("source_spans", [])
    segments, cursor = [], 0
    for anchor in spans:
        length = anchor["raw_end"] - anchor["raw_start"]
        if length <= 0 or any(
            anchor.get(k) != source.get(k)
            for k in ("paper_id", "document_id", "document_version")
        ):
            return fallback
        segments.append((cursor, cursor + length, anchor))
        cursor += length + 2
    if not segments or cursor - 2 != len(raw):
        return fallback
    child_ranges = {}
    for child in group:
        ranges = []
        for anchor in child.get("source_spans", []):
            match = next(
                (
                    seg
                    for seg in segments
                    if _span_key(seg[2]) == _span_key(anchor)
                    and seg[2]["raw_start"]
                    <= anchor["raw_start"]
                    < anchor["raw_end"]
                    <= seg[2]["raw_end"]
                ),
                None,
            )
            if match is None:
                return fallback
            start = match[0] + anchor["raw_start"] - match[2]["raw_start"]
            ranges.append([start, start + anchor["raw_end"] - anchor["raw_start"]])
        if not ranges or "\n\n".join(raw[a:b] for a, b in ranges) != child["excerpt"]:
            return fallback
        child_ranges[child["chunk_id"]] = ranges
    required = _merge_ranges([r for ranges in child_ranges.values() for r in ranges])
    # Decimal points do not split sentences; a boundary requires punctuation+space.
    ends = [m.end() for m in re.finditer(r"[.!?](?:[ \t]+|\n+)|\n\n", raw)]
    boundaries = sorted({0, len(raw), *ends})
    units = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    selected = {
        i for i, (a, b) in enumerate(units) if any(a < y and x < b for x, y in required)
    }

    def window(indices):
        return _merge_ranges([list(units[i]) for i in sorted(indices)])

    def cost(indices):
        return len("\n\n".join(raw[a:b] for a, b in window(indices)).encode())

    if cost(selected) > budget:
        # Preserve child boundaries rather than truncate a large sentence or number.
        return fallback
    neighbors = sorted(
        set(range(len(units))) - selected,
        key=lambda i: (min(abs(i - j) for j in selected), i),
    )
    for i in neighbors:
        if cost(selected | {i}) <= budget:
            selected.add(i)
    windows = window(selected)
    text, mapped, output_spans = "", [], []
    for a, b in windows:
        offset = len(text) + (2 if text else 0)
        text += ("\n\n" if text else "") + raw[a:b]
        mapped.append((a, b, offset))
        for left, right, anchor in segments:
            x, y = max(a, left), min(b, right)
            if x < y:
                output_spans.append(
                    {
                        **anchor,
                        "raw_start": anchor["raw_start"] + x - left,
                        "raw_end": anchor["raw_start"] + y - left,
                    }
                )
    citations = []
    for child in group:
        locations = []
        for a, b in child_ranges[child["chunk_id"]]:
            match = next((w for w in mapped if w[0] <= a < b <= w[1]), None)
            if match is None:
                return fallback
            locations.append([match[2] + a - match[0], match[2] + b - match[0]])
        citations.append({"id": child["chunk_id"], "ranges": locations})
    return {
        **fallback,
        "text": text,
        "citations": citations,
        "source_spans": output_spans,
        "expansion_status": "parent_window",
    }


def render_blocks(blocks):
    """Count the exact serialized context, including citable-range metadata."""
    fields = (
        "text",
        "citations",
        "paper_ids",
        "parent_ids",
        "document_version",
        "content_scope",
        "source_origins",
        "pages",
    )
    return "\n\n".join(
        json.dumps(
            {key: b[key] for key in fields}, ensure_ascii=False, separators=(",", ":")
        )
        for b in blocks
    )


def pack_context(
    sources, *, paper_ids=None, budget=None, parent_budget=None, expand=True, trace=None
):
    started = time.perf_counter()
    budget = settings.context_token_budget if budget is None else budget
    parent_budget = (
        settings.context_parent_budget if parent_budget is None else parent_budget
    )
    if not 1 <= budget <= 16384 or not 1 <= parent_budget <= 4096:
        raise ValueError("Invalid context budget")
    groups, admitted, decisions = [], [], []
    unique = list({s["chunk_id"]: s for s in reversed(sources)}.values())[::-1]
    # Explicit paper representatives come first for budget reservation.
    priority = []
    for paper_id in dict.fromkeys(paper_ids or []):
        match = next(
            (
                s
                for s in unique
                if s["paper_id"] == paper_id
                and s.get("evidence_role") not in {"bibliography", "page_marker"}
            ),
            None,
        )
        if match is not None:
            priority.append(match)
    priority += [s for s in unique if s not in priority]
    for source in priority:
        if source.get("evidence_role") in {"bibliography", "page_marker"}:
            decisions.append(
                {"child_id": source["chunk_id"], "reason": "non_body_evidence"}
            )
            continue
        proposed = [list(g) for g in groups]
        matches = [
            i
            for i, group in enumerate(proposed)
            if any(_same_group(source, s) for s in group)
        ]
        if matches:
            target = matches[0]
            proposed[target].append(source)
            for index in reversed(matches[1:]):
                proposed[target].extend(proposed.pop(index))
        else:
            proposed.append([source])
        blocks = [_child_block(group) for group in proposed]
        if len(render_blocks(blocks).encode()) > budget:
            decisions.append(
                {"child_id": source["chunk_id"], "reason": "child_exceeds_budget"}
            )
            continue
        groups = proposed
        admitted.append(source)
        decisions.append({"child_id": source["chunk_id"], "reason": "included"})
    blocks = [_child_block(group) for group in groups]
    if expand:
        for index, group in enumerate(groups):
            try:
                expanded = _parent_block(group, min(parent_budget, budget))
            except (KeyError, TypeError, ValueError):
                expanded = _child_block(group)
            proposed = list(blocks)
            proposed[index] = expanded
            if len(render_blocks(proposed).encode()) <= budget:
                blocks = proposed
    text = render_blocks(blocks)
    included = {s["chunk_id"] for s in admitted}
    coverage = {}
    for paper_id in dict.fromkeys(
        [*(paper_ids or []), *(s["paper_id"] for s in unique)]
    ):
        candidates = [s for s in unique if s["paper_id"] == paper_id]
        chosen = [s for s in candidates if s["chunk_id"] in included]
        coverage[paper_id] = {
            "candidate_status": "available" if candidates else "unknown",
            "candidate_count": len(candidates),
            "prompt_count": len(chosen),
            "prompt_status": "included" if chosen else "unknown",
            "support_status": "unknown",
        }
    if trace is not None:
        context_bytes = sum(len(b["text"].encode()) for b in blocks)
        child_bytes = sum(
            len(block["text"][a:b].encode())
            for block in blocks
            for a, b in _merge_ranges(
                [r for citation in block["citations"] for r in citation["ranges"]]
            )
        )
        trace.record["context_packing"] = {
            "budget_units": "utf8_bytes_conservative_bpe_bound",
            "budget": budget,
            "used_units": len(text.encode()),
            "parent_budget": parent_budget,
            "candidate_count": len(unique),
            "prompt_count": len(admitted),
            "prompt_evidence_coverage": len(admitted) / len(unique) if unique else None,
            "background_context_fraction": 1 - child_bytes / context_bytes
            if context_bytes
            else None,
            "decisions": decisions,
            "paper_coverage": {
                hashlib.sha256(k.encode()).hexdigest(): v for k, v in coverage.items()
            },
            "blocks": [
                {
                    key: block[key]
                    for key in (
                        "child_ids",
                        "parent_ids",
                        "source_spans",
                        "expansion_status",
                    )
                }
                for block in blocks
            ],
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        trace.record["stage_ms"]["packing"] = trace.record["context_packing"][
            "duration_ms"
        ]
    return PackedContext(
        [s for s in unique if s["chunk_id"] in included],
        blocks,
        text,
        coverage,
        len(text.encode()),
    )
