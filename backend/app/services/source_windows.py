"""Union already-validated original anchor slices; reject conflicting overlaps."""


def span_identity(span):
    return tuple(
        span.get(k)
        for k in ("paper_id", "document_id", "document_version", "page", "block_id")
    )


def _pieces(text, spans, source):
    if not spans:
        return None
    cursor, result = 0, []
    for anchor in spans:
        left, right = anchor.get("raw_start"), anchor.get("raw_end")
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left < right
            or any(
                anchor.get(k) != source.get(k)
                for k in ("paper_id", "document_id", "document_version")
            )
        ):
            return None
        length = right - left
        value = text[cursor : cursor + length]
        if len(value) != length:
            return None
        result.append((dict(anchor), value))
        cursor += length
        if cursor < len(text):
            if text[cursor : cursor + 2] != "\n\n":
                return None
            cursor += 2
    return result if cursor == len(text) else None


def original_union(sources, *, include_parents=False):
    signature = tuple(
        sources[0].get(k) for k in ("paper_id", "document_id", "document_version")
    )
    if any(
        tuple(s.get(k) for k in ("paper_id", "document_id", "document_version"))
        != signature
        for s in sources
    ):
        return None
    pieces, has_parent = [], False
    if include_parents:
        for source in sources:
            parent = source.get("parent")
            if parent and parent.get("id") == source.get("parent_id"):
                unpacked = _pieces(
                    parent.get("raw_text", ""), parent.get("source_spans", []), source
                )
                if unpacked:
                    pieces.extend(unpacked)
                    has_parent = True
        if not has_parent:
            return None
    for source in sources:
        unpacked = _pieces(source["excerpt"], source.get("source_spans", []), source)
        if unpacked is None:
            return None
        pieces.extend(unpacked)
    by_block = {}
    for anchor, text in pieces:
        by_block.setdefault(span_identity(anchor), []).append((anchor, text))
    merged = []
    for block in by_block.values():
        combined = []
        for anchor, text in sorted(
            block, key=lambda part: (part[0]["raw_start"], part[0]["raw_end"])
        ):
            if combined and anchor["raw_start"] <= combined[-1][0]["raw_end"]:
                previous, value = combined[-1]
                start = anchor["raw_start"] - previous["raw_start"]
                overlap = (
                    min(previous["raw_end"], anchor["raw_end"]) - anchor["raw_start"]
                )
                if value[start : start + overlap] != text[:overlap]:
                    return None
                if anchor["raw_end"] > previous["raw_end"]:
                    combined[-1] = (
                        {**previous, "raw_end": anchor["raw_end"]},
                        value + text[overlap:],
                    )
            else:
                combined.append((anchor, text))
        merged.extend(combined)
    # PDF raw offsets order blocks within a page. Extracted public text has no
    # page number, so retain the original parent span order for that case.
    if all(type(a.get("page")) is int for a, _ in merged):
        merged.sort(key=lambda item: (item[0]["page"], item[0]["raw_start"]))
    return "\n\n".join(text for _, text in merged), [a for a, _ in merged]


def citable_ranges(raw, spans, sources):
    segments, cursor = [], 0
    for anchor in spans:
        length = anchor["raw_end"] - anchor["raw_start"]
        segments.append((cursor, anchor))
        cursor += length + 2
    result = {}
    for source in sources:
        ranges = []
        for anchor in source.get("source_spans", []):
            match = next(
                (
                    item
                    for item in segments
                    if span_identity(item[1]) == span_identity(anchor)
                    and item[1]["raw_start"]
                    <= anchor["raw_start"]
                    < anchor["raw_end"]
                    <= item[1]["raw_end"]
                ),
                None,
            )
            if match is None:
                return None
            start = match[0] + anchor["raw_start"] - match[1]["raw_start"]
            ranges.append([start, start + anchor["raw_end"] - anchor["raw_start"]])
        if not ranges or "\n\n".join(raw[a:b] for a, b in ranges) != source["excerpt"]:
            return None
        result[source["chunk_id"]] = ranges
    return result
