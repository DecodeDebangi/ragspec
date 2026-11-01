"""Find citation markers such as [doc12] or [1, 3] in an answer and resolve them to context ids."""

from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_PATTERN = r"\[([^\[\]]+)\]"
_SPLIT = re.compile(r"\s*[,;]\s*")


def extract_markers(answer: str, pattern: str = DEFAULT_PATTERN) -> list[str]:
    """Raw citation tokens in order of first appearance: "see [a1, b2] and [a1]" -> [a1, b2]."""
    tokens: list[str] = []
    for match in re.finditer(pattern, answer):
        for token in _SPLIT.split(match.group(1).strip()):
            token = token.strip().strip("\"'")
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def resolve(
    tokens: Sequence[str], context_ids: Sequence[str], style: str = "auto"
) -> dict[str, str]:
    """Map each citation token to a context id.

    style="id": tokens are context ids.
    style="position": tokens are 1-based positions in the retrieved list ("[2]" = 2nd context).
    style="auto": tokens that match a context id map to themselves; if every other token is an
    integer, those are read as positions.
    Anything unresolvable is kept as-is and will count as an invalid citation.
    """
    ids = list(context_ids)
    known = set(ids) if style != "position" else set()
    unmatched = [t for t in tokens if t not in known]
    if style == "id":
        positional = False
    elif style == "position":
        positional = True
    else:
        positional = bool(unmatched) and all(t.isdigit() for t in unmatched)
    mapping: dict[str, str] = {}
    for token in tokens:
        if token in known:
            mapping[token] = token
        elif positional and token.isdigit() and 1 <= int(token) <= len(ids):
            mapping[token] = ids[int(token) - 1]
        else:
            mapping[token] = token
    return mapping
