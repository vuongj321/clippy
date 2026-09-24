from __future__ import annotations

import re


def has_keyword(text: str, keywords: list[str]) -> bool:
    return first_keyword(text, keywords) is not None


def first_keyword(text: str, keywords: list[str]) -> str | None:
    """Return the first keyword that matches as a whole phrase, or None."""
    if not text or not keywords:
        return None
    lowered = text.lower()
    for keyword in keywords:
        needle = keyword.lower().strip()
        if not needle:
            continue
        pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
        if re.search(pattern, lowered):
            return keyword
    return None
