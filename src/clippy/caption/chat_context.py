from __future__ import annotations

from collections import Counter
from typing import Any

from clippy.chat.models import ChatMessage

_EMOTE_ONLY = {
    "kekw",
    "kek",
    "lul",
    "lol",
    "lmao",
    "pog",
    "poggers",
    "pogchamp",
    "omegalul",
    "sadge",
    "copium",
    "based",
    "sheesh",
    "nice",
    "f",
    "gg",
    "pepe",
    "pepega",
    "monkas",
    "jebaited",
}


def slice_chat(
    messages: list[ChatMessage],
    *,
    start: float,
    end: float,
) -> list[ChatMessage]:
    return [m for m in messages if start <= m.ts <= end]


def is_emote_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    tokens = stripped.split()
    if len(tokens) != 1:
        return False
    token = tokens[0].lower().strip("!?.")
    if token in _EMOTE_ONLY:
        return True
    return not any(ch.isalnum() for ch in token)


def _has_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def build_chat_context(
    messages: list[ChatMessage],
    *,
    start: float,
    end: float,
    keywords: list[str] | None = None,
    max_messages: int = 40,
) -> dict[str, Any]:
    """Slice and despam chat for the caption LLM."""
    keywords = keywords or []
    window = slice_chat(messages, start=start, end=end)
    counts: Counter[str] = Counter()
    keyword_hits: list[dict[str, Any]] = []
    keyword_kept: list[ChatMessage] = []
    other_kept: list[ChatMessage] = []

    for msg in window:
        text = msg.text.strip()
        if not text:
            continue
        counts[text.lower()] += 1
        if _has_keyword(text, keywords):
            keyword_hits.append({"ts": msg.ts, "user": msg.user, "text": msg.text})
            keyword_kept.append(msg)
        elif not is_emote_only(text):
            other_kept.append(msg)

    if len(keyword_kept) + len(other_kept) > max_messages:
        mid = (start + end) / 2.0
        other_kept.sort(key=lambda m: abs(m.ts - mid))
        needed = max(0, max_messages - len(keyword_kept))
        kept = keyword_kept + other_kept[:needed]
    else:
        kept = keyword_kept + other_kept
    kept.sort(key=lambda m: m.ts)

    repeated = [
        {"text": text, "count": count}
        for text, count in counts.most_common(5)
        if count >= 2
    ]
    return {
        "messages": [{"ts": m.ts, "user": m.user, "text": m.text} for m in kept],
        "keyword_hits": keyword_hits[: max_messages],
        "repeated": repeated,
        "window_count": len(window),
    }
