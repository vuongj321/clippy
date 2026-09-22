from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ChatMessage:
    """Chat event on the shared stream-relative timeline (seconds)."""

    ts: float
    user: str
    text: str
    emotes: list[str] | None = None


def load_chat_json(path: Path) -> list[ChatMessage]:
    """
    Load chat dump JSON.

    Supported shapes:
    - list of {ts|offset_seconds|content_offset_seconds, user|username, text|message|body}
    - {"comments": [...]} 
    - {"messages": [...]}
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "comments" in raw:
            items = raw["comments"]
        elif "messages" in raw:
            items = raw["messages"]
        else:
            raise ValueError("Chat JSON object must contain 'comments' or 'messages'")
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Chat JSON must be a list or object")

    messages = [_parse_message(item) for item in items]
    messages.sort(key=lambda m: m.ts)
    return messages


def _parse_message(item: dict[str, Any]) -> ChatMessage:
    ts = item.get("ts")
    if ts is None:
        ts = item.get("offset_seconds")
    if ts is None:
        ts = item.get("content_offset_seconds")
    if ts is None:
        ts = item.get("contentOffsetSeconds")
    if ts is None:
        raise ValueError(f"Chat message missing timestamp field: {item!r}")

    user: Any = item.get("user") or item.get("username") or item.get("commenter") or "unknown"
    if isinstance(user, dict):
        user = user.get("display_name") or user.get("name") or "unknown"

    text: Any = item.get("text") or item.get("body") or item.get("message") or ""
    if isinstance(text, dict):
        text = text.get("body") or text.get("text") or ""

    emotes = item.get("emotes")
    if emotes is not None and not isinstance(emotes, list):
        emotes = None

    return ChatMessage(ts=float(ts), user=str(user), text=str(text), emotes=emotes)
