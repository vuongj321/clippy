from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clippy.chat.models import ChatMessage, load_chat_json


@dataclass
class VodIngestResult:
    """
    Offline / VOD ingest result.

    Timeline contract: chat timestamps and media PTS share stream-relative
    seconds with t=0 at media start (VOD start). Chat JSON must already use
    that clock (offset_seconds / content_offset_seconds / ts).
    """

    media_path: Path
    chat: list[ChatMessage]
    streamer_login: str
    display_name: str
    source_url: str | None = None
    vod_id: str | None = None


def ingest_local_vod(
    media_path: Path,
    chat_path: Path,
    *,
    streamer_login: str,
    display_name: str | None = None,
    source_url: str | None = None,
    vod_id: str | None = None,
) -> VodIngestResult:
    if not media_path.exists():
        raise FileNotFoundError(f"Media not found: {media_path}")
    if not chat_path.exists():
        raise FileNotFoundError(f"Chat JSON not found: {chat_path}")

    chat = load_chat_json(chat_path)
    return VodIngestResult(
        media_path=media_path.resolve(),
        chat=chat,
        streamer_login=streamer_login.lower(),
        display_name=display_name or streamer_login,
        source_url=source_url,
        vod_id=vod_id,
    )
