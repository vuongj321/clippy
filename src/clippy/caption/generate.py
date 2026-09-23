from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from clippy.caption.asr import transcribe_media
from clippy.caption.chat_context import build_chat_context
from clippy.chat.models import ChatMessage
from clippy.config import Settings

logger = logging.getLogger(__name__)

_MISSING_KEY_LOGGED = False

CAPTION_SYSTEM_PROMPT = """You write short clip captions for a Twitch highlight review tool.
Write 3-10 words describing what is happening in the clip.
Use the streamer name when it helps.
Do not invent events that are not supported by the transcript or chat.
Do not restate why the clip was extracted or mention detection scores.
If the evidence is too thin, reply with an empty string.
Reply with the caption only, no quotes or preamble."""


def _openai_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def generate_caption(
    *,
    streamer_display_name: str,
    streamer_login: str,
    extract_reason: str,
    transcript: str | None,
    chat_context: dict[str, Any],
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    timeout: float = 60.0,
) -> str:
    """Call a chat model and return a short caption. Raises on HTTP errors."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    user_payload = {
        "streamer_display_name": streamer_display_name,
        "streamer_login": streamer_login,
        "extract_reason_do_not_repeat": extract_reason,
        "transcript": transcript or "",
        "chat": chat_context,
    }
    response = httpx.post(
        url,
        headers=_openai_headers(api_key),
        json={
            "model": model,
            "temperature": 0.4,
            "messages": [
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content") or ""
    caption = str(content).strip().strip('"').strip("'")
    return caption


def annotate_extracted_candidate(
    *,
    media_path: Path | None,
    chat: list[ChatMessage],
    source_ts: float,
    pre_context_seconds: float,
    post_context_seconds: float,
    streamer_display_name: str,
    streamer_login: str,
    extract_reason: str,
    settings: Settings,
) -> tuple[str | None, str | None]:
    """
    Return (caption, transcript). Never raises for missing keys or API errors.
    """
    global _MISSING_KEY_LOGGED

    start = max(0.0, source_ts - pre_context_seconds)
    end = source_ts + post_context_seconds
    chat_context = build_chat_context(
        chat,
        start=start,
        end=end,
        keywords=settings.chat_keywords,
        max_messages=settings.caption_max_chat_messages,
    )

    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        if not _MISSING_KEY_LOGGED:
            logger.info(
                "CLIPPY_OPENAI_API_KEY is unset; skipping ASR and caption generation"
            )
            _MISSING_KEY_LOGGED = True
        return None, None

    transcript: str | None = None
    if media_path is not None:
        try:
            text = transcribe_media(
                media_path,
                api_key=api_key,
                base_url=settings.openai_base_url,
                model=settings.asr_model,
            )
            transcript = text or None
        except Exception:
            logger.exception("ASR failed for %s", media_path)

    try:
        caption = generate_caption(
            streamer_display_name=streamer_display_name,
            streamer_login=streamer_login,
            extract_reason=extract_reason,
            transcript=transcript,
            chat_context=chat_context,
            api_key=api_key,
            base_url=settings.openai_base_url,
            model=settings.caption_model,
        )
        return (caption or None), transcript
    except Exception:
        logger.exception("Caption generation failed")
        return None, transcript
