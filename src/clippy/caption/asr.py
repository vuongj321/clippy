from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_ASR_TIMEOUT = 180.0


def transcribe_media(
    media_path: Path,
    *,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "whisper-1",
    timeout: float = DEFAULT_ASR_TIMEOUT,
) -> str:
    """Transcribe a cut review window via an OpenAI-compatible Whisper endpoint."""
    if not media_path.exists() or media_path.stat().st_size == 0:
        raise FileNotFoundError(f"Media not found or empty: {media_path}")

    url = f"{base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    with media_path.open("rb") as fh:
        response = httpx.post(
            url,
            headers=headers,
            files={"file": (media_path.name, fh, "application/octet-stream")},
            data={"model": model},
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("text") if isinstance(payload, dict) else None
    if not text or not str(text).strip():
        return ""
    return str(text).strip()
