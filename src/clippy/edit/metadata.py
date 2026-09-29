"""Metadata generation for the final clip (M9).

Metadata is an **optimization layer**, never a quality mechanism: a render never fails
because of it, and every field is validated against the transcript, chat and streamer
identity so a model cannot invent a game, a person or an event that was not there.

Deterministic fallbacks exist for every field, so a clip always ships with usable text.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx

from clippy.config import Settings
from clippy.edit.render import escape_filter_path

logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTION_CHARS = 280
DEFAULT_TITLE_CHARS = 60
HASHTAG_TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "you", "your", "have", "just",
        "about", "there", "what", "when", "then", "them", "they", "from", "into",
        "like", "some", "been", "were", "will", "would", "could", "should", "here",
        "gonna", "really", "right", "well", "yeah", "okay", "know", "mean", "want",
        "gotta", "talk", "want", "come", "came", "make", "made", "much", "very",
        "lot", "get", "got", "get", "let", "lets", "one", "two", "three", "okay",
    }
)

METADATA_SYSTEM_PROMPT = """You write metadata for a short-form vertical video clipped
from a Twitch stream. Reply with strict JSON only:

{"title": str, "description": str, "hashtags": [str, ...]}

Rules:
- The title must be under 60 characters, punchy, and specific to what actually happens.
- Hashtags are lowercase, without the # character, 3-6 of them, relevant to the content
  and the streamer. Do not invent games, people or events that are not in the input.
- The description is one or two sentences.
- Never mention detection scores, timestamps, or that this was auto-generated.
"""


@dataclass
class ClipMetadata:
    title: str | None = None
    description: str | None = None
    hashtags: list[str] = field(default_factory=list)
    thumbnail_time: float | None = None
    source: str = "fallback"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "hashtags": list(self.hashtags),
            "thumbnail_time": (
                None if self.thumbnail_time is None else round(self.thumbnail_time, 3)
            ),
            "source": self.source,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipMetadata":
        time_value = data.get("thumbnail_time")
        return cls(
            title=data.get("title"),
            description=data.get("description"),
            hashtags=[str(item) for item in (data.get("hashtags") or [])],
            thumbnail_time=float(time_value) if time_value is not None else None,
            source=str(data.get("source", "fallback")),
            reason=data.get("reason"),
        )


def normalize_hashtag(value: str) -> str:
    """Lowercase, strip decoration and collapse to a single [a-z0-9_] token."""
    token = str(value).strip().lstrip("#").lower().replace(" ", "").replace("-", "")
    return "".join(HASHTAG_TOKEN_RE.findall(token))


def source_words(*texts: str) -> set[str]:
    """Every word that appears in the supplied text, lowercased."""
    words: set[str] = set()
    for text in texts:
        for token in HASHTAG_TOKEN_RE.findall(str(text).lower()):
            words.add(token)
    return words


def build_hashtags(
    *texts: str,
    streamer_login: str | None = None,
    keywords: Sequence[str] = (),
    limit: int = 6,
) -> list[str]:
    """
    Deterministic hashtags: the streamer, chat keywords that the transcript supports, then
    the most frequent content words.
    """
    counts: dict[str, int] = {}
    supported = source_words(*texts)
    for text in texts:
        for token in HASHTAG_TOKEN_RE.findall(str(text).lower()):
            if token in STOPWORDS or len(token) < 4 or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1

    tags: list[str] = []
    if streamer_login:
        handle = normalize_hashtag(streamer_login)
        if handle:
            tags.append(handle)
    for keyword in keywords:
        token = normalize_hashtag(str(keyword))
        if token and token in supported and token not in tags:
            tags.append(token)
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if token not in tags:
            tags.append(token)
        if len(tags) >= limit:
            break
    return tags[:limit]


def validate_metadata(
    proposal: dict[str, Any],
    *,
    transcript_text: str,
    chat_text: str = "",
    streamer_login: str = "",
    streamer_display_name: str = "",
    settings: Settings,
) -> ClipMetadata | None:
    """
    Accept a model proposal only where the facts are supported by the source material.

    Hashtags whose words never appear in the transcript, chat or streamer name are
    dropped - inventing a game or a person is the cheapest way for a model to be wrong
    here, and an unsupported tag is worse than no tag.
    """
    if not isinstance(proposal, dict):
        return None
    supported = source_words(
        transcript_text, chat_text, streamer_login, streamer_display_name
    )

    title = str(proposal.get("title") or "").strip()
    if not title:
        return None
    if len(title) > settings.metadata_title_max_chars:
        title = title[: settings.metadata_title_max_chars].rstrip()

    hashtags: list[str] = []
    for raw in proposal.get("hashtags") or []:
        token = normalize_hashtag(str(raw))
        if not token or token in hashtags:
            continue
        if token not in supported:
            logger.debug("Dropping unsupported hashtag %r", raw)
            continue
        hashtags.append(token)
    if not hashtags:
        return None

    description = str(proposal.get("description") or "").strip() or None
    if description and len(description) > DEFAULT_DESCRIPTION_CHARS:
        description = description[:DEFAULT_DESCRIPTION_CHARS].rstrip()

    return ClipMetadata(
        title=title,
        description=description,
        hashtags=hashtags[: settings.metadata_max_hashtags],
        source="llm",
    )


def _first_sentences(text: str, *, limit: int = 2) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return " ".join(parts[:limit]).strip()


def fallback_metadata(
    *,
    transcript_text: str,
    caption: str | None = None,
    chat_text: str = "",
    streamer_login: str = "",
    streamer_display_name: str = "",
    keywords: Sequence[str] = (),
    thumbnail_time: float | None = None,
    settings: Settings,
) -> ClipMetadata:
    """Deterministic metadata: always available, and never wrong about the facts."""
    title = (caption or "").strip() or _first_sentences(transcript_text, limit=1) or None
    if title and len(title) > settings.metadata_title_max_chars:
        title = title[: settings.metadata_title_max_chars].rstrip()

    description = _first_sentences(transcript_text) or None
    if description and streamer_display_name:
        description = f"{description} (via {streamer_display_name})"

    hashtags = build_hashtags(
        transcript_text,
        chat_text,
        streamer_login=streamer_login,
        keywords=keywords,
        limit=settings.metadata_max_hashtags,
    )
    return ClipMetadata(
        title=title,
        description=description,
        hashtags=hashtags,
        thumbnail_time=thumbnail_time,
        source="fallback",
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of model output, tolerating code fences and prose."""
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(cleaned[start : end + 1])
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def generate_metadata(
    *,
    transcript_text: str,
    chat_context: dict[str, Any],
    streamer_display_name: str,
    streamer_login: str,
    settings: Settings,
    caption: str | None = None,
    thumbnail_time: float | None = None,
    api_key: str = "",
    timeout: float = 60.0,
) -> ClipMetadata:
    """
    Ask a chat model for metadata, validate it, and fall back to deterministic text.

    Never raises: metadata is an optimization layer, so a failure leaves the clip with
    fallback text plus a reason instead of blocking the render.
    """
    chat_text = json.dumps(chat_context)[:2000]
    fallback = lambda reason=None: _fallback_with_reason(  # noqa: E731
        reason,
        transcript_text=transcript_text,
        caption=caption,
        chat_text=chat_text,
        streamer_login=streamer_login,
        streamer_display_name=streamer_display_name,
        keywords=settings.chat_keywords,
        thumbnail_time=thumbnail_time,
        settings=settings,
    )

    if not settings.metadata_enabled:
        return fallback("metadata disabled")
    if not (api_key or "").strip():
        return fallback("no API key: deterministic metadata used")

    try:
        response = httpx.post(
            f"{settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": settings.metadata_model or settings.caption_model,
                "temperature": 0.5,
                "messages": [
                    {"role": "system", "content": METADATA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "streamer": streamer_display_name,
                                "transcript": transcript_text[:4000],
                                "chat": chat_context,
                                "existing_caption": caption,
                            }
                        ),
                    },
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        validated = validate_metadata(
            _parse_json_object(content),
            transcript_text=transcript_text,
            chat_text=chat_text,
            streamer_login=streamer_login,
            streamer_display_name=streamer_display_name,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("Metadata generation failed")
        return fallback(f"llm metadata failed: {exc}")

    if validated is None:
        return fallback("llm metadata was unsupported by the transcript, used fallback")
    validated.thumbnail_time = thumbnail_time
    return validated


def _fallback_with_reason(reason: str | None, **kwargs: Any) -> ClipMetadata:
    metadata = fallback_metadata(**kwargs)
    metadata.reason = reason
    return metadata


def capture_thumbnail(
    media_path: Path,
    output: Path,
    *,
    time_seconds: float,
    ffmpeg_path: str = "ffmpeg",
    overlay_text: str | None = None,
    font_path: str = "C:/Windows/Fonts/arialbd.ttf",
) -> Path:
    """Grab one frame as a cover image, optionally with a short text overlay."""
    ffmpeg = shutil.which(ffmpeg_path) or (
        ffmpeg_path if Path(ffmpeg_path).exists() else None
    )
    if not ffmpeg:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(0.0, float(time_seconds)):.3f}",
        "-i",
        str(media_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
    ]
    if overlay_text:
        safe = escape_filter_path(str(overlay_text).replace("'", "").replace(":", ""))
        cmd += [
            "-vf",
            (
                f"drawtext=fontfile={escape_filter_path(font_path)}:text='{safe}'"
                ":fontcolor=white:fontsize=64:x=(w-text_w)/2:y=h-260"
                ":box=1:boxcolor=black@0.45:boxborderw=18"
            ),
        ]
    cmd.append(str(output))
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg thumbnail failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced an empty thumbnail: {output}")
    return output
