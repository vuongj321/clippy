from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class TranscriptWord:
    """One spoken word on the clip timeline."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptWord":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
        )


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
            words=[
                TranscriptWord.from_dict(item)
                for item in (data.get("words") or [])
                if isinstance(item, dict)
            ],
        )


@dataclass
class TranscriptPayload:
    """A transcript with whatever timing detail the endpoint was willing to give."""

    text: str = ""
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)

    def words(self) -> list[TranscriptWord]:
        """
        Flattened word timings.

        A server that only returns segment granularity still yields usable captions:
        word times are interpolated across each segment instead of giving up.
        """
        words: list[TranscriptWord] = []
        for segment in self.segments:
            words.extend(segment.words or _interpolate_words(segment))
        words.sort(key=lambda word: (word.start, word.end))
        return words

    def has_word_timings(self) -> bool:
        return any(segment.words for segment in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptPayload":
        return cls(
            text=str(data.get("text", "")),
            language=data.get("language"),
            duration=float(data["duration"]) if data.get("duration") else None,
            segments=[
                TranscriptSegment.from_dict(item)
                for item in (data.get("segments") or [])
                if isinstance(item, dict)
            ],
        )


def _interpolate_words(segment: TranscriptSegment) -> list[TranscriptWord]:
    """Spread a segment's words evenly across its span so captions still roughly sync."""
    tokens = segment.text.split()
    span = segment.end - segment.start
    if not tokens or span <= 0:
        return []
    step = span / len(tokens)
    return [
        TranscriptWord(
            start=segment.start + index * step,
            end=segment.start + (index + 1) * step,
            text=token,
        )
        for index, token in enumerate(tokens)
    ]


def _payload_from_response(payload: dict[str, Any]) -> TranscriptPayload:
    segments: list[TranscriptSegment] = []
    for item in payload.get("segments") or []:
        if not isinstance(item, dict):
            continue
        words = [
            TranscriptWord(
                start=float(entry.get("start", 0.0)),
                end=float(entry.get("end", 0.0)),
                text=str(entry.get("word", entry.get("text", ""))).strip(),
            )
            for entry in (item.get("words") or [])
            if isinstance(entry, dict)
        ]
        segments.append(
            TranscriptSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=str(item.get("text", "")).strip(),
                words=[word for word in words if word.text],
            )
        )
    language = payload.get("language")
    duration = payload.get("duration")
    return TranscriptPayload(
        text=str(payload.get("text") or "").strip(),
        language=str(language) if language else None,
        duration=float(duration) if duration else None,
        segments=segments,
    )


def _post_transcription(
    url: str,
    *,
    api_key: str,
    files: dict[str, Any],
    data: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _attach_top_level_words(
    payload: TranscriptPayload, raw: dict[str, Any]
) -> TranscriptPayload:
    """
    Fold a top-level `words` array into the segments it belongs to.

    Some OpenAI-compatible servers (Groq among them) accept `timestamp_granularities`
    but return the timings beside `segments` instead of inside them. Without this they
    would be dropped and needlessly re-interpolated, which costs exact karaoke timing.
    """
    entries = [item for item in (raw.get("words") or []) if isinstance(item, dict)]
    if not entries or payload.has_word_timings():
        return payload

    words = [
        TranscriptWord(
            start=float(entry.get("start", 0.0)),
            end=float(entry.get("end", 0.0)),
            text=str(entry.get("word", entry.get("text", ""))).strip(),
        )
        for entry in entries
    ]
    words = [word for word in words if word.text]
    if not words:
        return payload

    if not payload.segments:
        payload.segments = [
            TranscriptSegment(
                start=words[0].start,
                end=words[-1].end,
                text=payload.text,
                words=words,
            )
        ]
        return payload

    for segment in payload.segments:
        segment.words = [
            word for word in words if word.end > segment.start and word.start < segment.end
        ]
    return payload


def transcribe_words(
    media_path: Path,
    *,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "whisper-1",
    timeout: float = DEFAULT_ASR_TIMEOUT,
    word_timestamps: bool = True,
) -> TranscriptPayload:
    """
    Transcribe a clip with timings via an OpenAI-compatible endpoint.

    Requests `verbose_json` with word *and* segment granularity. A deployment that
    rejects word granularity (HTTP 400/422) is retried segment-only and the missing word
    times are interpolated, so a weaker server degrades instead of failing the render.
    """
    if not media_path.exists() or media_path.stat().st_size == 0:
        raise FileNotFoundError(f"Media not found or empty: {media_path}")

    url = f"{base_url.rstrip('/')}/audio/transcriptions"
    files = {
        "file": (media_path.name, media_path.read_bytes(), "application/octet-stream")
    }
    data: dict[str, Any] = {"model": model, "response_format": "verbose_json"}

    if word_timestamps:
        try:
            payload = _post_transcription(
                url,
                api_key=api_key,
                files=files,
                data={**data, "timestamp_granularities[]": ["word", "segment"]},
                timeout=timeout,
            )
            return _attach_top_level_words(_payload_from_response(payload), payload)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (400, 422):
                raise
            logger.info(
                "Endpoint rejected word-level timings (%s); retrying segment-only",
                exc.response.status_code,
            )

    payload = _post_transcription(
        url, api_key=api_key, files=files, data=data, timeout=timeout
    )
    return _attach_top_level_words(_payload_from_response(payload), payload)
