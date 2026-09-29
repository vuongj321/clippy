"""Caption stage for one edit (M5).

Order matters: transcription runs on ``trimmed.mp4``, i.e. *after* dead-air removal, so
every cue time already refers to the final timeline and nothing needs remapping.

The stage always degrades rather than failing a render: no API key, an ASR error, an
empty transcript or no cues each leave the plan saying why, and the render carries on
without burned-in text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from clippy.caption.align import Cue, build_cues, pick_emphasis_words
from clippy.caption.asr import TranscriptPayload, transcribe_words
from clippy.caption.ass import write_ass
from clippy.caption.styles import CaptionStyle, caption_anchor, resolve_style
from clippy.config import Settings
from clippy.edit.plan import CaptionsPlan, EditPaths, EditPlan

logger = logging.getLogger(__name__)


@dataclass
class CaptionResult:
    applied: bool
    cues: list[Cue] = field(default_factory=list)
    captions_path: Path | None = None
    transcript_path: Path | None = None
    style: CaptionStyle | None = None
    anchor: str = "bottom"
    reason: str | None = None
    emphasis: set[str] = field(default_factory=set)

    def commit(self, plan: EditPlan) -> None:
        """Record the outcome on the plan so later stages and the reviewer can see it."""
        plan.captions = CaptionsPlan(
            enabled=plan.captions.enabled,
            style=self.style.name if self.style else plan.captions.style,
            emphasis=plan.captions.emphasis,
            anchor=self.anchor,
            word_timestamps=plan.captions.word_timestamps,
            cue_count=len(self.cues),
            emphasis_words=sorted(self.emphasis),
            reason=self.reason,
        )


def _load_or_transcribe(
    paths: EditPaths,
    *,
    settings: Settings,
    api_key: str,
    force: bool,
) -> tuple[TranscriptPayload | None, str | None]:
    """Reuse a cached transcript so a re-render never pays for ASR twice."""
    if paths.transcript.exists() and not force:
        try:
            cached = TranscriptPayload.from_dict(
                json.loads(paths.transcript.read_text(encoding="utf-8"))
            )
        except Exception:
            logger.warning("Ignoring unreadable transcript cache %s", paths.transcript)
        else:
            if cached.segments:
                logger.info("Reusing cached transcript %s", paths.transcript)
                return cached, None

    try:
        payload = transcribe_words(
            paths.trimmed,
            api_key=api_key,
            base_url=settings.openai_base_url,
            model=settings.asr_model,
            word_timestamps=settings.asr_word_timestamps,
        )
    except Exception as exc:
        logger.exception("Transcription failed for %s", paths.trimmed)
        return None, f"transcription failed: {exc}"

    paths.transcript.write_text(
        json.dumps(payload.to_dict(), indent=2), encoding="utf-8"
    )
    return payload, None


def generate_captions(
    plan: EditPlan,
    paths: EditPaths,
    *,
    settings: Settings,
    force: bool = False,
    prefer_top: bool = False,
) -> CaptionResult:
    """
    Transcribe, align and write ``captions.ass`` for the trimmed clip.

    `prefer_top` comes from composition (M6/M7): when the subject sits in the lower band
    of the frame, captions move up so they never cover it. Emphasised words come from the
    transcript plus the chat keywords, and `caption_emphasis: off` disables them.
    """
    if not plan.captions.enabled:
        return CaptionResult(applied=False, reason="captions disabled for this render")

    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        return CaptionResult(applied=False, reason="no API key: captions skipped")
    if not paths.trimmed.exists():
        return CaptionResult(applied=False, reason="trimmed clip missing")

    payload, failure = _load_or_transcribe(
        paths, settings=settings, api_key=api_key, force=force
    )
    if payload is None:
        return CaptionResult(applied=False, reason=failure)

    style = resolve_style(settings.caption_style, settings)
    cues = build_cues(
        payload.words(),
        max_chars_per_line=style.max_chars_per_line,
        max_lines=style.max_lines,
        max_cue_seconds=style.max_cue_seconds,
        min_cue_seconds=style.min_cue_seconds,
        break_gap_seconds=style.break_gap_seconds,
        uppercase=style.uppercase,
    )
    if not cues:
        return CaptionResult(
            applied=False,
            transcript_path=paths.transcript,
            reason="transcript produced no cues",
        )

    emphasis: set[str] = set()
    if settings.caption_emphasis != "off":
        emphasis = pick_emphasis_words(cues, keywords=settings.chat_keywords)

    style, anchor = caption_anchor(
        style, prefer_top=prefer_top and settings.caption_safe_area == "auto"
    )
    write_ass(
        cues,
        style,
        path=paths.captions,
        width=settings.clip_target_width,
        height=settings.clip_target_height,
        emphasis=emphasis,
    )
    return CaptionResult(
        applied=True,
        cues=cues,
        captions_path=paths.captions,
        transcript_path=paths.transcript,
        style=style,
        anchor=anchor,
        emphasis=emphasis,
    )

