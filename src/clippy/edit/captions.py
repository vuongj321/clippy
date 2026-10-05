"""Caption stage for one edit.

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
from clippy.caption.emphasis import pick_emphasis_words_llm
from clippy.caption.styles import CaptionStyle, resolve_anchor, resolve_style
from clippy.config import Settings
from clippy.edit.plan import WARN_EMPHASIS_FALLBACK, CaptionsPlan, EditPaths, EditPlan

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
    emphasis_source: str = "none"
    emphasis_note: str | None = None

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
            emphasis_source=self.emphasis_source,
            reason=self.reason,
        )
        if self.emphasis_note:
            plan.add_warning(WARN_EMPHASIS_FALLBACK, self.emphasis_note)


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


def transcribe_window_words(
    media_path: Path,
    *,
    settings: Settings,
    cache_path: Path | None = None,
    force: bool = False,
) -> TranscriptPayload | None:
    """
    Word-timed ASR for a review window, cached on disk. Never raises.

    Boundary detection needs word timings *before* the clip is cut, so this transcribes the
    review window rather than the (not yet extracted) trimmed clip. Returns None when
    there is no API key, the media is missing, or the endpoint fails, leaving the caller to
    degrade to whatever evidence it already has.
    """
    api_key = (settings.openai_api_key or "").strip()
    if not api_key or not media_path.exists():
        return None

    if cache_path is not None and cache_path.exists() and not force:
        try:
            cached = TranscriptPayload.from_dict(
                json.loads(cache_path.read_text(encoding="utf-8"))
            )
        except Exception:
            logger.warning("Ignoring unreadable boundary transcript %s", cache_path)
        else:
            if cached.segments:
                logger.info("Reusing cached boundary transcript %s", cache_path)
                return cached

    try:
        payload = transcribe_words(
            media_path,
            api_key=api_key,
            base_url=settings.openai_base_url,
            model=settings.asr_model,
            word_timestamps=True,
        )
    except Exception:
        logger.exception("Boundary window transcription failed for %s", media_path)
        return None

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")
    return payload


def _resolve_emphasis(
    cues: list[Cue],
    *,
    mode: str,
    settings: Settings,
) -> tuple[set[str], str, str | None]:
    """
    Pick the emphasised words for the requested `caption_emphasis` mode.

    `heuristic` is free and deterministic (`caption.align`). `llm` asks the caption model and
    falls back to the heuristic when the answer is unusable, so a reviewer who asked for model
    emphasis still gets emphasised words rather than none - the fallback is recorded on the
    plan as a warning. `off` disables emphasis entirely.
    """
    if mode == "off":
        return set(), "off", None
    if mode != "llm":
        return pick_emphasis_words(cues, keywords=settings.chat_keywords), "heuristic", None

    requested, failure = pick_emphasis_words_llm(
        cues,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.caption_model,
    )
    if requested:
        return requested, "llm", None
    fallback = pick_emphasis_words(cues, keywords=settings.chat_keywords)
    return fallback, "heuristic_fallback", failure


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

    `prefer_top` comes from composition: when the subject sits in the lower band
    of the frame, captions move up so they never cover it. It only applies to
    `caption_safe_area: auto`; an explicit `top`/`middle`/`bottom` is a reviewer choice and
    wins over the frame evidence.

    The style and the emphasis mode are read from `plan.captions` rather than straight from
    `Settings`, because `build_plan` has already folded any `--caption-style` /
    `--caption-emphasis` override into the plan.
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

    style = resolve_style(plan.captions.style or settings.caption_style, settings)
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

    emphasis, emphasis_source, emphasis_note = _resolve_emphasis(
        cues, mode=plan.captions.emphasis or settings.caption_emphasis, settings=settings
    )

    style, anchor = resolve_anchor(
        style, safe_area=settings.caption_safe_area, prefer_top=prefer_top
    )
    write_ass(
        cues,
        style,
        path=paths.captions,
        # The plan is the record of the requested size, so a target-size override keeps the ASS
        # PlayRes (what libass scales against) in step with the canvas it is burned onto.
        width=plan.layout.width or settings.clip_target_width,
        height=plan.layout.height or settings.clip_target_height,
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
        emphasis_source=emphasis_source,
        emphasis_note=emphasis_note,
    )

