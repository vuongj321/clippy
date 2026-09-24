from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from clippy.audio.intensity import detect_audio_spikes
from clippy.caption.generate import annotate_extracted_candidate
from clippy.caption.reason import format_extract_reason
from clippy.chat.models import ChatMessage
from clippy.chat.signals import detect_chat_signals
from clippy.config import Settings
from clippy.detect.detector import coalesce_detections, combine_signal_events
from clippy.extract.ffmpeg_cut import extract_window, within_disk_budget
from clippy.ingest.live import LiveIngestSession, create_live_session
from clippy.ingest.vod import VodIngestResult, ingest_local_vod
from clippy.store.db import Database

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AnnotationJob:
    candidate_id: int
    media_path: Path
    source_ts: float
    score: float


def run_vod_pipeline(
    *,
    media_path: Path,
    chat_path: Path,
    streamer_login: str,
    settings: Settings,
    display_name: str | None = None,
    source_url: str | None = None,
    vod_id: str | None = None,
) -> dict:
    settings.ensure_dirs()
    db = Database(settings.resolved_db_path())
    ingest = ingest_local_vod(
        media_path,
        chat_path,
        streamer_login=streamer_login,
        display_name=display_name,
        source_url=source_url,
        vod_id=vod_id,
    )
    return _process_vod_like(ingest, settings=settings, db=db)


def _process_vod_like(
    ingest: VodIngestResult,
    *,
    settings: Settings,
    db: Database,
) -> dict:
    streamer = db.get_or_create_streamer(ingest.streamer_login, ingest.display_name)
    stream = db.create_stream(
        streamer.id,
        "vod",
        source_url=ingest.source_url,
        vod_id=ingest.vod_id,
        media_path=str(ingest.media_path),
    )

    chat_events = detect_chat_signals(
        ingest.chat,
        window_seconds=settings.chat_window_seconds,
        baseline_seconds=settings.chat_baseline_seconds,
        spike_multiplier=settings.chat_spike_multiplier,
        min_rate=settings.chat_min_rate,
        keywords=settings.chat_keywords,
        keyword_score=settings.chat_keyword_score,
        spike_score=settings.chat_spike_score,
    )
    logger.info("Chat signal events: %d", len(chat_events))

    audio_events = detect_audio_spikes(
        ingest.media_path,
        frame_seconds=settings.audio_frame_seconds,
        baseline_seconds=settings.audio_baseline_seconds,
        spike_multiplier=settings.audio_spike_multiplier,
        min_rms=settings.audio_min_rms,
        spike_score=settings.audio_spike_score,
        ffmpeg_path=settings.ffmpeg_path,
    )
    logger.info("Audio signal events: %d", len(audio_events))

    raw = combine_signal_events(chat_events, audio_events)
    coalesced = coalesce_detections(raw, gap_seconds=settings.coalesce_gap_seconds)
    logger.info("Coalesced candidates: %d", len(coalesced))

    media_dir = settings.resolved_media_dir()
    created = 0
    skipped_disk = 0
    extracted_jobs: list[_AnnotationJob] = []

    for item in coalesced:
        reason = format_extract_reason(item.signals)
        if not within_disk_budget(media_dir, settings.disk_budget_gb):
            skipped_disk += 1
            logger.warning("Disk budget exceeded; skipping remaining extracts")
            db.create_candidate(
                stream.id,
                item.ts,
                settings.pre_context_seconds,
                settings.post_context_seconds,
                item.signals,
                item.score,
                media_path=None,
                extract_reason=reason,
            )
            continue

        candidate = db.create_candidate(
            stream.id,
            item.ts,
            settings.pre_context_seconds,
            settings.post_context_seconds,
            item.signals,
            item.score,
            media_path=None,
            extract_reason=reason,
        )
        start = max(0.0, item.ts - settings.pre_context_seconds)
        duration = settings.pre_context_seconds + settings.post_context_seconds
        out = media_dir / f"candidate_{candidate.id}.mp4"
        try:
            extract_window(
                ingest.media_path,
                out,
                start_seconds=start,
                duration_seconds=duration,
                ffmpeg_path=settings.ffmpeg_path,
            )
            db.update_candidate_media(candidate.id, str(out))
            created += 1
            extracted_jobs.append(
                _AnnotationJob(
                    candidate_id=candidate.id,
                    media_path=out,
                    source_ts=item.ts,
                    score=item.score,
                )
            )
        except Exception:
            logger.exception("Failed to extract candidate %s", candidate.id)

    annotated = _run_caption_pass(
        db,
        extracted_jobs,
        chat=ingest.chat,
        settings=settings,
        streamer_display_name=ingest.display_name,
        streamer_login=ingest.streamer_login,
    )

    return {
        "stream_id": stream.id,
        "streamer": ingest.streamer_login,
        "chat_events": len(chat_events),
        "audio_events": len(audio_events),
        "candidates": len(coalesced),
        "extracted": created,
        "skipped_disk": skipped_disk,
        "annotated": annotated,
    }


def _select_annotation_jobs(
    jobs: list[_AnnotationJob],
    *,
    max_per_run: int,
) -> list[_AnnotationJob]:
    if max_per_run <= 0 or not jobs:
        return []
    ranked = sorted(jobs, key=lambda job: (-job.score, job.source_ts, job.candidate_id))
    return ranked[:max_per_run]


def _run_caption_pass(
    db: Database,
    jobs: list[_AnnotationJob],
    *,
    chat: list[ChatMessage],
    settings: Settings,
    streamer_display_name: str,
    streamer_login: str,
) -> int:
    if not (settings.openai_api_key or "").strip():
        logger.info("CLIPPY_OPENAI_API_KEY is unset; skipping ASR and caption generation")
        return 0
    selected = _select_annotation_jobs(jobs, max_per_run=settings.caption_max_per_run)
    if not selected:
        if jobs and settings.caption_max_per_run <= 0:
            logger.info(
                "caption_max_per_run=%s; skipping caption generation",
                settings.caption_max_per_run,
            )
        return 0
    if len(selected) < len(jobs):
        logger.info(
            "Annotating %d of %d extracted candidates (caption_max_per_run=%s)",
            len(selected),
            len(jobs),
            settings.caption_max_per_run,
        )
    annotated = 0
    for job in selected:
        if _save_caption(
            db,
            job.candidate_id,
            media_path=job.media_path,
            chat=chat,
            source_ts=job.source_ts,
            settings=settings,
            streamer_display_name=streamer_display_name,
            streamer_login=streamer_login,
        ):
            annotated += 1
    return annotated


def _save_caption(
    db: Database,
    candidate_id: int,
    *,
    media_path: Path,
    chat: list[ChatMessage],
    source_ts: float,
    settings: Settings,
    streamer_display_name: str,
    streamer_login: str,
) -> bool:
    try:
        caption, transcript = annotate_extracted_candidate(
            media_path=media_path,
            chat=chat,
            source_ts=source_ts,
            pre_context_seconds=settings.pre_context_seconds,
            post_context_seconds=settings.post_context_seconds,
            streamer_display_name=streamer_display_name,
            streamer_login=streamer_login,
            settings=settings,
        )
        if caption or transcript:
            db.update_candidate_caption(
                candidate_id,
                caption=caption,
                transcript=transcript,
            )
        return bool(caption)
    except Exception:
        logger.exception("Failed to annotate candidate %s", candidate_id)
        return False


def run_live_pipeline(
    *,
    channel_login: str,
    settings: Settings,
    duration_seconds: float = 300.0,
    poll_seconds: float = 15.0,
) -> dict:
    """
    Record live for `duration_seconds`, then run the same detect/extract path
    on the recorded file + collected chat (batch end-of-session for MVP simplicity).
    """
    if not settings.twitch_irc_nick or not settings.twitch_irc_oauth:
        raise RuntimeError(
            "Live mode requires CLIPPY_TWITCH_IRC_NICK and CLIPPY_TWITCH_IRC_OAUTH"
        )

    settings.ensure_dirs()
    db = Database(settings.resolved_db_path())
    session = create_live_session(
        channel_login,
        buffer_dir=settings.resolved_buffer_dir() / channel_login.lower(),
        nick=settings.twitch_irc_nick,
        oauth_token=settings.twitch_irc_oauth,
    )
    session.start()
    logger.info(
        "Live session started for %s; recording for %.0fs",
        channel_login,
        duration_seconds,
    )
    try:
        deadline = time.time() + duration_seconds
        while time.time() < deadline:
            time.sleep(min(poll_seconds, max(1.0, deadline - time.time())))
            logger.info(
                "Live elapsed=%.0fs chat_msgs=%d",
                session.elapsed(),
                len(session.chat),
            )
    finally:
        session.stop()

    if not session.output_path.exists() or session.output_path.stat().st_size == 0:
        raise RuntimeError(
            "Live recording produced no media. Is the channel live? Is streamlink working?"
        )

    # Persist chat snapshot next to media for reproducibility
    chat_path = session.output_path.with_suffix(".chat.json")
    _write_chat_json(chat_path, session.chat)

    ingest = VodIngestResult(
        media_path=session.output_path,
        chat=session.chat,
        streamer_login=channel_login.lower(),
        display_name=channel_login,
        source_url=f"https://twitch.tv/{channel_login}",
    )
    # Mark stream mode as live in DB by wrapping process
    result = _process_vod_like(ingest, settings=settings, db=db)
    # Fix mode to live
    with db.connection() as conn:
        conn.execute(
            "UPDATE streams SET mode = 'live' WHERE id = ?",
            (result["stream_id"],),
        )
    result["mode"] = "live"
    result["chat_path"] = str(chat_path)
    return result


def _write_chat_json(path: Path, messages: list) -> None:
    import json

    payload = [
        {"ts": m.ts, "user": m.user, "text": m.text, "emotes": m.emotes}
        for m in messages
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
