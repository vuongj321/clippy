"""Edit pipeline orchestration (Phase 2).

M0 scope: resolve candidates, build and persist an ``EditPlan``, and record a
``renders`` row of kind ``plan``. The encoding stages land in M2-M9 and raise
``NotImplementedError`` when ``dry_run=False``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from clippy.audio.intensity import probe_duration_seconds
from clippy.caption.chat_context import build_chat_context
from clippy.chat.models import ChatMessage, load_chat_json
from clippy.config import Settings
from clippy.edit.audio import normalize_audio
from clippy.edit.boundaries import detect_bounds, evidence_from_chat
from clippy.edit.captions import generate_captions
from clippy.edit.metadata import capture_thumbnail, generate_metadata
from clippy.edit.plan import (
    EditOverrides,
    EditPaths,
    EditPlan,
    MetadataPlan,
    build_plan,
)
from clippy.edit.render import apply_deadair, compose_vertical, extract_base, plan_composition
from clippy.store.db import Candidate, Database, Stream, Streamer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EditJob:
    """One candidate plus everything needed to reach its source footage."""

    candidate: Candidate
    stream: Stream
    source_path: Path
    streamer: Streamer | None = None


def select_edit_jobs(jobs: list[EditJob], *, max_per_run: int) -> list[EditJob]:
    """
    Rank jobs and cap the batch.

    Ranking matches the Phase 1 annotation pass: highest detection score first,
    then earliest timestamp, then id, so a capped run is deterministic.
    """
    if max_per_run <= 0 or not jobs:
        return []
    ranked = sorted(
        jobs,
        key=lambda job: (-job.candidate.score, job.candidate.source_ts, job.candidate.id),
    )
    return ranked[:max_per_run]


def resolve_jobs(
    db: Database,
    *,
    candidate_ids: list[int] | None = None,
    stream_id: int | None = None,
    status: str | None = None,
    source_path: Path | None = None,
) -> tuple[list[EditJob], list[str]]:
    """Turn a selection into jobs. Per-row problems become warnings, not failures."""
    warnings: list[str] = []

    if candidate_ids:
        candidates: list[Candidate] = []
        for candidate_id in candidate_ids:
            candidate = db.get_candidate(candidate_id)
            if candidate is None:
                warnings.append(f"candidate {candidate_id} not found")
                continue
            candidates.append(candidate)
    else:
        candidates = db.list_candidates(stream_id=stream_id, status=status, limit=10_000)

    jobs: list[EditJob] = []
    for candidate in candidates:
        stream = db.get_stream(candidate.stream_id)
        if stream is None:
            warnings.append(
                f"candidate {candidate.id}: stream {candidate.stream_id} not found"
            )
            continue
        resolved = source_path or (Path(stream.media_path) if stream.media_path else None)
        if resolved is None:
            warnings.append(f"candidate {candidate.id}: stream has no source media path")
            continue
        jobs.append(
            EditJob(
                candidate=candidate,
                stream=stream,
                source_path=resolved,
                streamer=db.get_streamer(stream.streamer_id),
            )
        )
    return jobs, warnings


def _thumbnail_time(cues: Sequence[Any], duration: float) -> float:
    """Middle of the longest cue: usually the moment worth showing as a cover."""
    if cues:
        longest = max(cues, key=lambda cue: cue.duration)
        return max(0.0, min(duration, longest.start + longest.duration / 2.0))
    return max(0.0, duration / 2.0)


def _existing_plan(settings: Settings, job: EditJob) -> EditPlan | None:
    """
    The plan a candidate already has, if it was built from the same source.

    A re-render without chat evidence cannot re-derive a chat-driven cut; reusing the recorded plan
    keeps the boundaries the reviewer already saw. A different source path means the recorded bounds
    describe footage that is no longer there, so they are discarded.
    """
    paths = EditPaths.for_candidate(settings, job.candidate.id)
    if not paths.plan.exists():
        return None
    try:
        existing = EditPlan.load(paths.plan)
    except Exception:
        logger.warning("Ignoring unreadable plan %s", paths.plan)
        return None
    if existing.source_path != str(job.source_path):
        return None
    return existing


def run_edit_pipeline(
    *,
    settings: Settings,
    candidate_ids: list[int] | None = None,
    stream_id: int | None = None,
    status: str | None = None,
    max_per_run: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    source_path: Path | None = None,
    source_offset_seconds: float | None = None,
    overrides: EditOverrides | None = None,
    chat_path: Path | None = None,
) -> dict:
    """
    Plan (and later render) short-form edits for a candidate selection.

    ``dry_run=True`` writes only ``plan.json`` plus a ``renders`` row, so it never
    touches ffmpeg or the network. ``chat_path`` supplies chat evidence so boundaries
    come from the chat reaction curve instead of the Phase 1 window.
    """
    settings.ensure_dirs()
    db = Database(settings.resolved_db_path())

    jobs, warnings = resolve_jobs(
        db,
        candidate_ids=candidate_ids,
        stream_id=stream_id,
        status=status,
        source_path=source_path,
    )
    cap = settings.edit_max_per_run if max_per_run is None else max_per_run
    selected = select_edit_jobs(jobs, max_per_run=cap)

    chat_messages: list[ChatMessage] = []
    if chat_path is not None:
        chat_messages = load_chat_json(chat_path)
        logger.info("Chat evidence loaded: %d messages", len(chat_messages))
    chat_times = [message.ts for message in chat_messages]

    planned = 0
    skipped = 0
    extracted = 0
    captioned = 0
    failed = 0
    deadair_removed = 0.0
    evidence_bounds = 0
    edit_dirs: list[str] = []

    for job in selected:
        if not job.source_path.exists():
            skipped += 1
            message = f"source media missing: {job.source_path}"
            warnings.append(f"candidate {job.candidate.id}: {message}")
            db.create_render(job.candidate.id, kind="plan", status="failed", error=message)
            continue

        decision = None
        if chat_times:
            window_start = max(
                0.0, job.candidate.source_ts - job.candidate.pre_context_seconds
            )
            window_end = job.candidate.source_ts + job.candidate.post_context_seconds
            decision = detect_bounds(
                candidate=job.candidate,
                settings=settings,
                evidence=evidence_from_chat(
                    chat_times, start=window_start, end=window_end
                ),
            )
            if decision.bounds.method != "phase1_window":
                evidence_bounds += 1

        # A re-plan with no chat evidence cannot re-derive the cut, and the Phase 1 window would
        # quietly replace a chat-derived one - the usual case for the UI render form, which passes
        # no chat unless the reviewer names a dump. Keep the boundaries the candidate already has
        # when the source is unchanged rather than silently moving the clip.
        if decision is not None:
            bounds = decision.bounds
            boundary_evidence = {
                "evidence": decision.evidence.to_dict(),
                "adjustments": list(decision.adjustments),
            }
        else:
            previous = _existing_plan(settings, job)
            bounds = previous.bounds if previous is not None else None
            boundary_evidence = (
                previous.boundary_evidence if previous is not None else None
            )

        plan = build_plan(
            candidate=job.candidate,
            stream=job.stream,
            source_path=job.source_path,
            settings=settings,
            overrides=overrides,
            bounds=bounds,
            boundary_evidence=boundary_evidence,
        )
        if source_offset_seconds is not None:
            plan.source_offset_seconds = source_offset_seconds

        paths = EditPaths.for_candidate(settings, job.candidate.id)
        paths.ensure_root()
        plan.set_stage("planned")
        plan.save(paths.plan)

        render_status = "ok"
        render_error: str | None = None
        if not dry_run:
            try:
                extract_base(plan, paths, settings=settings, force=force)
                apply_deadair(
                    plan,
                    paths,
                    settings=settings,
                    protect=(
                        max(0.0, plan.bounds.main_ts - plan.bounds.start),
                        max(0.0, plan.bounds.payoff_ts - plan.bounds.start),
                    ),
                    force=force,
                )
                trimmed_duration = probe_duration_seconds(
                    paths.trimmed, ffprobe_path=settings.ffprobe_path
                )
                # Resolve the framing before the captions: `caption_safe_area: auto` can only
                # move the caption band away from the subject once the frame has been read.
                composition = plan_composition(
                    plan, paths, settings=settings, duration=trimmed_duration
                )
                caption_result = generate_captions(
                    plan,
                    paths,
                    settings=settings,
                    force=force,
                    prefer_top=composition.caption_prefer_top,
                )
                caption_result.commit(plan)
                compose_vertical(
                    plan,
                    paths,
                    settings=settings,
                    duration=trimmed_duration,
                    ass_path=caption_result.captions_path,
                    force=force,
                    layout=composition,
                )
                plan.set_stage("composed")
                plan.save(paths.plan)
                audio_result = normalize_audio(
                    paths.vertical, paths.final, settings=settings, force=force
                )
                plan.audio.applied = audio_result.applied
                plan.audio.used_two_pass = audio_result.used_two_pass
                plan.audio.measured = dict(audio_result.measured)
                plan.audio.reason = audio_result.reason

                transcript_text = " ".join(cue.text for cue in caption_result.cues)
                chat_context = build_chat_context(
                    chat_messages,
                    start=plan.bounds.start,
                    end=plan.bounds.end,
                    keywords=settings.chat_keywords,
                    max_messages=settings.caption_max_chat_messages,
                )
                thumbnail_time = _thumbnail_time(caption_result.cues, trimmed_duration)
                clip_metadata = generate_metadata(
                    transcript_text=transcript_text,
                    chat_context=chat_context,
                    streamer_display_name=(
                        job.streamer.display_name if job.streamer else job.stream.login
                    ),
                    streamer_login=(
                        job.streamer.login if job.streamer else job.stream.login
                    ),
                    settings=settings,
                    caption=job.candidate.caption,
                    thumbnail_time=thumbnail_time,
                    api_key=settings.openai_api_key,
                )
                paths.metadata.write_text(
                    json.dumps(clip_metadata.to_dict(), indent=2), encoding="utf-8"
                )
                plan.metadata = MetadataPlan(
                    enabled=settings.metadata_enabled,
                    source=clip_metadata.source,
                    title=clip_metadata.title,
                    description=clip_metadata.description,
                    hashtags=list(clip_metadata.hashtags),
                    thumbnail=None,
                    reason=clip_metadata.reason,
                )
                if settings.thumbnail_enabled:
                    try:
                        capture_thumbnail(
                            paths.final,
                            paths.thumbnail,
                            time_seconds=thumbnail_time,
                            ffmpeg_path=settings.ffmpeg_path,
                            overlay_text=(
                                clip_metadata.title
                                if settings.thumbnail_overlay_text
                                else None
                            ),
                        )
                        plan.metadata.thumbnail = str(paths.thumbnail)
                    except Exception as exc:
                        logger.warning(
                            "Thumbnail failed for candidate %s: %s",
                            job.candidate.id,
                            exc,
                        )

                plan.set_stage("complete")
                plan.save(paths.plan)
                extracted += 1
                deadair_removed += plan.deadair.removed_seconds
                if caption_result.applied:
                    captioned += 1
                db.create_render(
                    job.candidate.id,
                    kind="final",
                    path=str(paths.final),
                    captions_path=(
                        str(caption_result.captions_path)
                        if caption_result.captions_path
                        else None
                    ),
                    width=plan.layout.width,
                    height=plan.layout.height,
                    duration=trimmed_duration,
                    plan_json=json.dumps(plan.to_dict()),
                    status="ok",
                )
                db.update_candidate_edit(
                    job.candidate.id,
                    edit_status="rendered",
                    edited_media_path=str(paths.final),
                )
            except Exception as exc:  # one bad candidate must not stop the batch
                logger.exception("Render stages failed for candidate %s", job.candidate.id)
                render_status = "failed"
                render_error = f"render failed: {exc}"
                failed += 1

        db.create_render(
            job.candidate.id,
            kind="plan",
            path=str(paths.plan),
            plan_json=json.dumps(plan.to_dict()),
            status=render_status,
            error=render_error,
        )
        if render_status == "ok":
            edit_dirs.append(str(paths.root))
            planned += 1
            logger.info(
                "Planned edit for candidate %s (%.1fs clip, stage=%s) -> %s",
                job.candidate.id,
                plan.bounds.duration,
                plan.stage,
                paths.plan,
            )

    return {
        "dry_run": dry_run,
        "candidates": len(jobs),
        "selected": len(selected),
        "planned": planned,
        "extracted": extracted,
        "captioned": captioned,
        "skipped": skipped,
        "failed": skipped + failed,
        "deadair_removed_seconds": round(deadair_removed, 3),
        "max_per_run": cap,
        "chat_evidence": len(chat_times),
        "evidence_bounds": evidence_bounds,
        "edit_dirs": edit_dirs,
        "warnings": warnings,
    }

