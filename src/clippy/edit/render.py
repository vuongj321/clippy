"""Render stages for one edit (Phase 2).

M3 scope: cut ``base.mp4`` from the source at the planned boundaries and verify what
was actually produced. Later milestones add dead-air removal, layout, caption burn-in
and the final encode here.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from clippy.audio.intensity import probe_duration_seconds
from clippy.config import Settings
from clippy.edit.deadair import (
    DeadAirCut,
    Span,
    build_deadair_filter,
    detect_silences,
    plan_deadair,
    segments_for_render,
)
from clippy.edit.plan import DeadAirPlan, EditPaths, EditPlan
from clippy.edit.layouts import BLUR_SIGMA, CompositionPlan, plan_layout
from clippy.edit.plan import WARN_LAYOUT_PENDING
from clippy.edit.track import track_subject
from clippy.extract.ffmpeg_cut import extract_window

logger = logging.getLogger(__name__)

WARN_EXTRACT_SHORT = "extract_short"
WARN_DEADAIR_LENGTH = "deadair_length_mismatch"
WARN_COMPOSE_LENGTH = "compose_length_mismatch"


def extract_base(
    plan: EditPlan,
    paths: EditPaths,
    *,
    settings: Settings,
    force: bool = False,
) -> Path:
    """
    Cut the planned range out of the source into ``base.mp4``.

    Cached: an existing ``base.mp4`` is reused unless ``force``. A cut that comes back
    short (source ended early, or a seek landed past the end) is reported and the plan
    is clamped to what was actually produced, so nothing downstream can silently assume
    frames that are not on disk.
    """
    source = Path(plan.source_path)
    if not source.exists():
        raise FileNotFoundError(f"source media not found: {source}")
    paths.ensure_root()
    if paths.base.exists() and not force:
        logger.info("Reusing cached %s", paths.base)
        return paths.base

    start, end = plan.source_range()
    expected = max(0.0, end - start)
    if expected <= 0:
        raise ValueError(f"plan has no duration: start={start} end={end}")

    extract_window(
        source,
        paths.base,
        start_seconds=start,
        duration_seconds=expected,
        ffmpeg_path=settings.ffmpeg_path,
        crf=settings.intermediate_crf,
        preset=settings.render_preset,
    )
    actual = probe_duration_seconds(paths.base, ffprobe_path=settings.ffprobe_path)
    if actual <= 0:
        raise RuntimeError(f"ffmpeg produced an empty clip: {paths.base}")

    if actual + settings.extract_duration_tolerance_seconds < expected:
        clamped_end = start + actual
        plan.bounds = replace(
            plan.bounds,
            end=clamped_end,
            main_ts=min(plan.bounds.main_ts, clamped_end),
            hook_ts=min(plan.bounds.hook_ts, clamped_end),
            payoff_ts=min(plan.bounds.payoff_ts, clamped_end),
        )
        plan.add_warning(
            WARN_EXTRACT_SHORT,
            (
                f"source produced only {actual:.2f}s of the planned {expected:.2f}s "
                f"from {start:.2f}s; bounds clamped to the media on disk"
            ),
        )
        logger.warning(
            "Extract short for candidate %s: %.2fs of %.2fs",
            plan.candidate_id,
            actual,
            expected,
        )

    return paths.base


def escape_filter_path(path: str) -> str:
    """
    Escape a filesystem path for use as an ffmpeg filter *option* value.

    ffmpeg unescapes the filtergraph string, then splits options on `:`, so a Windows
    drive colon must survive the first pass: it needs a **double** backslash. Verified on
    ffmpeg 9.0.2 here - `fontsdir=C:/...` and `fontsdir=C\\:/...` (single) both fail with
    "No option name near '/Windows/Fonts'", while `C\\\\:/...` parses.
    """
    return path.replace("\\", "/").replace(":", "\\\\:")


def build_composition_filter(
    layout: CompositionPlan,
    *,
    has_audio: bool = True,
    ass_filename: str | None = None,
    fonts_dir: str = "",
    blur_sigma: int = BLUR_SIGMA,
) -> str:
    """
    Build the `filter_complex` for the vertical canvas.

    One canvas is composed per layout segment (each with static layer rectangles), the
    segments are concatenated, and captions are burned last so they always sit on top.
    Audio is trimmed per segment with the video so a moving crop never desyncs sound.
    """
    parts: list[str] = []
    labels: list[str] = []
    for index, segment in enumerate(layout.segments):
        if segment.duration <= 0:
            continue
        start, end = segment.start, segment.end
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[sv{index}]"
        )
        if has_audio:
            parts.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[sa{index}]"
            )

        canvas = (
            f"color=c=black:s={layout.width}x{layout.height}:r={layout.fps}"
            f":d={segment.duration:.3f}[bg{index}]"
        )
        parts.append(canvas)
        current = f"bg{index}"

        for layer_index, layer in enumerate(sorted(segment.layers, key=lambda item: item.z)):
            src_x, src_y, src_w, src_h = (max(1.0, round(value)) for value in layer.src)
            dst_x, dst_y, dst_w, dst_h = (
                max(1.0, round(value)) for value in layer.dst
            )
            chain = (
                f"[sv{index}]crop={src_w:.0f}:{src_h:.0f}:{src_x:.0f}:{src_y:.0f},"
                f"scale={dst_w:.0f}:{dst_h:.0f}:flags=lanczos,setsar=1"
            )
            if layer.kind == "blur":
                chain += f",boxblur={blur_sigma}:1"
            parts.append(f"{chain}[ly{index}_{layer_index}]")
            out = f"mix{index}_{layer_index}"
            parts.append(
                f"[{current}][ly{index}_{layer_index}]overlay="
                f"{round(layer.dst[0])}:{round(layer.dst[1])}:shortest=1[{out}]"
            )
            current = out

        parts.append(f"[{current}]format=yuv420p[comp{index}]")
        labels.append(f"[comp{index}]")
        if has_audio:
            labels[-1] += f"[sa{index}]"

    if not labels:
        return ""

    if has_audio:
        parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=1[catv][cata]")
    else:
        parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[catv]")

    if ass_filename:
        ass = f"ass=filename={escape_filter_path(ass_filename)}"
        if fonts_dir:
            ass += f":fontsdir={escape_filter_path(fonts_dir)}"
        parts.append(f"[catv]{ass}[vout]")
    else:
        parts.append("[catv]null[vout]")
    if has_audio:
        parts.append("[cata]anull[aout]")
    return ";\n".join(parts)


def _require_ffmpeg(ffmpeg_path: str) -> str:
    ffmpeg = shutil.which(ffmpeg_path) or (
        ffmpeg_path if Path(ffmpeg_path).exists() else None
    )
    if not ffmpeg:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    return ffmpeg


def probe_streams(path: Path, *, ffprobe_path: str = "ffprobe") -> tuple[bool, bool]:
    """(has_video, has_audio): a filter graph must only map streams that exist."""
    probe = shutil.which(ffprobe_path) or (
        ffprobe_path if Path(ffprobe_path).exists() else None
    )
    if not probe:
        raise RuntimeError(
            f"ffprobe not found ({ffprobe_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    cmd = [
        probe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    payload = json.loads(proc.stdout or "{}")
    kinds = {stream.get("codec_type") for stream in payload.get("streams", [])}
    return ("video" in kinds, "audio" in kinds)


def _render_segments(
    base: Path,
    output: Path,
    *,
    segments: Sequence[tuple[float, float, float]],
    settings: Settings,
    force: bool,
) -> Path:
    """Concatenate `(start, end, speed)` segments into `output` in one pass."""
    if output.exists() and not force:
        logger.info("Reusing cached %s", output)
        return output
    has_video, has_audio = probe_streams(base, ffprobe_path=settings.ffprobe_path)
    if not has_video:
        raise RuntimeError(f"base clip has no video stream: {base}")
    filter_graph = build_deadair_filter(segments, has_audio=has_audio)
    if not filter_graph:
        raise ValueError("no renderable segments")

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _require_ffmpeg(settings.ffmpeg_path),
        "-y",
        "-i",
        str(base),
        "-filter_complex",
        filter_graph,
        "-map",
        "[vout]",
    ]
    if has_audio:
        cmd += ["-map", "[aout]"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        settings.render_preset,
        "-crf",
        str(settings.render_crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart", str(output)]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg dead-air render failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty output: {output}")
    return output


def apply_deadair(
    plan: EditPlan,
    paths: EditPaths,
    *,
    settings: Settings,
    protect: Span | None = None,
    words: Sequence[Span] | None = None,
    force: bool = False,
) -> DeadAirCut:
    """
    Remove dead air from ``base.mp4`` into ``trimmed.mp4`` and record what was cut.

    When nothing is cuttable the trimmed artifact mirrors the base clip, so every later
    stage can rely on one timeline instead of branching on whether a cut happened.
    """
    base = paths.base
    if not base.exists():
        raise FileNotFoundError(f"base clip not found: {base}")
    paths.ensure_root()

    duration = probe_duration_seconds(base, ffprobe_path=settings.ffprobe_path)
    if duration <= 0:
        raise RuntimeError(f"base clip has no duration: {base}")

    silences = detect_silences(
        base,
        noise_db=settings.deadair_noise_db,
        min_gap_seconds=settings.deadair_min_gap_seconds,
        ffmpeg_path=settings.ffmpeg_path,
        end_hint=duration,
    )
    cut = plan_deadair(
        silences=silences,
        duration=duration,
        settings=settings,
        protect=protect,
        words=words,
    )

    plan.deadair = DeadAirPlan(
        enabled=settings.deadair_enabled,
        applied=cut.applied,
        mode=cut.mode,
        keep_segments=[[round(start, 3), round(end, 3)] for start, end in cut.keep_segments],
        removed_seconds=round(cut.removed_seconds, 3),
        reason="; ".join(cut.notes) if cut.notes else None,
    )
    for note in cut.notes:
        logger.info("Dead air (candidate %s): %s", plan.candidate_id, note)

    if not cut.applied:
        if force or not paths.trimmed.exists():
            shutil.copy2(base, paths.trimmed)
        return cut

    _render_segments(
        base,
        paths.trimmed,
        segments=segments_for_render(
            cut, speed_factor=settings.deadair_speed_factor
        ),
        settings=settings,
        force=force,
    )
    actual = probe_duration_seconds(paths.trimmed, ffprobe_path=settings.ffprobe_path)
    expected = max(0.0, cut.duration - cut.removed_seconds)
    if abs(actual - expected) > max(1.0, settings.extract_duration_tolerance_seconds * 2):
        plan.add_warning(
            WARN_DEADAIR_LENGTH,
            f"trimmed clip is {actual:.2f}s but {expected:.2f}s was expected",
        )
        logger.warning(
            "Dead-air length mismatch for candidate %s: %.2fs vs %.2fs",
            plan.candidate_id,
            actual,
            expected,
        )
    return cut


def probe_dimensions(path: Path, *, ffprobe_path: str = "ffprobe") -> tuple[int, int]:
    """Width and height of the first video stream."""
    probe = shutil.which(ffprobe_path) or (
        ffprobe_path if Path(ffprobe_path).exists() else None
    )
    if not probe:
        raise RuntimeError(
            f"ffprobe not found ({ffprobe_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    cmd = [
        probe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    streams = json.loads(proc.stdout or "{}").get("streams") or [{}]
    stream = streams[0] if streams else {}
    return int(stream.get("width") or 0), int(stream.get("height") or 0)


def compose_vertical(
    plan: EditPlan,
    paths: EditPaths,
    *,
    settings: Settings,
    duration: float,
    ass_path: Path | None = None,
    force: bool = False,
) -> CompositionPlan:
    """
    Render ``vertical.mp4``: the vertical canvas, the chosen layout, captions burned in.

    The layout is planned from the real subject track of the trimmed clip, written to
    ``layout.json`` and summarised on the plan so a reviewer can see - and later override
    - the framing that was used. Video is encoded exactly once here; audio passes through
    untouched, because the loudness pass copies the video stream.
    """
    if not paths.trimmed.exists():
        raise FileNotFoundError(f"trimmed clip not found: {paths.trimmed}")

    width, height = probe_dimensions(paths.trimmed, ffprobe_path=settings.ffprobe_path)
    track = track_subject(
        paths.trimmed,
        duration_seconds=duration,
        settings=settings,
        ffmpeg_path=settings.ffmpeg_path,
    )
    layout = plan_layout(
        requested=settings.layout_strategy,
        source_width=width,
        source_height=height,
        width=settings.clip_target_width,
        height=settings.clip_target_height,
        fps=settings.clip_fps or 30,
        duration=duration,
        settings=settings,
        track=track,
        facecam_box=settings.parsed_facecam_box(),
    )
    paths.layout.write_text(json.dumps(layout.to_dict(), indent=2), encoding="utf-8")

    plan.layout.strategy = layout.strategy
    plan.layout.resolved_strategy = layout.resolved_strategy
    plan.layout.width = layout.width
    plan.layout.height = layout.height
    plan.layout.fps = layout.fps
    plan.layout.upscale_factor = layout.upscale_factor
    plan.layout.track_backend = settings.layout_track_backend
    plan.layout.layers = [
        layer.to_dict() for segment in layout.segments for layer in segment.layers
    ]
    # The plan was built before the strategy was known, so drop that now-stale warning:
    # plan.json must not keep claiming something the layout block itself disproves.
    plan.warnings = [
        warning
        for warning in plan.warnings
        if warning.code != WARN_LAYOUT_PENDING
    ]
    for code in layout.warnings:
        plan.add_warning(code, f"layout: {code.replace('_', ' ')}")

    if paths.vertical.exists() and not force:
        logger.info("Reusing cached %s", paths.vertical)
        return layout

    has_video, has_audio = probe_streams(
        paths.trimmed, ffprobe_path=settings.ffprobe_path
    )
    if not has_video:
        raise RuntimeError(f"trimmed clip has no video stream: {paths.trimmed}")

    ass_filename = ass_path.name if ass_path is not None and ass_path.exists() else None
    filter_graph = build_composition_filter(
        layout,
        has_audio=has_audio,
        ass_filename=ass_filename,
        fonts_dir=settings.caption_fonts_dir,
    )
    if not filter_graph:
        raise ValueError("layout produced no renderable segments")

    cmd = [
        _require_ffmpeg(settings.ffmpeg_path),
        "-y",
        "-i",
        str(paths.trimmed),
        "-filter_complex",
        filter_graph,
        "-map",
        "[vout]",
    ]
    if has_audio:
        cmd += ["-map", "[aout]"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        settings.render_preset,
        "-crf",
        str(settings.render_crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(settings.clip_fps or 30),
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart", str(paths.vertical)]

    # cwd is the artifact directory so the `ass` filter can use a plain relative
    # filename; Windows drive colons inside filter paths are a classic ffmpeg trap.
    proc = subprocess.run(cmd, capture_output=True, check=False, cwd=str(paths.root))
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg composition failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    if not paths.vertical.exists() or paths.vertical.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty output: {paths.vertical}")

    actual = probe_duration_seconds(paths.vertical, ffprobe_path=settings.ffprobe_path)
    if abs(actual - duration) > max(1.0, settings.extract_duration_tolerance_seconds * 2):
        plan.add_warning(
            WARN_COMPOSE_LENGTH,
            f"vertical clip is {actual:.2f}s but {duration:.2f}s was expected",
        )
    return layout
