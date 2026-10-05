"""Render stages for one edit (Phase 2).

M3 scope: cut ``base.mp4`` from the source at the planned boundaries and verify what
was actually produced. Later milestones add dead-air removal, layout, caption burn-in
and the final encode here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

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
from clippy.edit.layouts import (
    BLUR_SIGMA,
    WARN_CAPTION_BAND,
    WARN_FACECAM_AUTO_FAILED,
    WARN_FACECAM_DERIVED,
    CompositionPlan,
    facecam_box_from_track,
    gaming_panel_aspect,
    plan_layout,
)
from clippy.edit.plan import WARN_LAYOUT_PENDING
from clippy.edit.track import track_subject
from clippy.extract.ffmpeg_cut import extract_window

logger = logging.getLogger(__name__)

WARN_EXTRACT_SHORT = "extract_short"
WARN_DEADAIR_LENGTH = "deadair_length_mismatch"
WARN_COMPOSE_LENGTH = "compose_length_mismatch"

# Bump when the fingerprint of what `vertical.mp4` depends on changes shape, so an older
# sidecar can never be mistaken for a match.
COMPOSE_INPUTS_VERSION = 1


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
        str(settings.intermediate_crf),
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


def _require_trimmed(paths: EditPaths) -> None:
    if not paths.trimmed.exists():
        raise FileNotFoundError(f"trimmed clip not found: {paths.trimmed}")


def plan_composition(
    plan: EditPlan,
    paths: EditPaths,
    *,
    settings: Settings,
    duration: float,
) -> CompositionPlan:
    """
    Resolve the framing and record it, *before* anything burns captions into the frame.

    Tracking has to decode frames and the layout decides whether the caption band must move,
    so this runs ahead of the caption stage: `caption_safe_area: auto` can only follow the
    frame once the frame has been read. Writes ``layout.json`` and mirrors the result onto
    the plan, so the caption stage and the composition stage read one decision rather than
    resolving the strategy twice.
    """
    _require_trimmed(paths)

    width, height = probe_dimensions(paths.trimmed, ffprobe_path=settings.ffprobe_path)
    track = track_subject(
        paths.trimmed,
        duration_seconds=duration,
        settings=settings,
        ffmpeg_path=settings.ffmpeg_path,
    )
    # Every reviewer-facing choice is read from the PLAN, not `Settings`. `build_plan` resolved
    # `override or settings` once, so `plan.layout` already holds the strategy, size, bias and zoom
    # that were asked for; reading them back here is what makes a CLI/UI override reach the render
    # instead of being silently replaced by the config default.
    #
    # `facecam_box: auto` is resolved here because this is the first place a face track exists: the
    # box is derived from the clip's own tracked face, then falls through the exact same
    # `facecam_box` path as a typed one. A failure to derive leaves the box `None`, which the layout
    # reports as `facecam_box_auto_failed` and settles on `fit_blur`.
    canvas_width = plan.layout.width or settings.clip_target_width
    canvas_height = plan.layout.height or settings.clip_target_height
    facecam_box = plan.layout.facecam_box or settings.parsed_facecam_box()
    facecam_box_auto = False
    if facecam_box is None and settings.facecam_box_is_auto():
        facecam_box_auto = True
        facecam_box = facecam_box_from_track(
            track,
            source_width=width,
            source_height=height,
            pad=settings.facecam_pad,
            # Shape the tile to the panel it will fill, so `_crop_inside` does not re-crop it.
            aspect=gaming_panel_aspect(canvas_width, canvas_height),
        )
    layout = plan_layout(
        requested=plan.layout.strategy or settings.layout_strategy,
        source_width=width,
        source_height=height,
        width=canvas_width,
        height=canvas_height,
        fps=plan.layout.fps or (settings.clip_fps or 30),
        duration=duration,
        settings=settings,
        track=track,
        facecam_box=facecam_box,
        facecam_box_auto=facecam_box_auto,
        # `build_plan` already folded any style override in, so the caption band is measured
        # against the style that is actually going to be burned in.
        caption_style=plan.captions.style,
        crop_bias=plan.layout.crop_bias,
        zoom=plan.layout.zoom,
    )
    paths.layout.write_text(json.dumps(layout.to_dict(), indent=2), encoding="utf-8")
    _record_layout(plan, layout, settings=settings)
    return layout


def _record_layout(plan: EditPlan, layout: CompositionPlan, *, settings: Settings) -> None:
    """Mirror the resolved layout onto the plan so a reviewer reads one document."""
    plan.layout.strategy = layout.strategy
    plan.layout.resolved_strategy = layout.resolved_strategy
    plan.layout.width = layout.width
    plan.layout.height = layout.height
    plan.layout.fps = layout.fps
    plan.layout.upscale_factor = layout.upscale_factor
    plan.layout.track_backend = settings.layout_track_backend
    # Record the box the layout actually used (fractions of the source frame) and where it came
    # from. Writing the derived box back is also what keeps a re-render stable: `plan_composition`
    # reads it back and reuses it instead of deriving a new one.
    plan.layout.facecam_box = (
        [round(value, 4) for value in layout.facecam_box]
        if layout.facecam_box is not None
        else None
    )
    plan.layout.facecam_box_source = layout.facecam_box_source
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
        plan.add_warning(code, _layout_warning_message(code, layout, settings=settings))


def _layout_warning_message(code: str, layout: CompositionPlan, *, settings: Settings) -> str:
    """Turn a layout warning code into the sentence a reviewer can act on."""
    if code == WARN_CAPTION_BAND:
        return (
            "captions moved to the top band: subject motion sits in the bottom band for "
            f"{layout.caption_bottom_coverage:.0%} of frames (caption_avoid_ratio "
            f"{settings.caption_avoid_ratio:.0%})"
        )
    if code == WARN_FACECAM_DERIVED:
        box = layout.facecam_box
        rendered = (
            ",".join(f"{value:.3f}" for value in box) if box is not None else "unknown"
        )
        return (
            f"facecam box derived from face tracking ({rendered}); "
            "set facecam_box explicitly to pin it"
        )
    if code == WARN_FACECAM_AUTO_FAILED:
        return (
            "facecam_box: auto found no usable face, so the layout fell back to fit_blur; "
            "this needs layout_track_backend: opencv (and an OpenCV install)"
        )
    return f"layout: {code.replace('_', ' ')}"


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_stamp(path: Path) -> dict[str, Any] | None:
    """Cheap identity for a large media file: size plus mtime, not a full hash."""
    if not path.exists():
        return None
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _composition_inputs(
    paths: EditPaths,
    layout: CompositionPlan,
    *,
    settings: Settings,
    ass_path: Path | None,
) -> dict[str, Any]:
    """
    What ``vertical.mp4`` is made of: the pixels, the burned text and the framing.

    `trimmed.mp4` is identified by size and mtime (hashing a 100 MB clip on every run would
    cost more than it saves), while the two small text inputs are hashed - both are rewritten
    on every run whether or not anything changed, so their timestamps say nothing.
    """
    captions: str | None = None
    if ass_path is not None and ass_path.exists():
        captions = _text_digest(ass_path.read_text(encoding="utf-8"))
    return {
        "version": COMPOSE_INPUTS_VERSION,
        "trimmed": _file_stamp(paths.trimmed),
        "captions": captions,
        "layout": _text_digest(json.dumps(layout.to_dict(), sort_keys=True)),
        "encode": {
            "width": layout.width,
            "height": layout.height,
            "fps": layout.fps,
            "crf": settings.render_crf,
            "preset": settings.render_preset,
        },
    }


def _recorded_inputs(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Ignoring unreadable %s", path)
        return None
    return data if isinstance(data, dict) else None


def _cache_is_fresh(
    paths: EditPaths,
    layout: CompositionPlan,
    *,
    settings: Settings,
    ass_path: Path | None,
) -> tuple[bool, dict[str, Any]]:
    """
    Is the cached ``vertical.mp4`` still the render these inputs would produce?

    Returns the verdict and the fingerprint computed for this run, so the caller can record it
    after encoding without recomputing the hashes.
    """
    inputs = _composition_inputs(paths, layout, settings=settings, ass_path=ass_path)
    return _recorded_inputs(paths.compose_inputs) == inputs, inputs


def compose_vertical(
    plan: EditPlan,
    paths: EditPaths,
    *,
    settings: Settings,
    duration: float,
    ass_path: Path | None = None,
    force: bool = False,
    layout: CompositionPlan | None = None,
) -> CompositionPlan:
    """
    Render ``vertical.mp4``: the vertical canvas, the chosen layout, captions burned in.

    Pass ``layout`` when framing was already resolved (the pipeline resolves it before the
    caption stage so `caption_safe_area: auto` can follow the frame) to avoid tracking the
    clip twice; otherwise it is planned and recorded here. Either way ``layout.json`` holds
    the framing that was used, and video is encoded exactly once - audio passes through
    untouched, because the loudness pass copies the video stream.
    """
    _require_trimmed(paths)
    if layout is None:
        layout = plan_composition(plan, paths, settings=settings, duration=duration)

    fresh, inputs = _cache_is_fresh(paths, layout, settings=settings, ass_path=ass_path)
    if paths.vertical.exists() and not force and fresh:
        logger.info("Reusing cached %s (inputs unchanged)", paths.vertical)
        return layout
    if paths.vertical.exists() and not force:
        logger.info(
            "Cached %s no longer matches the current inputs; re-rendering",
            paths.vertical,
        )

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
        str(layout.fps or (settings.clip_fps or 30)),
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

    # Record what this render was made from, so the next run can tell whether it is still valid.
    paths.compose_inputs.write_text(json.dumps(inputs, indent=2), encoding="utf-8")

    actual = probe_duration_seconds(paths.vertical, ffprobe_path=settings.ffprobe_path)
    if abs(actual - duration) > max(1.0, settings.extract_duration_tolerance_seconds * 2):
        plan.add_warning(
            WARN_COMPOSE_LENGTH,
            f"vertical clip is {actual:.2f}s but {duration:.2f}s was expected",
        )
    return layout
