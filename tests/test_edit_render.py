from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from clippy.audio.intensity import probe_duration_seconds
from clippy.config import Settings
from clippy.edit.layouts import plan_layout
from clippy.edit.plan import EditPlan, EditPaths, build_plan
from clippy.edit.render import (
    WARN_EXTRACT_SHORT,
    _cache_is_fresh,
    _composition_inputs,
    apply_deadair,
    build_composition_filter,
    compose_vertical,
    escape_filter_path,
    extract_base,
)
from clippy.store.db import Candidate, Stream

FIXTURE = Path(__file__).resolve().parents[1] / "samples" / "sample_vod.mp4"
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4 (run scripts/make_sample_media.py)",
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "data_dir": tmp_path,
        "clip_min_seconds": 10.0,
        "clip_max_seconds": 45.0,
        "clip_target_seconds": 30.0,
        "extract_duration_tolerance_seconds": 0.5,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _plan(
    tmp_path: Path, *, ts: float, pre: float = 30.0, post: float = 30.0
) -> tuple[Settings, EditPlan, EditPaths]:
    settings = _settings(tmp_path)
    stream = Stream(
        id=1,
        streamer_id=1,
        mode="vod",
        source_url=None,
        vod_id=None,
        media_path=str(FIXTURE),
        started_at="2026-01-01T00:00:00+00:00",
        created_at="2026-01-01T00:00:00+00:00",
        source_width=640,
        source_height=360,
    )
    candidate = Candidate(
        id=1,
        stream_id=1,
        source_ts=ts,
        pre_context_seconds=pre,
        post_context_seconds=post,
        signals={"kind": "keyword"},
        score=0.5,
        media_path=None,
        status="pending",
        created_at="2026-01-01T00:00:00+00:00",
    )
    plan = build_plan(
        candidate=candidate, stream=stream, source_path=FIXTURE, settings=settings
    )
    return settings, plan, EditPaths.for_candidate(settings, candidate.id)


def test_extract_base_writes_media_at_the_planned_bounds(tmp_path: Path):
    settings, plan, paths = _plan(tmp_path, ts=120.0)
    assert plan.bounds.duration == pytest.approx(45.0)

    produced = extract_base(plan, paths, settings=settings)

    assert produced == paths.base
    assert produced.exists() and produced.stat().st_size > 0
    actual = probe_duration_seconds(produced, ffprobe_path=settings.ffprobe_path)
    assert actual == pytest.approx(plan.bounds.duration, abs=0.5)
    assert WARN_EXTRACT_SHORT not in plan.warning_codes()


def test_extract_base_clamps_when_the_source_ends_early(tmp_path: Path):
    settings, plan, paths = _plan(tmp_path, ts=250.0, pre=5.0, post=30.0)
    assert plan.bounds.end == pytest.approx(280.0)

    extract_base(plan, paths, settings=settings)

    assert WARN_EXTRACT_SHORT in plan.warning_codes()
    # The fixture is 260 s long, so the plan is clamped to what really exists.
    assert plan.bounds.end <= 261.0
    assert plan.bounds.duration < 30.0
    assert plan.bounds.main_ts <= plan.bounds.end
    assert plan.bounds.payoff_ts <= plan.bounds.end


def test_extract_base_reuses_cached_media_unless_forced(tmp_path: Path):
    settings, plan, paths = _plan(tmp_path, ts=120.0)
    first = extract_base(plan, paths, settings=settings)
    stamp = first.stat().st_mtime_ns

    again = extract_base(plan, paths, settings=settings)

    assert again == first
    assert again.stat().st_mtime_ns == stamp


def test_extract_base_rejects_a_zero_length_plan(tmp_path: Path):
    settings, plan, paths = _plan(tmp_path, ts=120.0)
    plan.bounds = replace(plan.bounds, end=plan.bounds.start)
    with pytest.raises(ValueError):
        extract_base(plan, paths, settings=settings)


def test_extract_base_requires_the_source(tmp_path: Path):
    settings, plan, paths = _plan(tmp_path, ts=120.0)
    plan.source_path = str(tmp_path / "missing.mp4")
    with pytest.raises(FileNotFoundError):
        extract_base(plan, paths, settings=settings)


def test_escape_filter_path_double_escapes_the_drive_colon():
    assert escape_filter_path("C:/Windows/Fonts") == r"C\\:/Windows/Fonts"
    assert escape_filter_path(r"C:\Windows\Fonts") == r"C\\:/Windows/Fonts"
    assert escape_filter_path("captions.ass") == "captions.ass"


def test_compose_vertical_reuses_the_cache_only_when_inputs_match(
    monkeypatch, tmp_path: Path
):
    """
    Framing is resolved once, and a cached render is reused only while its inputs still match.

    A pre-existing ``vertical.mp4`` plus a matching fingerprint returns the cached render without
    touching ffmpeg, which keeps this a pure orchestration test.
    """
    settings, plan, paths = _plan(tmp_path, ts=120.0)
    paths.ensure_root()
    paths.trimmed.write_bytes(b"trimmed")
    paths.vertical.write_bytes(b"cached")
    paths.captions.write_text("[captions v1]", encoding="utf-8")

    layout = plan_layout(
        requested="auto",
        source_width=1920,
        source_height=1080,
        width=1080,
        height=1920,
        fps=30,
        duration=10.0,
        settings=settings,
        facecam_box=(0.7, 0.6, 0.3, 0.4),
    )
    # The facecam panel owns the bottom of the frame; that is what moves the captions.
    assert layout.caption_prefer_top is True

    fresh, inputs = _cache_is_fresh(
        paths, layout, settings=settings, ass_path=paths.captions
    )
    assert fresh is False  # nothing recorded yet, so the render cannot be trusted
    paths.compose_inputs.write_text(json.dumps(inputs), encoding="utf-8")
    fresh, _ = _cache_is_fresh(paths, layout, settings=settings, ass_path=paths.captions)
    assert fresh is True

    def never(*args, **kwargs):
        raise AssertionError("composition must not re-resolve the framing it was given")

    monkeypatch.setattr("clippy.edit.render.plan_composition", never)
    result = compose_vertical(
        plan,
        paths,
        settings=settings,
        duration=10.0,
        layout=layout,
        ass_path=paths.captions,
    )

    assert result is layout


def test_composition_fingerprint_notices_every_input(tmp_path: Path):
    """A cached render is only valid while the pixels, the text and the framing are unchanged."""
    settings, _plan_obj, paths = _plan(tmp_path, ts=120.0)
    paths.ensure_root()
    paths.trimmed.write_bytes(b"trimmed")
    paths.captions.write_text("[v1]", encoding="utf-8")

    layout = plan_layout(
        requested="fit_blur",
        source_width=1920,
        source_height=1080,
        width=1080,
        height=1920,
        fps=30,
        duration=5.0,
        settings=settings,
    )
    first = _composition_inputs(paths, layout, settings=settings, ass_path=paths.captions)
    assert _composition_inputs(paths, layout, settings=settings, ass_path=paths.captions) == first

    # Rewriting the same caption text is not a change: both files are rewritten every run, so
    # the fingerprint compares content rather than timestamps.
    paths.captions.write_text("[v1]", encoding="utf-8")
    assert _composition_inputs(paths, layout, settings=settings, ass_path=paths.captions) == first

    paths.captions.write_text("[v2]", encoding="utf-8")
    assert _composition_inputs(paths, layout, settings=settings, ass_path=paths.captions) != first
    paths.captions.write_text("[v1]", encoding="utf-8")

    # No captions burned at all is a different render.
    assert _composition_inputs(paths, layout, settings=settings, ass_path=None) != first

    # Different framing, different pixels, different encode settings.
    other_layout = plan_layout(
        requested="irl",
        source_width=1920,
        source_height=1080,
        width=1080,
        height=1920,
        fps=30,
        duration=5.0,
        settings=settings,
    )
    assert (
        _composition_inputs(paths, other_layout, settings=settings, ass_path=paths.captions)
        != first
    )
    paths.trimmed.write_bytes(b"trimmed again")
    assert _composition_inputs(paths, layout, settings=settings, ass_path=paths.captions) != first
    louder = _settings(tmp_path, render_crf=18)
    assert _composition_inputs(paths, layout, settings=louder, ass_path=paths.captions) != first


def test_build_composition_filter_composes_and_burns_captions(tmp_path: Path):
    layout = plan_layout(
        requested="fit_blur",
        source_width=1920,
        source_height=1080,
        width=1080,
        height=1920,
        fps=30,
        duration=2.0,
        settings=Settings(data_dir=tmp_path),  # type: ignore[arg-type]
    )
    graph = build_composition_filter(
        layout,
        has_audio=True,
        ass_filename="captions.ass",
        fonts_dir="C:/Windows/Fonts",
    )

    assert "color=c=black:s=1080x1920:r=30:d=2.000[bg0]" in graph
    assert "[0:v]trim=start=0.000:end=2.000,setpts=PTS-STARTPTS[sv0]" in graph
    assert "[0:a]atrim=start=0.000:end=2.000,asetpts=PTS-STARTPTS[sa0]" in graph
    assert "boxblur=" in graph
    assert "ass=filename=captions.ass:fontsdir=C\\\\:/Windows/Fonts[vout]" in graph
    assert "concat=n=1:v=1:a=1[catv][cata]" in graph
    assert graph.endswith("[cata]anull[aout]")


def test_build_composition_filter_without_audio_or_captions(tmp_path: Path):
    layout = plan_layout(
        requested="fit_blur",
        source_width=1920,
        source_height=1080,
        width=1080,
        height=1920,
        fps=30,
        duration=1.0,
        settings=Settings(data_dir=tmp_path),  # type: ignore[arg-type]
    )
    graph = build_composition_filter(layout, has_audio=False)

    assert "[0:a]" not in graph
    assert "concat=n=1:v=1:a=0[catv]" in graph
    assert "[catv]null[vout]" in graph
    assert "ass=" not in graph


def _silence_fixture(tmp_path: Path) -> Path:
    """10s clip: speech-ish tone between 3s and 7s, near-silence either side."""
    out = tmp_path / "silence.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:d=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=10",
        "-filter_complex",
        "[1:a]volume='if(between(t,3,7),0.8,0.0005)':eval=frame[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _silence_plan(tmp_path: Path, media: Path, **overrides):
    settings = _settings(tmp_path, **overrides)
    stream = Stream(
        id=1,
        streamer_id=1,
        mode="vod",
        source_url=None,
        vod_id=None,
        media_path=str(media),
        started_at="2026-01-01T00:00:00+00:00",
        created_at="2026-01-01T00:00:00+00:00",
        source_width=320,
        source_height=240,
    )
    candidate = Candidate(
        id=1,
        stream_id=1,
        source_ts=5.0,
        pre_context_seconds=5.0,
        post_context_seconds=5.0,
        signals={"kind": "keyword"},
        score=0.5,
        media_path=None,
        status="pending",
        created_at="2026-01-01T00:00:00+00:00",
    )
    plan = build_plan(
        candidate=candidate, stream=stream, source_path=media, settings=settings
    )
    paths = EditPaths.for_candidate(settings, candidate.id)
    extract_base(plan, paths, settings=settings)
    return settings, plan, paths


def test_apply_deadair_trims_leading_and_trailing_silence(tmp_path: Path):
    media = _silence_fixture(tmp_path)
    settings, plan, paths = _silence_plan(tmp_path, media, deadair_max_removed_ratio=1.0)

    cut = apply_deadair(plan, paths, settings=settings)

    assert cut.applied is True
    assert cut.removed_seconds == pytest.approx(5.6, abs=0.8)
    assert plan.deadair.applied is True
    assert plan.deadair.removed_seconds > 4.0
    assert plan.deadair.keep_segments[0][0] == pytest.approx(2.8, abs=0.4)
    assert plan.deadair.reason is not None

    actual = probe_duration_seconds(paths.trimmed, ffprobe_path=settings.ffprobe_path)
    assert actual == pytest.approx(4.4, abs=0.8)


def test_apply_deadair_speed_mode_keeps_everything_but_compresses(tmp_path: Path):
    media = _silence_fixture(tmp_path)
    settings, plan, paths = _silence_plan(
        tmp_path,
        media,
        deadair_mode="speed",
        deadair_speed_factor=1.5,
        deadair_max_removed_ratio=1.0,
    )

    cut = apply_deadair(plan, paths, settings=settings)

    assert cut.mode == "speed"
    actual = probe_duration_seconds(paths.trimmed, ffprobe_path=settings.ffprobe_path)
    assert actual == pytest.approx(10.0 - cut.removed_seconds, abs=0.8)
    assert actual < 10.0


def _loud_fixture(tmp_path: Path) -> Path:
    """12s of continuous tone: no silence at all, so there is nothing to cut."""
    out = tmp_path / "loud.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:d=12",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=220:duration=12",
        "-filter_complex",
        "[1:a]volume=0.6[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_apply_deadair_copies_base_when_nothing_is_cuttable(tmp_path: Path):
    media = _loud_fixture(tmp_path)
    settings, plan, paths = _silence_plan(tmp_path, media)

    cut = apply_deadair(plan, paths, settings=settings)

    assert cut.applied is False
    assert plan.deadair.applied is False
    assert plan.deadair.keep_segments == [[0.0, 10.0]]
    assert "no silence detected" in (plan.deadair.reason or "")
    # The trimmed artifact still exists so later stages have one timeline to read.
    assert paths.trimmed.exists()
    assert paths.trimmed.stat().st_size == paths.base.stat().st_size

