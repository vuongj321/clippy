from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from clippy.audio.intensity import probe_duration_seconds
from clippy.config import Settings
from clippy.edit.layouts import WARN_FACECAM_DERIVED, plan_layout
from clippy.edit.plan import EditPlan, EditPaths, build_plan
from clippy.edit.track import TrackPoint
from clippy.edit.render import (
    WARN_EXTRACT_SHORT,
    _cache_is_fresh,
    _composition_inputs,
    _render_segments,
    apply_deadair,
    build_composition_filter,
    compose_vertical,
    escape_filter_path,
    extract_base,
    plan_composition,
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


def test_plan_composition_honours_the_plan_strategy(monkeypatch, tmp_path: Path):
    """
    The requested strategy lives in `plan.layout`, so composition must read the plan rather than
    the config default - otherwise a UI/CLI choice never reaches the render (the bug this covers).
    """
    settings, plan, paths = _plan(tmp_path, ts=120.0)
    paths.ensure_root()
    paths.trimmed.write_bytes(b"trimmed")
    # What `EditOverrides(strategy="fit_blur")` folds into the plan.
    plan.layout.strategy = "fit_blur"

    monkeypatch.setattr(
        "clippy.edit.render.probe_dimensions", lambda *args, **kwargs: (1920, 1080)
    )
    monkeypatch.setattr(
        "clippy.edit.render.track_subject", lambda *args, **kwargs: []
    )

    layout = plan_composition(plan, paths, settings=settings, duration=10.0)

    assert layout.strategy == "fit_blur"
    assert layout.resolved_strategy == "fit_blur"
    # `_record_layout` must keep the requested value, not overwrite it with the config default.
    assert plan.layout.strategy == "fit_blur"
    assert plan.layout.resolved_strategy == "fit_blur"
    assert paths.layout.exists()


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
    # fit_blur stacks a blurred fill and the video, so the reset source is split for both.
    assert (
        "[0:v]trim=start=0.000:end=2.000,setpts=PTS-STARTPTS,split=2[sv0_0][sv0_1]"
        in graph
    )
    assert "[sv0_0]crop=" in graph and "[sv0_1]crop=" in graph
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

    assert ",split=2[sv0_0][sv0_1]" in graph
    assert "[0:a]" not in graph
    assert "concat=n=1:v=1:a=0[catv]" in graph
    assert "[catv]null[vout]" in graph
    assert "ass=" not in graph


def test_build_composition_filter_gives_every_layer_its_own_source(tmp_path: Path):
    """
    Each layer reads its own split of the reset source, because ffmpeg hands a labelled stream to
    exactly one reader: a second reader is given the frames of a cloned chain that was fed the raw
    input, so its `setpts` never runs and it keeps whatever offset the cut carries (0.8167s on
    candidate 28). Reading `[sv0]` twice is what composited the facecam 0.82s late - the black
    panel over the first 0.83s, and then a layer that never caught up.
    """
    for requested, box in (
        ("gaming", [0.7, 0.05, 0.28, 0.3]),
        ("fit_blur", None),
        ("conversation", None),
    ):
        layout = plan_layout(
            requested=requested,
            source_width=1920,
            source_height=1080,
            width=1080,
            height=1920,
            fps=30,
            duration=2.0,
            settings=Settings(data_dir=tmp_path),  # type: ignore[arg-type]
            facecam_box=box,
        )
        layers = layout.segments[0].layers
        graph = build_composition_filter(layout, has_audio=True)

        assert len(layers) == 2, f"{requested} should stack two layers"
        assert ",split=2[sv0_0][sv0_1]" in graph
        for layer_index in range(len(layers)):
            assert f"[sv0_{layer_index}]crop=" in graph

        # Every label is written once and read once, except the two that are mapped out of the
        # graph by `compose_vertical`.
        for label, reads in Counter(re.findall(r"\[([A-Za-z0-9_]+)\]", graph)).items():
            if ":" in label:  # an input pad may be read once per segment
                continue
            expected = 1 if label in {"vout", "aout"} else 2
            assert reads == expected, f"[{label}] is read {reads} times in\n{graph}"


def test_composition_frame_locks_the_facecam_to_the_gameplay(tmp_path: Path):
    """
    The end-to-end shape of the defect: a cut whose video timestamps do not start at 0 (which is
    what `extract_window` used to write) must still compose every layer from the same instant. The
    facecam panel used to composite nothing until the offset elapsed, so the black canvas showed
    through it.
    """
    trimmed = tmp_path / "trimmed.mp4"
    # A colourful 3s window whose *video* timestamps do not start at 0, which is what
    # `extract_window` used to write: the audio starts at 0, the video 0.8167s later.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=25:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-vf",
            "setpts=PTS+0.816667/TB",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            str(trimmed),
        ],
        check=True,
        capture_output=True,
    )
    layout = plan_layout(
        requested="gaming",
        source_width=640,
        source_height=360,
        width=540,
        height=960,
        fps=25,
        duration=3.0,
        settings=_settings(tmp_path),
        facecam_box=[0.6, 0.05, 0.32, 0.3],
    )
    output = tmp_path / "vertical.mp4"
    graph = build_composition_filter(layout, has_audio=False)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(trimmed),
            "-filter_complex",
            graph,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            str(output),
        ],
        check=True,
        capture_output=True,
    )

    panel = max(layout.segments[0].layers, key=lambda layer: layer.z)
    panel_x, panel_y, panel_w, panel_h = (round(value) for value in panel.dst)
    probe = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(output),
            "-vf",
            f"crop={panel_w}:{panel_h}:{panel_x}:{panel_y},blackdetect=d=0.04:pix_th=0.10",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    assert "black_start" not in probe.stderr, probe.stderr[-2000:]


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


def test_deadair_render_keeps_the_intermediate_crf(monkeypatch, tmp_path: Path):
    """
    A cut re-encode produces an intermediate, so it keeps the master CRF rather than the
    deliverable one - otherwise the crf-16 `base.mp4` is spent on a crf-20 `trimmed.mp4`
    before the final encode ever runs.
    """
    settings = _settings(tmp_path, render_crf=20, intermediate_crf=16)
    base = tmp_path / "base.mp4"
    base.write_bytes(b"base")
    output = tmp_path / "trimmed.mp4"
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("clippy.edit.render.probe_streams", lambda *a, **k: (True, True))
    monkeypatch.setattr("clippy.edit.render.subprocess.run", fake_run)

    _render_segments(
        base, output, segments=[(0.0, 5.0, 1.0)], settings=settings, force=True
    )

    assert output.exists()
    crf = commands[-1][commands[-1].index("-crf") + 1]
    assert crf == "16"


def test_plan_composition_derives_a_facecam_box_when_asked(monkeypatch, tmp_path: Path):
    """
    `facecam_box: auto` is resolved from the clip's own face track, then folds into the same
    `gaming` path a typed box would - and the derived box is recorded so a reviewer can see it and
    a re-render reuses it instead of deriving a new one.
    """
    _, plan, paths = _plan(tmp_path, ts=120.0)
    settings = _settings(tmp_path, facecam_box="auto", layout_track_backend="opencv")
    paths.ensure_root()
    paths.trimmed.write_bytes(b"trimmed")

    face_track = [
        TrackPoint(t=index * 0.25, x=0.75, y=0.2, width=0.12, height=0.16)
        for index in range(8)
    ]
    monkeypatch.setattr(
        "clippy.edit.render.probe_dimensions", lambda *args, **kwargs: (1920, 1080)
    )
    monkeypatch.setattr(
        "clippy.edit.render.track_subject", lambda *args, **kwargs: list(face_track)
    )

    layout = plan_composition(plan, paths, settings=settings, duration=10.0)

    assert layout.resolved_strategy == "gaming"
    assert layout.facecam_box_source == "auto"
    assert layout.facecam_box is not None
    assert WARN_FACECAM_DERIVED in layout.warnings
    # A 22%-wide tile shaped to the gaming panel (1080:768, flush against the gameplay) and centred
    # on the face at (0.75, 0.2).
    assert plan.layout.facecam_box == pytest.approx([0.64, 0.0609, 0.22, 0.2781], abs=1e-3)
    assert plan.layout.facecam_box_source == "auto"
    # The whole point: the derived box must actually contain the face.
    x, y, w, h = plan.layout.facecam_box
    assert x <= 0.75 <= x + w
    assert y <= 0.2 <= y + h


def test_plan_composition_warns_when_auto_finds_no_face(monkeypatch, tmp_path: Path):
    _, plan, paths = _plan(tmp_path, ts=120.0)
    settings = _settings(tmp_path, facecam_box="auto", layout_track_backend="motion")
    paths.ensure_root()
    paths.trimmed.write_bytes(b"trimmed")

    monkeypatch.setattr(
        "clippy.edit.render.probe_dimensions", lambda *args, **kwargs: (1920, 1080)
    )
    # A motion track carries no box, so there is nothing to derive from.
    monkeypatch.setattr(
        "clippy.edit.render.track_subject",
        lambda *args, **kwargs: [TrackPoint(t=0.5, x=0.5, y=0.5)],
    )

    layout = plan_composition(plan, paths, settings=settings, duration=10.0)

    assert layout.resolved_strategy == "irl"
    assert layout.facecam_box is None
    assert "facecam_box_auto_failed" in layout.warnings
    assert plan.layout.facecam_box is None

