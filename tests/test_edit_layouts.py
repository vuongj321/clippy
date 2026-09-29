from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clippy.config import Settings
from clippy.edit.layouts import (
    WARN_FACECAM_MISSING,
    WARN_LOW_RES,
    WARN_TRACK_FALLBACK,
    WARN_UPSCALE,
    CompositionPlan,
    crop_rect,
    plan_layout,
    resolve_strategy,
    segments_from_track,
)
from clippy.edit.track import (
    TrackPoint,
    centroid_from_energy,
    resample_track,
    smooth_track,
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "data_dir": tmp_path,
        "clip_target_width": 1080,
        "clip_target_height": 1920,
        "clip_fps": 30,
        "quality_warn_upscale": 2.0,
        "layout_track_backend": "motion",
        "layout_smoothing": 0.12,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _plan(
    tmp_path: Path,
    *,
    requested="auto",
    source=(1920, 1080),
    track=(),
    facecam_box=None,
    **overrides,
):
    settings = _settings(tmp_path, **overrides)
    return settings, plan_layout(
        requested=requested,
        source_width=source[0],
        source_height=source[1],
        width=1080,
        height=1920,
        fps=30,
        duration=10.0,
        settings=settings,
        track=list(track),
        facecam_box=facecam_box,
    )


def test_crop_rect_follows_the_subject_and_stays_inside_the_frame():
    assert crop_rect(1920, 1080, 1080 / 1920, center_x=0.0)[0] == pytest.approx(0.0)
    middle = crop_rect(1920, 1080, 1080 / 1920, center_x=0.5)
    assert middle[0] == pytest.approx(656.25)
    assert middle[2] == pytest.approx(607.5)
    assert middle[3] == pytest.approx(1080.0)
    assert crop_rect(1920, 1080, 1080 / 1920, center_x=1.0)[0] == pytest.approx(1312.5)


def test_crop_rect_handles_a_source_that_is_already_the_target_ratio():
    assert crop_rect(1080, 1920, 1080 / 1920) == (0.0, 0.0, 1080.0, 1920.0)


def test_crop_rect_rejects_degenerate_input():
    assert crop_rect(0, 1080, 0.5) == (0.0, 0.0, 0.0, 0.0)


def test_centroid_from_energy_pulls_toward_the_centre():
    profile = np.zeros(10)
    profile[9] = 1.0
    # Raw centroid would be 0.95; the prior holds it back to a sane crop position.
    assert centroid_from_energy(profile, center_prior=0.35) == pytest.approx(0.7925)
    assert centroid_from_energy(profile, center_prior=1.0) == pytest.approx(0.5)
    assert centroid_from_energy(np.zeros(5)) == 0.5
    assert centroid_from_energy(np.array([])) == 0.5


def test_smooth_track_moves_gradually():
    assert smooth_track([]) == []
    assert smooth_track([0.0, 1.0], alpha=0.5) == [0.0, pytest.approx(0.5)]
    assert smooth_track([0.4, 0.4], alpha=0.5) == [0.4, pytest.approx(0.4)]


def test_resample_track_picks_the_nearest_point():
    points = [TrackPoint(t=0.0, x=0.1, y=0.5), TrackPoint(t=5.0, x=0.9, y=0.5)]
    sampled = resample_track(points, times=[0.4, 4.4])
    assert [round(point.x, 1) for point in sampled] == [0.1, 0.9]
    assert resample_track([], times=[1.0]) == []


def test_segments_from_track_collapses_stable_regions():
    track = [TrackPoint(t=index * 0.5, x=0.2, y=0.5) for index in range(6)]
    track += [TrackPoint(t=3.0 + index * 0.5, x=0.8, y=0.5) for index in range(4)]
    spans = segments_from_track(track, 5.0, deadband=0.02)
    assert spans == [
        (0.0, pytest.approx(3.0), pytest.approx(0.2)),
        (pytest.approx(3.0), 5.0, pytest.approx(0.8)),
    ]


def test_segments_from_track_defaults_to_a_centred_span():
    assert segments_from_track([], 10.0) == [(0.0, 10.0, 0.5)]
    assert segments_from_track([], 0.0) == []


def test_segments_from_track_caps_the_span_count():
    track = [
        TrackPoint(t=index * 0.5, x=0.2 if index % 2 else 0.8, y=0.5)
        for index in range(80)
    ]
    spans = segments_from_track(track, 40.0, max_segments=6)
    assert len(spans) <= 6
    assert spans[0][0] == 0.0
    assert spans[-1][1] == pytest.approx(40.0)


def test_resolve_strategy_auto_prefers_gaming_with_a_facecam_box(tmp_path: Path):
    settings = _settings(tmp_path)
    resolved, warnings = resolve_strategy(
        "auto",
        source_width=1920,
        source_height=1080,
        facecam_box=[0.7, 0.05, 0.28, 0.3],
        settings=settings,
    )
    assert resolved == "gaming"
    assert warnings == []


def test_resolve_strategy_auto_uses_irl_for_hd_sources(tmp_path: Path):
    resolved, warnings = resolve_strategy(
        "auto",
        source_width=1920,
        source_height=1080,
        facecam_box=None,
        settings=_settings(tmp_path),
    )
    assert resolved == "irl"
    assert warnings == []


def test_resolve_strategy_downgrades_low_resolution(tmp_path: Path):
    resolved, warnings = resolve_strategy(
        "auto",
        source_width=284,
        source_height=160,
        facecam_box=None,
        settings=_settings(tmp_path),
    )
    assert resolved == "fit_blur"
    assert WARN_LOW_RES in warnings

    # Even an explicit crop request is refused on a source that cannot support it.
    forced, forced_warnings = resolve_strategy(
        "irl",
        source_width=284,
        source_height=160,
        facecam_box=None,
        settings=_settings(tmp_path),
    )
    assert forced == "fit_blur"
    assert WARN_LOW_RES in forced_warnings


def test_resolve_strategy_notes_a_disabled_tracker(tmp_path: Path):
    resolved, warnings = resolve_strategy(
        "irl",
        source_width=1920,
        source_height=1080,
        facecam_box=None,
        settings=_settings(tmp_path, layout_track_backend="none"),
    )
    assert resolved == "irl"
    assert WARN_TRACK_FALLBACK in warnings


def test_plan_layout_fit_blur_fits_the_frame_and_fills_the_rest(tmp_path: Path):
    _settings_obj, plan = _plan(tmp_path, requested="fit_blur")

    assert plan.resolved_strategy == "fit_blur"
    assert len(plan.segments) == 1
    blur, video = plan.segments[0].layers
    assert blur.kind == "blur" and blur.z == 0
    assert blur.dst[2] >= 1080 and blur.dst[3] >= 1920
    assert video.kind == "video" and video.z == 1
    assert video.dst[2] == pytest.approx(1080.0)
    assert video.dst[3] == pytest.approx(607.5)
    assert video.dst[1] == pytest.approx((1920 - 607.5) / 2.0)


def test_plan_layout_irl_crop_fills_the_canvas(tmp_path: Path):
    _settings_obj, plan = _plan(tmp_path, requested="irl")

    segment = plan.segments[0]
    layer = segment.layers[0]
    assert layer.dst == (0.0, 0.0, 1080.0, 1920.0)
    assert layer.src[2] == pytest.approx(607.5)
    assert segment.crop_x == pytest.approx(0.5)
    # 1080 / 607.5 = 1.78x: normal for a 9:16 crop from 1080p, under the warning bar.
    assert plan.upscale_factor == pytest.approx(1.778, abs=0.01)
    assert WARN_UPSCALE not in plan.warnings


def test_plan_layout_flags_a_heavy_upscale(tmp_path: Path):
    _settings_obj, plan = _plan(tmp_path, requested="irl", source=(1280, 720))

    assert plan.upscale_factor == pytest.approx(2.667, abs=0.01)
    assert WARN_UPSCALE in plan.warnings


def test_plan_layout_gaming_stacks_gameplay_and_facecam(tmp_path: Path):
    _settings_obj, plan = _plan(
        tmp_path, requested="gaming", facecam_box=[0.7, 0.05, 0.28, 0.3]
    )

    assert plan.resolved_strategy == "gaming"
    gameplay, facecam = plan.segments[0].layers
    assert gameplay.dst == (0.0, 0.0, 1080.0, 1152.0)
    assert facecam.dst[0] == 0.0
    assert facecam.dst[1] == pytest.approx(1160.0)
    assert facecam.dst[3] == pytest.approx(760.0)
    assert plan.caption_prefer_top is True


def test_plan_layout_gaming_without_a_facecam_box_falls_back(tmp_path: Path):
    _settings_obj, plan = _plan(tmp_path, requested="gaming", facecam_box=None)

    assert plan.resolved_strategy == "fit_blur"
    assert WARN_FACECAM_MISSING in plan.warnings


def test_plan_layout_conversation_stacks_two_panels(tmp_path: Path):
    _settings_obj, plan = _plan(tmp_path, requested="conversation")

    left, right = plan.segments[0].layers
    assert left.dst[0] == 0.0 and left.dst[2] == pytest.approx(1080.0)
    assert right.dst[1] == pytest.approx(left.dst[3] + 8.0)
    assert left.src[2] <= 960.0
    assert plan.caption_prefer_top is False


def test_plan_layout_uses_the_track_for_multiple_segments(tmp_path: Path):
    track = [TrackPoint(t=index * 0.5, x=0.2, y=0.5) for index in range(6)]
    track += [TrackPoint(t=3.0 + index * 0.5, x=0.8, y=0.5) for index in range(4)]
    _settings_obj, plan = _plan(tmp_path, requested="irl", track=track)

    assert len(plan.segments) == 2
    assert plan.segments[0].crop_x == pytest.approx(0.2)
    assert plan.segments[1].crop_x == pytest.approx(0.8)
    # A right-of-centre subject means the crop starts further right.
    assert plan.segments[1].layers[0].src[0] > plan.segments[0].layers[0].src[0]


def test_composition_plan_round_trips_through_json(tmp_path: Path):
    track = [TrackPoint(t=0.5, x=0.4, y=0.5), TrackPoint(t=2.0, x=0.6, y=0.5)]
    _settings_obj, plan = _plan(tmp_path, requested="irl", track=track)
    restored = CompositionPlan.from_dict(plan.to_dict())
    assert restored.to_dict() == plan.to_dict()

