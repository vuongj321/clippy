from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clippy.config import Settings
from clippy.edit.layouts import (
    WARN_CAPTION_BAND,
    WARN_FACECAM_MISSING,
    WARN_LOW_RES,
    WARN_TRACK_FALLBACK,
    WARN_UPSCALE,
    CompositionPlan,
    LayoutLayer,
    LayoutSegment,
    caption_band_coverage,
    caption_bands,
    crop_rect,
    plan_layout,
    resolve_strategy,
    segments_from_track,
)
from clippy.edit.track import (
    TrackPoint,
    centroid_from_energy,
    crop_center_y,
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
    track = [TrackPoint(t=0.5, x=0.4, y=0.42), TrackPoint(t=2.0, x=0.6, y=0.58)]
    _settings_obj, plan = _plan(tmp_path, requested="irl", track=track)
    restored = CompositionPlan.from_dict(plan.to_dict())
    assert restored.to_dict() == plan.to_dict()
    # The vertical track survives the round trip too, because the caption band depends on it.
    assert [point.y for point in restored.subject_track] == pytest.approx([0.42, 0.58])


def test_crop_center_y_needs_a_subject_box(tmp_path: Path):
    """Motion evidence carries no box, so it must never move the crop vertically."""
    motion_points = [TrackPoint(t=0.5, x=0.5, y=0.9) for _ in range(4)]
    assert crop_center_y(motion_points, source_height=2400.0, crop_height=1920.0) == 0.5
    assert crop_center_y([], source_height=2400.0, crop_height=1920.0) == 0.5


def test_crop_center_y_has_no_headroom_on_a_16_by_9_source():
    """A 9:16 crop of a 16:9 frame already uses the whole height: nothing to move."""
    face_points = [TrackPoint(t=0.5, x=0.7, y=0.18, width=0.1, height=0.12)]
    assert crop_center_y(face_points, source_height=1080.0, crop_height=1080.0) == 0.5


def test_crop_center_y_follows_the_face_and_stays_inside_the_frame():
    # Median (not mean) position, so a single stray detection cannot drag the crop.
    points = [
        TrackPoint(t=0.5, x=0.7, y=0.05, width=0.1, height=0.12),
        TrackPoint(t=1.0, x=0.7, y=0.08, width=0.1, height=0.12),
        TrackPoint(t=1.5, x=0.7, y=0.99, width=0.1, height=0.12),
    ]
    # Median y is 0.08 -> 192px centre on a 2400px source, clamped to half the crop (960).
    assert crop_center_y(points, source_height=2400.0, crop_height=1920.0) == pytest.approx(
        960.0 / 2400.0
    )
    # A subject near the bottom clamps to the other edge.
    low = [TrackPoint(t=0.5, x=0.7, y=0.95, width=0.1, height=0.12)]
    assert crop_center_y(low, source_height=2400.0, crop_height=1920.0) == pytest.approx(
        1440.0 / 2400.0
    )


def test_plan_layout_uses_the_face_for_the_vertical_crop(tmp_path: Path):
    """
    With vertical headroom, a tall subject must be framed by its face, not by the canvas centre.

    The source is 1080x2400 (taller than 9:16), so the 1920-tall crop can move 480px. The face sits
    at 0.08 of the frame, near the top, so the crop has to ride up to keep it in shot.
    """
    track = [TrackPoint(t=0.5, x=0.5, y=0.08, width=0.12, height=0.12)]
    _settings_obj, plan = _plan(tmp_path, requested="irl", source=(1080, 2400), track=track)

    src = plan.segments[0].layers[0].src
    assert src[3] == pytest.approx(1920.0)
    assert src[1] == pytest.approx(0.0)  # clamped to the top, where the face is
    # A centred crop would have started at 240 and cut the top of the face off.
    assert 0.08 * 2400 >= src[1]


def test_caption_band_coverage_prefers_the_detected_box(tmp_path: Path):
    """
    A real face box replaces the assumed subject height, which changes the answer.

    On an identity mapping, the caption strip occupies rows 1525-1660. A detected box 5% of the
    frame tall centred on row 1400 ends at 1448 and clears the strip, while the *assumed* 25% box
    around the same point reaches 1640 and overlaps it.
    """
    segment = _canvas_segment()
    band = (1525.0, 1660.0)
    detected = [TrackPoint(t=0.5, x=0.5, y=1400.0 / 1920.0, width=0.1, height=0.05)]
    assumed = [TrackPoint(t=0.5, x=0.5, y=1400.0 / 1920.0)]

    assert (
        caption_band_coverage(
            detected,
            [segment],
            source_height=1920.0,
            canvas_height=1920.0,
            band=band,
        )
        == 0.0
    )
    assert (
        caption_band_coverage(
            assumed,
            [segment],
            source_height=1920.0,
            canvas_height=1920.0,
            band=band,
        )
        == 1.0
    )


def _canvas_segment(*, start: float = 0.0, end: float = 10.0) -> LayoutSegment:
    """One layer that copies the source straight onto a same-sized canvas."""
    return LayoutSegment(
        start=start,
        end=end,
        layers=[
            LayoutLayer(
                kind="video",
                src=(0.0, 0.0, 1080.0, 1920.0),
                dst=(0.0, 0.0, 1080.0, 1920.0),
            )
        ],
    )


def test_caption_bands_sit_inside_the_margins(tmp_path: Path):
    settings = _settings(tmp_path)
    bottom, top = caption_bands(1920, font_size=54, settings=settings)

    # 2 lines of 54px text at the default 1.25 line height, 260px in from the edge.
    assert bottom == (pytest.approx(1525.0), pytest.approx(1660.0))
    assert top == (pytest.approx(260.0), pytest.approx(395.0))

    # A bigger style draws a taller strip, which is the strip the subject has to avoid.
    big_bottom, big_top = caption_bands(1920, font_size=60, settings=settings)
    assert big_bottom[0] == pytest.approx(1510.0)  # 1920 - 260 - 150
    assert big_top[1] == pytest.approx(410.0)  # 260 + 150


def test_plan_layout_measures_the_band_with_the_chosen_style(tmp_path: Path):
    """
    The strip the subject is tested against is the one the chosen style will draw.

    On a 1920 canvas with `caption_margin_v` 260 and a 0.25-height subject box, the box around a
    subject at row 1300 reaches up to 1540. The bottom band starts at 1525 for `karaoke_highlight`
    (54px), 1550 for `minimal` (44px) and 1510 for `block_pop` (60px) - so row 1300 is inside the
    first and third, and clear of the second.
    """
    settings = _settings(tmp_path)
    track = [
        TrackPoint(t=(index + 0.5) / 6.0, x=0.5, y=1300.0 / 1920.0) for index in range(30)
    ]

    def build(style: str) -> CompositionPlan:
        return plan_layout(
            requested="irl",
            source_width=1080,
            source_height=1920,
            width=1080,
            height=1920,
            fps=30,
            duration=5.0,
            settings=settings,
            track=track,
            caption_style=style,
        )

    assert build("karaoke_highlight").caption_bottom_coverage == pytest.approx(1.0)
    assert build("block_pop").caption_bottom_coverage == pytest.approx(1.0)
    assert build("minimal").caption_bottom_coverage == 0.0


def test_caption_band_coverage_maps_motion_through_the_layout():
    band = (1525.0, 1660.0)
    low = [TrackPoint(t=1.0, x=0.5, y=0.9), TrackPoint(t=2.0, x=0.5, y=0.9)]
    half = [TrackPoint(t=1.0, x=0.5, y=0.5), TrackPoint(t=2.0, x=0.5, y=0.9)]

    # 0.9 of the source is row 1728, and the subject box around it overlaps the caption strip.
    assert (
        caption_band_coverage(
            low,
            [_canvas_segment()],
            source_height=1920.0,
            canvas_height=1920.0,
            band=band,
        )
        == 1.0
    )
    # Centred motion is nowhere near the strip.
    assert (
        caption_band_coverage(
            half,
            [_canvas_segment()],
            source_height=1920.0,
            canvas_height=1920.0,
            band=band,
        )
        == 0.5
    )


def test_caption_band_coverage_is_zero_without_evidence():
    band = (1525.0, 1660.0)
    low = [TrackPoint(t=1.0, x=0.5, y=0.9)]
    assert caption_band_coverage([], [_canvas_segment()], source_height=1920.0, canvas_height=1920.0, band=band) == 0.0
    assert caption_band_coverage(low, [], source_height=1920.0, canvas_height=1920.0, band=band) == 0.0
    assert caption_band_coverage(low, [_canvas_segment()], source_height=0.0, canvas_height=1920.0, band=band) == 0.0
    assert caption_band_coverage(low, [_canvas_segment()], source_height=1920.0, canvas_height=1920.0, band=(100.0, 100.0)) == 0.0
    # A moment no segment covers is not counted at all rather than counted as a miss.
    outside = [TrackPoint(t=30.0, x=0.5, y=0.9)]
    assert caption_band_coverage(outside, [_canvas_segment()], source_height=1920.0, canvas_height=1920.0, band=band) == 0.0


def test_plan_layout_moves_captions_when_the_subject_owns_the_band(tmp_path: Path):
    track = [TrackPoint(t=(index + 0.5) / 6.0, x=0.5, y=0.9) for index in range(60)]
    _settings_obj, plan = _plan(tmp_path, requested="irl", source=(1080, 1920), track=track)

    assert plan.resolved_strategy == "irl"
    assert plan.caption_bottom_coverage == pytest.approx(1.0)
    assert plan.caption_prefer_top is True
    assert WARN_CAPTION_BAND in plan.warnings


def test_plan_layout_keeps_captions_low_when_the_subject_is_centred(tmp_path: Path):
    track = [TrackPoint(t=(index + 0.5) / 6.0, x=0.5, y=0.5) for index in range(60)]
    _settings_obj, plan = _plan(tmp_path, requested="irl", source=(1080, 1920), track=track)

    assert plan.caption_bottom_coverage == 0.0
    assert plan.caption_prefer_top is False
    assert WARN_CAPTION_BAND not in plan.warnings


def test_plan_layout_tolerates_a_subject_dipping_into_the_band(tmp_path: Path):
    """`caption_avoid_ratio` is the tolerance for the subject merely passing through."""
    track = [
        TrackPoint(t=(index + 0.5) / 6.0, x=0.5, y=0.9 if index == 0 else 0.5)
        for index in range(10)
    ]
    _settings_obj, plan = _plan(tmp_path, requested="irl", source=(1080, 1920), track=track)
    assert plan.caption_bottom_coverage == pytest.approx(0.1)
    assert plan.caption_prefer_top is False

    # A reviewer who wants a tighter tolerance gets the flip.
    _other, strict = _plan(
        tmp_path,
        requested="irl",
        source=(1080, 1920),
        track=track,
        caption_avoid_ratio=0.05,
    )
    assert strict.caption_prefer_top is True
    assert WARN_CAPTION_BAND in strict.warnings


def test_plan_layout_keeps_captions_low_when_the_top_band_is_busier(tmp_path: Path):
    """Flipping into a busier band would trade one occluded caption for another."""
    busy_top = [
        TrackPoint(t=1.0, x=0.5, y=0.9),  # in the bottom band
        TrackPoint(t=2.0, x=0.5, y=0.9),
        TrackPoint(t=3.0, x=0.5, y=0.16),  # more motion up top than down below
        TrackPoint(t=4.0, x=0.5, y=0.16),
        TrackPoint(t=5.0, x=0.5, y=0.16),
    ]
    _settings_obj, plan = _plan(tmp_path, requested="irl", source=(1080, 1920), track=busy_top)
    assert plan.caption_bottom_coverage == pytest.approx(0.4)  # over the 0.25 tolerance
    assert plan.caption_prefer_top is False

    # The same bottom evidence with nothing competing at the top does move the band.
    quiet_top = [
        TrackPoint(t=1.0, x=0.5, y=0.9),
        TrackPoint(t=2.0, x=0.5, y=0.9),
        TrackPoint(t=3.0, x=0.5, y=0.5),
        TrackPoint(t=4.0, x=0.5, y=0.5),
        TrackPoint(t=5.0, x=0.5, y=0.5),
    ]
    _other, flipped = _plan(tmp_path, requested="irl", source=(1080, 1920), track=quiet_top)
    assert flipped.caption_bottom_coverage == pytest.approx(0.4)
    assert flipped.caption_prefer_top is True

