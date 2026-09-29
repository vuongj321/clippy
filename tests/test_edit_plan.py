from __future__ import annotations

import json
from pathlib import Path

import pytest

from clippy.config import Settings
from clippy.edit.plan import (
    PLAN_FILENAME,
    WARN_BOUNDARIES_PENDING,
    WARN_LAYOUT_PENDING,
    WARN_RESOLUTION_UNKNOWN,
    WARN_UPSCALE,
    ClipBounds,
    EditOverrides,
    EditPaths,
    EditPlan,
    build_plan,
    phase1_bounds,
)
from clippy.store.db import Candidate, Stream


def _candidate(**overrides) -> Candidate:
    base = {
        "id": 1,
        "stream_id": 1,
        "source_ts": 100.0,
        "pre_context_seconds": 30.0,
        "post_context_seconds": 30.0,
        "signals": {"kind": "keyword"},
        "score": 0.8,
        "media_path": None,
        "status": "pending",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


def _stream(**overrides) -> Stream:
    base = {
        "id": 1,
        "streamer_id": 1,
        "mode": "vod",
        "source_url": None,
        "vod_id": None,
        "media_path": "data/source/vod.ts",
        "started_at": "2026-01-01T00:00:00+00:00",
        "created_at": "2026-01-01T00:00:00+00:00",
        "source_width": 1280,
        "source_height": 720,
        "source_fps": 60.0,
        "capture_quality": "720p60",
    }
    base.update(overrides)
    return Stream(**base)  # type: ignore[arg-type]


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {"data_dir": tmp_path}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_phase1_bounds_clamps_60s_window_to_max(tmp_path: Path):
    settings = _settings(tmp_path, clip_max_seconds=45.0)
    bounds = phase1_bounds(_candidate(source_ts=100.0), settings)
    assert bounds.duration == pytest.approx(45.0)
    assert bounds.start == pytest.approx(77.5)
    assert bounds.end == pytest.approx(122.5)
    assert bounds.main_ts == pytest.approx(100.0)
    assert bounds.method == "phase1_window"


def test_phase1_bounds_expands_short_window_to_min(tmp_path: Path):
    settings = _settings(tmp_path, clip_min_seconds=10.0)
    bounds = phase1_bounds(
        _candidate(source_ts=100.0, pre_context_seconds=2.0, post_context_seconds=2.0),
        settings,
    )
    assert bounds.duration == pytest.approx(10.0)
    assert bounds.start == pytest.approx(95.0)
    assert bounds.end == pytest.approx(105.0)


def test_phase1_bounds_never_starts_before_zero(tmp_path: Path):
    settings = _settings(tmp_path)
    bounds = phase1_bounds(_candidate(source_ts=3.0), settings)
    assert bounds.start == 0.0
    assert bounds.end == pytest.approx(33.0)
    assert bounds.main_ts == pytest.approx(3.0)


def test_phase1_bounds_keeps_hook_and_payoff_inside(tmp_path: Path):
    settings = _settings(tmp_path, hook_lookback_seconds=8.0, reaction_tail_seconds=3.0)
    bounds = phase1_bounds(_candidate(source_ts=100.0), settings)
    assert bounds.start <= bounds.hook_ts <= bounds.main_ts
    assert bounds.main_ts <= bounds.payoff_ts <= bounds.end
    assert bounds.hook_ts == pytest.approx(92.0)
    assert bounds.payoff_ts == pytest.approx(103.0)


def test_clip_bounds_rejects_inverted_range():
    with pytest.raises(ValueError):
        ClipBounds(start=10.0, end=5.0, main_ts=7.0, hook_ts=6.0, payoff_ts=8.0)


def test_build_plan_marks_placeholder_stage_and_warnings(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = build_plan(
        candidate=_candidate(),
        stream=_stream(),
        source_path=Path("data/source/vod.ts"),
        settings=settings,
    )
    assert plan.stage == "planned"
    assert plan.bounds.method == "phase1_window"
    assert WARN_BOUNDARIES_PENDING in plan.warning_codes()
    assert WARN_LAYOUT_PENDING in plan.warning_codes()
    # 720p into a 1080-wide fit is a downscale, so no upscale warning.
    assert WARN_UPSCALE not in plan.warning_codes()
    # `auto` stays unresolved until composition resolves it - which is what the warning says.
    assert plan.layout.resolved_strategy == ""
    assert plan.layout.width == 1080
    assert plan.layout.height == 1920
    assert plan.audio.target_lufs == pytest.approx(-14.0)
    assert plan.captions.enabled is True
    assert plan.metadata.enabled is True


def test_build_plan_flags_low_res_upscale(tmp_path: Path):
    settings = _settings(tmp_path, quality_warn_upscale=2.0)
    plan = build_plan(
        candidate=_candidate(),
        stream=_stream(source_width=284, source_height=160, capture_quality="160p"),
        source_path=Path("data/source/low.ts"),
        settings=settings,
    )
    assert WARN_UPSCALE in plan.warning_codes()
    assert plan.layout.upscale_factor == pytest.approx(3.803, abs=0.01)
    message = next(w.message for w in plan.warnings if w.code == WARN_UPSCALE)
    assert "284x160" in message


def test_build_plan_warns_when_resolution_unknown(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = build_plan(
        candidate=_candidate(),
        stream=_stream(source_width=None, source_height=None),
        source_path=Path("data/source/unknown.ts"),
        settings=settings,
    )
    assert WARN_RESOLUTION_UNKNOWN in plan.warning_codes()
    assert plan.layout.upscale_factor == pytest.approx(1.0)


def test_build_plan_applies_overrides(tmp_path: Path):
    settings = _settings(tmp_path)
    overrides = EditOverrides(
        strategy="gaming",
        caption_style="block_pop",
        caption_emphasis="off",
        deadair_mode="speed",
        captions_enabled=False,
        zoom=1.5,
    )
    plan = build_plan(
        candidate=_candidate(),
        stream=_stream(),
        source_path=Path("data/source/vod.ts"),
        settings=settings,
        overrides=overrides,
    )
    assert plan.layout.strategy == "gaming"
    assert plan.layout.resolved_strategy == "gaming"
    assert plan.layout.zoom == pytest.approx(1.5)
    # Crop strategies cover the canvas: 1920 / 720 = 2.667, times zoom 1.5.
    assert plan.layout.upscale_factor == pytest.approx(4.0, abs=0.001)
    assert plan.captions.style == "block_pop"
    assert plan.captions.emphasis == "off"
    assert plan.captions.enabled is False
    assert plan.captions.reason == "captions disabled for this render"
    assert plan.deadair.mode == "speed"
    # An explicit strategy drops the auto-resolution warning.
    assert WARN_LAYOUT_PENDING not in plan.warning_codes()


def test_build_plan_rejects_invalid_override(tmp_path: Path):
    settings = _settings(tmp_path)
    with pytest.raises(ValueError):
        build_plan(
            candidate=_candidate(),
            stream=_stream(),
            source_path=Path("data/source/vod.ts"),
            settings=settings,
            overrides=EditOverrides(strategy="cinematic"),
        )


def test_plan_round_trip_through_json(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = build_plan(
        candidate=_candidate(),
        stream=_stream(),
        source_path=Path("data/source/vod.ts"),
        settings=settings,
    )
    payload = json.loads(json.dumps(plan.to_dict()))
    restored = EditPlan.from_dict(payload)
    assert restored.to_dict() == plan.to_dict()


def test_plan_save_and_load(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = build_plan(
        candidate=_candidate(),
        stream=_stream(),
        source_path=Path("data/source/vod.ts"),
        settings=settings,
    )
    paths = EditPaths.for_candidate(settings, plan.candidate_id)
    written = plan.save(paths.plan)
    assert written.name == PLAN_FILENAME
    loaded = EditPlan.load(paths.plan)
    assert loaded.candidate_id == plan.candidate_id
    assert loaded.bounds.duration == pytest.approx(plan.bounds.duration)
    assert loaded.warning_codes() == plan.warning_codes()


def test_add_warning_is_idempotent_per_code(tmp_path: Path):
    settings = _settings(tmp_path)
    plan = build_plan(
        candidate=_candidate(),
        stream=_stream(),
        source_path=Path("data/source/vod.ts"),
        settings=settings,
    )
    before = len(plan.warnings)
    plan.add_warning(WARN_LAYOUT_PENDING, "different message, same code")
    assert len(plan.warnings) == before


def test_source_range_applies_offset(tmp_path: Path):
    settings = _settings(tmp_path, capture_source_offset_seconds=12.5)
    plan = build_plan(
        candidate=_candidate(),
        stream=_stream(source_offset_seconds=0.0),
        source_path=Path("data/source/vod.ts"),
        settings=settings,
    )
    start, end = plan.source_range()
    assert start == pytest.approx(plan.bounds.start + 12.5)
    assert end == pytest.approx(plan.bounds.end + 12.5)


def test_edit_paths_live_under_edits_dir(tmp_path: Path):
    settings = _settings(tmp_path)
    paths = EditPaths.for_candidate(settings, 7)
    assert paths.root == settings.resolved_edits_dir() / "7"
    assert paths.plan == paths.root / PLAN_FILENAME
    assert paths.final.name == "final.mp4"
    assert paths.captions.name == "captions.ass"

