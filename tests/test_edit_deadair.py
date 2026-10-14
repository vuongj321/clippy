from __future__ import annotations

from pathlib import Path

import pytest

from clippy.config import Settings
from clippy.edit.deadair import (
    build_deadair_filter,
    complement_spans,
    merge_spans,
    parse_silences,
    plan_deadair,
    segments_for_render,
    shrink_spans,
    spans_seconds,
    subtract_span,
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "data_dir": tmp_path,
        "deadair_enabled": True,
        "deadair_mode": "cut",
        "deadair_noise_db": -30.0,
        "deadair_min_gap_seconds": 0.8,
        "deadair_keep_pad_seconds": 0.2,
        "deadair_min_keep_seconds": 0.5,
        "deadair_max_removed_ratio": 0.4,
        "deadair_speed_factor": 1.5,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_merge_spans_merges_touching_and_nearby_spans():
    assert merge_spans([(0.0, 1.0), (1.0, 2.0)]) == [(0.0, 2.0)]
    assert merge_spans([(0.0, 1.0), (3.0, 4.0)]) == [(0.0, 1.0), (3.0, 4.0)]
    assert merge_spans([(0.0, 1.0), (3.0, 4.0)], close_gap=2.0) == [(0.0, 4.0)]
    assert merge_spans([(2.0, 1.0), (-5.0, -1.0)]) == []


def test_complement_spans():
    assert complement_spans([(2.0, 4.0)], start=0.0, end=10.0) == [(0.0, 2.0), (4.0, 10.0)]
    assert complement_spans([], start=0.0, end=10.0) == [(0.0, 10.0)]
    assert complement_spans([(0.0, 20.0)], start=0.0, end=10.0) == []


def test_subtract_span():
    assert subtract_span([(0.0, 10.0)], (3.0, 5.0)) == [(0.0, 3.0), (5.0, 10.0)]
    assert subtract_span([(0.0, 10.0)], (0.0, 5.0)) == [(5.0, 10.0)]
    assert subtract_span([(0.0, 10.0)], (0.0, 20.0)) == []
    assert subtract_span([(1.0, 2.0), (5.0, 6.0)], (0.0, 10.0)) == []


def test_shrink_spans_keeps_the_guard_band():
    assert shrink_spans([(2.0, 6.0)], by=0.5) == [(2.5, 5.5)]
    assert shrink_spans([(2.0, 2.5)], by=0.5) == []
    assert shrink_spans([(2.0, 6.0)], by=0.0) == [(2.0, 6.0)]


def test_spans_seconds_sums_lengths():
    assert spans_seconds([(0.0, 1.5), (3.0, 4.0)]) == pytest.approx(2.5)
    assert spans_seconds([]) == 0.0


def test_parse_silences_reads_start_and_end_pairs():
    text = (
        "[silencedetect @ 0000] silence_start: 1.5\n"
        "[silencedetect @ 0000] silence_end: 3.25 | silence_duration: 1.75\n"
    )
    assert parse_silences(text) == [(1.5, 3.25)]


def test_parse_silences_closes_a_dangling_start_at_the_hint():
    text = "[silencedetect @ 0000] silence_start: 7.0\n"
    assert parse_silences(text, end_hint=10.0) == [(7.0, 10.0)]
    # Without a duration hint the trailing silence is unknowable, so it is dropped.
    assert parse_silences(text) == []


def test_build_deadair_filter_for_a_cut():
    graph = build_deadair_filter([(0.0, 3.0, 1.0), (5.0, 8.0, 1.0)], has_audio=True)
    assert "[0:v]trim=start=0.000:end=3.000,setpts=PTS-STARTPTS[v0]" in graph
    assert "[0:a]atrim=start=0.000:end=3.000,asetpts=PTS-STARTPTS[a0]" in graph
    assert "[0:v]trim=start=5.000:end=8.000,setpts=PTS-STARTPTS[v1]" in graph
    assert graph.endswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[vout][aout]")


def test_build_deadair_filter_for_speed_and_silent_video():
    fast = build_deadair_filter([(0.0, 2.0, 1.5)], has_audio=True)
    assert "setpts=(PTS-STARTPTS)/1.500" in fast
    assert "atempo=1.500" in fast

    silent = build_deadair_filter([(0.0, 1.0, 1.0)], has_audio=False)
    assert "[0:a]" not in silent
    assert silent.endswith("concat=n=1:v=1:a=0[vout]")

    assert build_deadair_filter([]) == ""


def test_plan_deadair_cuts_leading_and_trailing_gaps(tmp_path: Path):
    settings = _settings(tmp_path, deadair_max_removed_ratio=1.0)
    cut = plan_deadair(silences=[(0.0, 3.0), (7.0, 10.0)], duration=10.0, settings=settings)

    assert cut.applied is True
    assert cut.mode == "cut"
    assert cut.removed_seconds == pytest.approx(5.6)
    assert cut.keep_segments == [(pytest.approx(2.8), pytest.approx(7.2))]
    assert spans_seconds(cut.gaps) == pytest.approx(5.2)
    assert any("fragment" in note for note in cut.notes)


def test_plan_deadair_protects_the_payoff_region(tmp_path: Path):
    settings = _settings(tmp_path, deadair_max_removed_ratio=1.0)
    cut = plan_deadair(
        silences=[(2.0, 8.0)],
        duration=10.0,
        settings=settings,
        protect=(4.0, 7.0),
    )

    assert cut.removed_seconds == pytest.approx(1.6)
    assert cut.keep_segments == [
        (pytest.approx(0.0), pytest.approx(2.2)),
        (pytest.approx(3.8), pytest.approx(10.0)),
    ]
    assert any("protected" in note for note in cut.notes)


def test_plan_deadair_vetoes_gaps_containing_speech(tmp_path: Path):
    settings = _settings(tmp_path, deadair_max_removed_ratio=1.0)
    cut = plan_deadair(
        silences=[(1.0, 5.0)],
        duration=10.0,
        settings=settings,
        words=[(2.0, 3.0)],
    )

    assert cut.removed_seconds == pytest.approx(1.6)
    assert any("speech" in note for note in cut.notes)


def test_plan_deadair_caps_total_removal(tmp_path: Path):
    settings = _settings(tmp_path, deadair_max_removed_ratio=0.4)
    cut = plan_deadair(silences=[(0.0, 3.0), (7.0, 10.0)], duration=10.0, settings=settings)

    # The cap is 4s, so only one of the two 2.6s gaps is cut and the clip keeps 7.2s.
    assert cut.removed_seconds == pytest.approx(2.8)
    assert cut.gaps == [(pytest.approx(7.2), pytest.approx(9.8))]
    assert any("capped" in note for note in cut.notes)


def test_plan_deadair_speed_mode_compresses_gaps(tmp_path: Path):
    settings = _settings(
        tmp_path, deadair_mode="speed", deadair_speed_factor=1.5, deadair_max_removed_ratio=1.0
    )
    cut = plan_deadair(silences=[(0.0, 3.0), (7.0, 10.0)], duration=10.0, settings=settings)

    assert cut.applied is True
    assert cut.mode == "speed"
    # Nothing is dropped: the gaps are shortened, so 5.2s of gap saves ~1.73s.
    assert cut.keep_segments == [(pytest.approx(0.0), pytest.approx(10.0))]
    assert cut.removed_seconds == pytest.approx(1.733, abs=0.01)
    assert any("compressed" in note for note in cut.notes)

    segments = segments_for_render(cut, speed_factor=1.5)
    assert len(segments) == 5
    assert segments[0] == (pytest.approx(0.0), pytest.approx(0.2), 1.0)
    assert segments[1][2] == 1.5
    assert segments[-1] == (pytest.approx(9.8), pytest.approx(10.0), 1.0)


def test_plan_deadair_reports_when_there_is_nothing_to_do(tmp_path: Path):
    settings = _settings(tmp_path)
    quiet = plan_deadair(silences=[], duration=10.0, settings=settings)
    assert quiet.applied is False
    assert quiet.keep_segments == [(0.0, 10.0)]
    assert any("no silence" in note for note in quiet.notes)

    disabled = plan_deadair(
        silences=[(0.0, 3.0)],
        duration=10.0,
        settings=_settings(tmp_path, deadair_enabled=False),
    )
    assert disabled.applied is False
    assert any("disabled" in note for note in disabled.notes)

    empty = plan_deadair(silences=[(0.0, 3.0)], duration=0.0, settings=settings)
    assert empty.applied is False
    assert any("no duration" in note for note in empty.notes)

    all_vetoed = plan_deadair(
        silences=[(1.0, 2.0)], duration=10.0, settings=settings
    )
    assert all_vetoed.applied is False
    assert any("too short" in note for note in all_vetoed.notes)


def test_segments_for_render_in_cut_mode_uses_the_keeps(tmp_path: Path):
    settings = _settings(tmp_path, deadair_max_removed_ratio=1.0)
    cut = plan_deadair(silences=[(0.0, 3.0)], duration=10.0, settings=settings)
    segments = segments_for_render(cut)
    assert segments == [(start, end, 1.0) for start, end in cut.keep_segments]

