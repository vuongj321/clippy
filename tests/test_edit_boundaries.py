from __future__ import annotations

from pathlib import Path

import pytest

from clippy.caption.asr import TranscriptWord
from clippy.config import Settings
from clippy.edit.boundaries import (
    ContextEvidence,
    Word,
    adjust_end,
    adjust_start,
    apply_llm_bounds,
    audio_decay_ts,
    build_context_evidence,
    chat_burst_end_ts,
    detect_bounds,
    enforce_constraints,
    evidence_from_chat,
    refine_bounds_with_llm,
    speech_gaps,
    utterance_containing,
    words_to_evidence,
    words_to_utterances,
)
from clippy.store.db import Candidate


def _candidate(**overrides) -> Candidate:
    base = {
        "id": 5,
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


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "data_dir": tmp_path,
        "clip_min_seconds": 10.0,
        "clip_max_seconds": 45.0,
        "clip_target_seconds": 30.0,
        "hook_lookback_seconds": 8.0,
        "min_context_seconds": 1.5,
        "reaction_tail_seconds": 3.0,
        "chat_window_seconds": 5.0,
        "chat_min_rate": 0.5,
        "audio_baseline_seconds": 30.0,
        "audio_min_rms": 0.02,
        "boundary_llm_refine": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _words(spec: list[tuple[float, float, str]]) -> list[Word]:
    return [Word(start=start, end=end, text=text) for start, end, text in spec]


def test_words_to_utterances_splits_on_the_configured_gap():
    words = _words([(0.0, 1.0, "a"), (1.1, 2.0, "b"), (3.0, 4.0, "c")])
    spans = words_to_utterances(words, gap_seconds=0.6)
    assert [(s.start, s.end) for s in spans] == [(0.0, 2.0), (3.0, 4.0)]
    assert spans[0].text == "a b"


def test_utterance_containing_only_matches_inside_speech():
    spans = words_to_utterances(_words([(0.0, 1.0, "a"), (1.1, 2.0, "b"), (3.0, 4.0, "c")]))
    assert utterance_containing(spans, 1.5) is not None
    assert utterance_containing(spans, 2.5) is None


def test_speech_gaps_are_clipped_to_the_search_span():
    spans = words_to_utterances(_words([(0.0, 2.0, "a"), (3.0, 4.0, "b")]))
    gaps = speech_gaps(spans, span_start=2.5, span_end=5.0)
    # The leading gap starts at the span floor, and the trailing gap runs to the ceiling.
    assert gaps == [(2.5, 3.0), (4.0, 5.0)]


def test_adjust_start_moves_back_out_of_a_word():
    words = _words([(90.0, 92.0, "so"), (92.2, 93.5, "anyway")])
    start, note = adjust_start(
        91.0, words=words, gaps=[], tolerance_seconds=1.5, floor=70.0
    )
    assert start == pytest.approx(90.0)
    assert note is not None and "beginning of a word" in note


def test_adjust_start_snaps_into_nearby_silence():
    words = _words([(90.0, 92.0, "so"), (92.2, 93.5, "anyway")])
    start, note = adjust_start(
        93.7, words=words, gaps=[(93.5, 94.0)], tolerance_seconds=1.5, floor=70.0
    )
    assert start == pytest.approx(93.5)
    assert note is not None and "silence" in note


def test_adjust_start_leaves_a_distant_target_alone():
    words = _words([(90.0, 92.0, "so")])
    start, note = adjust_start(
        96.0, words=words, gaps=[(92.0, 93.0)], tolerance_seconds=1.5, floor=70.0
    )
    assert start == pytest.approx(96.0)
    assert note is None


def test_adjust_end_extends_to_finish_a_word():
    words = _words([(102.0, 104.0, "wild")])
    end, note = adjust_end(103.0, words=words, gaps=[], tolerance_seconds=2.5)
    assert end == pytest.approx(104.0)
    assert note is not None and "finish a word" in note


def test_adjust_end_snaps_into_nearby_silence():
    words = _words([(102.0, 104.0, "wild")])
    end, note = adjust_end(
        104.2, words=words, gaps=[(104.0, 106.0)], tolerance_seconds=2.5
    )
    assert end == pytest.approx(106.0)
    assert note is not None and "silence" in note


def test_enforce_moves_a_start_that_begins_after_the_main_event(tmp_path: Path):
    settings = _settings(tmp_path)
    start, end, notes = enforce_constraints(
        105.0, 120.0, main_ts=100.0, payoff_ts=101.0, settings=settings, floor=70.0, ceiling=130.0
    )
    assert (start, end) == (pytest.approx(98.5), pytest.approx(120.0))
    assert any("must not begin after the main event" in note for note in notes)


def test_enforce_extends_the_end_to_keep_the_payoff(tmp_path: Path):
    settings = _settings(tmp_path)
    start, end, notes = enforce_constraints(
        80.0, 95.0, main_ts=100.0, payoff_ts=104.0, settings=settings, floor=70.0, ceiling=130.0
    )
    assert (start, end) == (pytest.approx(80.0), pytest.approx(104.0))
    assert any("keep the payoff" in note for note in notes)


def test_enforce_trims_the_start_first_for_the_ceiling(tmp_path: Path):
    settings = _settings(tmp_path)
    start, end, notes = enforce_constraints(
        70.0, 130.0, main_ts=120.0, payoff_ts=125.0, settings=settings, floor=60.0, ceiling=140.0
    )
    assert end - start == pytest.approx(45.0)
    assert start == pytest.approx(85.0)
    assert any("start trimmed 15.00s" in note for note in notes)


def test_enforce_trims_the_end_when_the_start_cannot_move_far_enough(tmp_path: Path):
    settings = _settings(tmp_path)
    start, end, notes = enforce_constraints(
        110.0, 180.0, main_ts=115.0, payoff_ts=140.0, settings=settings, floor=110.0, ceiling=200.0
    )
    # The start stops at main - min_context, and the tail absorbs the rest.
    assert start == pytest.approx(113.5)
    assert end == pytest.approx(158.5)
    assert any("payoff may be cut" in note for note in notes)


def test_enforce_extends_to_the_minimum_length(tmp_path: Path):
    settings = _settings(tmp_path)
    start, end, notes = enforce_constraints(
        98.0, 101.0, main_ts=100.0, payoff_ts=101.0, settings=settings, floor=70.0, ceiling=130.0
    )
    assert (start, end) == (pytest.approx(98.0), pytest.approx(108.0))
    assert any("minimum length" in note for note in notes)


def test_enforce_reaches_backwards_when_there_is_no_room_forward(tmp_path: Path):
    settings = _settings(tmp_path)
    start, end, _notes = enforce_constraints(
        126.0, 129.0, main_ts=127.0, payoff_ts=127.5, settings=settings, floor=70.0, ceiling=130.0
    )
    assert end - start == pytest.approx(10.0)
    assert start == pytest.approx(119.0)


def test_enforce_accepts_a_short_clip_when_the_window_is_too_small(tmp_path: Path):
    settings = _settings(tmp_path)
    start, end, notes = enforce_constraints(
        101.0, 102.0, main_ts=101.5, payoff_ts=102.0, settings=settings, floor=100.0, ceiling=104.0
    )
    assert (start, end) == (pytest.approx(100.0), pytest.approx(104.0))
    assert any("shorter than the configured minimum" in note for note in notes)


def test_enforce_swaps_an_inverted_range(tmp_path: Path):
    settings = _settings(tmp_path)
    start, end, _notes = enforce_constraints(
        120.0, 100.0, main_ts=110.0, payoff_ts=110.0, settings=settings, floor=0.0, ceiling=200.0
    )
    assert (start, end) == (pytest.approx(100.0), pytest.approx(120.0))


def test_chat_burst_end_detects_decay(tmp_path: Path):
    settings = _settings(tmp_path, chat_window_seconds=5.0, chat_min_rate=0.5)
    dense = [95.0 + index * 0.5 for index in range(13)]  # 2 messages/s through 101.0
    decayed = chat_burst_end_ts(
        dense, peak_ts=100.0, settings=settings, search_seconds=3.0
    )
    assert decayed == pytest.approx(102.5)

    sustained = [95.0 + index * 0.5 for index in range(21)]  # still busy at the cap
    capped = chat_burst_end_ts(
        sustained, peak_ts=100.0, settings=settings, search_seconds=3.0
    )
    assert capped == pytest.approx(103.0)


def test_chat_burst_end_without_chat(tmp_path: Path):
    settings = _settings(tmp_path)
    assert (
        chat_burst_end_ts([], peak_ts=100.0, settings=settings, search_seconds=3.0) is None
    )


def test_audio_decay_finds_the_drop_back_to_baseline(tmp_path: Path):
    settings = _settings(tmp_path, audio_baseline_seconds=30.0, audio_min_rms=0.02)
    times = [float(t) for t in range(90, 111)]
    values = [0.02] * 10 + [0.3] * 5 + [0.028] * 6  # loud 100..104, quiet from 105
    decay = audio_decay_ts(
        times, values, peak_ts=100.0, settings=settings, search_seconds=10.0
    )
    assert decay == pytest.approx(105.0)


def test_audio_decay_rejects_mismatched_series(tmp_path: Path):
    settings = _settings(tmp_path)
    assert (
        audio_decay_ts([1.0, 2.0], [0.1], peak_ts=1.0, settings=settings, search_seconds=5.0)
        is None
    )


def _chat_burst_decision(tmp_path: Path) -> tuple[Settings, object]:
    settings = _settings(tmp_path)
    chat = [95.0 + index * 0.5 for index in range(21)]
    return settings, detect_bounds(
        candidate=_candidate(),
        settings=settings,
        evidence=evidence_from_chat(chat, start=70.0, end=130.0),
    )


def test_detect_bounds_falls_back_without_evidence(tmp_path: Path):
    settings = _settings(tmp_path)
    decision = detect_bounds(candidate=_candidate(), settings=settings)
    assert decision.bounds.method == "phase1_window"
    assert decision.bounds.duration == pytest.approx(45.0)
    assert "Phase 1" in decision.adjustments[0]


def test_detect_bounds_uses_chat_decay_without_a_transcript(tmp_path: Path):
    settings, decision = _chat_burst_decision(tmp_path)
    assert decision.bounds.method == "signal_evidence"
    assert decision.bounds.start == pytest.approx(73.0)
    assert decision.bounds.end == pytest.approx(103.0)
    assert decision.bounds.duration == pytest.approx(30.0)
    assert decision.evidence.chat_burst_end_ts == pytest.approx(103.0)


def test_detect_bounds_uses_the_setup_utterance_as_the_hook(tmp_path: Path):
    settings = _settings(tmp_path)
    words = _words(
        [
            (90.0, 92.0, "so"),
            (92.2, 95.0, "anyway"),
            (98.0, 100.0, "he"),
            (100.0, 102.0, "actually"),
            (102.0, 104.0, "did"),
            (106.0, 108.0, "wow"),
            (108.0, 110.0, "insane"),
        ]
    )
    decision = detect_bounds(
        candidate=_candidate(), settings=settings, evidence=ContextEvidence(words=words)
    )
    assert decision.bounds.method == "signal_evidence"
    assert decision.bounds.start == pytest.approx(92.0)
    assert decision.bounds.end == pytest.approx(106.0)
    assert decision.bounds.payoff_ts == pytest.approx(104.0)
    assert decision.evidence.hook_utterance_start_ts == pytest.approx(90.0)
    assert decision.evidence.natural_end_ts == pytest.approx(104.0)


def test_detect_bounds_never_starts_mid_word(tmp_path: Path):
    settings = _settings(tmp_path)
    words = _words([(89.0, 93.0, "sooo"), (98.0, 100.0, "he"), (100.0, 102.0, "did")])
    decision = detect_bounds(
        candidate=_candidate(), settings=settings, evidence=ContextEvidence(words=words)
    )
    assert decision.bounds.start == pytest.approx(89.0)
    assert any("beginning of a word" in note for note in decision.adjustments)


def test_detect_bounds_extends_for_the_reaction_within_the_tail(tmp_path: Path):
    settings = _settings(tmp_path)
    words = _words([(99.0, 100.0, "wow")])
    times = [98.0, 99.0, 100.0, 101.0, 102.0, 103.0]
    values = [0.02, 0.02, 0.3, 0.3, 0.3, 0.028]
    decision = detect_bounds(
        candidate=_candidate(),
        settings=settings,
        evidence=ContextEvidence(words=words, rms_times=times, rms_values=values),
    )
    assert decision.bounds.payoff_ts == pytest.approx(103.0)
    assert decision.bounds.end == pytest.approx(103.0)
    assert decision.evidence.audio_decay_ts == pytest.approx(103.0)


def test_apply_llm_bounds_clamps_a_start_after_the_main_event(tmp_path: Path):
    settings, decision = _chat_burst_decision(tmp_path)
    refined = apply_llm_bounds(
        {"start": 120.0, "end": 130.0, "main": 100.0, "payoff": 101.0, "rationale": "trust me"},
        decision,  # type: ignore[arg-type]
        settings=settings,
        floor=70.0,
        ceiling=130.0,
    )
    assert refined.bounds.method == "llm_refined"
    assert refined.bounds.start == pytest.approx(98.5)
    assert refined.bounds.end == pytest.approx(130.0)
    assert refined.bounds.notes == "trust me"
    assert any("clamped" in note for note in refined.adjustments)


def test_apply_llm_bounds_keeps_the_decision_for_missing_values(tmp_path: Path):
    settings, decision = _chat_burst_decision(tmp_path)
    refined = apply_llm_bounds(
        {"start": "loud", "end": None},
        decision,  # type: ignore[arg-type]
        settings=settings,
        floor=70.0,
        ceiling=130.0,
    )
    assert refined.bounds.start == pytest.approx(decision.bounds.start)  # type: ignore[attr-defined]
    assert refined.bounds.end == pytest.approx(decision.bounds.end)  # type: ignore[attr-defined]
    assert refined.bounds.method == "llm_refined"


def test_refine_bounds_with_llm_is_off_by_default(tmp_path: Path):
    settings, decision = _chat_burst_decision(tmp_path)
    calls: list[dict] = []

    def call_llm(payload: dict):
        calls.append(payload)
        return {"start": 90.0}

    result = refine_bounds_with_llm(
        decision,  # type: ignore[arg-type]
        call_llm=call_llm,
        settings=settings,
        floor=70.0,
        ceiling=130.0,
        payload={"transcript": "..."},
    )
def test_refine_bounds_with_llm_keeps_the_result_on_error(tmp_path: Path):
    settings, decision = _chat_burst_decision(tmp_path)
    enabled = settings.model_copy(update={"boundary_llm_refine": True})

    def boom(payload: dict):
        raise RuntimeError("429")

    result = refine_bounds_with_llm(
        decision,  # type: ignore[arg-type]
        call_llm=boom,
        settings=enabled,
        floor=70.0,
        ceiling=130.0,
        payload={},
    )
    assert result is decision


def test_refine_bounds_with_llm_ignores_a_non_mapping(tmp_path: Path):
    settings, decision = _chat_burst_decision(tmp_path)
    enabled = settings.model_copy(update={"boundary_llm_refine": True})
    result = refine_bounds_with_llm(
        decision,  # type: ignore[arg-type]
        call_llm=lambda payload: "nope",
        settings=enabled,
        floor=70.0,
        ceiling=130.0,
        payload={},
    )
    assert result is decision


def test_refine_bounds_with_llm_applies_a_valid_proposal(tmp_path: Path):
    settings, decision = _chat_burst_decision(tmp_path)
    enabled = settings.model_copy(update={"boundary_llm_refine": True})
    refined = refine_bounds_with_llm(
        decision,  # type: ignore[arg-type]
        call_llm=lambda payload: {
            "start": 96.0,
            "end": 108.0,
            "main": 100.0,
            "payoff": 104.0,
            "rationale": "setup then payoff",
        },
        settings=enabled,
        floor=70.0,
        ceiling=130.0,
        payload={"transcript": "..."},
    )
    assert refined.bounds.method == "llm_refined"
    assert refined.bounds.start == pytest.approx(96.0)
    assert refined.bounds.end == pytest.approx(108.0)
    assert refined.bounds.notes == "setup then payoff"


def test_evidence_from_chat_filters_to_the_window():
    evidence = evidence_from_chat([10.0, 50.0, 120.0, 300.0], start=40.0, end=130.0)
    assert evidence.chat_times == [50.0, 120.0]
    assert evidence.has_transcript() is False
    assert evidence.has_audio() is False


def test_context_evidence_requires_paired_audio_series():
    assert ContextEvidence(rms_times=[1.0], rms_values=[0.1]).has_audio() is True
    assert ContextEvidence(rms_times=[1.0, 2.0], rms_values=[0.1]).has_audio() is False
    assert ContextEvidence(words=[Word(start=0.0, end=1.0)]).has_transcript() is True


def test_words_to_evidence_offsets_onto_the_source_timeline():
    window = [TranscriptWord(0.0, 0.5, "he"), TranscriptWord(0.6, 1.0, "did")]
    words = words_to_evidence(window, window_start=90.0)
    assert [(word.start, word.end) for word in words] == [(90.0, 90.5), (90.6, 91.0)]
    assert [word.text for word in words] == ["he", "did"]


def test_words_to_evidence_drops_zero_length_words():
    window = [TranscriptWord(1.0, 1.0, "uh"), TranscriptWord(2.0, 2.5, "ok")]
    words = words_to_evidence(window, window_start=10.0)
    assert [(word.start, word.end) for word in words] == [(12.0, 12.5)]


def test_build_context_evidence_filters_and_keeps_audio_paired():
    evidence = build_context_evidence(
        start=90.0,
        end=100.0,
        chat_times=[80.0, 95.0, 140.0],
        words=_words([(85.0, 89.0, "before"), (92.0, 93.0, "in"), (120.0, 121.0, "after")]),
        rms_times=[88.0, 92.0, 120.0],
        rms_values=[0.01, 0.3, 0.2],
    )
    assert evidence.chat_times == [95.0]
    assert [word.text for word in evidence.words] == ["in"]
    assert evidence.rms_times == [92.0]
    assert evidence.rms_values == [0.3]
    assert evidence.has_audio() is True


def test_words_to_evidence_drive_detect_bounds(tmp_path: Path):
    """Word timings on the source timeline make the hook snap to the setup line."""
    settings = _settings(tmp_path, hook_lookback_seconds=30.0)
    window = [
        TranscriptWord(0.0, 2.0, "so"),
        TranscriptWord(2.2, 5.0, "anyway"),
        TranscriptWord(8.0, 10.0, "he"),
        TranscriptWord(10.0, 12.0, "did"),
        TranscriptWord(16.0, 18.0, "wow"),
    ]
    evidence = build_context_evidence(
        start=70.0, end=130.0, words=words_to_evidence(window, window_start=90.0)
    )
    decision = detect_bounds(
        candidate=_candidate(), settings=settings, evidence=evidence
    )
    assert decision.bounds.method == "signal_evidence"
    # hook walks back to the setup utterance (90.0) rather than a fixed offset
    assert decision.bounds.start == pytest.approx(90.0)
    assert decision.evidence.hook_utterance_start_ts == pytest.approx(90.0)
    # the end snaps to the gap after the sentence that carries the main event
    assert decision.bounds.end == pytest.approx(102.0)



