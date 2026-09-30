from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from clippy.caption.align import (
    Cue,
    build_cues,
    cue_text,
    ends_sentence,
    pick_emphasis_words,
)
from clippy.caption.asr import (
    TranscriptPayload,
    TranscriptSegment,
    TranscriptWord,
    _payload_from_response,
    transcribe_words,
)
from clippy.caption.ass import escape_ass_text, format_timestamp, write_ass
from clippy.caption.emphasis import (
    parse_emphasis_response,
    transcript_word_forms,
    validate_emphasis_words,
)
from clippy.caption.styles import caption_anchor, resolve_anchor, resolve_style
from clippy.config import Settings
from clippy.edit.captions import generate_captions
from clippy.edit.plan import WARN_EMPHASIS_FALLBACK, EditPaths, EditPlan, build_plan
from clippy.store.db import Candidate, Stream


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "data_dir": tmp_path,
        "caption_enabled": True,
        "caption_style": "karaoke_highlight",
        "caption_emphasis": "heuristic",
        "caption_max_chars_per_line": 18,
        "caption_max_lines": 2,
        "caption_max_cue_seconds": 2.2,
        "caption_min_cue_seconds": 0.5,
        "caption_break_gap_seconds": 0.35,
        "caption_uppercase": True,
        "caption_safe_area": "auto",
        "asr_word_timestamps": True,
        "openai_api_key": "sk-test",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _w(text: str, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(start=start, end=end, text=text)


def test_build_cues_breaks_on_pauses_and_sentence_endings():
    words = [
        _w("Hello", 0.0, 0.4),
        _w("there.", 0.5, 1.0),
        _w("Next", 2.0, 2.4),
        _w("one", 2.5, 2.9),
    ]
    cues = build_cues(words, uppercase=False)
    assert len(cues) == 2
    assert cues[0].text == "Hello there."
    assert cues[1].text == "Next one"
    assert cues[0].end <= cues[1].start


def test_build_cues_respects_the_character_budget():
    words = [_w("word", index * 0.5, index * 0.5 + 0.4) for index in range(12)]
    cues = build_cues(words, max_chars_per_line=10, max_lines=2, uppercase=False)
    assert all(len(cue.text) <= 20 for cue in cues)
    assert len(cues) >= 3


def test_build_cues_respects_the_duration_ceiling():
    words = [_w("slow", index * 1.0, index * 1.0 + 0.9) for index in range(4)]
    cues = build_cues(words, max_cue_seconds=1.5, uppercase=False)
    assert all(cue.duration <= 1.5 + 0.2 for cue in cues)


def test_build_cues_never_overlaps_and_stretches_short_cues():
    words = [_w("a", 0.0, 0.05), _w("b", 0.1, 0.12), _w("c", 0.2, 3.5)]
    cues = build_cues(words, min_cue_seconds=0.5, uppercase=False)
    assert [cue.start for cue in cues] == sorted(cue.start for cue in cues)
    for index, cue in enumerate(cues[:-1]):
        assert cue.end <= cues[index + 1].start
    assert all(cue.duration > 0 for cue in cues)


def test_build_cues_uppercases_when_asked():
    cues = build_cues([_w("hello", 0.0, 0.5), _w("world", 0.6, 1.0)], uppercase=True)
    assert cues[0].text == "HELLO WORLD"


def test_ends_sentence_detects_punctuation():
    assert ends_sentence("done!") is True
    assert ends_sentence("done") is False
    assert ends_sentence("") is False


def test_pick_emphasis_words_scores_shouting_numbers_and_keywords():
    cues = build_cues(
        [
            _w("THIS", 0.0, 0.3),
            _w("is", 0.4, 0.6),
            _w("1000", 0.7, 1.0),
            _w("clip", 1.1, 1.4),
            _w("that", 1.5, 1.7),
            _w("moment", 1.8, 2.4),
        ],
        uppercase=False,
    )
    emphasis = pick_emphasis_words(cues, keywords=["clip", "clip that"])
    assert "this" in emphasis
    assert "1000" in emphasis
    assert "clip" in emphasis
    # A multi-word phrase must not leak its parts, or every "that" lights up.
    assert "that" not in emphasis


def test_cue_text_joins_the_track():
    cues = build_cues([_w("one", 0.0, 0.4), _w("two", 0.5, 0.9)], uppercase=False)
    assert cue_text(cues) == "one two"


def test_format_timestamp_handles_carries():
    assert format_timestamp(0) == "0:00:00.00"
    assert format_timestamp(61.234) == "0:01:01.23"
    assert format_timestamp(3599.999) == "1:00:00.00"
    assert format_timestamp(-5) == "0:00:00.00"


def test_escape_ass_text_neutralises_control_characters():
    assert escape_ass_text("{hello}") == "(hello)"
    assert escape_ass_text("back\\slash") == "back\\\\slash"
    assert escape_ass_text("two\nlines") == "two lines"


def test_write_ass_emits_a_karaoke_track(tmp_path: Path):
    settings = _settings(tmp_path)
    style = resolve_style("karaoke_highlight", settings)
    cues = build_cues(
        [_w("THIS", 0.0, 0.5), _w("is", 0.5, 0.8)], uppercase=style.uppercase
    )
    path = write_ass(
        cues, style, path=tmp_path / "captions.ass", emphasis={"this"}
    )

    text = path.read_text(encoding="utf-8")
    assert "[Script Info]" in text
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "Style: Caption,Arial" in text
    assert text.count("Dialogue: 0,") == len(cues)
    assert "\\k50" in text  # 0.5s word as karaoke centiseconds
    assert style.highlight_colour in text
    assert "\\fscx" not in text  # karaoke presets carry no pop scale


def test_write_ass_plain_style_has_no_karaoke_tags(tmp_path: Path):
    settings = _settings(tmp_path)
    style = resolve_style("minimal", settings)
    cues = build_cues([_w("Hello", 0.0, 0.6), _w("world", 0.6, 1.2)], uppercase=False)
    text = write_ass(cues, style, path=tmp_path / "c.ass", emphasis=set()).read_text(
        encoding="utf-8"
    )
    assert "\\k" not in text
    assert "HELLO" not in text


def test_write_ass_with_no_cues_is_still_valid(tmp_path: Path):
    settings = _settings(tmp_path)
    style = resolve_style("karaoke_highlight", settings)
    text = write_ass([], style, path=tmp_path / "empty.ass").read_text(encoding="utf-8")
    assert "[Events]" in text
    assert "Dialogue:" not in text


def test_resolve_style_falls_back_for_unknown_names(tmp_path: Path):
    settings = _settings(tmp_path, caption_style="karaoke_highlight")
    style = resolve_style("nonsense", settings)
    assert style.name == "karaoke_highlight"
    assert style.karaoke is True


def test_caption_anchor_flips_to_the_top_band(tmp_path: Path):
    settings = _settings(tmp_path)
    style = resolve_style("karaoke_highlight", settings)
    flipped, anchor = caption_anchor(style, prefer_top=True)
    assert anchor == "top"
    assert flipped.alignment == 8
    assert caption_anchor(style, prefer_top=False)[1] == "bottom"


def test_payload_from_response_parses_words_and_interpolates_missing_ones():
    payload = _payload_from_response(
        {
            "text": "hi there",
            "language": "en",
            "duration": 2.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hi",
                    "words": [{"word": "hi", "start": 0.1, "end": 0.4}],
                },
                {"start": 1.0, "end": 2.0, "text": "there"},
            ],
        }
    )
    assert payload.language == "en"
    assert payload.has_word_timings() is True
    words = payload.words()
    assert [word.text for word in words] == ["hi", "there"]
    assert words[0].start == pytest.approx(0.1)
    # The segment without word timings is interpolated across its own span.
    assert words[1].start == pytest.approx(1.0)
    assert words[1].end == pytest.approx(2.0)


def test_transcribe_words_requests_word_granularity(monkeypatch, tmp_path: Path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x" * 32)
    seen: dict = {}

    def fake_post(url, *, api_key, files, data, timeout):
        seen["url"] = url
        seen["data"] = data
        return {
            "text": "ok",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "ok", "words": [{"word": "ok", "start": 0.0, "end": 0.5}]}
            ],
        }

    monkeypatch.setattr("clippy.caption.asr._post_transcription", fake_post)
    payload = transcribe_words(media, api_key="sk-test", model="whisper-large-v3")

    assert seen["data"]["response_format"] == "verbose_json"
    assert seen["data"]["timestamp_granularities[]"] == ["word", "segment"]
    assert seen["data"]["model"] == "whisper-large-v3"
    assert payload.has_word_timings() is True


def test_transcribe_words_falls_back_when_words_are_rejected(monkeypatch, tmp_path: Path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x" * 32)
    calls: list[dict] = []

    def fake_post(url, *, api_key, files, data, timeout):
        calls.append(data)
        if len(calls) == 1:
            request = httpx.Request("POST", url)
            response = httpx.Response(400, request=request)
            raise httpx.HTTPStatusError("bad request", request=request, response=response)
        return {"text": "hello world", "segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}]}

    monkeypatch.setattr("clippy.caption.asr._post_transcription", fake_post)
    payload = transcribe_words(media, api_key="sk-test")

    assert len(calls) == 2
    assert "timestamp_granularities[]" not in calls[1]
    assert payload.has_word_timings() is False
    words = payload.words()
    assert [word.text for word in words] == ["hello", "world"]
    assert words[0].start == pytest.approx(0.0)
    assert words[1].end == pytest.approx(2.0)


def test_transcribe_words_reads_top_level_word_timings(monkeypatch, tmp_path: Path):
    """Groq accepts the granularity parameter but returns `words` beside `segments`."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x" * 32)

    def fake_post(url, *, api_key, files, data, timeout):
        return {
            "text": "hi there",
            "segments": [{"start": 0.0, "end": 2.0, "text": "hi there"}],
            "words": [
                {"word": "hi", "start": 0.1, "end": 0.4},
                {"word": "there", "start": 0.5, "end": 0.9},
            ],
        }

    monkeypatch.setattr("clippy.caption.asr._post_transcription", fake_post)
    payload = transcribe_words(media, api_key="sk-test")

    assert payload.has_word_timings() is True
    words = payload.words()
    assert [word.text for word in words] == ["hi", "there"]
    assert words[0].start == pytest.approx(0.1)
    assert words[1].end == pytest.approx(0.9)


def _captions_plan(tmp_path: Path, **overrides):
    settings = _settings(tmp_path, **overrides)
    stream = Stream(
        id=1,
        streamer_id=1,
        mode="vod",
        source_url=None,
        vod_id=None,
        media_path="vod.ts",
        started_at="2026-01-01T00:00:00+00:00",
        created_at="2026-01-01T00:00:00+00:00",
    )
    candidate = Candidate(
        id=1,
        stream_id=1,
        source_ts=5.0,
        pre_context_seconds=5.0,
        post_context_seconds=5.0,
        signals={},
        score=0.5,
        media_path=None,
        status="pending",
        created_at="2026-01-01T00:00:00+00:00",
    )
    plan = build_plan(
        candidate=candidate, stream=stream, source_path=Path("vod.ts"), settings=settings
    )
    paths = EditPaths.for_candidate(settings, candidate.id)
    paths.ensure_root()
    paths.trimmed.write_bytes(b"fake trimmed clip")
    return settings, plan, paths


def _payload() -> TranscriptPayload:
    return TranscriptPayload(
        text="this is crazy",
        language="en",
        duration=2.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="this is crazy",
                words=[
                    TranscriptWord(0.0, 0.4, "THIS"),
                    TranscriptWord(0.5, 0.8, "is"),
                    TranscriptWord(0.9, 1.4, "crazy"),
                ],
            )
        ],
    )


def test_generate_captions_writes_the_ass_and_transcript(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path)
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())

    result = generate_captions(plan, paths, settings=settings)

    assert result.applied is True
    assert result.captions_path == paths.captions
    assert paths.captions.exists()
    assert paths.transcript.exists()
    assert "Dialogue:" in paths.captions.read_text(encoding="utf-8")

    result.commit(plan)
    assert plan.captions.cue_count == len(result.cues)
    assert plan.captions.reason is None
    assert plan.captions.anchor == "bottom"
    assert plan.captions.emphasis_words  # "crazy" is an emotion word


def test_generate_captions_can_move_to_the_top_band(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path)
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())

    result = generate_captions(plan, paths, settings=settings, prefer_top=True)

    assert result.anchor == "top"
    assert "Style: Caption,Arial,54" in paths.captions.read_text(encoding="utf-8")


def test_generate_captions_degrades_without_a_key(tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path, openai_api_key="")
    result = generate_captions(plan, paths, settings=settings)
    assert result.applied is False
    assert result.reason is not None and "no API key" in result.reason


def test_generate_captions_degrades_when_disabled(tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path, caption_enabled=False)
    result = generate_captions(plan, paths, settings=settings)
    assert result.applied is False
    assert result.reason is not None and "disabled" in result.reason


def test_generate_captions_reports_a_transcription_failure(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("429 too many requests")

    monkeypatch.setattr("clippy.edit.captions.transcribe_words", boom)
    result = generate_captions(plan, paths, settings=settings)

    assert result.applied is False
    assert result.reason is not None and result.reason.startswith("transcription failed")


def test_generate_captions_reuses_a_cached_transcript(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path)
    paths.transcript.write_text(json.dumps(_payload().to_dict()), encoding="utf-8")

    def never(*args, **kwargs):
        raise AssertionError("ASR must not run when a transcript is cached")

    monkeypatch.setattr("clippy.edit.captions.transcribe_words", never)
    result = generate_captions(plan, paths, settings=settings)
    assert result.applied is True


def test_generate_captions_reports_an_empty_transcript(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path)
    monkeypatch.setattr(
        "clippy.edit.captions.transcribe_words",
        lambda *a, **k: TranscriptPayload(text="", segments=[]),
    )
    result = generate_captions(plan, paths, settings=settings)
    assert result.applied is False
    assert result.reason is not None and "no cues" in result.reason


def test_generate_captions_respects_emphasis_off(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path, caption_emphasis="off")
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())
    result = generate_captions(plan, paths, settings=settings)
    assert result.applied is True
    assert result.emphasis == set()





def _style_alignment(path: Path) -> str:
    """Pull the Alignment field out of the generated ASS style line."""
    line = next(
        item
        for item in path.read_text(encoding="utf-8").splitlines()
        if item.startswith("Style: Caption")
    )
    return line.split(",")[18]


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_resolve_anchor_maps_every_band(tmp_path: Path):
    settings = _settings(tmp_path)
    style = resolve_style("karaoke_highlight", settings)

    assert resolve_anchor(style, safe_area="bottom", prefer_top=False)[1] == "bottom"
    assert resolve_anchor(style, safe_area="middle", prefer_top=False)[1] == "middle"
    assert resolve_anchor(style, safe_area="top", prefer_top=False)[1] == "top"
    assert resolve_anchor(style, safe_area="bottom", prefer_top=False)[0].alignment == 2
    assert resolve_anchor(style, safe_area="middle", prefer_top=False)[0].alignment == 5
    assert resolve_anchor(style, safe_area="top", prefer_top=False)[0].alignment == 8


def test_resolve_anchor_explicit_band_ignores_the_subject(tmp_path: Path):
    """`auto` follows the frame; an explicit choice is the reviewer's and must win."""
    settings = _settings(tmp_path)
    style = resolve_style("karaoke_highlight", settings)

    # prefer_top asks for the top band, but the reviewer pinned the captions to the bottom.
    pinned, anchor = resolve_anchor(style, safe_area="bottom", prefer_top=True)
    assert anchor == "bottom"
    assert pinned.alignment == 2

    # `auto` keeps the frame-aware behaviour.
    assert resolve_anchor(style, safe_area="auto", prefer_top=True)[1] == "top"
    assert resolve_anchor(style, safe_area="auto", prefer_top=False)[1] == "bottom"


def test_resolve_anchor_unknown_value_falls_back_to_bottom(tmp_path: Path):
    """A stale config value must not fail a render."""
    settings = _settings(tmp_path)
    style = resolve_style("karaoke_highlight", settings)
    resolved, anchor = resolve_anchor(style, safe_area="nonsense", prefer_top=False)
    assert anchor == "bottom"
    assert resolved.alignment == 2


def test_generate_captions_burns_the_middle_band(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path, caption_safe_area="middle")
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())

    result = generate_captions(plan, paths, settings=settings)

    assert result.anchor == "middle"
    assert _style_alignment(paths.captions) == "5"
    result.commit(plan)
    assert plan.captions.anchor == "middle"


def test_generate_captions_burns_the_top_band_without_frame_evidence(
    monkeypatch, tmp_path: Path
):
    settings, plan, paths = _captions_plan(tmp_path, caption_safe_area="top")
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())

    result = generate_captions(plan, paths, settings=settings)

    assert result.anchor == "top"
    assert _style_alignment(paths.captions) == "8"


def test_parse_emphasis_response_handles_bullets_and_punctuation():
    tokens = parse_emphasis_response("Here you go:\n- CRAZY,\n1. CLIP IT\n")
    # Order is preserved, punctuation is gone, and list markers become plain tokens that
    # validation then rejects because they were never spoken.
    assert tokens == ["here", "you", "go", "crazy", "1", "clip", "it"]


def test_validate_emphasis_words_drops_anything_that_was_not_said():
    allowed = {"this", "is", "crazy"}
    # "banana" was never said, "clip that" is not one word, and duplicates collapse.
    kept = validate_emphasis_words(
        ["CRAZY", "banana", "clip that", "crazy", "is"], allowed=allowed, limit=12
    )
    assert kept == {"crazy", "is"}


def test_validate_emphasis_words_respects_the_cap():
    allowed = {"a", "b", "c"}
    assert len(validate_emphasis_words(["a", "b", "c"], allowed=allowed, limit=2)) == 2


def test_transcript_word_forms_matches_the_ass_writer():
    cues = build_cues([_w("THIS", 0.0, 0.3), _w("crazy!", 0.4, 0.9)], uppercase=False)
    # The ASS writer matches on `word.strip().strip(".,!?\"'()[]").lower()`.
    assert transcript_word_forms(cues) == {"this", "crazy"}




def test_generate_captions_uses_the_llm_emphasis_when_asked(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path, caption_emphasis="llm")
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())

    seen: dict = {}

    def fake_post(url, *, headers, json, timeout):
        seen["url"] = url
        seen["model"] = json["model"]
        return _FakeResp({"choices": [{"message": {"content": "crazy\nbanana"}}]})

    monkeypatch.setattr("clippy.caption.emphasis.httpx.post", fake_post)

    result = generate_captions(plan, paths, settings=settings)

    assert result.applied is True
    # "banana" was never spoken, so only the real word survives validation.
    assert result.emphasis == {"crazy"}
    assert result.emphasis_source == "llm"
    assert result.emphasis_note is None
    assert seen["url"].endswith("/chat/completions")
    assert seen["model"] == settings.caption_model
    result.commit(plan)
    assert plan.captions.emphasis_source == "llm"
    assert WARN_EMPHASIS_FALLBACK not in plan.warning_codes()


def test_generate_captions_falls_back_to_heuristic_emphasis(monkeypatch, tmp_path: Path):
    settings, plan, paths = _captions_plan(tmp_path, caption_emphasis="llm")
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())

    def boom(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr("clippy.caption.emphasis.httpx.post", boom)

    result = generate_captions(plan, paths, settings=settings)

    assert result.applied is True
    assert "crazy" in result.emphasis  # the heuristic still produced words
    assert result.emphasis_source == "heuristic_fallback"
    assert result.emphasis_note is not None
    result.commit(plan)
    assert WARN_EMPHASIS_FALLBACK in plan.warning_codes()


def test_generate_captions_falls_back_when_the_model_says_nothing_usable(
    monkeypatch, tmp_path: Path
):
    settings, plan, paths = _captions_plan(tmp_path, caption_emphasis="llm")
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())
    monkeypatch.setattr(
        "clippy.caption.emphasis.httpx.post",
        lambda *a, **k: _FakeResp({"choices": [{"message": {"content": "banana"}}]}),
    )

    result = generate_captions(plan, paths, settings=settings)

    assert result.emphasis_source == "heuristic_fallback"
    assert result.emphasis_note is not None
    assert "no word from the transcript" in result.emphasis_note


def test_generate_captions_honours_the_plan_emphasis_override(monkeypatch, tmp_path: Path):
    """`--caption-emphasis off` lands in the plan, so the stage has to read the plan."""
    settings, plan, paths = _captions_plan(tmp_path)
    plan.captions.emphasis = "off"
    monkeypatch.setattr("clippy.edit.captions.transcribe_words", lambda *a, **k: _payload())

    result = generate_captions(plan, paths, settings=settings)

    assert settings.caption_emphasis == "heuristic"
    assert result.emphasis == set()
    assert result.emphasis_source == "off"
