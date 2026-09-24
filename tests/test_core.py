from __future__ import annotations

from pathlib import Path

from clippy.chat.models import ChatMessage, load_chat_json
from clippy.chat.signals import detect_chat_signals
from clippy.detect.detector import RawDetection, coalesce_detections, combine_signal_events
from clippy.store.db import Database


def test_load_chat_sample():
    path = Path(__file__).resolve().parents[1] / "samples" / "chat_sample.json"
    messages = load_chat_json(path)
    assert len(messages) > 0
    assert messages[0].ts <= messages[-1].ts


def test_chat_spike_and_keyword():
    path = Path(__file__).resolve().parents[1] / "samples" / "chat_sample.json"
    messages = load_chat_json(path)
    events = detect_chat_signals(
        messages,
        window_seconds=5.0,
        baseline_seconds=60.0,
        spike_multiplier=2.0,
        min_rate=0.3,
        keywords=["clip it", "clip"],
    )
    kinds = {e.kind for e in events}
    assert "keyword" in kinds
    assert any(e.kind == "rate_spike" for e in events) or any(
        e.kind == "keyword" for e in events
    )


def test_detect_ignores_clip_substring():
    messages = [
        ChatMessage(ts=1.0, user="a", text="clippers are winning"),
        ChatMessage(ts=2.0, user="b", text="unclipped vod"),
        ChatMessage(ts=3.0, user="c", text="please clip it"),
    ]
    events = detect_chat_signals(
        messages,
        keywords=["clip it", "clip that", "clip this", "clip"],
    )
    texts = [e.details["text"] for e in events if e.kind == "keyword"]
    assert texts == ["please clip it"]


def test_coalesce_merges_nearby():
    dets = [
        RawDetection(ts=10.0, score=0.5, signals={"kind": "a"}),
        RawDetection(ts=15.0, score=0.9, signals={"kind": "b"}),
        RawDetection(ts=50.0, score=0.4, signals={"kind": "c"}),
    ]
    out = coalesce_detections(dets, gap_seconds=20.0)
    assert len(out) == 2
    assert out[0].score == 0.9
    assert out[0].ts == 15.0


def test_combine_boosts_chat_audio():
    from types import SimpleNamespace

    chat = [SimpleNamespace(ts=10.0, kind="keyword", score=0.7, details={"keyword": "clip"})]
    audio = [
        SimpleNamespace(ts=11.0, kind="intensity_spike", score=0.75, details={"rms": 0.1})
    ]
    raw = combine_signal_events(chat, audio, proximity_seconds=5.0)
    assert len(raw) == 1
    assert raw[0].signals["kind"] == "chat_audio"
    assert raw[0].score > 0.7


def test_database_review_roundtrip(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(streamer.id, "vod", vod_id="123")
    cand = db.create_candidate(
        stream.id,
        source_ts=42.0,
        pre_context_seconds=30,
        post_context_seconds=30,
        signals={"kind": "keyword"},
        score=0.8,
        extract_reason="Chat asked to clip it",
    )
    db.update_candidate_caption(
        cand.id,
        caption="Jason reacts to GG EZ",
        transcript="I can't believe that",
    )
    db.review_candidate(cand.id, "approved")
    stats = db.stats()
    assert stats["approved"] == 1
    assert stats["approve_rate"] == 1.0
    loaded = db.get_candidate(cand.id)
    assert loaded is not None
    assert loaded.extract_reason == "Chat asked to clip it"
    assert loaded.caption == "Jason reacts to GG EZ"
    exported = db.export_reviews()
    assert exported[0]["decision"] == "approved"
    assert exported[0]["caption"] == "Jason reacts to GG EZ"
    assert exported[0]["extract_reason"] == "Chat asked to clip it"
    assert exported[0]["transcript"] == "I can't believe that"


def test_update_candidate_caption_partial_does_not_null_other(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(streamer.id, "vod")
    cand = db.create_candidate(
        stream.id,
        source_ts=1.0,
        pre_context_seconds=30,
        post_context_seconds=30,
        signals={"kind": "keyword"},
        score=0.5,
        extract_reason="Chat asked to clip it",
    )
    db.update_candidate_caption(
        cand.id,
        caption="first caption",
        transcript="keep this transcript",
    )
    db.update_candidate_caption(cand.id, caption="second caption")
    loaded = db.get_candidate(cand.id)
    assert loaded is not None
    assert loaded.caption == "second caption"
    assert loaded.transcript == "keep this transcript"
    assert loaded.extract_reason == "Chat asked to clip it"
    db.update_candidate_caption(cand.id)
    loaded = db.get_candidate(cand.id)
    assert loaded is not None
    assert loaded.caption == "second caption"
    assert loaded.transcript == "keep this transcript"


def test_candidate_view_reads_caption_fields(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(streamer.id, "vod")
    cand = db.create_candidate(
        stream.id,
        source_ts=1.0,
        pre_context_seconds=30,
        post_context_seconds=30,
        signals={"kind": "keyword"},
        score=0.5,
        extract_reason="Chat asked to clip it",
    )
    db.update_candidate_caption(cand.id, caption="cap", transcript="tr")
    views = db.list_candidate_views(status=None)
    assert views[0].candidate.id == cand.id
    assert views[0].candidate.extract_reason == "Chat asked to clip it"
    assert views[0].candidate.caption == "cap"
    assert views[0].candidate.transcript == "tr"
