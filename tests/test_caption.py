from __future__ import annotations

import sqlite3
from pathlib import Path

from clippy.caption.chat_context import build_chat_context, is_emote_only, slice_chat
from clippy.caption.generate import annotate_extracted_candidate, generate_caption
from clippy.caption.reason import format_extract_reason
from clippy.chat.models import ChatMessage
from clippy.config import Settings
from clippy.pipeline import (
    _AnnotationJob,
    _annotate_extracted_candidates,
    _select_annotation_jobs,
)
from clippy.store.db import Database


def test_format_extract_reason_keyword():
    text = format_extract_reason(
        {"kind": "keyword", "keyword": "clip that", "user": "alice", "text": "CLIP THAT"}
    )
    assert "clip that" in text
    assert "alice" in text


def test_format_extract_reason_rate_and_audio():
    text = format_extract_reason(
        {
            "kinds": ["rate_spike", "intensity_spike"],
            "events": [
                {"kind": "rate_spike", "multiplier": 3.2, "window_rate": 3.2, "baseline_rate": 1.0},
                {"kind": "intensity_spike", "multiplier": 2.8},
            ],
        }
    )
    assert "3.2x" in text
    assert "2.8x" in text


def test_format_extract_reason_chat_audio_nested():
    text = format_extract_reason(
        {
            "kind": "chat_audio",
            "kinds": ["chat_audio"],
            "events": [
                {
                    "kind": "chat_audio",
                    "chat": {"kind": "keyword", "keyword": "clip it", "user": "bob"},
                    "audio": {"kind": "intensity_spike", "multiplier": 2.5},
                }
            ],
        }
    )
    assert "clip it" in text
    assert "bob" in text
    assert "2.5x" in text


def test_format_extract_reason_empty():
    assert format_extract_reason({}) == "Flagged by detection signals"


def test_format_rate_spike_derives_multiplier_from_rates():
    text = format_extract_reason(
        {"kind": "rate_spike", "window_rate": 4.0, "baseline_rate": 2.0}
    )
    assert "2.0x" in text


def test_format_rate_spike_treats_zero_window_as_present():
    text = format_extract_reason(
        {"kind": "rate_spike", "window_rate": 0.0, "baseline_rate": 1.0}
    )
    assert "0.0x" in text


def test_format_rate_spike_zero_baseline_uses_fallback():
    text = format_extract_reason(
        {"kind": "rate_spike", "window_rate": 3.0, "baseline_rate": 0.0}
    )
    assert text == "Chat rate jumped vs the last minute"


def test_format_audio_spike_bad_multiplier_uses_fallback():
    text = format_extract_reason({"kind": "intensity_spike", "multiplier": "loud"})
    assert text == "Audio got louder than the recent baseline"


def test_slice_and_despam_chat():
    messages = [
        ChatMessage(ts=1.0, user="a", text="hello there"),
        ChatMessage(ts=2.0, user="b", text="KEKW"),
        ChatMessage(ts=3.0, user="c", text="clip that"),
        ChatMessage(ts=4.0, user="d", text="KEKW"),
        ChatMessage(ts=50.0, user="e", text="outside"),
    ]
    window = slice_chat(messages, start=0.0, end=10.0)
    assert [m.user for m in window] == ["a", "b", "c", "d"]
    assert is_emote_only("KEKW")
    assert not is_emote_only("clip that")

    ctx = build_chat_context(
        messages,
        start=0.0,
        end=10.0,
        keywords=["clip that"],
        max_messages=10,
    )
    texts = [m["text"] for m in ctx["messages"]]
    assert "hello there" in texts
    assert "clip that" in texts
    assert "KEKW" not in texts
    assert ctx["keyword_hits"][0]["text"] == "clip that"
    assert any(item["text"] == "kekw" and item["count"] == 2 for item in ctx["repeated"])


def test_build_chat_context_caps_and_keeps_keywords():
    messages = [
        ChatMessage(ts=float(i), user="u", text=f"message {i}") for i in range(30)
    ]
    messages.append(ChatMessage(ts=30.0, user="x", text="please clip it"))
    ctx = build_chat_context(
        messages,
        start=0.0,
        end=30.0,
        keywords=["clip it"],
        max_messages=5,
    )
    assert len(ctx["messages"]) <= 5
    assert any(m["text"] == "please clip it" for m in ctx["messages"])


def test_generate_caption_parses_response(monkeypatch):
    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '"Jason reacts to GG EZ"'}}]}

    monkeypatch.setattr("clippy.caption.generate.httpx.post", lambda *a, **k: FakeResp())
    caption = generate_caption(
        streamer_display_name="Jason",
        streamer_login="jason",
        extract_reason="Chat asked to clip it",
        transcript="I can't believe that",
        chat_context={"messages": []},
        api_key="test-key",
    )
    assert caption == "Jason reacts to GG EZ"


def test_select_annotation_jobs_ranks_by_score_then_caps():
    jobs = [
        _AnnotationJob(1, Path("a.mp4"), 10.0, 0.4, "r"),
        _AnnotationJob(2, Path("b.mp4"), 20.0, 0.9, "r"),
        _AnnotationJob(3, Path("c.mp4"), 30.0, 0.9, "r"),
    ]
    selected = _select_annotation_jobs(jobs, max_per_run=2)
    assert [job.candidate_id for job in selected] == [2, 3]


def test_select_annotation_jobs_zero_cap():
    jobs = [_AnnotationJob(1, Path("a.mp4"), 10.0, 0.9, "r")]
    assert _select_annotation_jobs(jobs, max_per_run=0) == []


def test_annotate_extracted_skips_without_api_key(monkeypatch):
    called: list[int] = []
    monkeypatch.setattr(
        "clippy.pipeline._annotate_candidate",
        lambda *a, **k: called.append(1),
    )
    jobs = [_AnnotationJob(1, Path("a.mp4"), 10.0, 0.9, "r")]
    count = _annotate_extracted_candidates(
        object(),  # type: ignore[arg-type]
        jobs,
        chat=[],
        settings=Settings(openai_api_key=""),
        streamer_display_name="Jason",
        streamer_login="jason",
    )
    assert count == 0
    assert called == []


def test_annotate_extracted_respects_cap(monkeypatch):
    called: list[int] = []

    def fake_annotate(db, candidate_id, **kwargs):
        called.append(candidate_id)

    monkeypatch.setattr("clippy.pipeline._annotate_candidate", fake_annotate)
    jobs = [
        _AnnotationJob(1, Path("a.mp4"), 10.0, 0.2, "r"),
        _AnnotationJob(2, Path("b.mp4"), 20.0, 0.9, "r"),
        _AnnotationJob(3, Path("c.mp4"), 30.0, 0.5, "r"),
    ]
    count = _annotate_extracted_candidates(
        object(),  # type: ignore[arg-type]
        jobs,
        chat=[],
        settings=Settings(openai_api_key="sk-test", caption_max_per_run=2),
        streamer_display_name="Jason",
        streamer_login="jason",
    )
    assert count == 2
    assert called == [2, 3]


def test_annotate_skips_without_api_key():
    caption, transcript = annotate_extracted_candidate(
        media_path=None,
        chat=[],
        source_ts=10.0,
        pre_context_seconds=5.0,
        post_context_seconds=5.0,
        streamer_display_name="Jason",
        streamer_login="jason",
        extract_reason="Chat asked to clip it",
        settings=Settings(openai_api_key=""),
    )
    assert caption is None
    assert transcript is None


def test_migrates_legacy_candidates_table(tmp_path: Path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id INTEGER NOT NULL,
            source_ts REAL NOT NULL,
            pre_context_seconds REAL NOT NULL,
            post_context_seconds REAL NOT NULL,
            signals TEXT NOT NULL DEFAULT '{}',
            score REAL NOT NULL DEFAULT 0,
            media_path TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    with db.connection() as migrated:
        cols = {row[1] for row in migrated.execute("PRAGMA table_info(candidates)")}
    assert {"extract_reason", "caption", "transcript"}.issubset(cols)
