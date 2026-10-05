from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from clippy.store.db import (
    CANDIDATE_COLUMN_MIGRATIONS,
    REJECTION_REASONS,
    STREAM_COLUMN_MIGRATIONS,
    Database,
)


def _seed(tmp_path: Path) -> tuple[Database, int, int]:
    db = Database(tmp_path / "clippy.db")
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(streamer.id, "vod", vod_id="123")
    candidate = db.create_candidate(
        stream.id,
        source_ts=42.0,
        pre_context_seconds=30.0,
        post_context_seconds=30.0,
        signals={"kind": "keyword"},
        score=0.8,
        extract_reason="Chat asked to clip it",
    )
    return db, stream.id, candidate.id


def test_render_revision_becomes_current_only_for_final_ok(tmp_path: Path):
    db, _, candidate_id = _seed(tmp_path)
    plan = db.create_render(candidate_id, kind="plan", path="plan.json")
    assert plan.revision == 1
    assert plan.is_current == 0

    rough = db.create_render(candidate_id, kind="rough", path="vertical.mp4")
    assert rough.revision == 2
    assert rough.is_current == 0

    final = db.create_render(
        candidate_id,
        kind="final",
        path="final.mp4",
        width=1080,
        height=1920,
        duration=42.5,
    )
    assert final.revision == 3
    assert final.is_current == 1

    second_final = db.create_render(candidate_id, kind="final", path="final_v2.mp4")
    assert second_final.revision == 4
    assert second_final.is_current == 1

    rerendered = db.list_renders(candidate_id)
    current = [r for r in rerendered if r.is_current == 1]
    assert [r.id for r in current] == [second_final.id]
    assert db.get_current_render(candidate_id).id == second_final.id


def test_failed_final_does_not_displace_current(tmp_path: Path):
    db, _, candidate_id = _seed(tmp_path)
    good = db.create_render(candidate_id, kind="final", path="final.mp4")
    bad = db.create_render(
        candidate_id,
        kind="final",
        status="failed",
        error="ffmpeg exploded",
    )
    assert bad.is_current == 0
    assert db.get_current_render(candidate_id).id == good.id
    assert bad.error == "ffmpeg exploded"


def test_list_renders_orders_newest_first(tmp_path: Path):
    db, _, candidate_id = _seed(tmp_path)
    for index in range(3):
        db.create_render(candidate_id, kind="plan", path=f"plan{index}.json")
    revisions = [r.revision for r in db.list_renders(candidate_id)]
    assert revisions == [3, 2, 1]


def test_update_candidate_edit_is_partial(tmp_path: Path):
    db, _, candidate_id = _seed(tmp_path)
    db.update_candidate_edit(candidate_id, edit_status="rendering")
    db.update_candidate_edit(candidate_id, edited_media_path="data/edits/1/final.mp4")
    loaded = db.get_candidate(candidate_id)
    assert loaded is not None
    assert loaded.edit_status == "rendering"
    assert loaded.edited_media_path == "data/edits/1/final.mp4"

    db.update_candidate_edit(candidate_id, edit_status="rendered")
    loaded = db.get_candidate(candidate_id)
    assert loaded is not None
    assert loaded.edit_status == "rendered"
    assert loaded.edited_media_path == "data/edits/1/final.mp4"


def test_candidate_view_exposes_edit_fields(tmp_path: Path):
    db, _, candidate_id = _seed(tmp_path)
    db.update_candidate_edit(
        candidate_id, edit_status="rendered", edited_media_path="data/edits/1/final.mp4"
    )
    views = db.list_candidate_views(status=None)
    assert views[0].candidate.edit_status == "rendered"
    assert views[0].candidate.edited_media_path == "data/edits/1/final.mp4"


def test_list_candidates_filters(tmp_path: Path):
    db, stream_id, candidate_id = _seed(tmp_path)
    db.create_candidate(
        stream_id,
        source_ts=99.0,
        pre_context_seconds=30.0,
        post_context_seconds=30.0,
        signals={"kind": "intensity_spike"},
        score=0.5,
    )
    assert len(db.list_candidates(stream_id=stream_id)) == 2
    assert len(db.list_candidates(ids=[candidate_id])) == 1
    assert db.list_candidates(ids=[9999]) == []
    # Highest score first, matching the detection queue order.
    assert db.list_candidates(stream_id=stream_id)[0].id == candidate_id
    db.review_candidate(candidate_id, "approved")
    assert [c.id for c in db.list_candidates(stream_id=stream_id, status="approved")] == [
        candidate_id
    ]
    assert len(db.list_candidates(stream_id=stream_id, status="pending")) == 1


def test_stream_source_metadata_round_trip(tmp_path: Path):
    db = Database(tmp_path / "clippy.db")
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(
        streamer.id,
        "vod",
        media_path="data/source/vod.ts",
        source_width=1920,
        source_height=1080,
        source_fps=60.0,
        capture_quality="1080p60",
        source_offset_seconds=1.5,
        source_bytes=8_000_000_000,
    )
    loaded = db.get_stream(stream.id)
    assert loaded is not None
    assert loaded.source_width == 1920
    assert loaded.source_height == 1080
    assert loaded.source_fps == pytest.approx(60.0)
    assert loaded.capture_quality == "1080p60"
    assert loaded.source_offset_seconds == pytest.approx(1.5)
    assert loaded.source_bytes == 8_000_000_000
    assert db.get_stream(9999) is None


def test_legacy_database_is_migrated(tmp_path: Path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE streamers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            login TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL
        );
        CREATE TABLE streams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            streamer_id INTEGER NOT NULL REFERENCES streamers(id),
            mode TEXT NOT NULL CHECK(mode IN ('vod', 'live')),
            source_url TEXT,
            vod_id TEXT,
            media_path TEXT,
            started_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_id INTEGER NOT NULL REFERENCES streams(id),
            source_ts REAL NOT NULL,
            pre_context_seconds REAL NOT NULL,
            post_context_seconds REAL NOT NULL,
            signals TEXT NOT NULL DEFAULT '{}',
            score REAL NOT NULL DEFAULT 0,
            media_path TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO streamers (login, display_name) VALUES ('tester', 'Tester')"
    )
    conn.execute(
        """
        INSERT INTO streams (streamer_id, mode, started_at, created_at)
        VALUES (1, 'vod', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO candidates
            (stream_id, source_ts, pre_context_seconds, post_context_seconds,
             signals, score, created_at)
        VALUES (1, 10.0, 30.0, 30.0, '{"kind": "keyword"}', 0.7, '2026-01-01T00:00:00+00:00')
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    with db.connection() as migrated:
        candidate_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(candidates)")
        }
        stream_columns = {row[1] for row in migrated.execute("PRAGMA table_info(streams)")}
        render_table = migrated.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'renders'"
        ).fetchone()
    assert {name for name, _ in CANDIDATE_COLUMN_MIGRATIONS}.issubset(candidate_columns)
    assert {name for name, _ in STREAM_COLUMN_MIGRATIONS}.issubset(stream_columns)
    assert render_table is not None

    migrated_candidate = db.get_candidate(1)
    assert migrated_candidate is not None
    assert migrated_candidate.extract_reason is None
    assert migrated_candidate.edit_status == "unrendered"
    assert migrated_candidate.edited_media_path is None

    migrated_stream = db.get_stream(1)
    assert migrated_stream is not None
    assert migrated_stream.source_offset_seconds == pytest.approx(0.0)

    # A plan render can be attached to a legacy row without touching media.
    render = db.create_render(1, kind="plan", path="data/edits/1/plan.json")
    assert render.revision == 1


def test_update_stream_source_is_partial(tmp_path: Path):
    db = Database(tmp_path / "clippy.db")
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(streamer.id, "vod", media_path="old.ts")

    db.update_stream_source(stream.id, media_path="data/source/new.ts")
    db.update_stream_source(
        stream.id,
        source_width=1920,
        source_height=1080,
        source_fps=60.0,
        capture_quality="1080p60",
        source_bytes=123,
    )
    loaded = db.get_stream(stream.id)
    assert loaded is not None
    assert loaded.media_path == "data/source/new.ts"
    assert loaded.source_width == 1920
    assert loaded.source_height == 1080
    assert loaded.source_fps == pytest.approx(60.0)
    assert loaded.capture_quality == "1080p60"
    assert loaded.source_bytes == 123
    assert loaded.source_offset_seconds == pytest.approx(0.0)

    db.update_stream_source(stream.id, source_offset_seconds=-12.5)
    reloaded = db.get_stream(stream.id)
    assert reloaded is not None
    assert reloaded.source_offset_seconds == pytest.approx(-12.5)
    assert reloaded.media_path == "data/source/new.ts"


def test_edit_rejection_reasons_are_valid(tmp_path: Path):
    db, _, candidate_id = _seed(tmp_path)
    assert "bad_edit" in REJECTION_REASONS
    assert "bad_captions" in REJECTION_REASONS
    review = db.review_candidate(candidate_id, "rejected", reason_code="bad_edit")
    assert review.reason_code == "bad_edit"
    with pytest.raises(ValueError):
        db.review_candidate(candidate_id, "rejected", reason_code="not_a_reason")

