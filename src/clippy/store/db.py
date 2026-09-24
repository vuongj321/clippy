from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

CandidateStatus = Literal["pending", "approved", "rejected"]
StreamMode = Literal["vod", "live"]
Decision = Literal["approved", "rejected"]

REJECTION_REASONS = (
    "false_alarm",
    "needs_context",
    "too_long",
    "boring",
    "unsafe",
    "other",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS streamers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    login TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS streams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streamer_id INTEGER NOT NULL REFERENCES streamers(id),
    mode TEXT NOT NULL CHECK(mode IN ('vod', 'live')),
    source_url TEXT,
    vod_id TEXT,
    media_path TEXT,
    started_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id INTEGER NOT NULL REFERENCES streams(id),
    source_ts REAL NOT NULL,
    pre_context_seconds REAL NOT NULL,
    post_context_seconds REAL NOT NULL,
    signals TEXT NOT NULL DEFAULT '{}',
    score REAL NOT NULL DEFAULT 0,
    media_path TEXT,
    extract_reason TEXT,
    caption TEXT,
    transcript TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'approved', 'rejected')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL UNIQUE REFERENCES candidates(id),
    decision TEXT NOT NULL CHECK(decision IN ('approved', 'rejected')),
    reason_code TEXT,
    notes TEXT,
    reviewed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidates_stream_score
    ON candidates(stream_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_status
    ON candidates(status);
"""

CANDIDATE_COLUMN_MIGRATIONS = (
    ("extract_reason", "TEXT"),
    ("caption", "TEXT"),
    ("transcript", "TEXT"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Streamer:
    id: int
    login: str
    display_name: str


@dataclass
class Stream:
    id: int
    streamer_id: int
    mode: StreamMode
    source_url: str | None
    vod_id: str | None
    media_path: str | None
    started_at: str
    created_at: str


@dataclass
class Candidate:
    id: int
    stream_id: int
    source_ts: float
    pre_context_seconds: float
    post_context_seconds: float
    signals: dict[str, Any]
    score: float
    media_path: str | None
    status: CandidateStatus
    created_at: str
    extract_reason: str | None = None
    caption: str | None = None
    transcript: str | None = None


@dataclass
class Review:
    id: int
    candidate_id: int
    decision: Decision
    reason_code: str | None
    notes: str | None
    reviewed_at: str


@dataclass
class CandidateView:
    candidate: Candidate
    streamer_login: str
    streamer_display_name: str
    stream_mode: StreamMode
    review: Review | None = None


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
        }
        for name, col_type in CANDIDATE_COLUMN_MIGRATIONS:
            if name not in existing:
                conn.execute(f"ALTER TABLE candidates ADD COLUMN {name} {col_type}")

    def get_or_create_streamer(self, login: str, display_name: str | None = None) -> Streamer:
        login = login.lower().strip()
        display = display_name or login
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM streamers WHERE login = ?", (login,)
            ).fetchone()
            if row:
                if display_name and row["display_name"] != display_name:
                    conn.execute(
                        "UPDATE streamers SET display_name = ? WHERE id = ?",
                        (display_name, row["id"]),
                    )
                    row = conn.execute(
                        "SELECT * FROM streamers WHERE id = ?", (row["id"],)
                    ).fetchone()
                return Streamer(**dict(row))
            cur = conn.execute(
                "INSERT INTO streamers (login, display_name) VALUES (?, ?)",
                (login, display),
            )
            return Streamer(id=cur.lastrowid, login=login, display_name=display)

    def create_stream(
        self,
        streamer_id: int,
        mode: StreamMode,
        *,
        source_url: str | None = None,
        vod_id: str | None = None,
        media_path: str | None = None,
        started_at: str | None = None,
    ) -> Stream:
        now = utc_now()
        started = started_at or now
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO streams
                    (streamer_id, mode, source_url, vod_id, media_path, started_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (streamer_id, mode, source_url, vod_id, media_path, started, now),
            )
            return Stream(
                id=cur.lastrowid,
                streamer_id=streamer_id,
                mode=mode,
                source_url=source_url,
                vod_id=vod_id,
                media_path=media_path,
                started_at=started,
                created_at=now,
            )

    def create_candidate(
        self,
        stream_id: int,
        source_ts: float,
        pre_context_seconds: float,
        post_context_seconds: float,
        signals: dict[str, Any],
        score: float,
        media_path: str | None = None,
        extract_reason: str | None = None,
        caption: str | None = None,
        transcript: str | None = None,
    ) -> Candidate:
        now = utc_now()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO candidates (
                    stream_id, source_ts, pre_context_seconds, post_context_seconds,
                    signals, score, media_path, status, created_at,
                    extract_reason, caption, transcript
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    stream_id,
                    source_ts,
                    pre_context_seconds,
                    post_context_seconds,
                    json.dumps(signals),
                    score,
                    media_path,
                    now,
                    extract_reason,
                    caption,
                    transcript,
                ),
            )
            return self._candidate_from_row(
                conn.execute(
                    "SELECT * FROM candidates WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
            )

    def update_candidate_media(self, candidate_id: int, media_path: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE candidates SET media_path = ? WHERE id = ?",
                (media_path, candidate_id),
            )

    def update_candidate_caption(
        self,
        candidate_id: int,
        *,
        caption: str | None = None,
        transcript: str | None = None,
    ) -> None:
        assignments: list[str] = []
        params: list[Any] = []
        if caption is not None:
            assignments.append("caption = ?")
            params.append(caption)
        if transcript is not None:
            assignments.append("transcript = ?")
            params.append(transcript)
        if not assignments:
            return
        params.append(candidate_id)
        with self.connection() as conn:
            conn.execute(
                f"UPDATE candidates SET {', '.join(assignments)} WHERE id = ?",
                params,
            )

    def get_candidate(self, candidate_id: int) -> Candidate | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            return self._candidate_from_row(row) if row else None

    def list_candidate_views(
        self,
        *,
        status: CandidateStatus | None = "pending",
        stream_id: int | None = None,
        limit: int = 200,
    ) -> list[CandidateView]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("c.status = ?")
            params.append(status)
        if stream_id is not None:
            clauses.append("c.stream_id = ?")
            params.append(stream_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        sql = f"""
            SELECT
                c.*,
                s.mode AS stream_mode,
                st.login AS streamer_login,
                st.display_name AS streamer_display_name,
                r.id AS review_id,
                r.decision AS review_decision,
                r.reason_code AS review_reason_code,
                r.notes AS review_notes,
                r.reviewed_at AS review_reviewed_at
            FROM candidates c
            JOIN streams s ON s.id = c.stream_id
            JOIN streamers st ON st.id = s.streamer_id
            LEFT JOIN reviews r ON r.candidate_id = c.id
            {where}
            ORDER BY c.score DESC, c.source_ts ASC
            LIMIT ?
        """
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._candidate_view_from_row(row) for row in rows]

    def review_candidate(
        self,
        candidate_id: int,
        decision: Decision,
        reason_code: str | None = None,
        notes: str | None = None,
    ) -> Review:
        if decision == "rejected" and reason_code and reason_code not in REJECTION_REASONS:
            raise ValueError(f"Invalid reason_code: {reason_code}")
        now = utc_now()
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT id FROM reviews WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE reviews
                    SET decision = ?, reason_code = ?, notes = ?, reviewed_at = ?
                    WHERE candidate_id = ?
                    """,
                    (decision, reason_code, notes, now, candidate_id),
                )
                review_id = existing["id"]
            else:
                cur = conn.execute(
                    """
                    INSERT INTO reviews (candidate_id, decision, reason_code, notes, reviewed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (candidate_id, decision, reason_code, notes, now),
                )
                review_id = cur.lastrowid
            conn.execute(
                "UPDATE candidates SET status = ? WHERE id = ?",
                (decision, candidate_id),
            )
            row = conn.execute(
                "SELECT * FROM reviews WHERE id = ?", (review_id,)
            ).fetchone()
            return Review(**dict(row))

    def stats(self) -> dict[str, Any]:
        with self.connection() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]
            pending = conn.execute(
                "SELECT COUNT(*) AS n FROM candidates WHERE status = 'pending'"
            ).fetchone()["n"]
            approved = conn.execute(
                "SELECT COUNT(*) AS n FROM candidates WHERE status = 'approved'"
            ).fetchone()["n"]
            rejected = conn.execute(
                "SELECT COUNT(*) AS n FROM candidates WHERE status = 'rejected'"
            ).fetchone()["n"]
            reasons = conn.execute(
                """
                SELECT reason_code, COUNT(*) AS n
                FROM reviews
                WHERE decision = 'rejected' AND reason_code IS NOT NULL
                GROUP BY reason_code
                ORDER BY n DESC
                """
            ).fetchall()
        reviewed = approved + rejected
        precision = (approved / reviewed) if reviewed else None
        return {
            "total_candidates": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "reviewed": reviewed,
            "approve_rate": precision,
            "rejection_reasons": {r["reason_code"]: r["n"] for r in reasons},
        }

    def export_reviews(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.id AS review_id,
                    r.decision,
                    r.reason_code,
                    r.notes,
                    r.reviewed_at,
                    c.id AS candidate_id,
                    c.source_ts,
                    c.score,
                    c.signals,
                    c.status,
                    c.media_path,
                    c.extract_reason,
                    c.caption,
                    c.transcript,
                    s.id AS stream_id,
                    s.mode,
                    s.vod_id,
                    s.source_url,
                    st.login AS streamer_login,
                    st.display_name AS streamer_display_name
                FROM reviews r
                JOIN candidates c ON c.id = r.candidate_id
                JOIN streams s ON s.id = c.stream_id
                JOIN streamers st ON st.id = s.streamer_id
                ORDER BY r.reviewed_at ASC
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["signals"] = json.loads(item["signals"] or "{}")
            out.append(item)
        return out

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> Candidate:
        data = dict(row)
        data["signals"] = json.loads(data["signals"] or "{}")
        return Candidate(**data)

    def _candidate_view_from_row(self, row: sqlite3.Row) -> CandidateView:
        candidate = Candidate(
            id=row["id"],
            stream_id=row["stream_id"],
            source_ts=row["source_ts"],
            pre_context_seconds=row["pre_context_seconds"],
            post_context_seconds=row["post_context_seconds"],
            signals=json.loads(row["signals"] or "{}"),
            score=row["score"],
            media_path=row["media_path"],
            status=row["status"],
            created_at=row["created_at"],
            extract_reason=row["extract_reason"] if "extract_reason" in row.keys() else None,
            caption=row["caption"] if "caption" in row.keys() else None,
            transcript=row["transcript"] if "transcript" in row.keys() else None,
        )
        review = None
        if row["review_id"] is not None:
            review = Review(
                id=row["review_id"],
                candidate_id=candidate.id,
                decision=row["review_decision"],
                reason_code=row["review_reason_code"],
                notes=row["review_notes"],
                reviewed_at=row["review_reviewed_at"],
            )
        return CandidateView(
            candidate=candidate,
            streamer_login=row["streamer_login"],
            streamer_display_name=row["streamer_display_name"],
            stream_mode=row["stream_mode"],
            review=review,
        )
