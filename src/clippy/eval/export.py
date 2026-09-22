from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from clippy.store.db import Database


def export_reviews_json(db: Database, path: Path) -> Path:
    rows = db.export_reviews()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def export_reviews_csv(db: Database, path: Path) -> Path:
    rows = db.export_reviews()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "candidate_id",
        "decision",
        "reason_code",
        "notes",
        "reviewed_at",
        "source_ts",
        "score",
        "streamer_login",
        "stream_id",
        "mode",
        "vod_id",
        "status",
        "media_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def precision_report(db: Database) -> dict[str, Any]:
    stats = db.stats()
    clip_it_candidates = 0
    with db.connection() as conn:
        rows = conn.execute("SELECT signals FROM candidates").fetchall()
    for row in rows:
        signals = json.loads(row["signals"] or "{}")
        blob = json.dumps(signals).lower()
        if "clip it" in blob or '"keyword": "clip' in blob:
            clip_it_candidates += 1
    stats["clip_keyword_candidates"] = clip_it_candidates
    return stats
