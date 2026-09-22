from __future__ import annotations

import shutil
from pathlib import Path


class RollingMediaBuffer:
    """
    Time-bounded local media cache for live ingest.

    VOD mode can point at a single file; live mode stores segment files and
    deletes those older than retention_seconds.
    """

    def __init__(self, root: Path, *, retention_seconds: float = 300.0) -> None:
        self.root = root
        self.retention_seconds = retention_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self._source_media: Path | None = None

    def set_source_media(self, path: Path) -> None:
        self._source_media = path

    @property
    def source_media(self) -> Path | None:
        return self._source_media

    def add_segment(self, src: Path, *, stream_ts: float) -> Path:
        dest = self.root / f"seg_{stream_ts:012.3f}_{src.name}"
        shutil.copy2(src, dest)
        self.prune(current_ts=stream_ts)
        return dest

    def prune(self, *, current_ts: float) -> None:
        cutoff = current_ts - self.retention_seconds
        for path in self.root.glob("seg_*"):
            try:
                # seg_{ts}_...
                ts_str = path.name.split("_")[1]
                ts = float(ts_str)
            except (IndexError, ValueError):
                continue
            if ts < cutoff:
                path.unlink(missing_ok=True)
