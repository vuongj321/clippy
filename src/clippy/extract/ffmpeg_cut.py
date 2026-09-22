from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def media_dir_size_bytes(media_dir: Path) -> int:
    if not media_dir.exists():
        return 0
    total = 0
    for path in media_dir.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def within_disk_budget(media_dir: Path, budget_gb: float) -> bool:
    used = media_dir_size_bytes(media_dir)
    return used < budget_gb * (1024**3)


def extract_window(
    source_media: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    ffmpeg = shutil.which(ffmpeg_path) or (
        ffmpeg_path if Path(ffmpeg_path).exists() else None
    )
    if not ffmpeg:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, start_seconds)
    duration = max(0.1, duration_seconds)

    # Re-encode for reliable cuts (input seeking can be inaccurate on some VODs)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source_media),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg extract failed: {proc.stderr.decode('utf-8', errors='replace')}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty output: {output_path}")
    return output_path
