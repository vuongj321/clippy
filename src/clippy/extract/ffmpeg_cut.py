from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Input seeking can only start the *video* on a keyframe: the demuxer seeks forward to the next
# one, so a cut landing inside a GOP would silently begin up to a whole GOP late. The audio carries
# no keyframes and starts exactly where it was asked to, which would leave the picture running
# ahead of the sound. Every cut therefore seeks `preroll_seconds` early and trims both streams to
# the same window, so the preroll has to reach back past one GOP.
DEFAULT_PREROLL_SECONDS = 3.0
RETRY_PREROLL_FACTOR = 2.0
MAX_PREROLL_ATTEMPTS = 3

FIRST_FRAME_PTS_RE = re.compile(r"n:\s+0 pts:\s+\d+ pts_time:(\S+)")


@dataclass(frozen=True)
class ExtractResult:
    """
    The cut, plus how much of its head the source could not supply.

    `head_seconds` is 0 whenever the window starts where it was asked to. It only rises above 0
    when the source holds no decodable video at the requested start even after seeking back a whole
    GOP - a damaged keyframe - in which case the clip starts that much later so the picture and the
    sound still begin on the same instant.
    """

    path: Path
    head_seconds: float = 0.0


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


def _require_ffmpeg(ffmpeg_path: str) -> str:
    ffmpeg = shutil.which(ffmpeg_path) or (
        ffmpeg_path if Path(ffmpeg_path).exists() else None
    )
    if not ffmpeg:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    return ffmpeg


def probe_video_head_seconds(
    source_media: Path,
    *,
    start_seconds: float,
    ffmpeg_path: str = "ffmpeg",
) -> float | None:
    """
    How far after `start_seconds` the first *decodable* video frame actually sits.

    ffmpeg's own labels are the answer: under input seeking they are relative to the requested
    position, so the first frame it delivers reports the distance to the keyframe the demuxer had
    to start on - 0.816 s for candidate 28, whose cut falls 0.816 s before the next keyframe.
    `None` means the probe could not tell, and the caller then assumes the video starts where it
    was asked to.
    """
    ffmpeg = _require_ffmpeg(ffmpeg_path)
    cmd = [
        ffmpeg,
        "-v",
        "info",
        "-ss",
        f"{max(0.0, float(start_seconds)):.3f}",
        "-i",
        str(source_media),
        # `-frames:v 1`, not `-t`: the frame that carries the offset sits *after* the requested
        # position on the same labelled timeline, so a duration limit would stop before it arrives.
        "-frames:v",
        "1",
        "-vf",
        "showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    match = FIRST_FRAME_PTS_RE.search(proc.stderr.decode("utf-8", errors="replace"))
    if not match:
        return None
    try:
        return max(0.0, float(match.group(1)))
    except ValueError:
        return None


def _plan_window(
    source_media: Path,
    *,
    start: float,
    preroll: float,
    ffmpeg_path: str,
) -> tuple[float, float, float]:
    """
    Pick the seek position and the window (in seek-relative seconds) both streams are cut to.

    Returns `(window_start, seek_at, head)`. The probe is retried with a longer preroll while the
    video still begins after the requested position, which is what a GOP longer than the configured
    preroll looks like.
    """
    attempt = preroll
    seek_at = start
    window_start = 0.0
    head = 0.0
    for _ in range(MAX_PREROLL_ATTEMPTS):
        seek_at = max(0.0, start - attempt)
        requested = start - seek_at  # the requested start, in this seek's label space
        observed = probe_video_head_seconds(
            source_media, start_seconds=seek_at, ffmpeg_path=ffmpeg_path
        )
        if observed is None:
            # No answer: keep the requested window, so the trim is an identity.
            return requested, seek_at, 0.0
        head = max(0.0, observed - requested)
        window_start = requested + head
        if head <= 0.0:
            return window_start, seek_at, 0.0
        logger.info(
            "Source %s has no decodable video at %.3fs (first frame %.3fs later); "
            "retrying with a longer preroll",
            source_media.name,
            start,
            head,
        )
        attempt *= RETRY_PREROLL_FACTOR
    return window_start, seek_at, head


def extract_window(
    source_media: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_path: str = "ffmpeg",
    crf: int = 23,
    preset: str = "veryfast",
    preroll_seconds: float = DEFAULT_PREROLL_SECONDS,
) -> ExtractResult:
    """
    Cut ``duration_seconds`` of ``source_media`` into ``output_path``, starting at ``start_seconds``.

    The cut is re-encoded, and both streams are trimmed to *the same* window. A video stream can
    only restart on a keyframe, so seeking straight to the start would begin the picture at the
    next keyframe - ahead of the sound - and, because `-t` is enforced on that same labelled
    timeline, also drop the matching frames from the tail. Seeking `preroll_seconds` early and
    trimming the window back keeps the whole picture window *and* keeps picture and sound together;
    `head_seconds` is non-zero only when even that early seek could not reach the start.
    """
    ffmpeg = _require_ffmpeg(ffmpeg_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, start_seconds)
    duration = max(0.1, duration_seconds)

    window, seek_at, head = _plan_window(
        source_media,
        start=start,
        preroll=max(0.0, float(preroll_seconds)),
        ffmpeg_path=ffmpeg_path,
    )

    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{seek_at:.3f}",
        "-i",
        str(source_media),
        "-vf",
        f"trim=start={window:.3f}:end={window + duration:.3f},setpts=PTS-STARTPTS",
        "-af",
        f"atrim=start={window:.3f}:end={window + duration:.3f},asetpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
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
    return ExtractResult(path=output_path, head_seconds=head)
