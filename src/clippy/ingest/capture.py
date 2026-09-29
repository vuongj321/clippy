"""HQ VOD capture (M1).

Same shape as ``ingest/live.py``: build a downloader argv, run it as a
subprocess, then probe what landed so the ``streams`` row can record real source
metadata instead of guesses.

Downloads are written to ``stdout=sys.stderr`` so the CLI's stdout stays pure
JSON while streamlink's progress bar remains visible.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

STREAMLINK_SEGMENT_THREADS = 4
STREAMLINK_SEGMENT_ATTEMPTS = 5
STREAMLINK_SEGMENT_TIMEOUT = 30
STREAMLINK_READ_TIMEOUT = 60
STREAMLINK_RETRY_DELAY = 2
STREAMLINK_RETRY_MAX = 3
INSTALL_HINTS = {
    "streamlink": "uv tool install streamlink  (or: winget install --id Streamlink.Streamlink)",
    "yt-dlp": "uv tool install yt-dlp  (or: winget install --id yt-dlp.yt-dlp)",
}


@dataclass(frozen=True)
class MediaMetadata:
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


@dataclass(frozen=True)
class CaptureResult:
    path: Path
    url: str
    downloader: str
    quality: str
    argv: list[str]
    bytes: int
    media: MediaMetadata = field(default_factory=MediaMetadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "url": self.url,
            "downloader": self.downloader,
            "quality": self.quality,
            "bytes": self.bytes,
            **self.media.to_dict(),
        }


def normalize_downloader(name: str | None) -> str:
    """Accept ``yt-dlp`` / ``yt_dlp`` / ``streamlink`` and return the config value."""
    raw = (name or "streamlink").strip().lower().replace("-", "_")
    if raw not in ("streamlink", "yt_dlp"):
        raise ValueError(f"Unknown downloader {name!r}; expected streamlink or yt-dlp")
    return raw


def vod_url(vod: str) -> str:
    """Accept a full URL, a ``videos/<id>`` path, or a bare VOD id."""
    raw = (vod or "").strip()
    if not raw:
        raise ValueError("Empty VOD reference")
    if raw.startswith(("http://", "https://")):
        return raw
    if raw.startswith("videos/"):
        return f"https://www.twitch.tv/{raw}"
    if raw.isdigit():
        return f"https://www.twitch.tv/videos/{raw}"
    return f"https://www.twitch.tv/{raw}"


def default_output_path(source_dir: Path, vod: str) -> Path:
    raw = (vod or "").strip().rstrip("/")
    slug = raw.rsplit("/", 1)[-1] or "vod"
    slug = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in slug)
    return source_dir / f"{slug}.ts"


def resolve_downloader(downloader: str) -> str:
    binary = "yt-dlp" if downloader == "yt_dlp" else "streamlink"
    found = shutil.which(binary)
    if not found:
        raise RuntimeError(
            f"{binary} not found on PATH. Install it with: {INSTALL_HINTS[binary]}"
        )
    return found


def ytdlp_format(quality: str) -> str:
    """Map a streamlink-style quality selector onto a yt-dlp format selector."""
    raw = (quality or "best").strip().lower()
    if raw in ("best", "bestvideo", "source", "high"):
        return "bv*+ba/b"
    if raw == "worst":
        return "worst"
    if raw in ("audio_only", "audio"):
        return "ba/b"
    height = "".join(ch for ch in raw.split("p")[0] if ch.isdigit())
    if height:
        return f"bv*[height<={height}]+ba/b[height<={height}]"
    return "bv*+ba/b"


def build_capture_argv(
    *,
    downloader: str,
    url: str,
    output: Path,
    quality: str,
    write_chat: bool = False,
) -> list[str]:
    """Build the downloader argv (pure function: no PATH lookup, no execution)."""
    if downloader == "yt_dlp":
        argv = [
            "yt-dlp",
            url,
            "-o",
            str(output),
            "-f",
            ytdlp_format(quality),
            "--no-part",
            "--force-overwrites",
        ]
        if write_chat:
            argv.append("--write-chat")
        return argv
    if downloader != "streamlink":
        raise ValueError(f"Unknown downloader {downloader!r}; expected streamlink or yt_dlp")
    return [
        "streamlink",
        url,
        quality or "best",
        "-o",
        str(output),
        "--force",
        "--stream-segment-threads",
        str(STREAMLINK_SEGMENT_THREADS),
        "--stream-segment-attempts",
        str(STREAMLINK_SEGMENT_ATTEMPTS),
        "--stream-segment-timeout",
        str(STREAMLINK_SEGMENT_TIMEOUT),
        "--stream-timeout",
        str(STREAMLINK_READ_TIMEOUT),
        # The live path waits for a channel that may come online; a VOD either
        # exists or it does not, so keep stream-list retries short and fail fast.
        # Resilience for a multi-GB download comes from the segment options above.
        "--retry-streams",
        str(STREAMLINK_RETRY_DELAY),
        "--retry-max",
        str(STREAMLINK_RETRY_MAX),
    ]


def probe_video_metadata(path: Path, *, ffprobe_path: str = "ffprobe") -> MediaMetadata:
    """Read duration/width/height/fps so the stream row records real numbers."""
    probe = shutil.which(ffprobe_path) or (
        ffprobe_path if Path(ffprobe_path).exists() else None
    )
    if not probe:
        raise RuntimeError(
            f"ffprobe not found ({ffprobe_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    cmd = [
        probe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or [{}]
    stream = streams[0] if isinstance(streams, list) and streams else {}
    duration_raw = (payload.get("format") or {}).get("duration")
    return MediaMetadata(
        duration_seconds=_as_float(duration_raw),
        width=_as_int(stream.get("width")),
        height=_as_int(stream.get("height")),
        fps=_parse_fps(stream.get("r_frame_rate")),
    )


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    parsed = _as_float(value)
    return None if parsed is None else int(parsed)


def _parse_fps(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return _as_float(value)
    numerator, _, denominator = value.partition("/")
    top = _as_float(numerator)
    bottom = _as_float(denominator)
    if top is None or not bottom:
        return None
    return round(top / bottom, 3)


def _run_downloader(argv: list[str]) -> None:
    """Run the downloader with progress on stderr, keeping our stdout pure JSON."""
    proc = subprocess.run(argv, stdout=sys.stderr, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed with exit code {proc.returncode}")


def capture_vod(
    *,
    vod: str,
    output: Path,
    quality: str = "best",
    downloader: str = "streamlink",
    ffprobe_path: str = "ffprobe",
    write_chat: bool = False,
) -> CaptureResult:
    """Download a VOD at `quality` and return the file plus its real metadata."""
    url = vod_url(vod)
    normalized = normalize_downloader(downloader)
    binary = resolve_downloader(normalized)
    argv = build_capture_argv(
        downloader=normalized,
        url=url,
        output=output,
        quality=quality,
        write_chat=write_chat,
    )
    argv[0] = binary
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Capturing %s -> %s (quality=%s, downloader=%s)", url, output, quality, normalized)
    _run_downloader(argv)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Capture produced no media: {output}")

    return CaptureResult(
        path=output,
        url=url,
        downloader=normalized,
        quality=quality or "best",
        argv=argv,
        bytes=output.stat().st_size,
        media=probe_video_metadata(output, ffprobe_path=ffprobe_path),
    )


def source_dir_bytes(source_dir: Path) -> int:
    if not source_dir.exists():
        return 0
    return sum(path.stat().st_size for path in source_dir.glob("*") if path.is_file())


def prune_sources(
    source_dir: Path,
    *,
    budget_gb: float,
    keep: Iterable[Path] = (),
) -> list[Path]:
    """
    Delete oldest captures (never anything in `keep`) until the budget fits.

    ``budget_gb <= 0`` disables pruning entirely rather than deleting everything.
    """
    if budget_gb <= 0 or not source_dir.exists():
        return []
    protected = {Path(path).resolve() for path in keep}
    files = sorted(
        (path for path in source_dir.glob("*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    total = sum(path.stat().st_size for path in files)
    budget_bytes = budget_gb * (1024**3)
    removed: list[Path] = []
    for path in files:
        if total <= budget_bytes:
            break
        if path.resolve() in protected:
            continue
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        removed.append(path)
        total -= size
    return removed

