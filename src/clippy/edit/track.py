"""Subject tracking for vertical framing (M6).

The goal is narrow and honest: keep whatever is moving near the centre of a 9:16 crop,
without a face detector. Frames are decoded as tiny grayscale images, frame-to-frame
motion is accumulated into a per-column energy profile, and the profile's centroid
(with a centre prior, then smoothed) becomes the crop's horizontal position.

Everything except `sample_motion_profile` is pure numpy/stdlib, so the maths - centroid,
smoothing, resampling - is testable without ffmpeg.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from clippy.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_FPS = 6.0
SAMPLE_WIDTH = 64
SAMPLE_HEIGHT = 36
CENTER_PRIOR = 0.35
DEFAULT_SMOOTHING = 0.12
NONE_BACKEND = "none"
MOTION_BACKEND = "motion"
SUPPORTED_BACKENDS = (NONE_BACKEND, MOTION_BACKEND, "opencv", "mediapipe")


@dataclass(frozen=True)
class TrackPoint:
    """Normalised subject position (0..1) at a time on the clip timeline."""

    t: float
    x: float
    y: float


def centroid_from_energy(energy: np.ndarray, *, center_prior: float = CENTER_PRIOR) -> float:
    """
    Motion centroid in 0..1, pulled toward the middle of the frame.

    The prior matters because gameplay and IRL footage often have motion at the frame
    edges (HUD, crowds); without it the crop would chase noise instead of the subject.
    """
    profile = np.asarray(energy, dtype=np.float64).reshape(-1)
    if profile.size == 0:
        return 0.5
    total = float(profile.sum())
    if total <= 1e-9:
        return 0.5
    positions = (np.arange(profile.size) + 0.5) / profile.size
    raw = float((profile * positions).sum() / total)
    prior = min(max(center_prior, 0.0), 1.0)
    return (1.0 - prior) * raw + prior * 0.5


def smooth_track(values: Sequence[float], *, alpha: float = DEFAULT_SMOOTHING) -> list[float]:
    """Exponential smoothing, so the crop pans instead of jumping."""
    if not values:
        return []
    rate = min(max(alpha, 1e-3), 1.0)
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(smoothed[-1] + rate * (float(value) - smoothed[-1]))
    return smoothed


def resample_track(
    points: Sequence[TrackPoint], *, times: Sequence[float]
) -> list[TrackPoint]:
    """Nearest-sample the track at the requested times (used for keyframes)."""
    if not points:
        return []
    ordered = sorted(points, key=lambda point: point.t)
    chosen: list[TrackPoint] = []
    for moment in times:
        nearest = min(ordered, key=lambda point: abs(point.t - moment))
        chosen.append(TrackPoint(t=float(moment), x=nearest.x, y=nearest.y))
    return chosen


def _require_ffmpeg(ffmpeg_path: str) -> str:
    ffmpeg = shutil.which(ffmpeg_path) or (
        ffmpeg_path if Path(ffmpeg_path).exists() else None
    )
    if not ffmpeg:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    return ffmpeg


def sample_motion_profile(
    media_path: Path,
    *,
    duration_seconds: float,
    start_seconds: float = 0.0,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    width: int = SAMPLE_WIDTH,
    height: int = SAMPLE_HEIGHT,
    ffmpeg_path: str = "ffmpeg",
) -> np.ndarray:
    """
    Frame-to-frame motion energy per column, shaped ``(frames - 1, width)``.

    Decoding to a 64x36 grayscale stream keeps this cheap (a 30 s clip is a few MB of raw
    frames) and the diff between consecutive frames is what the crop follows.
    """
    ffmpeg = _require_ffmpeg(ffmpeg_path)
    cmd = [ffmpeg, "-v", "error"]
    if start_seconds:
        cmd += ["-ss", f"{max(0.0, float(start_seconds)):.3f}"]
    cmd += [
        "-i",
        str(media_path),
        "-t",
        f"{max(0.1, float(duration_seconds)):.3f}",
        "-vf",
        f"fps={sample_fps},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg motion decode failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    frame_bytes = width * height
    frame_count = len(proc.stdout) // frame_bytes
    if frame_count < 2:
        return np.zeros((0, width), dtype=np.float64)
    buffer = np.frombuffer(proc.stdout[: frame_count * frame_bytes], dtype=np.uint8)
    volume = buffer.reshape(frame_count, height, width).astype(np.float64)
    return np.abs(np.diff(volume, axis=0)).sum(axis=1)


def track_subject(
    media_path: Path,
    *,
    duration_seconds: float,
    start_seconds: float = 0.0,
    settings: Settings,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    ffmpeg_path: str = "ffmpeg",
) -> list[TrackPoint]:
    """
    Follow the subject's horizontal position across a clip.

    Returns an empty track (which means "crop centred") when tracking is disabled or a CV
    backend was requested but is not installed - a layout must never fail because of it.
    Vertical position is deliberately not tracked: without face detection it would chase
    noise, and a centred vertical crop is the safer default.
    """
    backend = settings.layout_track_backend
    if backend == NONE_BACKEND:
        return []
    if backend not in (MOTION_BACKEND,):
        logger.info(
            "Track backend %r is unavailable; falling back to motion tracking", backend
        )

    profiles = sample_motion_profile(
        media_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
        ffmpeg_path=ffmpeg_path,
    )
    if profiles.shape[0] == 0:
        return []

    raw = [centroid_from_energy(row) for row in profiles]
    smoothed = smooth_track(raw, alpha=settings.layout_smoothing)
    step = 1.0 / max(sample_fps, 1e-6)
    return [
        TrackPoint(t=(index + 0.5) * step, x=value, y=0.5)
        for index, value in enumerate(smoothed)
    ]

