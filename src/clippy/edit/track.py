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
FACE_BACKEND = "opencv"
SUPPORTED_BACKENDS = (NONE_BACKEND, MOTION_BACKEND, FACE_BACKEND, "mediapipe")


@dataclass(frozen=True)
class MotionProfile:
    """Frame-to-frame motion energy per column and per row, from one decode."""

    columns: np.ndarray  # (frames - 1, width)
    rows: np.ndarray  # (frames - 1, height)

    @property
    def frame_count(self) -> int:
        return int(self.columns.shape[0])


@dataclass(frozen=True)
class TrackPoint:
    """Normalised subject position (0..1) at a time on the clip timeline."""

    t: float
    x: float
    y: float
    # Subject box size as fractions of the frame. `0.0` means "unknown", which is what the
    # motion tracker reports: the caption band then assumes a default subject height.
    width: float = 0.0
    height: float = 0.0


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
        chosen.append(
            TrackPoint(
                t=float(moment),
                x=nearest.x,
                y=nearest.y,
                width=nearest.width,
                height=nearest.height,
            )
        )
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


def sample_motion_profiles(
    media_path: Path,
    *,
    duration_seconds: float,
    start_seconds: float = 0.0,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    width: int = SAMPLE_WIDTH,
    height: int = SAMPLE_HEIGHT,
    ffmpeg_path: str = "ffmpeg",
) -> MotionProfile:
    """
    Frame-to-frame motion energy per column and per row, from a single decode.

    Decoding to a 64x36 grayscale stream keeps this cheap (a 30 s clip is a few MB of raw
    frames). Both axes come from the same frames: the horizontal profile drives the crop, and
    the vertical profile answers the much coarser question of which part of the frame the
    motion lives in.
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
    frame_bytes = width * height
    frame_count = len(proc.stdout) // frame_bytes
    if frame_count < 2:
        if proc.returncode != 0 and frame_count == 0:
            raise RuntimeError(
                "ffmpeg motion decode failed: "
                f"{proc.stderr.decode('utf-8', errors='replace')}"
            )
        return MotionProfile(
            columns=np.zeros((0, width), dtype=np.float64),
            rows=np.zeros((0, height), dtype=np.float64),
        )
    if proc.returncode != 0:
        # Twitch VODs carry damaged packets, which ffmpeg reports as a non-zero exit even though
        # it decoded usable frames. Frames on disk beat a clean exit code.
        logger.warning(
            "ffmpeg reported %s while sampling motion from %s; using the %d frames it produced",
            proc.returncode,
            media_path.name,
            frame_count,
        )
    buffer = np.frombuffer(proc.stdout[: frame_count * frame_bytes], dtype=np.uint8)
    volume = buffer.reshape(frame_count, height, width).astype(np.float64)
    motion = np.abs(np.diff(volume, axis=0))
    return MotionProfile(columns=motion.sum(axis=1), rows=motion.sum(axis=2))


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
    """Per-column motion energy, for callers that only need the horizontal profile."""
    return sample_motion_profiles(
        media_path,
        duration_seconds=duration_seconds,
        start_seconds=start_seconds,
        sample_fps=sample_fps,
        width=width,
        height=height,
        ffmpeg_path=ffmpeg_path,
    ).columns


def crop_center_y(
    track: Sequence[TrackPoint],
    *,
    source_height: float,
    crop_height: float,
) -> float:
    """
    Vertical crop position that keeps the subject in frame (0.5 = centred).

    Two guards keep this honest. Without a subject *box* - which is what the motion tracker
    reports - there is no trustworthy vertical position, so the crop stays centred. And a crop
    that already spans the full source height has no slack to move within, which is the normal
    case for a 16:9 capture: a 9:16 crop of a 16:9 frame uses all of its height. Vertical
    headroom only exists for sources *taller* than 9:16.

    The median of the tracked positions is used rather than the mean, so one stray detection
    cannot drag the crop, and the result is clamped so the crop never leaves the frame.
    """
    if not track or source_height <= 0 or crop_height >= source_height:
        return 0.5
    if not any(point.height > 0 for point in track):
        return 0.5

    positions = sorted(point.y for point in track if point.height > 0)
    median = positions[len(positions) // 2] * source_height
    half = crop_height / 2.0
    centre = min(max(median, half), source_height - half)
    return min(max(centre / source_height, 0.0), 1.0)


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
    Follow the subject across a clip: horizontally for the crop, vertically for the captions.

    `x` drives framing. `y` is a coarse motion centroid, deliberately not used to position a
    crop - without face detection a vertical crop would chase noise - but it is exactly the
    right granularity for the caption-band decision, which only needs to know which slice of
    the canvas the motion occupies *over the whole clip*. The same centre prior is applied to
    both axes, which keeps the caption decision conservative: the motion has to be consistently
    low in the frame before captions move.

    Returns an empty track (which means "crop centred", and no caption evidence) when tracking
    is disabled or a CV backend was requested but is not installed - a layout must never fail
    because of it.
    """
    backend = settings.layout_track_backend
    if backend == NONE_BACKEND:
        return []
    if backend == FACE_BACKEND:
        # Imported here so the edit package stays importable - and the suite runnable - without
        # OpenCV installed.
        from clippy.edit import faces

        if faces.available():
            try:
                face_points = faces.face_track(
                    media_path,
                    duration_seconds=duration_seconds,
                    start_seconds=start_seconds,
                    sample_fps=sample_fps,
                    width=settings.face_detection_width,
                    min_size_ratio=settings.face_min_size_ratio,
                    min_hit_ratio=settings.face_min_hit_ratio,
                    min_hits=settings.face_min_hits,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=settings.ffprobe_path,
                )
            except Exception:  # a detector failure must never fail a render
                logger.exception("Face tracking failed; falling back to motion tracking")
            else:
                if face_points:
                    return face_points
                logger.info(
                    "No usable face in %s; falling back to motion tracking",
                    media_path.name,
                )
        else:
            logger.info(
                "layout_track_backend=%r needs opencv-python-headless, which is not "
                "importable; falling back to motion tracking",
                backend,
            )
    elif backend != MOTION_BACKEND:
        logger.info(
            "Track backend %r is not implemented; falling back to motion tracking", backend
        )

    profiles = sample_motion_profiles(
        media_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
        ffmpeg_path=ffmpeg_path,
    )
    if profiles.frame_count == 0:
        return []

    horizontal = smooth_track(
        [centroid_from_energy(row) for row in profiles.columns],
        alpha=settings.layout_smoothing,
    )
    vertical = smooth_track(
        [centroid_from_energy(row) for row in profiles.rows],
        alpha=settings.layout_smoothing,
    )
    step = 1.0 / max(sample_fps, 1e-6)
    return [
        TrackPoint(t=(index + 0.5) * step, x=value, y=vertical[index])
        for index, value in enumerate(horizontal)
    ]

