from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioSignalEvent:
    ts: float
    kind: str  # "intensity_spike"
    score: float
    details: dict


def _require_ffmpeg(ffmpeg_path: str) -> str:
    path = shutil.which(ffmpeg_path) or (ffmpeg_path if Path(ffmpeg_path).exists() else None)
    if not path:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    return path


def extract_mono_pcm(
    media_path: Path,
    *,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
) -> np.ndarray:
    """Decode audio to mono float32 PCM via ffmpeg."""
    ffmpeg = _require_ffmpeg(ffmpeg_path)
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(media_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio decode failed: {proc.stderr.decode('utf-8', errors='replace')}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def compute_rms_series(
    samples: np.ndarray,
    *,
    sample_rate: int = 16000,
    frame_seconds: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_seconds, rms_per_frame)."""
    frame_len = max(1, int(sample_rate * frame_seconds))
    n = len(samples)
    if n == 0:
        return np.array([]), np.array([])
    n_frames = n // frame_len
    if n_frames == 0:
        rms = np.array([float(np.sqrt(np.mean(samples**2)))])
        return np.array([0.0]), rms
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed**2, axis=1))
    times = (np.arange(n_frames) + 0.5) * frame_seconds
    return times, rms


def detect_audio_spikes(
    media_path: Path,
    *,
    frame_seconds: float = 0.5,
    baseline_seconds: float = 30.0,
    spike_multiplier: float = 2.5,
    min_rms: float = 0.02,
    spike_score: float = 0.75,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
) -> list[AudioSignalEvent]:
    samples = extract_mono_pcm(
        media_path, sample_rate=sample_rate, ffmpeg_path=ffmpeg_path
    )
    times, rms = compute_rms_series(
        samples, sample_rate=sample_rate, frame_seconds=frame_seconds
    )
    if len(rms) == 0:
        return []

    baseline_frames = max(1, int(baseline_seconds / frame_seconds))
    events: list[AudioSignalEvent] = []
    last_ts = -1e18

    for i, (t, value) in enumerate(zip(times, rms)):
        start = max(0, i - baseline_frames)
        baseline = float(np.mean(rms[start:i])) if i > start else float(np.mean(rms[: i + 1]))
        if value < min_rms:
            continue
        if baseline <= 1e-8:
            continue
        if value < baseline * spike_multiplier:
            continue
        if t - last_ts < frame_seconds * 2:
            continue
        last_ts = float(t)
        events.append(
            AudioSignalEvent(
                ts=float(t),
                kind="intensity_spike",
                score=spike_score,
                details={
                    "rms": round(float(value), 5),
                    "baseline_rms": round(baseline, 5),
                    "multiplier": round(float(value / baseline), 3),
                },
            )
        )
    return events


def probe_duration_seconds(media_path: Path, *, ffprobe_path: str = "ffprobe") -> float:
    path = shutil.which(ffprobe_path) or (
        ffprobe_path if Path(ffprobe_path).exists() else None
    )
    if not path:
        raise RuntimeError(
            f"ffprobe not found ({ffprobe_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    cmd = [
        path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    return float(proc.stdout.strip())
