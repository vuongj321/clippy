"""Timeline alignment between chat activity and audio energy (M1).

Contract: ``estimate_timeline_offset`` returns the value that maps
stream-relative seconds onto the captured media timeline::

    media_seconds = stream_seconds + offset_seconds

A full VOD download starts at VOD t=0, so the expected answer is ~0. A partial
or late-started capture shows up as a non-zero offset, which is reported (and can
be applied via ``--source-offset``) instead of silently mis-cutting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np

from clippy.audio.intensity import compute_rms_series, extract_mono_pcm
from clippy.chat.models import ChatMessage

logger = logging.getLogger(__name__)

DEFAULT_BIN_SECONDS = 1.0
DEFAULT_SEARCH_SECONDS = 60.0
DEFAULT_ALIGN_WINDOW_SECONDS = 1800.0
MIN_MEANINGFUL_SCORE = 0.05
MIN_ACTIVE_BINS = 3

CLIP_VERIFY_FRAME_SECONDS = 0.05
CLIP_VERIFY_MARGIN_SECONDS = 45.0
CLIP_VERIFY_SPAN_SECONDS = 180.0
CLIP_VERIFY_MIN_SCORE = 0.6
CLIP_VERIFY_DEFAULT_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class TimelineAlignment:
    """Result of aligning the chat clock against the captured media clock."""

    offset_seconds: float
    score: float
    bin_seconds: float
    samples: int

    def is_meaningful(self, *, min_score: float = MIN_MEANINGFUL_SCORE) -> bool:
        """False when the curves carried too little signal to say anything."""
        return self.score >= min_score

    def to_dict(self) -> dict[str, float | int]:
        return {
            "offset_seconds": round(self.offset_seconds, 3),
            "score": round(self.score, 4),
            "bin_seconds": self.bin_seconds,
            "samples": self.samples,
        }


def _histogram(
    times: Sequence[float],
    values: Sequence[float] | None,
    *,
    bin_seconds: float,
    bins: int,
) -> np.ndarray:
    """Bin `times` by count (values=None) or by mean value (values given)."""
    times_array = np.asarray(times, dtype=np.float64).reshape(-1)
    if times_array.size == 0:
        return np.zeros(bins, dtype=np.float64)
    indices = np.clip((times_array / bin_seconds).astype(np.int64), 0, bins - 1)
    if values is None:
        counts = np.zeros(bins, dtype=np.float64)
        np.add.at(counts, indices, 1.0)
        return counts
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    if value_array.size != times_array.size:
        raise ValueError("values and times must be the same length")
    sums = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.float64)
    np.add.at(sums, indices, value_array)
    np.add.at(counts, indices, 1.0)
    return np.divide(sums, counts, out=np.zeros(bins, dtype=np.float64), where=counts > 0)


def _normalize(curve: np.ndarray) -> np.ndarray:
    if curve.size == 0:
        return curve
    centered = curve - float(np.mean(curve))
    deviation = float(np.std(centered))
    if deviation <= 1e-9:
        return np.zeros_like(curve)
    return centered / deviation


def estimate_timeline_offset(
    chat_times: Sequence[float],
    rms_times: Sequence[float],
    rms_values: Sequence[float],
    *,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
    search_seconds: float = DEFAULT_SEARCH_SECONDS,
) -> TimelineAlignment:
    """
    Cross-correlate chat activity with audio energy to recover the source offset.

    Both curves are binned on their own absolute clock (chat on stream-relative
    seconds, audio on media-relative seconds) and zero-mean normalized, so the lag
    that maximizes correlation is exactly ``media_seconds - stream_seconds``.
    """
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")
    chat = np.asarray(chat_times, dtype=np.float64).reshape(-1)
    audio_times = np.asarray(rms_times, dtype=np.float64).reshape(-1)
    span = float(
        max(
            chat.max(initial=0.0),
            audio_times.max(initial=0.0),
            0.0,
        )
    )
    bins = max(2, int(np.ceil(span / bin_seconds)) + 1)
    empty = TimelineAlignment(
        offset_seconds=0.0, score=0.0, bin_seconds=bin_seconds, samples=bins
    )

    chat_counts = _histogram(chat, None, bin_seconds=bin_seconds, bins=bins)
    audio_curve = _histogram(audio_times, rms_values, bin_seconds=bin_seconds, bins=bins)
    # Refuse to make a clock claim from too little evidence: a handful of bins
    # produces a confident-looking correlation out of pure binning structure.
    if (
        int(np.count_nonzero(chat_counts)) < MIN_ACTIVE_BINS
        or int(np.count_nonzero(audio_curve)) < MIN_ACTIVE_BINS
    ):
        return empty

    chat_curve = _normalize(chat_counts)
    audio_curve = _normalize(audio_curve)
    if not np.any(chat_curve) or not np.any(audio_curve):
        return empty
    correlation = np.correlate(audio_curve, chat_curve, mode="full")
    lags = np.arange(-(bins - 1), bins, dtype=np.float64) * bin_seconds
    allowed = np.abs(lags) <= search_seconds + 1e-9
    if not np.any(allowed):
        return empty
    best = int(np.argmax(np.where(allowed, correlation, -np.inf)))
    return TimelineAlignment(
        offset_seconds=float(lags[best]),
        score=max(0.0, float(correlation[best]) / float(bins)),
        bin_seconds=bin_seconds,
        samples=bins,
    )


def alignment_warning(
    alignment: TimelineAlignment, *, tolerance_seconds: float
) -> str | None:
    """Warn (never silently fix) when the capture clock disagrees with the chat clock."""
    if not alignment.is_meaningful():
        return None
    if abs(alignment.offset_seconds) <= tolerance_seconds + 1e-9:
        return None
    return (
        f"captured media is {alignment.offset_seconds:+.2f}s off the stream clock "
        f"(tolerance {tolerance_seconds:.2f}s); apply --source-offset "
        f"{alignment.offset_seconds:.2f}"
    )


def probe_alignment(
    media_path: Path,
    messages: Sequence[ChatMessage],
    *,
    window_start_seconds: float = 0.0,
    window_seconds: float = DEFAULT_ALIGN_WINDOW_SECONDS,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
    search_seconds: float = DEFAULT_SEARCH_SECONDS,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
) -> TimelineAlignment:
    """
    Estimate the offset from a bounded decode of the captured media.

    Only ``window_seconds`` of media are decoded (a 6 h VOD would otherwise need
    over 1 GB of PCM in memory) and chat is restricted to the stream range that
    window can possibly cover, including the +/- search margin.

    This is a **heuristic health check, not proof**: chat bursts and loud moments are
    both spiky, so a low score can peak at the wrong lag. It may warn, but a
    confidently *applied* offset must come from `verify_offset_with_clips`.
    """
    samples = extract_mono_pcm(
        media_path,
        sample_rate=sample_rate,
        ffmpeg_path=ffmpeg_path,
        start_seconds=window_start_seconds,
        duration_seconds=window_seconds,
    )
    times, rms = compute_rms_series(
        samples, sample_rate=sample_rate, frame_seconds=bin_seconds
    )
    window_end = window_start_seconds + window_seconds
    chat_times = [
        message.ts
        for message in messages
        if -search_seconds <= message.ts <= window_end + search_seconds
    ]
    return estimate_timeline_offset(
        chat_times,
        times + window_start_seconds,
        rms,
        bin_seconds=bin_seconds,
        search_seconds=search_seconds,
    )


@dataclass(frozen=True)
class OffsetVerification:
    """A decisive offset measurement made by comparing identical audio content."""

    offset_seconds: float
    score: float
    checked: int
    per_clip: list[tuple[int, float, float]] = field(default_factory=list)

    def is_confident(self, *, min_score: float = CLIP_VERIFY_MIN_SCORE) -> bool:
        return self.checked > 0 and self.score >= min_score

    def to_dict(self) -> dict[str, float | int]:
        return {
            "offset_seconds": round(self.offset_seconds, 3),
            "score": round(self.score, 4),
            "checked": self.checked,
        }


def audio_envelope(
    samples: np.ndarray,
    *,
    sample_rate: int = CLIP_VERIFY_DEFAULT_SAMPLE_RATE,
    frame_seconds: float = CLIP_VERIFY_FRAME_SECONDS,
) -> np.ndarray:
    """RMS envelope, so alignment can work on loudness shape instead of raw samples."""
    step = max(1, int(sample_rate * frame_seconds))
    count = len(samples) // step
    if count == 0:
        return np.zeros(1, dtype=np.float64)
    trimmed = np.asarray(samples[: count * step], dtype=np.float64).reshape(count, step)
    return np.sqrt((trimmed**2).mean(axis=1))


def envelope_offset(
    clip_env: np.ndarray,
    wide_env: np.ndarray,
    *,
    frame_seconds: float = CLIP_VERIFY_FRAME_SECONDS,
) -> tuple[int, float]:
    """Best lag (in frames) and its normalized correlation score."""
    a = clip_env - float(np.mean(clip_env))
    b = wide_env - float(np.mean(wide_env))
    a_norm = float(np.linalg.norm(a))
    if a_norm <= 1e-9 or len(a) > len(b):
        return 0, 0.0
    best_index, best_score = 0, -1.0
    for lag in range(0, len(b) - len(a) + 1):
        segment = b[lag : lag + len(a)]
        denom = float(np.linalg.norm(segment)) * a_norm
        score = float(np.dot(segment, a)) / denom if denom > 1e-9 else 0.0
        if score > best_score:
            best_index, best_score = lag, score
    return best_index, best_score


def verify_offset_with_clips(
    *,
    media_path: Path,
    clips: Sequence[tuple[Path, float]],
    margin_seconds: float = CLIP_VERIFY_MARGIN_SECONDS,
    span_seconds: float = CLIP_VERIFY_SPAN_SECONDS,
    sample_rate: int = CLIP_VERIFY_DEFAULT_SAMPLE_RATE,
    ffmpeg_path: str = "ffmpeg",
    min_score: float = CLIP_VERIFY_MIN_SCORE,
) -> OffsetVerification:
    """
    Recover the true source offset by correlating existing clips against the capture.

    ``clips`` are ``(clip_path, clip_start_seconds)`` where the start is on the stream
    clock (Phase 1 cut each candidate as ``[ts - pre_context, ts + post_context]``).
    Because both sides are the *same* audio content, a confident score is decisive in
    a way chat-vs-audio correlation never is. The median offset of the confident
    clips is returned so one bad clip cannot move the answer.
    """
    offsets: list[float] = []
    scores: list[float] = []
    per_clip: list[tuple[int, float, float]] = []

    for index, (clip_path, clip_start) in enumerate(clips):
        if not Path(clip_path).exists():
            continue
        wide_start = max(0.0, clip_start - margin_seconds)
        clip_env = audio_envelope(
            extract_mono_pcm(Path(clip_path), sample_rate=sample_rate, ffmpeg_path=ffmpeg_path),
            sample_rate=sample_rate,
        )
        wide_env = audio_envelope(
            extract_mono_pcm(
                media_path,
                sample_rate=sample_rate,
                ffmpeg_path=ffmpeg_path,
                start_seconds=wide_start,
                duration_seconds=span_seconds,
            ),
            sample_rate=sample_rate,
        )
        lag, score = envelope_offset(clip_env, wide_env)
        offset = lag * CLIP_VERIFY_FRAME_SECONDS + wide_start - clip_start
        per_clip.append((index, round(offset, 3), round(score, 4)))
        scores.append(score)
        if score >= min_score:
            offsets.append(offset)

    if not offsets:
        return OffsetVerification(
            offset_seconds=0.0,
            score=max(scores) if scores else 0.0,
            checked=0,
            per_clip=per_clip,
        )
    return OffsetVerification(
        offset_seconds=float(median(offsets)),
        score=float(sum(scores) / len(scores)),
        checked=len(offsets),
        per_clip=per_clip,
    )

