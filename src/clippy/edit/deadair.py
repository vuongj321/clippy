"""Dead-air removal: cutting the parts of a clip nobody wants to watch.

Two rules shape this stage:

1. Only *dead* air is cuttable. A gap must be detected as silence by ffmpeg and - once
   word timings exist - must not contain words, so quiet speech on a quiet stream
   is never mistaken for dead air.
2. The payoff is sacred. Everything between `main_ts` and `payoff_ts` is protected, and
   total removal is capped so a clip cannot be gutted into something unwatchable.

All times here are seconds on the **base clip** timeline (0 == `base.mp4` start), which
is the artifact this stage reads. Every later stage works on the trimmed timeline.

Media work lives behind `detect_silences`; the decisions are pure functions so they can
be tested without ffmpeg.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from clippy.config import Settings

logger = logging.getLogger(__name__)

SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")

Span = tuple[float, float]


@dataclass
class DeadAirCut:
    """What to keep, what to remove, and why."""

    keep_segments: list[Span]
    removed_seconds: float
    duration: float
    applied: bool
    mode: str
    gaps: list[Span] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def removed_ratio(self) -> float:
        return (self.removed_seconds / self.duration) if self.duration > 0 else 0.0

    @property
    def kept_seconds(self) -> float:
        return max(0.0, self.duration - self.removed_seconds)


def merge_spans(spans: Sequence[Span], *, close_gap: float = 0.0) -> list[Span]:
    """Sort, clip negatives, and merge spans that touch or are within `close_gap`."""
    cleaned = sorted(
        (start, end)
        for start, end in ((max(0.0, s), max(0.0, e)) for s, e in spans)
        if end > start
    )
    merged: list[Span] = []
    for start, end in cleaned:
        if merged and start - merged[-1][1] <= close_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def complement_spans(spans: Sequence[Span], *, start: float, end: float) -> list[Span]:
    """The parts of [start, end] not covered by `spans`."""
    result: list[Span] = []
    cursor = start
    for span_start, span_end in merge_spans(spans):
        if span_end <= start:
            continue
        if span_start >= end:
            break
        if span_start > cursor:
            result.append((cursor, min(span_start, end)))
        cursor = max(cursor, min(span_end, end))
    if cursor < end:
        result.append((cursor, end))
    return [(s, e) for s, e in result if e > s]


def subtract_span(spans: Sequence[Span], block: Span) -> list[Span]:
    """Remove `block` from every span, splitting spans it lands inside."""
    block_start, block_end = block
    result: list[Span] = []
    for start, end in merge_spans(spans):
        if block_end <= start or block_start >= end:
            result.append((start, end))
            continue
        if block_start > start:
            result.append((start, block_start))
        if block_end < end:
            result.append((block_end, end))
    return result


def shrink_spans(spans: Sequence[Span], *, by: float) -> list[Span]:
    """Trim `by` seconds off both ends of each span (the guard band around speech)."""
    if by <= 0:
        return merge_spans(spans)
    return merge_spans([(s + by, e - by) for s, e in spans])


def spans_seconds(spans: Sequence[Span]) -> float:
    return sum(end - start for start, end in spans)


def total_length(spans: Sequence[Span]) -> float:
    return max(0.0, spans_seconds(merge_spans(spans)))


def parse_silences(text: str, *, end_hint: float | None = None) -> list[Span]:
    """
    Parse `silencedetect` output into spans.

    ffmpeg logs `silence_start: X` and `silence_end: Y`; when a stream ends while still
    silent only the start is logged, so a dangling start is closed at `end_hint` (the
    clip duration) - otherwise a trailing silence would survive every cut.
    """
    spans: list[Span] = []
    pending_start: float | None = None
    for line in text.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            spans.append((max(0.0, pending_start), float(end_match.group(1))))
            pending_start = None
    if pending_start is not None and end_hint is not None:
        spans.append((max(0.0, pending_start), float(end_hint)))
    return merge_spans(spans)


def detect_silences(
    media_path: Path,
    *,
    noise_db: float = -30.0,
    min_gap_seconds: float = 0.8,
    ffmpeg_path: str = "ffmpeg",
    end_hint: float | None = None,
) -> list[Span]:
    """Ask ffmpeg for the silent stretches of a clip (base-clip coordinates)."""
    ffmpeg = shutil.which(ffmpeg_path) or (
        ffmpeg_path if Path(ffmpeg_path).exists() else None
    )
    if not ffmpeg:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    cmd = [
        ffmpeg,
        "-v",
        "info",
        "-i",
        str(media_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_gap_seconds}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg silencedetect failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    return parse_silences(
        proc.stderr.decode("utf-8", errors="replace"), end_hint=end_hint
    )


def build_deadair_filter(
    segments: Sequence[tuple[float, float, float]],
    *,
    has_audio: bool = True,
) -> str:
    """
    Build the `filter_complex` that concatenates `(start, end, speed)` segments.

    `speed == 1` keeps the range as-is; `speed > 1` compresses it with `setpts`/`atempo`
    for `deadair_mode: speed`. Returns "" when there is nothing to render.
    """
    parts: list[str] = []
    labels: list[str] = []
    for index, (start, end, speed) in enumerate(segments):
        if end <= start:
            continue
        rate = max(1.0, float(speed))
        expr = "PTS-STARTPTS" if rate == 1.0 else f"(PTS-STARTPTS)/{rate:.3f}"
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts={expr}[v{index}]"
        )
        labels.append(f"[v{index}]")
        if has_audio:
            audio = (
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS"
            )
            if rate > 1.0:
                audio += f",atempo={rate:.3f}"
            parts.append(f"{audio}[a{index}]")
            labels[-1] += f"[a{index}]"
    if not labels:
        return ""
    outputs = "[vout][aout]" if has_audio else "[vout]"
    parts.append(
        f"{''.join(labels)}concat=n={len(labels)}:v=1:a={1 if has_audio else 0}{outputs}"
    )
    return ";\n".join(parts)


def plan_deadair(
    *,
    silences: Sequence[Span],
    duration: float,
    settings: Settings,
    protect: Span | None = None,
    words: Sequence[Span] | None = None,
) -> DeadAirCut:
    """
    Decide which gaps to cut and which parts to keep.

    `silences` are candidate gaps from ffmpeg; `words` veto any gap they overlap so
    quiet speech is never cut; `protect` marks the payoff region that must survive
    untouched.
    """
    duration = max(0.0, float(duration))
    mode = settings.deadair_mode
    if not settings.deadair_enabled or duration <= 0:
        return DeadAirCut(
            [(0.0, duration)],
            0.0,
            duration,
            False,
            mode,
            notes=["dead-air removal disabled"]
            if duration > 0
            else ["clip has no duration"],
        )

    candidates = merge_spans(
        [(max(0.0, start), min(duration, end)) for start, end in silences],
        close_gap=settings.deadair_keep_pad_seconds,
    )
    if not candidates:
        return DeadAirCut(
            [(0.0, duration)],
            0.0,
            duration,
            False,
            mode,
            notes=["no silence detected: nothing worth cutting"],
        )

    notes: list[str] = []
    if words:
        word_spans = merge_spans(words)
        for word_span in word_spans:
            candidates = subtract_span(candidates, word_span)
        if not candidates:
            return DeadAirCut(
                [(0.0, duration)],
                0.0,
                duration,
                False,
                mode,
                notes=["every silent gap contained speech, so nothing was cut"],
            )
        notes.append("gaps overlapping speech were left alone")

    if protect is not None and protect[1] > protect[0]:
        candidates = subtract_span(candidates, protect)
        notes.append("the payoff region was protected from cutting")
        if not candidates:
            return DeadAirCut(
                [(0.0, duration)],
                0.0,
                duration,
                False,
                mode,
                notes=[*notes, "the only silence was inside the payoff region"],
            )

    # Keep a guard band around speech so a cut never clips a word onset.
    candidates = [
        span
        for span in shrink_spans(candidates, by=settings.deadair_keep_pad_seconds)
        if (span[1] - span[0]) >= settings.deadair_min_gap_seconds
    ]
    if not candidates:
        return DeadAirCut(
            [(0.0, duration)],
            0.0,
            duration,
            False,
            mode,
            notes=[*notes, "every silent gap was too short to be worth cutting"],
        )

    budget = settings.deadair_max_removed_ratio * duration
    if spans_seconds(candidates) > budget:
        kept_gaps = list(candidates)
        while kept_gaps and spans_seconds(kept_gaps) > budget:
            # Longest gap first; on a length tie the *earlier* gap is restored, so the
            # opening context survives and the choice is not float-order dependent.
            longest = max(
                kept_gaps,
                key=lambda span: (round(span[1] - span[0], 3), -span[0]),
            )
            kept_gaps.remove(longest)
        candidates = sorted(kept_gaps)
        notes.append(
            "removal capped at "
            f"{settings.deadair_max_removed_ratio:.0%} of the clip: the largest gaps won"
        )
        if not candidates:
            return DeadAirCut(
                [(0.0, duration)],
                0.0,
                duration,
                False,
                mode,
                notes=[*notes, "the cap left nothing cuttable"],
            )

    if mode == "speed" and candidates:
        rate = max(1.0, settings.deadair_speed_factor)
        saved = spans_seconds(candidates) * (1.0 - 1.0 / rate)
        return DeadAirCut(
            [(0.0, duration)],
            saved,
            duration,
            True,
            "speed",
            gaps=sorted(candidates),
            notes=[*notes, f"compressed gaps at {rate:.2f}x instead of cutting"],
        )

    keeps = complement_spans(candidates, start=0.0, end=duration)
    fragments = [span for span in keeps if (span[1] - span[0]) < settings.deadair_min_keep_seconds]
    if fragments:
        keeps = [span for span in keeps if span not in fragments]
        notes.append(f"dropped {len(fragments)} fragment(s) shorter than the minimum keep")
    if not keeps:
        return DeadAirCut(
            [(0.0, duration)],
            0.0,
            duration,
            False,
            mode,
            notes=[*notes, "cutting would have removed the whole clip"],
        )

    kept_seconds = spans_seconds(keeps)
    return DeadAirCut(
        keep_segments=keeps,
        removed_seconds=max(0.0, duration - kept_seconds),
        duration=duration,
        applied=True,
        mode="cut",
        gaps=sorted(candidates),
        notes=notes,
    )


def segments_for_render(
    cut: DeadAirCut,
    *,
    speed_factor: float = 1.5,
) -> list[tuple[float, float, float]]:
    """(start, end, speed) triples for the filter graph, in chronological order."""
    if cut.mode != "speed" or not cut.gaps:
        return [(start, end, 1.0) for start, end in cut.keep_segments]

    rate = max(1.0, speed_factor)
    segments: list[tuple[float, float, float]] = []
    cursor = 0.0
    for start, end in cut.gaps:
        if start > cursor:
            segments.append((cursor, start, 1.0))
        segments.append((start, end, rate))
        cursor = end
    if cursor < cut.duration:
        segments.append((cursor, cut.duration, 1.0))
    return segments


