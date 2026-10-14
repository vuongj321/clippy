"""Caption alignment: word timings into readable cues.

Captions have to be readable on a phone *and* stay in sync, so this module does the
unglamorous work of grouping words: at most a couple of lines, bounded characters and
duration, no flicker, and no cue crossing another. Everything here is pure logic so the
rules can be tested without a model or a media file.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from clippy.caption.asr import TranscriptWord

EMPHASIS_MIN_LENGTH = 9
DEFAULT_EMPHASIS_LIMIT = 12
DEFAULT_EMOTION_WORDS = frozenset(
    {
        "crazy", "insane", "unreal", "unbelievable", "never", "impossible",
        "actually", "finally", "literally", "what", "stop", "wait", "look",
        "help", "wow", "holy", "bro", "dude", "clutch", "cooked", "ripped",
        "wild", "caught", "banned", "broke", "money", "win", "won", "lost", "died",
    }
)


@dataclass(frozen=True)
class Cue:
    """One caption shown on screen, with the words that make it up."""

    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = ()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
        }


def ends_sentence(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[-1] in ".!?…"


def _join(words: Sequence[TranscriptWord], *, uppercase: bool) -> str:
    text = " ".join(word.text.strip() for word in words if word.text.strip()).strip()
    return text.upper() if uppercase else text


def _make_cue(words: Sequence[TranscriptWord], *, uppercase: bool) -> Cue:
    return Cue(
        start=words[0].start,
        end=max(words[-1].end, words[0].start),
        text=_join(words, uppercase=uppercase),
        words=tuple(words),
    )


def build_cues(
    words: Iterable[TranscriptWord],
    *,
    max_chars_per_line: int = 18,
    max_lines: int = 2,
    max_cue_seconds: float = 2.2,
    min_cue_seconds: float = 0.5,
    break_gap_seconds: float = 0.35,
    uppercase: bool = False,
    max_cue_chars: int | None = None,
) -> list[Cue]:
    """
    Group words into cues a viewer can actually read in the time available.

    Breaks on: exceeding the character budget, exceeding the duration ceiling, a pause of
    at least `break_gap_seconds`, or the end of a sentence. Cues that are too short are
    then extended and overlapping ones merged/clamped, so the track stays monotonic.
    """
    budget = max_cue_chars or max(1, max_chars_per_line * max_lines)
    ordered = sorted(
        (word for word in words if word.text.strip()), key=lambda w: (w.start, w.end)
    )

    cues: list[Cue] = []
    current: list[TranscriptWord] = []
    for word in ordered:
        if current:
            gap = word.start - current[-1].end
            candidate = _join([*current, word], uppercase=uppercase)
            if (
                len(candidate) > budget
                or (word.end - current[0].start) > max_cue_seconds
                or gap >= break_gap_seconds
                or ends_sentence(current[-1].text)
            ):
                cues.append(_make_cue(current, uppercase=uppercase))
                current = []
        current.append(word)
    if current:
        cues.append(_make_cue(current, uppercase=uppercase))
    return _normalise(cues, min_cue_seconds=min_cue_seconds, uppercase=uppercase)


def _normalise(
    cues: Sequence[Cue], *, min_cue_seconds: float, uppercase: bool
) -> list[Cue]:
    """Merge overlaps, kill flicker-length cues, and never let cues cross."""
    merged: list[Cue] = []
    for cue in cues:
        if merged and cue.start < merged[-1].end:
            previous = merged.pop()
            merged.append(
                _make_cue(list(previous.words) + list(cue.words), uppercase=uppercase)
            )
            continue
        merged.append(cue)

    stretched = [
        replace(cue, end=cue.start + min_cue_seconds)
        if cue.duration < min_cue_seconds
        else cue
        for cue in merged
    ]

    clamped: list[Cue] = []
    for index, cue in enumerate(stretched):
        limit = stretched[index + 1].start if index + 1 < len(stretched) else None
        end = cue.end if limit is None else min(cue.end, max(cue.start, limit))
        clamped.append(cue if end == cue.end else replace(cue, end=end))
    return clamped


def pick_emphasis_words(
    cues: Sequence[Cue],
    *,
    keywords: Sequence[str] = (),
    limit: int = DEFAULT_EMPHASIS_LIMIT,
) -> set[str]:
    """
    Heuristic word emphasis: shouting, numbers, chat keywords and emotion words.

    Returns lowercase forms so the ASS writer can highlight any word that matches.
    """
    # Only single-word keywords count: a phrase like "clip that" must not cause every
    # "that" for the rest of the clip to be highlighted.
    keyword_tokens = {
        str(phrase).lower().strip(".,!?")
        for phrase in keywords
        if len(str(phrase).split()) == 1 and str(phrase).strip(".,!?")
    }
    scored: Counter[str] = Counter()
    for cue in cues:
        for word in cue.words:
            core = word.text.strip().strip(".,!?\"'()[]")
            if not core:
                continue
            lowered = core.lower()
            score = 0
            if core.isupper() and len(core) > 1:
                score += 3
            if any(char.isdigit() for char in core):
                score += 2
            if lowered in keyword_tokens:
                score += 2
            if lowered in DEFAULT_EMOTION_WORDS:
                score += 2
            if len(core) >= EMPHASIS_MIN_LENGTH:
                score += 1
            if score:
                scored[lowered] += score
    return {word for word, _ in scored.most_common(limit)}


def cue_text(cues: Sequence[Cue]) -> str:
    return " ".join(cue.text for cue in cues).strip()

