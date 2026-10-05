"""Boundary detection: deciding where a clip starts and ends.

All times are stream-relative seconds, like every other stage.

The decision is deterministic and explainable first. Evidence comes from the
candidate's own signals plus whatever the caller can supply (word timings, chat
activity, audio RMS). The constraint pass then enforces the plan's "never cut
here" rules, and the optional LLM refinement (`boundary_llm_refine`) may only
propose values that the constraint pass re-validates.

Landmarks the plan names, mapped to code:

| Plan name        | Field             |
| ---------------- | ----------------- |
| Hook / start     | `bounds.start`    |
| Context          | `hook_ts`         |
| Main event       | `main_ts`         |
| Reaction/payoff  | `payoff_ts`       |
| Natural ending   | `bounds.end`      |
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Sequence

from clippy.caption.asr import TranscriptWord
from clippy.config import Settings
from clippy.edit.plan import ClipBounds, review_window_bounds
from clippy.store.db import Candidate

logger = logging.getLogger(__name__)

DEFAULT_UTTERANCE_GAP_SECONDS = 0.6


@dataclass(frozen=True)
class Word:
    """One spoken word on the trimmed/final timeline."""

    start: float
    end: float
    text: str = ""


@dataclass
class ContextEvidence:
    """Everything the decision is allowed to look at, gathered by the caller."""

    signals: dict[str, Any] = field(default_factory=dict)
    words: list[Word] = field(default_factory=list)
    chat_times: list[float] = field(default_factory=list)
    rms_times: list[float] = field(default_factory=list)
    rms_values: list[float] = field(default_factory=list)

    def has_transcript(self) -> bool:
        return bool(self.words)

    def has_audio(self) -> bool:
        return len(self.rms_times) == len(self.rms_values) and bool(self.rms_times)


@dataclass
class BoundaryEvidence:
    """The landmarks the decision was made from (serialized into plan.json)."""

    signal_peak_ts: float
    main_end_ts: float | None = None
    utterance_end_ts: float | None = None
    chat_burst_end_ts: float | None = None
    audio_decay_ts: float | None = None
    natural_end_ts: float | None = None
    hook_utterance_start_ts: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def rounded(value: float | None) -> float | None:
            return None if value is None else round(float(value), 3)

        return {
            "signal_peak_ts": rounded(self.signal_peak_ts),
            "main_end_ts": rounded(self.main_end_ts),
            "utterance_end_ts": rounded(self.utterance_end_ts),
            "chat_burst_end_ts": rounded(self.chat_burst_end_ts),
            "audio_decay_ts": rounded(self.audio_decay_ts),
            "natural_end_ts": rounded(self.natural_end_ts),
            "hook_utterance_start_ts": rounded(self.hook_utterance_start_ts),
            "notes": list(self.notes),
        }


@dataclass
class BoundaryDecision:
    """Bounds plus the reasoning the reviewer sees in the UI."""

    bounds: ClipBounds
    evidence: BoundaryEvidence
    adjustments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": self.bounds.to_dict(),
            "evidence": self.evidence.to_dict(),
            "adjustments": list(self.adjustments),
        }


@dataclass(frozen=True)
class Utterance:
    """A run of words with no gap wider than the utterance gap."""

    start: float
    end: float
    text: str = ""


def words_to_utterances(
    words: Sequence[Word], *, gap_seconds: float = DEFAULT_UTTERANCE_GAP_SECONDS
) -> list[Utterance]:
    """Merge words into speech spans: a gap >= `gap_seconds` starts a new span."""
    ordered = sorted((w for w in words if w.end > w.start), key=lambda w: w.start)
    spans: list[Utterance] = []
    for word in ordered:
        if spans and word.start - spans[-1].end < gap_seconds:
            last = spans[-1]
            spans[-1] = Utterance(
                start=last.start,
                end=max(last.end, word.end),
                text=f"{last.text} {word.text}".strip(),
            )
        else:
            spans.append(Utterance(start=word.start, end=word.end, text=word.text))
    return spans


def utterance_containing(
    utterances: Sequence[Utterance], ts: float
) -> Utterance | None:
    """The speech span that contains `ts` (the main event is usually mid-sentence)."""
    for utterance in utterances:
        if utterance.start <= ts <= utterance.end:
            return utterance
    return None


def speech_gaps(
    utterances: Sequence[Utterance], *, span_start: float, span_end: float
) -> list[tuple[float, float]]:
    """Silent gaps between utterances, clipped to the search span."""
    gaps: list[tuple[float, float]] = []
    cursor = span_start
    for utterance in utterances:
        if utterance.start > cursor:
            gaps.append((max(cursor, span_start), min(utterance.start, span_end)))
        cursor = max(cursor, utterance.end)
    if cursor < span_end:
        gaps.append((cursor, span_end))
    return [(start, end) for start, end in gaps if end > start]


def word_containing(words: Sequence[Word], ts: float) -> Word | None:
    for word in words:
        if word.start < ts < word.end:
            return word
    return None


def _nearest_gap_edge(
    target: float, gaps: Sequence[tuple[float, float]], *, prefer: str
) -> float | None:
    """
    Nearest usable silence boundary to `target`.

    Only one edge per gap is usable, and which one depends on the direction of the
    cut: a clip *start* uses the gap's left edge (the moment the previous word ended)
    and a clip *end* uses its right edge (the moment the next word begins), so neither
    end of the clip lands in the middle of a word.
    """
    if prefer == "start":
        edges = [start for start, _ in gaps]
    else:
        edges = [end for _, end in gaps]
    if not edges:
        return None
    return min(edges, key=lambda value: abs(value - target))


def adjust_start(
    start: float,
    *,
    words: Sequence[Word],
    gaps: Sequence[tuple[float, float]],
    tolerance_seconds: float,
    floor: float,
) -> tuple[float, str | None]:
    """
    Never begin mid-word; otherwise snap into the nearest silence when close.

    A start inside a word moves back to that word's beginning so the opening
    clause stays intact (cutting before necessary context is the failure mode the
    plan warns about).
    """
    inside = word_containing(words, start)
    if inside is not None:
        moved = max(floor, inside.start)
        return moved, f"start moved back {start - moved:.2f}s to the beginning of a word"
    if tolerance_seconds > 0:
        edge = _nearest_gap_edge(start, gaps, prefer="start")
        if edge is not None and 0 < abs(edge - start) <= tolerance_seconds:
            moved = max(floor, edge)
            return moved, f"start snapped {abs(moved - start):.2f}s into silence"
    return start, None


def adjust_end(
    end: float,
    *,
    words: Sequence[Word],
    gaps: Sequence[tuple[float, float]],
    tolerance_seconds: float,
) -> tuple[float, str | None]:
    """Never end mid-word; otherwise snap into the nearest silence when close."""
    inside = word_containing(words, end)
    if inside is not None:
        return inside.end, f"end extended {inside.end - end:.2f}s to finish a word"
    if tolerance_seconds > 0:
        edge = _nearest_gap_edge(end, gaps, prefer="end")
        if edge is not None and 0 < abs(edge - end) <= tolerance_seconds:
            return edge, f"end snapped {abs(edge - end):.2f}s into silence"
    return end, None


DECAY_FACTOR = 1.5


def chat_burst_end_ts(
    chat_times: Sequence[float],
    *,
    peak_ts: float,
    settings: Settings,
    search_seconds: float,
) -> float | None:
    """
    When does the chat spike die back down after the peak?

    Uses the same windowed rate and floor as `chat.signals` so both stages agree on
    what "chat is busy" means. Returns the end of the last busy window at/after the
    peak, capped by `search_seconds`, or None when there was no chat to look at.
    """
    if not chat_times:
        return None
    window = max(0.5, settings.chat_window_seconds)
    horizon = peak_ts + max(window, search_seconds)
    relevant = sorted(t for t in chat_times if peak_ts - window <= t <= horizon)
    if not relevant:
        return None

    step = max(0.5, window / 2.0)
    last_busy = peak_ts
    moment = peak_ts
    while moment <= horizon:
        count = sum(1 for t in relevant if moment - window < t <= moment)
        if count / window >= settings.chat_min_rate:
            last_busy = moment
        moment += step
    return min(last_busy, peak_ts + search_seconds)


def audio_decay_ts(
    rms_times: Sequence[float],
    rms_values: Sequence[float],
    *,
    peak_ts: float,
    settings: Settings,
    search_seconds: float,
) -> float | None:
    """
    When does the loud moment fall back toward the local baseline?

    The baseline is the median RMS over `audio_baseline_seconds` before the peak;
    decay is the first frame at/after the peak that drops below `baseline * 1.5`
    (i.e. is no longer notably loud). None when there is no usable audio.
    """
    if not rms_times or len(rms_times) != len(rms_values):
        return None
    baseline_window = max(1.0, settings.audio_baseline_seconds)
    before = [
        value
        for moment, value in zip(rms_times, rms_values)
        if peak_ts - baseline_window <= moment <= peak_ts
    ]
    baseline = median(before) if before else settings.audio_min_rms
    threshold = max(settings.audio_min_rms, baseline * DECAY_FACTOR)
    horizon = peak_ts + search_seconds
    for moment, value in zip(rms_times, rms_values):
        if moment < peak_ts:
            continue
        if moment > horizon:
            break
        if value < threshold:
            return float(moment)
    return None


def enforce_constraints(
    start: float,
    end: float,
    *,
    main_ts: float,
    payoff_ts: float,
    settings: Settings,
    floor: float,
    ceiling: float,
) -> tuple[float, float, list[str]]:
    """
    Apply the plan's hard "never cut here" rules.

    1. A clip may not begin after the main event (context lost).
    2. The payoff and its reaction must be inside the clip.
    3. The duration ceiling wins over the tail, but the start is trimmed first.
    4. The duration floor extends forward when there is room, else backward.
    """
    if end < start:
        start, end = end, start
    notes: list[str] = []
    max_seconds = max(settings.clip_min_seconds, settings.clip_max_seconds)
    min_seconds = min(settings.clip_min_seconds, max_seconds)

    if start > main_ts:
        start = max(floor, min(main_ts, start) - settings.min_context_seconds)
        notes.append("start moved back: a clip must not begin after the main event")

    if end < payoff_ts:
        notes.append(f"end extended to keep the payoff at {payoff_ts:.2f}s")
        end = payoff_ts

    if end - start > max_seconds:
        # Pull the start forward -- but never past the main event minus the minimum
        # context, so the ceiling cannot silently produce a context-free clip.
        latest_start = max(floor, main_ts - settings.min_context_seconds)
        trimmed_start = min(max(start, end - max_seconds), latest_start)
        if trimmed_start > start:
            notes.append(
                f"start trimmed {trimmed_start - start:.2f}s to fit the "
                f"{max_seconds:.0f}s ceiling"
            )
            start = trimmed_start
    if end - start > max_seconds:
        end = start + max_seconds
        notes.append(
            f"end trimmed to fit the {max_seconds:.0f}s ceiling; the payoff may be cut"
        )

    if end - start < min_seconds:
        desired_end = min(ceiling, start + min_seconds)
        if desired_end - start >= min_seconds:
            notes.append(f"end extended to reach the {min_seconds:.0f}s minimum length")
            end = desired_end
        else:
            desired_start = max(floor, end - min_seconds)
            if desired_start < start:
                notes.append(
                    f"start moved back to reach the {min_seconds:.0f}s minimum length"
                )
                start = desired_start
            end = min(ceiling, start + min_seconds)
            if end - start < min_seconds:
                notes.append(
                    "clip is shorter than the configured minimum: the search window is too small"
                )

    start = max(floor, start)
    end = min(ceiling, max(end, start))
    return start, end, notes


START_SNAP_TOLERANCE_SECONDS = 1.5
END_SNAP_TOLERANCE_SECONDS = 2.5


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_bounds(
    *,
    candidate: Candidate,
    settings: Settings,
    evidence: ContextEvidence | None = None,
) -> BoundaryDecision:
    """
    Decide the final bounds, degrading to the review window when evidence is thin.

    Bounds come back on the source (stream-relative) timeline with the evidence and
    every adjustment the constraint pass made, so a reviewer can see why the clip
    starts and ends where it does.
    """
    context = evidence or ContextEvidence()
    notes: list[str] = []
    adjustments: list[str] = []
    fallback = review_window_bounds(candidate, settings)

    if not (context.has_transcript() or context.has_audio() or context.chat_times):
        return BoundaryDecision(
            bounds=fallback,
            evidence=BoundaryEvidence(
                signal_peak_ts=candidate.source_ts,
                notes=["no transcript, audio or chat evidence available"],
            ),
            adjustments=["fell back to the review window: no boundary evidence available"],
        )

    search_start = max(0.0, candidate.source_ts - candidate.pre_context_seconds)
    search_end = candidate.source_ts + candidate.post_context_seconds
    tail = max(0.0, settings.reaction_tail_seconds)

    utterances = words_to_utterances(context.words)
    gaps = (
        speech_gaps(utterances, span_start=search_start, span_end=search_end)
        if utterances
        else []
    )
    if not context.has_transcript():
        notes.append("no transcript: boundaries use signal, audio and chat evidence")
    if not context.has_audio():
        notes.append("no audio series available")

    main_ts = candidate.source_ts
    main_utterance = utterance_containing(utterances, main_ts)
    main_end = main_ts
    if main_utterance is not None and main_utterance.end > main_ts:
        main_end = main_utterance.end
        adjustments.append(
            f"main event extended {main_end - main_ts:.2f}s to finish the sentence"
        )

    chat_end = (
        chat_burst_end_ts(
            context.chat_times, peak_ts=main_ts, settings=settings, search_seconds=tail
        )
        if context.chat_times
        else None
    )
    audio_end = (
        audio_decay_ts(
            context.rms_times,
            context.rms_values,
            peak_ts=main_ts,
            settings=settings,
            search_seconds=tail,
        )
        if context.has_audio()
        else None
    )

    # The reaction tail caps how far the *reaction* extends; it must never truncate
    # a sentence that is still being spoken.
    reaction_candidates = [end for end in (chat_end, audio_end) if end is not None]
    reaction_end = min(max(reaction_candidates or [main_ts]), main_ts + tail)
    payoff_ts = max(main_end, reaction_end)
    natural_end = next((start for start, _ in gaps if start >= payoff_ts), None)

    # Hook: the setup line is the utterance immediately before the money line. With
    # no transcript at all, aim for the configured clip length instead.
    lookback_floor = main_ts - settings.hook_lookback_seconds
    context_floor = main_ts - settings.min_context_seconds
    hook_utterance_start = None
    if main_utterance is not None:
        for position, utterance in enumerate(utterances):
            if utterance is main_utterance:
                if position > 0:
                    hook_utterance_start = utterances[position - 1].start
                break
    if hook_utterance_start is not None:
        hook_ts = max(lookback_floor, min(hook_utterance_start, context_floor))
    elif not utterances:
        target = max(settings.clip_min_seconds, settings.clip_target_seconds)
        hook_ts = main_ts - max(0.0, target - tail)
    else:
        hook_ts = lookback_floor
    hook_ts = max(search_start, min(hook_ts, main_ts))

    start_ts, note = adjust_start(
        hook_ts,
        words=context.words,
        gaps=gaps,
        tolerance_seconds=START_SNAP_TOLERANCE_SECONDS,
        floor=search_start,
    )
    if note:
        adjustments.append(note)

    end_ts = natural_end if natural_end is not None else payoff_ts
    end_ts, note = adjust_end(
        end_ts,
        words=context.words,
        gaps=gaps,
        tolerance_seconds=END_SNAP_TOLERANCE_SECONDS,
    )
    if note:
        adjustments.append(note)

    start_ts, end_ts, constraint_notes = enforce_constraints(
        start_ts,
        end_ts,
        main_ts=main_ts,
        payoff_ts=payoff_ts,
        settings=settings,
        floor=search_start,
        ceiling=search_end,
    )
    adjustments.extend(constraint_notes)

    evidence_out = BoundaryEvidence(
        signal_peak_ts=main_ts,
        main_end_ts=main_end,
        utterance_end_ts=main_utterance.end if main_utterance else None,
        chat_burst_end_ts=chat_end,
        audio_decay_ts=audio_end,
        natural_end_ts=natural_end,
        hook_utterance_start_ts=hook_utterance_start,
        notes=notes,
    )

    if end_ts - start_ts <= 0:
        adjustments.append("fell back to the review window: no usable boundary evidence")
        return BoundaryDecision(bounds=fallback, evidence=evidence_out, adjustments=adjustments)

    bounds = ClipBounds(
        start=start_ts,
        end=end_ts,
        main_ts=main_ts,
        hook_ts=min(max(hook_ts, start_ts), end_ts),
        payoff_ts=min(max(payoff_ts, start_ts), end_ts),
        method="signal_evidence",
        notes="; ".join(adjustments[:2]),
    )
    return BoundaryDecision(bounds=bounds, evidence=evidence_out, adjustments=adjustments)


def apply_llm_bounds(
    proposal: dict[str, Any],
    decision: BoundaryDecision,
    *,
    settings: Settings,
    floor: float,
    ceiling: float,
) -> BoundaryDecision:
    """
    Clamp an LLM proposal through the same constraint pass as the deterministic path.

    A proposal is only ever a suggestion: values that would start the clip after the
    payoff, drop the reaction, or break the duration limits are corrected exactly as
    if the deterministic path had produced them.
    """
    start = _as_float(proposal.get("start"))
    if start is None:
        start = _as_float(proposal.get("hook"))
    end = _as_float(proposal.get("end"))
    main_ts = _as_float(proposal.get("main")) or decision.bounds.main_ts
    payoff_ts = _as_float(proposal.get("payoff")) or decision.bounds.payoff_ts

    start = decision.bounds.start if start is None else start
    end = decision.bounds.end if end is None else end

    start, end, notes = enforce_constraints(
        start,
        end,
        main_ts=main_ts,
        payoff_ts=payoff_ts,
        settings=settings,
        floor=floor,
        ceiling=ceiling,
    )
    rationale = proposal.get("rationale")
    bounds = ClipBounds(
        start=start,
        end=end,
        main_ts=main_ts,
        hook_ts=min(max(decision.bounds.hook_ts, start), end),
        payoff_ts=min(max(payoff_ts, start), end),
        method="llm_refined",
        notes=str(rationale)[:200] if rationale else "LLM proposal, clamped",
    )
    return BoundaryDecision(
        bounds=bounds,
        evidence=decision.evidence,
        adjustments=[
            *decision.adjustments,
            "LLM proposal clamped through the constraint pass",
            *notes,
        ],
    )


def refine_bounds_with_llm(
    decision: BoundaryDecision,
    *,
    call_llm: Callable[[dict[str, Any]], Any],
    settings: Settings,
    floor: float,
    ceiling: float,
    payload: dict[str, Any],
) -> BoundaryDecision:
    """
    Ask a model for `{start, end, main, payoff, rationale}` and clamp the answer.

    `call_llm` is injected (same shape as the caption/metadata HTTP helpers) so this
    stays offline-testable. Any error or malformed answer keeps the deterministic
    result, following the degrade-never-drop convention.
    """
    if not settings.boundary_llm_refine:
        return decision
    try:
        proposal = call_llm(payload)
    except Exception:
        logger.exception("Boundary LLM refinement failed; keeping deterministic bounds")
        return decision
    if not isinstance(proposal, dict):
        logger.info("Boundary LLM refinement returned %s; ignoring", type(proposal).__name__)
        return decision
    return apply_llm_bounds(
        proposal, decision, settings=settings, floor=floor, ceiling=ceiling
    )


def evidence_from_chat(
    chat_times: Sequence[float], *, start: float, end: float
) -> ContextEvidence:
    """Chat-only evidence (no transcript, no audio decode) for planning a window."""
    return ContextEvidence(
        chat_times=[float(t) for t in chat_times if start <= float(t) <= end]
    )


def words_to_evidence(
    words: Sequence[TranscriptWord], *, window_start: float
) -> list[Word]:
    """Map ASR word timings from a window-relative timeline onto the source timeline."""
    return [
        Word(
            start=window_start + float(word.start),
            end=window_start + float(word.end),
            text=word.text,
        )
        for word in words
        if word.end > word.start
    ]


def build_context_evidence(
    *,
    start: float,
    end: float,
    chat_times: Sequence[float] = (),
    words: Sequence[Word] = (),
    rms_times: Sequence[float] = (),
    rms_values: Sequence[float] = (),
) -> ContextEvidence:
    """
    Combine every evidence source into one `ContextEvidence`, filtered to [start, end].

    Word and chat times are on the source (stream-relative) timeline, so callers map ASR
    output with `words_to_evidence` first. The RMS series stays index-paired, because
    `ContextEvidence.has_audio` only trusts a series whose times and values line up.
    """
    audio = [
        (float(moment), float(value))
        for moment, value in zip(rms_times, rms_values)
        if start <= float(moment) <= end
    ]
    return ContextEvidence(
        words=[w for w in words if w.end > start and w.start < end],
        chat_times=[float(t) for t in chat_times if start <= float(t) <= end],
        rms_times=[moment for moment, _ in audio],
        rms_values=[value for _, value in audio],
    )



