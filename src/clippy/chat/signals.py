from __future__ import annotations

from dataclasses import dataclass

from clippy.chat.keywords import first_keyword
from clippy.chat.models import ChatMessage


@dataclass
class ChatSignalEvent:
    ts: float
    kind: str  # "rate_spike" | "keyword"
    score: float
    details: dict


def detect_chat_signals(
    messages: list[ChatMessage],
    *,
    window_seconds: float = 5.0,
    baseline_seconds: float = 60.0,
    spike_multiplier: float = 3.0,
    min_rate: float = 0.5,
    keywords: list[str] | None = None,
    keyword_score: float = 0.7,
    spike_score: float = 0.85,
) -> list[ChatSignalEvent]:
    """Emit chat rate-spike and keyword events on the stream timeline."""
    if not messages:
        return []

    default_keywords = ["clip it", "clip that", "clip this", "clip"]
    keywords = [k.lower() for k in (keywords if keywords is not None else default_keywords)]
    events: list[ChatSignalEvent] = []

    # Keyword hits
    for msg in messages:
        hit = first_keyword(msg.text, keywords)
        if hit:
            events.append(
                ChatSignalEvent(
                    ts=msg.ts,
                    kind="keyword",
                    score=keyword_score,
                    details={
                        "keyword": hit,
                        "user": msg.user,
                        "text": msg.text,
                    },
                )
            )

    # Rate spikes via sliding windows sampled at message times
    times = [m.ts for m in messages]
    if len(times) < 2:
        return _dedupe_nearby(events)

    # Precompute cumulative counts for O(log n) window queries via two pointers
    n = len(times)
    left_w = 0
    left_b = 0
    last_spike_ts = -1e18

    for i, t in enumerate(times):
        while times[left_w] < t - window_seconds:
            left_w += 1
        while times[left_b] < t - baseline_seconds:
            left_b += 1

        window_count = i - left_w + 1
        baseline_count = i - left_b + 1
        window_rate = window_count / max(window_seconds, 1e-6)
        baseline_rate = baseline_count / max(baseline_seconds, 1e-6)

        if window_rate < min_rate:
            continue
        if baseline_rate <= 0:
            continue
        if window_rate < baseline_rate * spike_multiplier:
            continue
        # Suppress near-duplicate spike events
        if t - last_spike_ts < window_seconds:
            continue

        last_spike_ts = t
        events.append(
            ChatSignalEvent(
                ts=t,
                kind="rate_spike",
                score=spike_score,
                details={
                    "window_rate": round(window_rate, 3),
                    "baseline_rate": round(baseline_rate, 3),
                    "multiplier": round(window_rate / baseline_rate, 3),
                    "window_count": window_count,
                },
            )
        )

    events.sort(key=lambda e: e.ts)
    return _dedupe_nearby(events)


def _dedupe_nearby(events: list[ChatSignalEvent], gap: float = 1.0) -> list[ChatSignalEvent]:
    """Keep highest-scoring event within small gaps (same kind)."""
    if not events:
        return []
    events = sorted(events, key=lambda e: (e.kind, e.ts))
    out: list[ChatSignalEvent] = []
    for ev in events:
        if (
            out
            and out[-1].kind == ev.kind
            and ev.ts - out[-1].ts < gap
        ):
            if ev.score > out[-1].score:
                out[-1] = ev
            continue
        out.append(ev)
    out.sort(key=lambda e: e.ts)
    return out
