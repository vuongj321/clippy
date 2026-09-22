from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDetection:
    ts: float
    score: float
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoalescedCandidate:
    ts: float
    score: float
    signals: dict[str, Any]


def coalesce_detections(
    detections: list[RawDetection],
    *,
    gap_seconds: float = 20.0,
) -> list[CoalescedCandidate]:
    """Merge overlapping/nearby detections into one candidate per moment."""
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.ts)
    clusters: list[list[RawDetection]] = [[ordered[0]]]
    for det in ordered[1:]:
        if det.ts - clusters[-1][-1].ts <= gap_seconds:
            clusters[-1].append(det)
        else:
            clusters.append([det])

    result: list[CoalescedCandidate] = []
    for cluster in clusters:
        best = max(cluster, key=lambda d: d.score)
        # Representative timestamp: peak score; fall back to mean of times
        peak = best.ts
        merged_signals: dict[str, Any] = {
            "events": [d.signals for d in cluster],
            "event_count": len(cluster),
            "kinds": sorted(
                {
                    d.signals.get("kind")
                    for d in cluster
                    if d.signals.get("kind")
                }
            ),
        }
        # Preserve primary event details from best
        for key, value in best.signals.items():
            if key not in merged_signals:
                merged_signals[key] = value
        result.append(
            CoalescedCandidate(ts=peak, score=best.score, signals=merged_signals)
        )
    return result


def combine_signal_events(
    chat_events: list[Any],
    audio_events: list[Any],
    *,
    proximity_seconds: float = 5.0,
) -> list[RawDetection]:
    """
    Turn chat/audio signal events into raw detections.

    Nearby chat+audio pairs are boosted into a single higher-score detection.
    """
    detections: list[RawDetection] = []
    used_audio: set[int] = set()

    for chat in chat_events:
        partner_idx = None
        partner = None
        best_dist = proximity_seconds
        for i, audio in enumerate(audio_events):
            if i in used_audio:
                continue
            dist = abs(audio.ts - chat.ts)
            if dist <= best_dist:
                best_dist = dist
                partner_idx = i
                partner = audio
        if partner is not None and partner_idx is not None:
            used_audio.add(partner_idx)
            score = min(1.0, chat.score + partner.score * 0.5)
            detections.append(
                RawDetection(
                    ts=chat.ts,
                    score=score,
                    signals={
                        "kind": "chat_audio",
                        "chat": {"kind": chat.kind, "score": chat.score, **chat.details},
                        "audio": {
                            "kind": partner.kind,
                            "score": partner.score,
                            **partner.details,
                        },
                    },
                )
            )
        else:
            detections.append(
                RawDetection(
                    ts=chat.ts,
                    score=chat.score,
                    signals={"kind": chat.kind, **chat.details},
                )
            )

    for i, audio in enumerate(audio_events):
        if i in used_audio:
            continue
        detections.append(
            RawDetection(
                ts=audio.ts,
                score=audio.score,
                signals={"kind": audio.kind, **audio.details},
            )
        )

    detections.sort(key=lambda d: d.ts)
    return detections
