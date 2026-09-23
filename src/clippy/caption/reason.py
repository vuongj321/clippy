from __future__ import annotations

from typing import Any


def format_extract_reason(signals: dict[str, Any] | None) -> str:
    """Turn coalesced candidate signals into a short human-readable reason."""
    if not signals:
        return "Flagged by detection signals"

    parts: list[str] = []
    keyword = _first_keyword(signals)
    spike = _first_matching(signals, "rate_spike")
    audio = _first_matching(signals, "intensity_spike")
    kinds = _all_kinds(signals)

    if keyword or "keyword" in kinds:
        parts.append(_format_keyword(keyword))
    if spike or "rate_spike" in kinds:
        parts.append(_format_rate_spike(spike))
    if audio or "intensity_spike" in kinds:
        parts.append(_format_audio_spike(audio))

    if not parts:
        kind = signals.get("kind") or (next(iter(kinds), None) if kinds else None)
        if kind:
            return f"Flagged by {str(kind).replace('_', ' ')} signal"
        return "Flagged by detection signals"

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}; {parts[1]}"
    return f"{parts[0]}; {parts[1]}; {parts[2]}"


def _all_kinds(signals: dict[str, Any]) -> set[str]:
    kinds: set[str] = set()
    raw = signals.get("kinds")
    if isinstance(raw, list):
        kinds.update(str(k) for k in raw if k)
    for event in _iter_event_dicts(signals):
        kind = event.get("kind")
        if kind:
            kinds.add(str(kind))
    return kinds


def _iter_event_dicts(signals: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [signals]
    events = signals.get("events")
    if isinstance(events, list):
        for item in events:
            if isinstance(item, dict):
                out.append(item)
                for nested_key in ("chat", "audio"):
                    nested = item.get(nested_key)
                    if isinstance(nested, dict):
                        out.append(nested)
    for nested_key in ("chat", "audio"):
        nested = signals.get(nested_key)
        if isinstance(nested, dict):
            out.append(nested)
    return out


def _first_matching(signals: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for event in _iter_event_dicts(signals):
        if event.get("kind") == kind:
            return event
    return None


def _first_keyword(signals: dict[str, Any]) -> dict[str, Any] | None:
    found = _first_matching(signals, "keyword")
    if found:
        return found
    if signals.get("keyword"):
        return signals
    return None


def _format_keyword(event: dict[str, Any] | None) -> str:
    if not event:
        return "Chat asked to clip it"
    keyword = event.get("keyword") or "clip"
    user = event.get("user")
    quoted = f'"{keyword}"'
    if user:
        return f"Chat asked to clip it ({quoted} from {user})"
    return f"Chat asked to clip it ({quoted})"


def _format_rate_spike(event: dict[str, Any] | None) -> str:
    if not event:
        return "Chat rate jumped vs the last minute"
    multiplier = event.get("multiplier")
    if multiplier is None:
        window = event.get("window_rate")
        baseline = event.get("baseline_rate")
        if window and baseline:
            try:
                multiplier = float(window) / float(baseline)
            except (TypeError, ValueError, ZeroDivisionError):
                multiplier = None
    if multiplier is not None:
        try:
            return f"Chat rate jumped {float(multiplier):.1f}x vs the last minute"
        except (TypeError, ValueError):
            pass
    return "Chat rate jumped vs the last minute"


def _format_audio_spike(event: dict[str, Any] | None) -> str:
    if not event:
        return "Audio got louder than the recent baseline"
    multiplier = event.get("multiplier")
    if multiplier is not None:
        try:
            return f"Audio got {float(multiplier):.1f}x louder than the recent baseline"
        except (TypeError, ValueError):
            pass
    return "Audio got louder than the recent baseline"
