from __future__ import annotations

from typing import Any

_KIND_PRIORITY = ("keyword", "rate_spike", "intensity_spike", "chat_audio")


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
        kind = _fallback_kind(signals, kinds)
        if kind:
            return f"Flagged by {kind.replace('_', ' ')} signal"
        return "Flagged by detection signals"

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}; {parts[1]}"
    return f"{parts[0]}; {parts[1]}; {parts[2]}"


def _fallback_kind(signals: dict[str, Any], kinds: set[str]) -> str | None:
    kind = signals.get("kind")
    if kind:
        return str(kind)
    for preferred in _KIND_PRIORITY:
        if preferred in kinds:
            return preferred
    return min(kinds) if kinds else None


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


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_rate_spike(event: dict[str, Any] | None) -> str:
    fallback = "Chat rate jumped vs the last minute"
    if not event:
        return fallback
    multiplier = _as_float(event.get("multiplier"))
    if multiplier is None:
        window = _as_float(event.get("window_rate"))
        baseline = _as_float(event.get("baseline_rate"))
        if window is not None and baseline is not None and baseline != 0.0:
            multiplier = window / baseline
    if multiplier is None:
        return fallback
    return f"Chat rate jumped {multiplier:.1f}x vs the last minute"


def _format_audio_spike(event: dict[str, Any] | None) -> str:
    fallback = "Audio got louder than the recent baseline"
    if not event:
        return fallback
    multiplier = _as_float(event.get("multiplier"))
    if multiplier is None:
        return fallback
    return f"Audio got {multiplier:.1f}x louder than the recent baseline"
