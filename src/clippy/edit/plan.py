"""Edit plan: the serializable contract for one automated short-form edit.

Every edit stage reads and writes the same ``EditPlan`` document
(``data/edits/{candidate_id}/plan.json``), so a render stays reproducible and
reviewable without consulting the pipeline implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from clippy.config import Settings
from clippy.store.db import Candidate, Stream, utc_now

PLAN_VERSION = 1
PLAN_FILENAME = "plan.json"

PlanStage = Literal[
    "planned",
    "extracted",
    "trimmed",
    "captioned",
    "composed",
    "normalized",
    "metadata",
    "complete",
    "failed",
]
BoundsMethod = Literal["review_window", "signal_evidence", "llm_refined"]

STRATEGIES = ("auto", "fit_blur", "irl", "gaming", "conversation")
CAPTION_STYLES = ("karaoke_highlight", "block_pop", "minimal")
CAPTION_EMPHASIS_MODES = ("heuristic", "llm", "off")
CAPTION_ANCHORS = ("auto", "top", "middle", "bottom")
DEADAIR_MODES = ("cut", "speed")

WARN_BOUNDARIES_PENDING = "boundaries_pending"
WARN_LAYOUT_PENDING = "layout_pending"
WARN_RESOLUTION_UNKNOWN = "source_resolution_unknown"
WARN_UPSCALE = "upscale_exceeds_threshold"
WARN_SOURCE_MISSING = "source_missing"
WARN_EMPHASIS_FALLBACK = "emphasis_llm_fallback"


def _r(value: float | int) -> float:
    return round(float(value), 3)


def _value(data: dict[str, Any], key: str, default: Any) -> Any:
    value = data.get(key)
    return default if value is None else value


@dataclass
class PlanWarning:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanWarning":
        return cls(
            code=str(_value(data, "code", "unknown")),
            message=str(_value(data, "message", "")),
        )


@dataclass
class ClipBounds:
    """Final clip boundaries on the source (stream-relative) timeline."""

    start: float
    end: float
    main_ts: float
    hook_ts: float
    payoff_ts: float
    method: BoundsMethod = "review_window"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"bounds end ({self.end}) precedes start ({self.start})")

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _r(self.start),
            "end": _r(self.end),
            "duration": _r(self.duration),
            "main_ts": _r(self.main_ts),
            "hook_ts": _r(self.hook_ts),
            "payoff_ts": _r(self.payoff_ts),
            "method": self.method,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipBounds":
        return cls(
            start=float(_value(data, "start", 0.0)),
            end=float(_value(data, "end", 0.0)),
            main_ts=float(_value(data, "main_ts", 0.0)),
            hook_ts=float(_value(data, "hook_ts", 0.0)),
            payoff_ts=float(_value(data, "payoff_ts", 0.0)),
            method=_value(data, "method", "review_window"),
            notes=str(_value(data, "notes", "")),
        )


@dataclass
class DeadAirPlan:
    enabled: bool = True
    applied: bool = False
    mode: str = "cut"
    keep_segments: list[list[float]] = field(default_factory=list)
    removed_seconds: float = 0.0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "mode": self.mode,
            "keep_segments": [[_r(s), _r(e)] for s, e in self.keep_segments],
            "removed_seconds": _r(self.removed_seconds),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeadAirPlan":
        segments = [
            [float(pair[0]), float(pair[1])]
            for pair in _value(data, "keep_segments", [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        return cls(
            enabled=bool(_value(data, "enabled", True)),
            applied=bool(_value(data, "applied", False)),
            mode=str(_value(data, "mode", "cut")),
            keep_segments=segments,
            removed_seconds=float(_value(data, "removed_seconds", 0.0)),
            reason=_value(data, "reason", None),
        )


@dataclass
class LayoutPlan:
    strategy: str = "auto"
    resolved_strategy: str = ""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    upscale_factor: float = 1.0
    track_backend: str = "motion"
    crop_bias: float = 0.0
    zoom: float = 1.0
    facecam_box: list[float] | None = None
    # "config" (typed) or "auto" (derived from the face track). Blank when no box was used.
    facecam_box_source: str = ""
    layers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "resolved_strategy": self.resolved_strategy,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "upscale_factor": _r(self.upscale_factor),
            "track_backend": self.track_backend,
            "crop_bias": _r(self.crop_bias),
            "zoom": _r(self.zoom),
            "facecam_box": (
                [_r(v) for v in self.facecam_box] if self.facecam_box is not None else None
            ),
            "facecam_box_source": self.facecam_box_source,
            "layers": self.layers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutPlan":
        box = _value(data, "facecam_box", None)
        return cls(
            strategy=str(_value(data, "strategy", "auto")),
            resolved_strategy=str(_value(data, "resolved_strategy", "")),
            width=int(_value(data, "width", 1080)),
            height=int(_value(data, "height", 1920)),
            fps=int(_value(data, "fps", 30)),
            upscale_factor=float(_value(data, "upscale_factor", 1.0)),
            track_backend=str(_value(data, "track_backend", "motion")),
            crop_bias=float(_value(data, "crop_bias", 0.0)),
            zoom=float(_value(data, "zoom", 1.0)),
            facecam_box=[float(v) for v in box] if isinstance(box, (list, tuple)) else None,
            facecam_box_source=str(_value(data, "facecam_box_source", "")),
            layers=[layer for layer in _value(data, "layers", []) if isinstance(layer, dict)],
        )


@dataclass
class CaptionsPlan:
    enabled: bool = True
    style: str = "karaoke_highlight"
    emphasis: str = "heuristic"
    anchor: str = "bottom"
    word_timestamps: bool = True
    cue_count: int = 0
    emphasis_words: list[str] = field(default_factory=list)
    emphasis_source: str = ""
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "style": self.style,
            "emphasis": self.emphasis,
            "anchor": self.anchor,
            "word_timestamps": self.word_timestamps,
            "cue_count": self.cue_count,
            "emphasis_words": list(self.emphasis_words),
            "emphasis_source": self.emphasis_source,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptionsPlan":
        return cls(
            enabled=bool(_value(data, "enabled", True)),
            style=str(_value(data, "style", "karaoke_highlight")),
            emphasis=str(_value(data, "emphasis", "heuristic")),
            anchor=str(_value(data, "anchor", "bottom")),
            word_timestamps=bool(_value(data, "word_timestamps", True)),
            cue_count=int(_value(data, "cue_count", 0)),
            emphasis_words=[str(w) for w in _value(data, "emphasis_words", [])],
            emphasis_source=str(_value(data, "emphasis_source", "")),
            reason=_value(data, "reason", None),
        )


@dataclass
class AudioPlan:
    normalize: bool = True
    target_lufs: float = -14.0
    true_peak: float = -1.5
    limiter: bool = True
    applied: bool = False
    used_two_pass: bool = False
    measured: dict[str, float] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalize": self.normalize,
            "target_lufs": _r(self.target_lufs),
            "true_peak": _r(self.true_peak),
            "limiter": self.limiter,
            "applied": self.applied,
            "used_two_pass": self.used_two_pass,
            "measured": {k: _r(v) for k, v in self.measured.items()},
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioPlan":
        measured = _value(data, "measured", {})
        return cls(
            normalize=bool(_value(data, "normalize", True)),
            target_lufs=float(_value(data, "target_lufs", -14.0)),
            true_peak=float(_value(data, "true_peak", -1.5)),
            limiter=bool(_value(data, "limiter", True)),
            applied=bool(_value(data, "applied", False)),
            used_two_pass=bool(_value(data, "used_two_pass", False)),
            measured={str(k): float(v) for k, v in measured.items()},
            reason=_value(data, "reason", None),
        )


@dataclass
class MetadataPlan:
    enabled: bool = True
    source: str | None = None
    title: str | None = None
    description: str | None = None
    hashtags: list[str] = field(default_factory=list)
    thumbnail: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "title": self.title,
            "description": self.description,
            "hashtags": list(self.hashtags),
            "thumbnail": self.thumbnail,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadataPlan":
        return cls(
            enabled=bool(_value(data, "enabled", True)),
            source=_value(data, "source", None),
            title=_value(data, "title", None),
            description=_value(data, "description", None),
            hashtags=[str(h) for h in _value(data, "hashtags", [])],
            thumbnail=_value(data, "thumbnail", None),
            reason=_value(data, "reason", None),
        )


@dataclass
class EditPlan:
    """Everything the edit stages need, plus what the reviewer needs to trust them."""

    candidate_id: int
    stream_id: int
    source_path: str
    bounds: ClipBounds
    version: int = PLAN_VERSION
    stage: PlanStage = "planned"
    source_offset_seconds: float = 0.0
    deadair: DeadAirPlan = field(default_factory=DeadAirPlan)
    layout: LayoutPlan = field(default_factory=LayoutPlan)
    captions: CaptionsPlan = field(default_factory=CaptionsPlan)
    audio: AudioPlan = field(default_factory=AudioPlan)
    metadata: MetadataPlan = field(default_factory=MetadataPlan)
    boundary_evidence: dict[str, Any] | None = None
    warnings: list[PlanWarning] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def add_warning(self, code: str, message: str) -> None:
        """Record a warning once per code (idempotent re-plans stay diffable)."""
        if any(w.code == code for w in self.warnings):
            return
        self.warnings.append(PlanWarning(code=code, message=message))

    def warning_codes(self) -> list[str]:
        return [w.code for w in self.warnings]

    def touch(self) -> None:
        self.updated_at = utc_now()

    def set_stage(self, stage: PlanStage) -> None:
        self.stage = stage
        self.touch()

    def source_range(self) -> tuple[float, float]:
        """Bounds mapped onto the source media timeline."""
        offset = self.source_offset_seconds
        return (self.bounds.start + offset, self.bounds.end + offset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "candidate_id": self.candidate_id,
            "stream_id": self.stream_id,
            "source_path": self.source_path,
            "source_offset_seconds": _r(self.source_offset_seconds),
            "stage": self.stage,
            "bounds": self.bounds.to_dict(),
            "deadair": self.deadair.to_dict(),
            "layout": self.layout.to_dict(),
            "captions": self.captions.to_dict(),
            "audio": self.audio.to_dict(),
            "metadata": self.metadata.to_dict(),
            "boundary_evidence": self.boundary_evidence,
            "warnings": [w.to_dict() for w in self.warnings],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditPlan":
        return cls(
            candidate_id=int(_value(data, "candidate_id", 0)),
            stream_id=int(_value(data, "stream_id", 0)),
            source_path=str(_value(data, "source_path", "")),
            bounds=ClipBounds.from_dict(_value(data, "bounds", {})),
            version=int(_value(data, "version", PLAN_VERSION)),
            stage=_value(data, "stage", "planned"),
            source_offset_seconds=float(_value(data, "source_offset_seconds", 0.0)),
            deadair=DeadAirPlan.from_dict(_value(data, "deadair", {})),
            layout=LayoutPlan.from_dict(_value(data, "layout", {})),
            captions=CaptionsPlan.from_dict(_value(data, "captions", {})),
            audio=AudioPlan.from_dict(_value(data, "audio", {})),
            metadata=MetadataPlan.from_dict(_value(data, "metadata", {})),
            boundary_evidence=_value(data, "boundary_evidence", None),
            warnings=[PlanWarning.from_dict(w) for w in _value(data, "warnings", [])],
            created_at=str(_value(data, "created_at", utc_now())),
            updated_at=str(_value(data, "updated_at", utc_now())),
        )

    def save(self, path: Path) -> Path:
        self.touch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "EditPlan":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


ARTIFACT_NAMES = {
    "plan": PLAN_FILENAME,
    "base": "base.mp4",
    "trimmed": "trimmed.mp4",
    "transcript": "transcript.json",
    "boundary_transcript": "boundary_transcript.json",
    "captions": "captions.ass",
    "layout": "layout.json",
    "vertical": "vertical.mp4",
    "final": "final.mp4",
    "thumbnail": "thumbnail.jpg",
    "metadata": "metadata.json",
    "compose_inputs": "compose.inputs.json",
}


@dataclass(frozen=True)
class EditPaths:
    """Canonical artifact locations for one candidate's edit."""

    root: Path
    plan: Path
    base: Path
    trimmed: Path
    transcript: Path
    boundary_transcript: Path
    captions: Path
    layout: Path
    vertical: Path
    final: Path
    thumbnail: Path
    metadata: Path
    compose_inputs: Path

    @classmethod
    def for_candidate(cls, settings: Settings, candidate_id: int) -> "EditPaths":
        root = settings.resolved_edits_dir() / str(candidate_id)
        return cls(
            root=root,
            plan=root / ARTIFACT_NAMES["plan"],
            base=root / ARTIFACT_NAMES["base"],
            trimmed=root / ARTIFACT_NAMES["trimmed"],
            transcript=root / ARTIFACT_NAMES["transcript"],
            boundary_transcript=root / ARTIFACT_NAMES["boundary_transcript"],
            captions=root / ARTIFACT_NAMES["captions"],
            layout=root / ARTIFACT_NAMES["layout"],
            vertical=root / ARTIFACT_NAMES["vertical"],
            final=root / ARTIFACT_NAMES["final"],
            thumbnail=root / ARTIFACT_NAMES["thumbnail"],
            metadata=root / ARTIFACT_NAMES["metadata"],
            compose_inputs=root / ARTIFACT_NAMES["compose_inputs"],
        )

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root


@dataclass
class EditOverrides:
    """Human/CLI overrides applied on top of `Settings` when building a plan."""

    strategy: str | None = None
    caption_style: str | None = None
    caption_emphasis: str | None = None
    deadair_mode: str | None = None
    captions_enabled: bool | None = None
    crop_bias: float | None = None
    zoom: float | None = None
    target_width: int | None = None
    target_height: int | None = None

    def validate(self) -> None:
        checks = (
            ("strategy", self.strategy, STRATEGIES),
            ("caption_style", self.caption_style, CAPTION_STYLES),
            ("caption_emphasis", self.caption_emphasis, CAPTION_EMPHASIS_MODES),
            ("deadair_mode", self.deadair_mode, DEADAIR_MODES),
        )
        for name, value, allowed in checks:
            if value is not None and value not in allowed:
                raise ValueError(
                    f"Invalid {name} {value!r}; expected one of {', '.join(allowed)}"
                )


def _clamp_span(
    *,
    start: float,
    end: float,
    anchor: float,
    min_seconds: float,
    max_seconds: float,
) -> tuple[float, float]:
    """Shrink or grow [start, end] around `anchor` to fit the allowed duration."""
    span = end - start
    if span > max_seconds:
        start = max(0.0, anchor - max_seconds / 2.0)
        end = start + max_seconds
        return start, end
    if span < min_seconds:
        start = max(0.0, anchor - min_seconds / 2.0)
        end = start + min_seconds
        return start, end
    return start, end


def review_window_bounds(candidate: Candidate, settings: Settings) -> ClipBounds:
    """
    The detector's review window, clamped to the configured short-form duration.

    The recorded `method` states these bounds came from the detection pass rather
    than from evidence, so no reviewer mistakes a placeholder for a decision.
    """
    start = max(0.0, candidate.source_ts - candidate.pre_context_seconds)
    end = candidate.source_ts + candidate.post_context_seconds
    start, end = _clamp_span(
        start=start,
        end=end,
        anchor=candidate.source_ts,
        min_seconds=settings.clip_min_seconds,
        max_seconds=settings.clip_max_seconds,
    )
    main_ts = min(max(candidate.source_ts, start), end)
    hook_ts = max(start, main_ts - settings.hook_lookback_seconds)
    payoff_ts = min(end, main_ts + settings.reaction_tail_seconds)
    return ClipBounds(
        start=start,
        end=end,
        main_ts=main_ts,
        hook_ts=hook_ts,
        payoff_ts=payoff_ts,
        method="review_window",
        notes="review window clamped to the clip duration",
    )


def _upscale_factor(
    *,
    stream: Stream,
    width: int,
    height: int,
    resolved_strategy: str,
    zoom: float,
) -> float:
    """
    Estimate how much the source has to grow for the target frame.

    ``fit_blur`` scales the whole frame down/up to fit, so the binding axis is
    ``min``; crop strategies must cover the full canvas, so the binding axis is
    ``max``.
    """
    source_w = stream.source_width or 0
    source_h = stream.source_height or 0
    if source_w <= 0 or source_h <= 0:
        return 1.0
    width_scale = width / float(source_w)
    height_scale = height / float(source_h)
    base = min(width_scale, height_scale) if resolved_strategy == "fit_blur" else max(
        width_scale, height_scale
    )
    return round(base * max(1.0, zoom), 3)


def _build_layout(
    settings: Settings,
    overrides: EditOverrides,
    *,
    stream: Stream,
    width: int,
    height: int,
    fps: int,
) -> LayoutPlan:
    strategy = overrides.strategy or settings.layout_strategy
    # `auto` cannot be resolved before the source is probed and the subject tracked, so the
    # plan records it as *unresolved* and composition fills in the real value. The interim
    # `fit_blur` assumption is used only for the pre-render upscale estimate.
    estimate = strategy if strategy != "auto" else "fit_blur"
    zoom = overrides.zoom if overrides.zoom is not None else settings.layout_zoom
    facecam = settings.parsed_facecam_box()
    return LayoutPlan(
        strategy=strategy,
        resolved_strategy="" if strategy == "auto" else strategy,
        width=width,
        height=height,
        fps=fps,
        upscale_factor=_upscale_factor(
            stream=stream,
            width=width,
            height=height,
            resolved_strategy=estimate,
            zoom=zoom,
        ),
        track_backend=settings.layout_track_backend,
        crop_bias=overrides.crop_bias if overrides.crop_bias is not None else 0.0,
        zoom=zoom,
        facecam_box=list(facecam) if facecam else None,
    )


def _apply_quality_warnings(plan: EditPlan, *, stream: Stream, settings: Settings) -> None:
    if not stream.source_height or not stream.source_width:
        plan.add_warning(
            WARN_RESOLUTION_UNKNOWN,
            "Source resolution is not recorded on the stream row; upscale quality is unchecked.",
        )
    elif plan.layout.upscale_factor > settings.quality_warn_upscale:
        plan.add_warning(
            WARN_UPSCALE,
            (
                f"{plan.layout.upscale_factor:.2f}x upscale for "
                f"{plan.layout.resolved_strategy} from "
                f"{stream.source_width}x{stream.source_height}; capture higher quality."
            ),
        )


def build_plan(
    *,
    candidate: Candidate,
    stream: Stream,
    source_path: Path,
    settings: Settings,
    overrides: EditOverrides | None = None,
    bounds: ClipBounds | None = None,
    boundary_evidence: dict[str, Any] | None = None,
) -> EditPlan:
    """
    Build the initial (planned-stage) edit plan for one candidate.

    `bounds`/`boundary_evidence` come from `edit.boundaries.detect_bounds`; when they
    are omitted the review window is used and the plan says so.
    """
    overrides = overrides or EditOverrides()
    overrides.validate()

    width = overrides.target_width or settings.clip_target_width
    height = overrides.target_height or settings.clip_target_height
    fps = settings.clip_fps if settings.clip_fps > 0 else 30

    captions_enabled = (
        overrides.captions_enabled
        if overrides.captions_enabled is not None
        else settings.caption_enabled
    )

    plan = EditPlan(
        candidate_id=candidate.id,
        stream_id=candidate.stream_id,
        source_path=str(source_path),
        bounds=bounds or review_window_bounds(candidate, settings),
        boundary_evidence=boundary_evidence,
        source_offset_seconds=(
            stream.source_offset_seconds
            if stream.source_offset_seconds
            else settings.capture_source_offset_seconds
        ),
        deadair=DeadAirPlan(
            enabled=settings.deadair_enabled,
            applied=False,
            mode=overrides.deadair_mode or settings.deadair_mode,
        ),
        layout=_build_layout(
            settings, overrides, stream=stream, width=width, height=height, fps=fps
        ),
        captions=CaptionsPlan(
            enabled=captions_enabled,
            style=overrides.caption_style or settings.caption_style,
            emphasis=overrides.caption_emphasis or settings.caption_emphasis,
            anchor=settings.caption_safe_area,
            word_timestamps=settings.asr_word_timestamps,
            reason=None if captions_enabled else "captions disabled for this render",
        ),
        audio=AudioPlan(
            normalize=settings.audio_normalize,
            target_lufs=settings.audio_target_lufs,
            true_peak=settings.audio_true_peak,
            limiter=settings.audio_limiter,
        ),
        metadata=MetadataPlan(enabled=settings.metadata_enabled),
    )

    if bounds is None:
        plan.add_warning(
            WARN_BOUNDARIES_PENDING,
            "Boundary detection has not run; bounds are the review window clamped to the clip duration.",
        )
    elif plan.bounds.method == "review_window":
        plan.add_warning(
            WARN_BOUNDARIES_PENDING,
            "boundary detection found no usable evidence; bounds are the review window",
        )
    if plan.layout.strategy == "auto" and not plan.layout.resolved_strategy:
        plan.add_warning(
            WARN_LAYOUT_PENDING,
            "Layout strategy 'auto' is resolved during composition; fit_blur is the "
            "fallback if composition has not run yet.",
        )
    _apply_quality_warnings(plan, stream=stream, settings=settings)
    return plan




