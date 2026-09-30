"""Vertical layout planning (M6).

A layout is a list of segments, each holding layers that map a source rectangle onto a
vertical canvas rectangle. Plans are serialisable (`layout.json`) and the composition
stage turns them into one filter graph, so a reviewer can see - and override - exactly
what the render will do.

| strategy | shape |
| --- | --- |
| `fit_blur` | whole frame fitted, a blurred copy filling the rest |
| `irl` | tracked 9:16 crop that keeps the subject near the centre |
| `gaming` | gameplay crop plus a facecam panel |
| `conversation` | two stacked panels, or a follow crop when only one region moves |

Tracking is expressed as *segments* with a static crop each (rather than a moving crop
expression) because that renders in a single pass alongside dead-air removal and caption
burn-in, and because a crop that only updates when the subject actually moves produces a
calmer result than a continuously drifting one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from clippy.caption.styles import resolve_style
from clippy.edit.track import TrackPoint, crop_center_y

logger = logging.getLogger(__name__)

STRATEGY_AUTO = "auto"
STRATEGY_FIT = "fit_blur"
STRATEGY_IRL = "irl"
STRATEGY_GAMING = "gaming"
STRATEGY_CONVERSATION = "conversation"
ALL_STRATEGIES = (
    STRATEGY_AUTO,
    STRATEGY_FIT,
    STRATEGY_IRL,
    STRATEGY_GAMING,
    STRATEGY_CONVERSATION,
)

WARN_LOW_RES = "low_resolution_layout"
WARN_TRACK_FALLBACK = "tracking_unavailable"
WARN_UPSCALE = "upscale_exceeds_threshold"
WARN_CAPTION_BAND = "caption_band_moved"

MIN_TRACKED_SOURCE_HEIGHT = 720
BLUR_SIGMA = 24
PANEL_GAP = 8
TRACK_DEADBAND = 0.02
MAX_TRACK_SEGMENTS = 24

# Caption-band geometry, used only to decide whether a caption would land on the subject.
# ASS default line spacing is close enough for a strip this size, and the subject is modelled
# as a box around the motion centroid rather than a point, because the plan is to avoid the
# subject, not to measure it.
CAPTION_LINE_HEIGHT = 1.25
SUBJECT_BOX_HEIGHT_RATIO = 0.25


@dataclass
class LayoutLayer:
    """One drawn element: a source rectangle placed on the canvas."""

    kind: str  # "video" | "blur" | "panel"
    src: tuple[float, float, float, float]
    dst: tuple[float, float, float, float]
    z: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "src": [round(value, 2) for value in self.src],
            "dst": [round(value, 2) for value in self.dst],
            "z": self.z,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutLayer":
        src = list(data.get("src") or [0, 0, 0, 0])
        dst = list(data.get("dst") or [0, 0, 0, 0])
        return cls(
            kind=str(data.get("kind", "video")),
            src=(float(src[0]), float(src[1]), float(src[2]), float(src[3])),
            dst=(float(dst[0]), float(dst[1]), float(dst[2]), float(dst[3])),
            z=int(data.get("z", 0)),
        )


@dataclass
class LayoutSegment:
    """A stretch of the clip rendered with one static set of layers."""

    start: float
    end: float
    layers: list[LayoutLayer] = field(default_factory=list)
    crop_x: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "crop_x": None if self.crop_x is None else round(self.crop_x, 4),
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutSegment":
        crop_x = data.get("crop_x")
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            layers=[
                LayoutLayer.from_dict(item)
                for item in (data.get("layers") or [])
                if isinstance(item, dict)
            ],
            crop_x=float(crop_x) if crop_x is not None else None,
        )


@dataclass
class CompositionPlan:
    """Everything the composition stage needs to build a filter graph."""

    strategy: str
    resolved_strategy: str
    width: int
    height: int
    fps: int
    source_width: int
    source_height: int
    upscale_factor: float
    segments: list[LayoutSegment] = field(default_factory=list)
    caption_prefer_top: bool = False
    caption_bottom_coverage: float = 0.0
    subject_track: list[TrackPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(segment.duration for segment in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "resolved_strategy": self.resolved_strategy,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "upscale_factor": round(self.upscale_factor, 3),
            "caption_prefer_top": self.caption_prefer_top,
            "caption_bottom_coverage": round(self.caption_bottom_coverage, 4),
            "segment_count": len(self.segments),
            "segments": [segment.to_dict() for segment in self.segments],
            "track": [
                [
                    round(point.t, 3),
                    round(point.x, 4),
                    round(point.y, 4),
                    round(point.width, 4),
                    round(point.height, 4),
                ]
                for point in self.subject_track
            ],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositionPlan":
        return cls(
            strategy=str(data.get("strategy", STRATEGY_FIT)),
            resolved_strategy=str(data.get("resolved_strategy", STRATEGY_FIT)),
            width=int(data.get("width", 1080)),
            height=int(data.get("height", 1920)),
            fps=int(data.get("fps", 30)),
            source_width=int(data.get("source_width", 0)),
            source_height=int(data.get("source_height", 0)),
            upscale_factor=float(data.get("upscale_factor", 1.0)),
            segments=[
                LayoutSegment.from_dict(item)
                for item in (data.get("segments") or [])
                if isinstance(item, dict)
            ],
            caption_prefer_top=bool(data.get("caption_prefer_top", False)),
            caption_bottom_coverage=float(data.get("caption_bottom_coverage", 0.0)),
            subject_track=[
                TrackPoint(
                    t=float(item[0]),
                    x=float(item[1]),
                    y=float(item[2]) if len(item) > 2 else 0.5,
                    width=float(item[3]) if len(item) > 3 else 0.0,
                    height=float(item[4]) if len(item) > 4 else 0.0,
                )
                for item in (data.get("track") or [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            ],
            warnings=[str(item) for item in (data.get("warnings") or [])],
        )


def clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return min(max(value, low), high)


def crop_rect(
    source_width: float,
    source_height: float,
    target_ratio: float,
    *,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> tuple[float, float, float, float]:
    """Largest target-ratio rect *inside* the source, centred on a normalised point."""
    if source_width <= 0 or source_height <= 0 or target_ratio <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    if source_width / source_height > target_ratio:
        height = source_height
        width = height * target_ratio
    else:
        width = source_width
        height = width / target_ratio
    x = clamp(center_x * source_width - width / 2.0, 0.0, max(0.0, source_width - width))
    y = clamp(center_y * source_height - height / 2.0, 0.0, max(0.0, source_height - height))
    return (x, y, width, height)


def resolve_strategy(
    requested: str,
    *,
    source_width: int,
    source_height: int,
    facecam_box: Sequence[float] | None,
    settings: Settings,
) -> tuple[str, list[str]]:
    """
    Pick a strategy, downgrading to `fit_blur` when the source cannot support a crop.

    `auto` prefers `gaming` when a facecam box is configured, `irl` for a tall-enough
    source, and `fit_blur` otherwise - which is the honest answer for low-resolution
    footage, where a 9:16 crop is mostly upscaled noise.
    """
    warnings: list[str] = []
    if requested != STRATEGY_AUTO:
        chosen = requested
    elif facecam_box:
        chosen = STRATEGY_GAMING
    elif min(source_width, source_height) >= MIN_TRACKED_SOURCE_HEIGHT:
        chosen = STRATEGY_IRL
    else:
        chosen = STRATEGY_FIT
        warnings.append(WARN_LOW_RES)

    if chosen in (STRATEGY_IRL, STRATEGY_CONVERSATION) and source_height < MIN_TRACKED_SOURCE_HEIGHT:
        if WARN_LOW_RES not in warnings:
            warnings.append(WARN_LOW_RES)
        chosen = STRATEGY_FIT
    if chosen in (STRATEGY_IRL, STRATEGY_CONVERSATION) and settings.layout_track_backend == "none":
        warnings.append(WARN_TRACK_FALLBACK)
    return chosen, warnings


def segments_from_track(
    track: Sequence[TrackPoint],
    duration: float,
    *,
    deadband: float = TRACK_DEADBAND,
    max_segments: int = MAX_TRACK_SEGMENTS,
) -> list[tuple[float, float, float]]:
    """
    Collapse a dense track into ``(start, end, crop_x)`` spans.

    A crop only moves when the subject genuinely moves (`deadband`), which keeps the pan
    calm, and the span count is capped so the filter graph stays small.
    """
    if duration <= 0:
        return []
    if not track:
        return [(0.0, duration, 0.5)]

    points = sorted(track, key=lambda point: point.t)
    spans: list[list[float]] = [[0.0, points[0].x]]
    for point in points:
        current_x = spans[-1][1]
        if abs(point.x - current_x) >= deadband:
            spans.append([max(0.0, point.t), point.x])

    while len(spans) > 1 and len(spans) > max_segments:
        # Merge the neighbouring pair with the smallest position change.
        index = min(
            range(len(spans) - 1),
            key=lambda i: abs(spans[i + 1][1] - spans[i][1]),
        )
        spans[index][1] = spans[index + 1][1]
        del spans[index + 1]

    result: list[tuple[float, float, float]] = []
    for index, (start, x) in enumerate(spans):
        end = spans[index + 1][0] if index + 1 < len(spans) else duration
        result.append((float(start), float(end), float(x)))
    return [
        (start, end, x) for start, end, x in result if end > start
    ] or [(0.0, duration, 0.5)]


WARN_FACECAM_MISSING = "facecam_box_missing"


def caption_bands(
    height: float,
    *,
    font_size: float,
    settings: Settings,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Vertical span of the caption band at the bottom and at the top of the canvas.

    Mirrors what the ASS writer does: `caption_margin_v` is the distance from the edge, and the
    band is as tall as `caption_max_lines` lines of the *resolved style's* text, so the test is
    made against the strip that will actually be drawn.
    """
    band = font_size * CAPTION_LINE_HEIGHT * settings.caption_max_lines
    margin = float(settings.caption_margin_v)
    bottom = (max(0.0, height - margin - band), max(0.0, height - margin))
    top = (margin, min(float(height), margin + band))
    return bottom, top


def caption_band_coverage(
    track: Sequence[TrackPoint],
    segments: Sequence[LayoutSegment],
    *,
    source_height: float,
    canvas_height: float,
    band: tuple[float, float],
) -> float:
    """
    Fraction of sampled frames whose subject box overlaps `band` on the canvas.

    The tracked vertical position is in source space, so it is mapped through the layer that
    actually carries it: the point is looked up in each layer's source rectangle for the
    segment covering that moment, and a frame counts once any matching layer puts the subject
    - modelled as a box of `SUBJECT_BOX_HEIGHT_RATIO` of the canvas around the motion centroid -
    inside the band.

    With no usable evidence (an empty track, no segments, a degenerate band or frame) the result
    is 0.0, which leaves `auto` on the bottom band: being wrong about the subject is worse than
    being conservative about the caption.
    """
    if not track or not segments or source_height <= 0 or canvas_height <= 0:
        return 0.0
    band_top, band_bottom = band
    if band_bottom <= band_top:
        return 0.0

    hits = 0
    sampled = 0
    for point in track:
        segment = next(
            (item for item in segments if item.start <= point.t < item.end), None
        )
        if segment is None:
            continue
        sampled += 1
        source_y = min(max(point.y, 0.0), 1.0) * source_height
        # A detected subject box is what the tracker really saw. Without one - the motion
        # tracker reports no box - the subject is assumed to be `SUBJECT_BOX_HEIGHT_RATIO` of the
        # frame tall, which keeps the motion path conservative rather than confident.
        half_subject = (
            max(0.01, point.height) / 2.0
            if point.height > 0
            else SUBJECT_BOX_HEIGHT_RATIO / 2.0
        )
        source_top = min(max(point.y - half_subject, 0.0), 1.0) * source_height
        source_bottom = min(max(point.y + half_subject, 0.0), 1.0) * source_height
        for layer in segment.layers:
            if layer.kind == "blur" or layer.src[3] <= 0:
                continue
            src_y, src_height = layer.src[1], layer.src[3]
            if not (src_y <= source_y <= src_y + src_height):
                continue
            scale = layer.dst[3] / src_height
            canvas_top = layer.dst[1] + (source_top - src_y) * scale
            canvas_bottom = layer.dst[1] + (source_bottom - src_y) * scale
            if canvas_top <= band_bottom and canvas_bottom >= band_top:
                hits += 1
                break
    if sampled == 0:
        return 0.0
    return hits / sampled


def plan_layout(
    *,
    requested: str,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
    fps: int,
    duration: float,
    settings: Settings,
    track: Sequence[TrackPoint] = (),
    facecam_box: Sequence[float] | None = None,
    caption_style: str = "",
) -> CompositionPlan:
    """
    Build the composition plan for one clip (pure: it never touches media).

    `caption_style` is the style that will be burned in (empty means `settings.caption_style`).
    It is needed here because the caption band is measured against the real text height, and
    avoiding that band is what can move the captions to the top.

    Returns segments with static layers, plus the warnings a reviewer needs to judge the
    result: a low-resolution source, a missing facecam box, or a crop that upscales more
    than `quality_warn_upscale`.
    """
    resolved, warnings = resolve_strategy(
        requested,
        source_width=source_width,
        source_height=source_height,
        facecam_box=facecam_box,
        settings=settings,
    )
    facecam = _facecam_rect(facecam_box, source_width, source_height)
    if resolved == STRATEGY_GAMING and facecam is None:
        warnings.append(WARN_FACECAM_MISSING)
        resolved = STRATEGY_FIT

    duration = max(0.0, float(duration))
    prefer_top = False
    if resolved == STRATEGY_IRL:
        # Vertical headroom is a property of the geometry, not of the tracker: see `crop_center_y`.
        crop_height = crop_rect(source_width, source_height, width / height)[3]
        crop_y = crop_center_y(
            track, source_height=source_height, crop_height=crop_height
        )
        segments = [
            LayoutSegment(
                start=start,
                end=end,
                crop_x=crop_x,
                layers=_irl_layers(
                    source_width,
                    source_height,
                    width,
                    height,
                    crop_x=crop_x,
                    crop_y=crop_y,
                ),
            )
            for start, end, crop_x in segments_from_track(track, duration)
        ]
    elif resolved == STRATEGY_GAMING:
        segments = [
            LayoutSegment(
                start=0.0,
                end=duration,
                layers=_gaming_layers(
                    source_width, source_height, width, height, facecam=facecam
                ),
            )
        ]
        prefer_top = True  # the facecam panel owns the bottom of the frame
    elif resolved == STRATEGY_CONVERSATION:
        segments = [
            LayoutSegment(
                start=0.0,
                end=duration,
                layers=_conversation_layers(source_width, source_height, width, height),
            )
        ]
    else:
        segments = [
            LayoutSegment(
                start=0.0,
                end=duration,
                layers=_fit_layers(source_width, source_height, width, height),
            )
        ]

    factor = 1.0
    for segment in segments:
        for layer in segment.layers:
            if layer.kind == "blur":
                continue
            factor = max(factor, _upscale_factor(layer.src, layer.dst))
    if factor > settings.quality_warn_upscale and WARN_UPSCALE not in warnings:
        warnings.append(WARN_UPSCALE)

    # Caption band. A facecam panel owns the bottom of the frame by construction; otherwise the
    # subject track can say the same thing for a full-frame layout. `caption_avoid_ratio` is the
    # tolerance for the subject merely dipping into the band, and anything unmeasurable leaves the
    # captions where they were.
    bottom_band, top_band = caption_bands(
        height,
        font_size=resolve_style(caption_style or settings.caption_style, settings).font_size,
        settings=settings,
    )
    bottom_coverage = caption_band_coverage(
        track,
        segments,
        source_height=source_height,
        canvas_height=height,
        band=bottom_band,
    )
    if (
        not prefer_top
        and bottom_coverage > settings.caption_avoid_ratio
        and bottom_coverage
        >= caption_band_coverage(
            track,
            segments,
            source_height=source_height,
            canvas_height=height,
            band=top_band,
        )
    ):
        prefer_top = True
        warnings.append(WARN_CAPTION_BAND)

    return CompositionPlan(
        strategy=requested,
        resolved_strategy=resolved,
        width=width,
        height=height,
        fps=fps,
        source_width=source_width,
        source_height=source_height,
        upscale_factor=round(factor, 3),
        segments=segments,
        caption_prefer_top=prefer_top,
        caption_bottom_coverage=round(bottom_coverage, 4),
        subject_track=list(track),
        warnings=warnings,
    )



def _facecam_rect(
    box: Sequence[float] | None, source_width: float, source_height: float
) -> tuple[float, float, float, float] | None:
    """Resolve a facecam box: fractions of the frame when every value is <= 1, else pixels."""
    if not box or len(box) != 4:
        return None
    x, y, w, h = (float(value) for value in box)
    if all(0.0 <= value <= 1.0 for value in (x, y, w, h)):
        x, y, w, h = x * source_width, y * source_height, w * source_width, h * source_height
    x = clamp(x, 0.0, max(0.0, source_width - 1.0))
    y = clamp(y, 0.0, max(0.0, source_height - 1.0))
    w = clamp(w, 1.0, max(1.0, source_width - x))
    h = clamp(h, 1.0, max(1.0, source_height - y))
    return (x, y, w, h)


def _crop_inside(
    rect: tuple[float, float, float, float],
    target_ratio: float,
    *,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> tuple[float, float, float, float]:
    """Crop `target_ratio` inside an arbitrary rect (used for panels)."""
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return rect
    if w / h > target_ratio:
        height = h
        width = height * target_ratio
    else:
        width = w
        height = width / target_ratio
    return (
        x + clamp(center_x * w - width / 2.0, 0.0, max(0.0, w - width)),
        y + clamp(center_y * h - height / 2.0, 0.0, max(0.0, h - height)),
        width,
        height,
    )


def _fit_layers(
    source_width: float, source_height: float, width: float, height: float
) -> list[LayoutLayer]:
    """Whole frame fitted inside the canvas with a blurred copy filling the rest."""
    source_ratio = source_width / source_height if source_height else width / height
    fitted_w = min(width, height * source_ratio)
    fitted_h = fitted_w / source_ratio
    cover_w = max(width, height * source_ratio)
    cover_h = cover_w / source_ratio
    return [
        LayoutLayer(
            kind="blur",
            src=(0.0, 0.0, source_width, source_height),
            dst=((width - cover_w) / 2.0, (height - cover_h) / 2.0, cover_w, cover_h),
            z=0,
        ),
        LayoutLayer(
            kind="video",
            src=(0.0, 0.0, source_width, source_height),
            dst=((width - fitted_w) / 2.0, (height - fitted_h) / 2.0, fitted_w, fitted_h),
            z=1,
        ),
    ]


def _irl_layers(
    source_width: float,
    source_height: float,
    width: float,
    height: float,
    *,
    crop_x: float,
    crop_y: float = 0.5,
) -> list[LayoutLayer]:
    """A canvas-shaped crop of the source, centred on the tracked subject."""
    src = crop_rect(
        source_width,
        source_height,
        width / height,
        center_x=crop_x,
        center_y=crop_y,
    )
    return [
        LayoutLayer(kind="video", src=src, dst=(0.0, 0.0, width, height), z=1)
    ]


def _gaming_layers(
    source_width: float,
    source_height: float,
    width: float,
    height: float,
    *,
    facecam: tuple[float, float, float, float] | None,
    gameplay_ratio: float = 0.6,
) -> list[LayoutLayer]:
    """Gameplay on top, facecam panel below."""
    gameplay_h = round(height * gameplay_ratio)
    gameplay_src = crop_rect(
        source_width, source_height, width / gameplay_h, center_x=0.5
    )
    layers = [
        LayoutLayer(
            kind="video",
            src=gameplay_src,
            dst=(0.0, 0.0, width, float(gameplay_h)),
            z=1,
        )
    ]
    if facecam is not None:
        panel_h = float(height - gameplay_h - PANEL_GAP)
        panel_src = _crop_inside(facecam, width / panel_h, center_x=0.5, center_y=0.5)
        layers.append(
            LayoutLayer(
                kind="panel",
                src=panel_src,
                dst=(0.0, float(gameplay_h + PANEL_GAP), width, panel_h),
                z=2,
            )
        )
    return layers


def _conversation_layers(
    source_width: float, source_height: float, width: float, height: float
) -> list[LayoutLayer]:
    """
    Two stacked panels: left half above, right half below.

    A true speaker-tracking version needs per-region detection (a documented gap); this
    keeps both sides of a conversation visible without guessing who is talking.
    """
    panel_h = (height - PANEL_GAP) / 2.0
    ratio = width / panel_h
    left = _crop_inside((0.0, 0.0, source_width / 2.0, source_height), ratio)
    right = _crop_inside(
        (source_width / 2.0, 0.0, source_width / 2.0, source_height), ratio
    )
    return [
        LayoutLayer(kind="panel", src=left, dst=(0.0, 0.0, width, panel_h), z=1),
        LayoutLayer(
            kind="panel", src=right, dst=(0.0, panel_h + PANEL_GAP, width, panel_h), z=2
        ),
    ]


def _upscale_factor(
    src: tuple[float, float, float, float], dst: tuple[float, float, float, float]
) -> float:
    """How much a layer has to grow; the larger axis decides."""
    sx = dst[2] / src[2] if src[2] > 0 else 1.0
    sy = dst[3] / src[3] if src[3] > 0 else 1.0
    return max(sx, sy)


