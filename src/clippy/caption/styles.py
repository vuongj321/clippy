"""Caption styles: the knobs a reviewer can tune without touching ASS (M5)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from clippy.config import Settings

# ASS colours are &HAABBGGRR (alpha inverted: 00 is opaque).
PRESET_NAMES = ("karaoke_highlight", "block_pop", "minimal")

# ASS alignment: 1-3 sit at the bottom, 4-6 in the middle, 7-9 at the top. Clippy always
# centres horizontally, so only the band matters. These are the values a reviewer picks
# through `caption_safe_area`, minus `auto` (which is resolved from the frame instead).
ANCHOR_ALIGNMENTS = {
    "bottom": 2,
    "middle": 5,
    "top": 8,
}


@dataclass(frozen=True)
class CaptionStyle:
    """Resolved caption look, ready for the ASS writer."""

    name: str
    font: str
    font_size: int
    primary_colour: str
    highlight_colour: str
    outline_colour: str
    outline: int
    shadow: int
    margin_v: int
    margin_l: int
    margin_r: int
    alignment: int
    bold: int
    uppercase: bool
    karaoke: bool
    max_chars_per_line: int
    max_lines: int
    max_cue_seconds: float
    min_cue_seconds: float
    break_gap_seconds: float
    pop_scale: int = 0


_PRESETS: dict[str, dict[str, object]] = {
    "karaoke_highlight": {
        "font": "Arial",
        "font_size": 54,
        "alignment": 2,
        "bold": 1,
        "uppercase": True,
        "karaoke": True,
        "outline": 4,
        "shadow": 2,
        "pop_scale": 0,
    },
    "block_pop": {
        "font": "Arial",
        "font_size": 60,
        "alignment": 2,
        "bold": 1,
        "uppercase": True,
        "karaoke": False,
        "outline": 5,
        "shadow": 3,
        "pop_scale": 108,
    },
    "minimal": {
        "font": "Arial",
        "font_size": 44,
        "alignment": 2,
        "bold": 0,
        "uppercase": False,
        "karaoke": False,
        "outline": 3,
        "shadow": 1,
        "pop_scale": 0,
    },
}


def resolve_style(name: str, settings: Settings) -> CaptionStyle:
    """
    Apply `Settings` overrides on top of a named preset.

    Unknown names fall back to `karaoke_highlight` rather than raising, so a stale config
    value can never block a render.
    """
    preset = _PRESETS.get(name) or _PRESETS["karaoke_highlight"]
    return CaptionStyle(
        name=name if name in _PRESETS else "karaoke_highlight",
        font=settings.caption_font or str(preset["font"]),
        font_size=settings.caption_font_size or int(preset["font_size"]),  # type: ignore[arg-type]
        primary_colour=settings.caption_primary_color,
        highlight_colour=settings.caption_highlight_color,
        outline_colour="&H00000000",
        outline=int(preset["outline"]),  # type: ignore[arg-type]
        shadow=int(preset["shadow"]),  # type: ignore[arg-type]
        margin_v=settings.caption_margin_v,
        margin_l=60,
        margin_r=60,
        alignment=int(preset["alignment"]),  # type: ignore[arg-type]
        bold=int(preset["bold"]),  # type: ignore[arg-type]
        uppercase=bool(preset["uppercase"]) and settings.caption_uppercase,
        karaoke=bool(preset["karaoke"]),
        max_chars_per_line=settings.caption_max_chars_per_line,
        max_lines=settings.caption_max_lines,
        max_cue_seconds=settings.caption_max_cue_seconds,
        min_cue_seconds=settings.caption_min_cue_seconds,
        break_gap_seconds=settings.caption_break_gap_seconds,
        pop_scale=int(preset["pop_scale"]),  # type: ignore[arg-type]
    )


def caption_anchor(style: CaptionStyle, *, prefer_top: bool) -> tuple[CaptionStyle, str]:
    """
    Flip the caption band when the subject occupies the bottom of the frame.

    Returns the adjusted style and the anchor name, so `plan.json` can record what
    happened instead of the choice being invisible.
    """
    if not prefer_top:
        return style, "bottom"
    return replace(style, alignment=8), "top"


def resolve_anchor(
    style: CaptionStyle,
    *,
    safe_area: str,
    prefer_top: bool,
) -> tuple[CaptionStyle, str]:
    """
    Turn `caption_safe_area` into a concrete ASS alignment.

    `auto` keeps the frame-aware behaviour: bottom band, flipping to the top band when the
    subject owns the bottom of the frame. `top`, `middle` and `bottom` are explicit reviewer
    choices, so they ignore `prefer_top` entirely. An unrecognised value falls back to the
    bottom band rather than raising, because a stale config value must not fail a render.

    Returns the adjusted style and the anchor name for `plan.json`.
    """
    if safe_area != "auto":
        alignment = ANCHOR_ALIGNMENTS.get(safe_area)
        if alignment is not None:
            return replace(style, alignment=alignment), safe_area
    return caption_anchor(style, prefer_top=prefer_top)
