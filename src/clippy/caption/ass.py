"""ASS subtitle writing for burned-in captions (M5).

libass is the renderer (ffmpeg's `ass` filter), so a generated file is the contract
between the caption logic and the encoder. Two details matter:

1. Every cue is escaped. `{`, `}` and `\\` are ASS control characters, and a stray brace
   from a transcript would corrupt the whole track.
2. Emphasis uses karaoke tags (`\\k`) when the style wants word-level highlighting: the
   style's PrimaryColour is the "spoken" colour and SecondaryColour the base colour, so
   words light up as they are said without needing one event per word.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from clippy.caption.align import Cue
from clippy.caption.styles import CaptionStyle

logger = logging.getLogger(__name__)

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},{primary},{secondary},{outline_colour},&H64000000,{bold},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def escape_ass_text(text: str) -> str:
    """
    Neutralise ASS control characters.

    Braces become parentheses because libass treats `{...}` as override tags; losing a
    stray brace from a transcript is far better than corrupting every later cue.
    """
    return (
        text.replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def format_timestamp(seconds: float) -> str:
    """ASS timestamps are H:MM:SS.cc (centiseconds), rounded with carries handled."""
    total_centis = max(0, int(round(float(seconds) * 100)))
    hours, remainder = divmod(total_centis, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _emphasis_prefix(style: CaptionStyle) -> str:
    return (
        f"\\fscx{style.pop_scale}\\fscy{style.pop_scale}"
        if style.pop_scale
        else ""
    )


def karaoke_text(cue: Cue, *, style: CaptionStyle, emphasis: set[str]) -> str:
    """One Dialogue text with per-word `\\k` timings so words highlight as spoken."""
    if not cue.words:
        return escape_ass_text(cue.text)

    parts: list[str] = []
    scale = _emphasis_prefix(style)
    for word in cue.words:
        core = word.text.strip()
        if not core:
            continue
        centis = max(1, int(round(max(0.0, word.end - word.start) * 100)))
        text = escape_ass_text(core)
        if core.strip(".,!?\"'()[]").lower() in emphasis:
            parts.append(f"{{{scale}\\b1}}{{\\k{centis}}}{text}{{\\r}}")
        else:
            parts.append(f"{{\\k{centis}}}{text}")
    # Every word was stripped of its surrounding whitespace above, so the separator has to be
    # put back here. Joining the tagged words with no space ran the whole cue together and the
    # burned-in captions read "Ohshit" instead of "Oh shit".
    return " ".join(parts)


def plain_text(cue: Cue, *, style: CaptionStyle, emphasis: set[str]) -> str:
    """One Dialogue text with emphasised words recoloured (no karaoke timing)."""
    if not cue.words or not emphasis:
        return escape_ass_text(cue.text)

    parts: list[str] = []
    scale = _emphasis_prefix(style)
    for word in cue.words:
        core = word.text.strip()
        if not core:
            continue
        text = escape_ass_text(core)
        if core.strip(".,!?\"'()[]").lower() in emphasis:
            parts.append(
                f"{{{scale}\\c{style.highlight_colour}\\b1}}{text}"
                f"{{\\c{style.primary_colour}\\b{style.bold}}}"
            )
        else:
            parts.append(text)
    return " ".join(parts)


def write_ass(
    cues: Sequence[Cue],
    style: CaptionStyle,
    *,
    path: Path,
    width: int = 1080,
    height: int = 1920,
    emphasis: Iterable[str] = (),
) -> Path:
    """
    Write a complete ASS track: one Dialogue event per cue.

    Karaoke styles put the emphasis colour in PrimaryColour and the base colour in
    SecondaryColour (libass switches between them at each `\\k` boundary); plain styles
    use the base colour for both. An empty cue list still produces a valid file, so the
    encoder never has to special-case "captions that ended up empty".
    """
    emphasis_set = {item.strip().lower() for item in emphasis if item.strip()}
    if style.karaoke:
        primary, secondary = style.highlight_colour, style.primary_colour
    else:
        primary = secondary = style.primary_colour

    lines = [
        ASS_HEADER.format(
            width=width,
            height=height,
            font=style.font,
            size=style.font_size,
            primary=primary,
            secondary=secondary,
            outline_colour=style.outline_colour,
            bold=style.bold,
            outline=style.outline,
            shadow=style.shadow,
            alignment=style.alignment,
            margin_l=style.margin_l,
            margin_r=style.margin_r,
            margin_v=style.margin_v,
        )
    ]
    for cue in cues:
        text = (
            karaoke_text(cue, style=style, emphasis=emphasis_set)
            if style.karaoke
            else plain_text(cue, style=style, emphasis=emphasis_set)
        )
        if not text:
            continue
        start = format_timestamp(cue.start)
        end = format_timestamp(cue.end)
        lines.append(
            f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d cue(s) to %s", len(lines) - 1, path)
    return path

