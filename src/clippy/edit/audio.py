"""Audio normalization for the final clip (M8).

Two-pass EBU R128: measure the clip, then apply `loudnorm` with the measured values
(far more accurate than one-pass on a 30 s clip) plus a limiter guard.

The video stream is **copied**, so normalizing loudness never costs another generation of
video quality - which is why composition encodes the picture once and this stage only
touches sound.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clippy.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_LRA = 11.0
LIMITER_CEILING = 0.95
MEASURE_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")


def _json_blocks(text: str) -> list[dict[str, Any]]:
    """Every balanced JSON object in `text`, in order (ffmpeg prints them multi-line)."""
    blocks: list[dict[str, Any]] = []
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    blocks.append(payload)
                start = -1
    return blocks


def parse_loudnorm_output(text: str) -> dict[str, float]:
    """
    Pull the measurement JSON out of ffmpeg's loudnorm stderr.

    ffmpeg pretty-prints a **multi-line** object after the `[Parsed_loudnorm_*]` banner, so
    the last balanced object containing `input_i` is the measurement. A line-oriented parse
    returns nothing here, which quietly downgrades every render to one-pass loudnorm - it
    did, until a real run showed an empty `measured`.
    """
    measurements: dict[str, float] = {}
    for block in _json_blocks(text):
        if "input_i" not in block:
            continue
        for key in MEASURE_KEYS:
            value = block.get(key)
            if value is None:
                continue
            try:
                measurements[key] = float(value)
            except (TypeError, ValueError):
                logger.debug("Ignoring non-numeric loudnorm field %s=%r", key, value)
    return measurements


def build_loudnorm_filter(
    measured: dict[str, float] | None,
    *,
    target_lufs: float = -14.0,
    true_peak: float = -1.5,
    limiter: bool = True,
) -> str:
    """
    EBU R128 filter chain: two-pass when measurements exist, one-pass otherwise.

    `linear=true` is only set for the measured pass: without measurements linear mode can
    push the signal past the true-peak ceiling instead of respecting it.
    """
    chain = f"loudnorm=I={target_lufs:.1f}:TP={true_peak:.1f}:LRA={DEFAULT_LRA:.0f}"
    values = measured or {}
    required = ("input_i", "input_tp", "input_lra", "input_thresh")
    if all(key in values for key in required):
        chain += (
            f":measured_I={values['input_i']:.2f}"
            f":measured_TP={values['input_tp']:.2f}"
            f":measured_LRA={values['input_lra']:.2f}"
            f":measured_thresh={values['input_thresh']:.2f}"
        )
        if "target_offset" in values:
            chain += f":offset={values['target_offset']:.2f}"
        chain += ":linear=true"
    if limiter:
        chain += f",alimiter=limit={LIMITER_CEILING}"
    return chain


@dataclass
class AudioResult:
    applied: bool
    measured: dict[str, float] = field(default_factory=dict)
    used_two_pass: bool = False
    reason: str | None = None


def _require_ffmpeg(ffmpeg_path: str) -> str:
    ffmpeg = shutil.which(ffmpeg_path) or (
        ffmpeg_path if Path(ffmpeg_path).exists() else None
    )
    if not ffmpeg:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    return ffmpeg


def measure_loudness(source: Path, *, settings: Settings) -> dict[str, float]:
    """First loudnorm pass: analyse the clip and return what it measured."""
    cmd = [
        _require_ffmpeg(settings.ffmpeg_path),
        "-v",
        "info",
        "-i",
        str(source),
        "-af",
        (
            f"loudnorm=I={settings.audio_target_lufs:.1f}"
            f":TP={settings.audio_true_peak:.1f}:LRA={DEFAULT_LRA:.0f}:print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg loudnorm measurement failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    return parse_loudnorm_output(proc.stderr.decode("utf-8", errors="replace"))


def normalize_audio(
    source: Path,
    output: Path,
    *,
    settings: Settings,
    force: bool = False,
) -> AudioResult:
    """
    Loudness-normalize `source` into `output`, copying the video stream.

    Returns what happened so the plan can explain it. A failed measurement degrades to
    single-pass loudnorm instead of shipping unnormalized audio.
    """
    if not source.exists():
        raise FileNotFoundError(f"source clip not found: {source}")
    if output.exists() and not force:
        logger.info("Reusing cached %s", output)
        return AudioResult(applied=True, reason="reused cached final clip")

    ffmpeg = _require_ffmpeg(settings.ffmpeg_path)
    measured: dict[str, float] = {}
    reason: str | None = None
    if settings.audio_normalize:
        try:
            measured = measure_loudness(source, settings=settings)
        except Exception as exc:
            logger.warning("Loudness measurement failed: %s", exc)
            reason = f"measurement failed, single-pass loudnorm used: {exc}"
        else:
            if not measured:
                reason = "loudness measurement was unusable, single-pass loudnorm used"
                logger.warning("loudnorm measurement produced no numbers for %s", source)

    filter_chain = (
        build_loudnorm_filter(
            measured,
            target_lufs=settings.audio_target_lufs,
            true_peak=settings.audio_true_peak,
            limiter=settings.audio_limiter,
        )
        if settings.audio_normalize
        else "anull"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "copy",
        "-af",
        filter_chain,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio normalization failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty output: {output}")

    if not settings.audio_normalize:
        reason = "audio normalization disabled"
    return AudioResult(
        applied=settings.audio_normalize,
        measured=measured,
        used_two_pass=bool(measured),
        reason=reason,
    )
