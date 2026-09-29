from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from clippy.config import Settings
from clippy.edit.audio import (
    _json_blocks,
    build_loudnorm_filter,
    measure_loudness,
    normalize_audio,
    parse_loudnorm_output,
)

FIXTURE = Path(__file__).resolve().parents[1] / "samples" / "sample_vod.mp4"

# Verbatim shape of ffmpeg's loudnorm measurement: a *pretty-printed multi-line* object.
FFMPEG_LOUDNORM_STDERR = """
ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers
  built with gcc 16.2.0 (Rev3, Built by MSYS2 project)
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'vertical.mp4':
  Duration: 00:00:25.70, start: 0.000000, bitrate: 5287 kb/s
Stream mapping:
  Stream #0:1 -> #0:0 (aac (native) -> pcm_s16le (native))
[Parsed_loudnorm_0 @ 000001c0f5e3a0c0]
{
\t"input_i" : "-23.45",
\t"input_tp" : "-3.21",
\t"input_lra" : "4.60",
\t"input_thresh" : "-34.57",
\t"output_i" : "-14.00",
\t"output_tp" : "-1.40",
\t"output_lra" : "4.20",
\t"output_thresh" : "-24.12",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.55"
}
"""


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "data_dir": tmp_path,
        "audio_normalize": True,
        "audio_target_lufs": -14.0,
        "audio_true_peak": -1.5,
        "audio_limiter": True,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_parse_loudnorm_output_reads_a_multiline_block():
    measured = parse_loudnorm_output(FFMPEG_LOUDNORM_STDERR)

    assert measured["input_i"] == pytest.approx(-23.45)
    assert measured["input_tp"] == pytest.approx(-3.21)
    assert measured["input_lra"] == pytest.approx(4.6)
    assert measured["input_thresh"] == pytest.approx(-34.57)
    assert measured["target_offset"] == pytest.approx(0.55)


def test_parse_loudnorm_output_ignores_unrelated_json_and_garbage():
    assert parse_loudnorm_output("") == {}
    assert parse_loudnorm_output("no json at all") == {}
    assert parse_loudnorm_output('{"something": 1}') == {}
    # A stray JSON object before the measurement must not win.
    text = '{"something": 1}\n' + FFMPEG_LOUDNORM_STDERR
    assert parse_loudnorm_output(text)["input_i"] == pytest.approx(-23.45)


def test_json_blocks_handles_nesting():
    blocks = _json_blocks('log {"a": {"b": 1}, "c": "{literal}"} tail')
    assert blocks == [{"a": {"b": 1}, "c": "{literal}"}]


def test_build_loudnorm_filter_uses_measurements_when_available():
    measured = parse_loudnorm_output(FFMPEG_LOUDNORM_STDERR)
    chain = build_loudnorm_filter(measured, target_lufs=-14.0, true_peak=-1.5)

    assert chain.startswith("loudnorm=I=-14.0:TP=-1.5:LRA=11")
    assert "measured_I=-23.45" in chain
    assert "measured_TP=-3.21" in chain
    assert "measured_LRA=4.60" in chain
    assert "measured_thresh=-34.57" in chain
    assert "offset=0.55" in chain
    assert "linear=true" in chain
    assert chain.endswith("alimiter=limit=0.95")


def test_build_loudnorm_filter_without_measurements_stays_one_pass():
    chain = build_loudnorm_filter({}, target_lufs=-14.0, true_peak=-1.5)

    assert "measured_I" not in chain
    assert "linear=true" not in chain
    assert "alimiter=limit=0.95" in chain


def test_build_loudnorm_filter_can_skip_the_limiter():
    assert "alimiter" not in build_loudnorm_filter(None, limiter=False)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)
def test_measure_loudness_on_a_real_file(tmp_path: Path):
    """The regression that mattered: real ffmpeg output must parse, not come back empty."""
    measured = measure_loudness(FIXTURE, settings=_settings(tmp_path))
    assert "input_i" in measured


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)
def test_normalize_audio_records_the_two_pass_mode(tmp_path: Path):
    settings = _settings(tmp_path)
    result = normalize_audio(FIXTURE, tmp_path / "final.mp4", settings=settings)

    assert result.applied is True
    assert result.used_two_pass is True
    assert result.measured
    assert result.reason is None
    assert (tmp_path / "final.mp4").stat().st_size > 0


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)
def test_normalize_audio_can_be_disabled(tmp_path: Path):
    settings = _settings(tmp_path, audio_normalize=False)
    result = normalize_audio(FIXTURE, tmp_path / "final.mp4", settings=settings)

    assert result.applied is False
    assert result.used_two_pass is False
    assert result.reason == "audio normalization disabled"
    assert (tmp_path / "final.mp4").exists()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)
def test_normalize_audio_reuses_a_cached_output(tmp_path: Path):
    output = tmp_path / "final.mp4"
    output.write_bytes(b"cached")
    settings = _settings(tmp_path)
    result = normalize_audio(FIXTURE, output, settings=settings)
    assert output.read_bytes() == b"cached"
    assert result.reason == "reused cached final clip"


def test_normalize_audio_requires_a_source(tmp_path: Path):
    settings = _settings(tmp_path)
    with pytest.raises(FileNotFoundError):
        normalize_audio(tmp_path / "missing.mp4", tmp_path / "out.mp4", settings=settings)
