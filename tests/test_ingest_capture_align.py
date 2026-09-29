from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clippy.chat.models import ChatMessage
from clippy.extract.ffmpeg_cut import extract_window
from clippy.ingest.align import (
    TimelineAlignment,
    alignment_warning,
    audio_envelope,
    envelope_offset,
    estimate_timeline_offset,
    probe_alignment,
    verify_offset_with_clips,
)
from clippy.ingest.capture import (
    MediaMetadata,
    build_capture_argv,
    capture_vod,
    default_output_path,
    normalize_downloader,
    probe_video_metadata,
    prune_sources,
    resolve_downloader,
    ytdlp_format,
    vod_url,
)

FIXTURE = Path(__file__).resolve().parents[1] / "samples" / "sample_vod.mp4"


def test_vod_url_accepts_id_path_and_full_url():
    assert vod_url("12345") == "https://www.twitch.tv/videos/12345"
    assert vod_url("videos/12345") == "https://www.twitch.tv/videos/12345"
    assert vod_url("https://www.twitch.tv/videos/12345") == "https://www.twitch.tv/videos/12345"
    with pytest.raises(ValueError):
        vod_url("   ")


def test_normalize_downloader_accepts_dash_and_underscore():
    assert normalize_downloader("streamlink") == "streamlink"
    assert normalize_downloader("yt-dlp") == "yt_dlp"
    assert normalize_downloader("YT_DLP") == "yt_dlp"
    assert normalize_downloader(None) == "streamlink"
    with pytest.raises(ValueError):
        normalize_downloader("curl")


def test_default_output_path_slugifies_the_vod_reference():
    path = default_output_path(Path("data/source"), "https://www.twitch.tv/videos/12345?t=1")
    assert path.parent == Path("data/source")
    assert path.name == "12345_t_1.ts"


def test_build_capture_argv_streamlink():
    argv = build_capture_argv(
        downloader="streamlink",
        url="https://www.twitch.tv/videos/12345",
        output=Path("out.ts"),
        quality="720p",
    )
    assert argv[0] == "streamlink"
    assert argv[1] == "https://www.twitch.tv/videos/12345"
    assert argv[2] == "720p"
    assert argv[argv.index("-o") + 1] == "out.ts"
    assert "--force" in argv
    assert argv[argv.index("--stream-segment-threads") + 1] == "4"
    # Segment-level resilience for a long download, but fast failure on a bad VOD.
    assert argv[argv.index("--stream-segment-attempts") + 1] == "5"
    assert argv[argv.index("--stream-segment-timeout") + 1] == "30"
    assert argv[argv.index("--retry-streams") + 1] == "2"
    assert argv[argv.index("--retry-max") + 1] == "3"


def test_build_capture_argv_ytdlp_maps_quality_and_writes_chat():
    argv = build_capture_argv(
        downloader="yt_dlp",
        url="https://www.twitch.tv/videos/12345",
        output=Path("out.ts"),
        quality="720p60",
        write_chat=True,
    )
    assert argv[0] == "yt-dlp"
    assert argv[argv.index("-f") + 1] == "bv*[height<=720]+ba/b[height<=720]"
    assert "--write-chat" in argv
    assert "--no-part" in argv
    assert ytdlp_format("best") == "bv*+ba/b"
    assert ytdlp_format("worst") == "worst"
    assert ytdlp_format("") == "bv*+ba/b"


def test_build_capture_argv_rejects_unknown_downloader():
    with pytest.raises(ValueError):
        build_capture_argv(
            downloader="wget", url="u", output=Path("o.ts"), quality="best"
        )


def test_resolve_downloader_missing_names_the_install_command(monkeypatch):
    monkeypatch.setattr("clippy.ingest.capture.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        resolve_downloader("streamlink")
    assert "uv tool install streamlink" in str(excinfo.value)

    with pytest.raises(RuntimeError) as ytdlp_error:
        resolve_downloader("yt_dlp")
    assert "uv tool install yt-dlp" in str(ytdlp_error.value)


def test_capture_vod_runs_downloader_and_probes(monkeypatch, tmp_path: Path):
    output = tmp_path / "out.ts"
    seen: dict[str, object] = {}

    def fake_run(argv: list[str]) -> None:
        seen["argv"] = argv
        output.write_bytes(b"x" * 64)

    monkeypatch.setattr("clippy.ingest.capture.resolve_downloader", lambda name: "streamlink")
    monkeypatch.setattr("clippy.ingest.capture._run_downloader", fake_run)
    monkeypatch.setattr(
        "clippy.ingest.capture.probe_video_metadata",
        lambda path, *, ffprobe_path: MediaMetadata(22211.8, 1920, 1080, 60.0),
    )

    result = capture_vod(vod="12345", output=output, quality="best")

    assert result.path == output
    assert result.url == "https://www.twitch.tv/videos/12345"
    assert result.downloader == "streamlink"
    assert result.bytes == 64
    assert result.media.width == 1920
    assert result.media.fps == pytest.approx(60.0)
    assert seen["argv"][0] == "streamlink"
    assert result.to_dict()["height"] == 1080


def test_capture_vod_rejects_empty_output(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("clippy.ingest.capture.resolve_downloader", lambda name: "streamlink")
    monkeypatch.setattr("clippy.ingest.capture._run_downloader", lambda argv: None)
    with pytest.raises(RuntimeError) as excinfo:
        capture_vod(vod="12345", output=tmp_path / "missing.ts")
    assert "produced no media" in str(excinfo.value)


def test_probe_video_metadata_parses_ffprobe_json(monkeypatch, tmp_path: Path):
    media = tmp_path / "clip.ts"
    media.write_bytes(b"\x00")
    payload = {
        "streams": [{"width": 1280, "height": 720, "r_frame_rate": "60/1"}],
        "format": {"duration": "123.456"},
    }

    def fake_run(cmd, capture_output=True, text=True, check=False):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("clippy.ingest.capture.shutil.which", lambda name: "ffprobe")
    monkeypatch.setattr("clippy.ingest.capture.subprocess.run", fake_run)

    metadata = probe_video_metadata(media)
    assert metadata.width == 1280
    assert metadata.height == 720
    assert metadata.fps == pytest.approx(60.0)
    assert metadata.duration_seconds == pytest.approx(123.456)


def test_probe_video_metadata_without_ffprobe_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("clippy.ingest.capture.shutil.which", lambda name: None)
    monkeypatch.setattr("clippy.ingest.capture.Path.exists", lambda self: False)
    with pytest.raises(RuntimeError) as excinfo:
        probe_video_metadata(tmp_path / "clip.ts")
    assert "ffprobe not found" in str(excinfo.value)


@pytest.mark.skipif(not FIXTURE.exists(), reason="run scripts/make_sample_media.py first")
def test_probe_video_metadata_on_local_fixture():
    metadata = probe_video_metadata(FIXTURE)
    assert metadata.width == 640
    assert metadata.height == 360
    assert metadata.duration_seconds == pytest.approx(260.0, abs=1.0)
    assert metadata.fps is not None and metadata.fps > 0


def _fake_source(path: Path, size: int, mtime: float) -> Path:
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def test_prune_sources_deletes_oldest_first(tmp_path: Path):
    older = _fake_source(tmp_path / "older.ts", 1000, 1_000_000)
    newer = _fake_source(tmp_path / "newer.ts", 1000, 2_000_000)
    budget_gb = 1500 / (1024**3)

    removed = prune_sources(tmp_path, budget_gb=budget_gb)
    assert removed == [older]
    assert not older.exists()
    assert newer.exists()


def test_prune_sources_never_deletes_protected_files(tmp_path: Path):
    older = _fake_source(tmp_path / "older.ts", 1000, 1_000_000)
    newer = _fake_source(tmp_path / "newer.ts", 1000, 2_000_000)

    removed = prune_sources(tmp_path, budget_gb=500 / (1024**3), keep=[older])
    assert removed == [newer]
    assert older.exists()
    assert not newer.exists()


def test_prune_sources_disabled_for_zero_budget(tmp_path: Path):
    kept = _fake_source(tmp_path / "keep.ts", 1000, 1_000_000)
    assert prune_sources(tmp_path, budget_gb=0) == []
    assert kept.exists()
    assert prune_sources(tmp_path / "missing", budget_gb=1.0) == []


def test_estimate_timeline_offset_recovers_a_known_shift():
    chat_times = [100.0, 200.0, 300.0]
    media_times = [95.0, 195.0, 295.0]  # media = stream - 5
    values = [0.5, 0.5, 0.5]

    alignment = estimate_timeline_offset(
        chat_times, media_times, values, search_seconds=10.0
    )

    assert alignment.offset_seconds == pytest.approx(-5.0, abs=1.0)
    assert alignment.is_meaningful()
    assert alignment.samples > 300


def test_estimate_timeline_offset_is_zero_when_clocks_agree():
    chat_times = [100.0, 200.0, 300.0]
    alignment = estimate_timeline_offset(
        chat_times, list(chat_times), [0.4, 0.4, 0.4], search_seconds=10.0
    )
    assert alignment.offset_seconds == pytest.approx(0.0, abs=1.0)


def test_estimate_timeline_offset_respects_the_search_radius():
    chat_times = [100.0, 200.0, 300.0]
    media_times = [40.0, 140.0, 240.0]  # true shift -60, outside a 10s search
    alignment = estimate_timeline_offset(
        chat_times, media_times, [0.5, 0.5, 0.5], search_seconds=10.0
    )
    assert abs(alignment.offset_seconds) <= 10.0


def test_estimate_timeline_offset_ignores_flat_curves():
    empty = estimate_timeline_offset([], [], [])
    assert empty.offset_seconds == 0.0
    assert empty.is_meaningful() is False
    assert empty.to_dict()["score"] == 0.0


def test_estimate_timeline_offset_validates_bin_seconds():
    with pytest.raises(ValueError):
        estimate_timeline_offset([1.0], [1.0], [0.1], bin_seconds=0)


def test_alignment_warning_only_fires_beyond_tolerance():
    within = TimelineAlignment(offset_seconds=0.4, score=0.5, bin_seconds=1.0, samples=10)
    assert alignment_warning(within, tolerance_seconds=1.0) is None

    beyond = TimelineAlignment(offset_seconds=-7.0, score=0.5, bin_seconds=1.0, samples=10)
    message = alignment_warning(beyond, tolerance_seconds=1.0)
    assert message is not None
    assert "--source-offset -7.00" in message

    weak = TimelineAlignment(offset_seconds=-7.0, score=0.0, bin_seconds=1.0, samples=10)
    assert alignment_warning(weak, tolerance_seconds=1.0) is None


def test_probe_alignment_windows_the_decode_and_filters_chat(monkeypatch, tmp_path: Path):
    captured: dict[str, float] = {}

    def fake_extract(
        media_path, *, sample_rate, ffmpeg_path, start_seconds, duration_seconds
    ):
        captured["start"] = float(start_seconds)
        captured["duration"] = float(duration_seconds)
        return np.ones(sample_rate * 4, dtype=np.float32)

    monkeypatch.setattr("clippy.ingest.align.extract_mono_pcm", fake_extract)
    messages = [
        ChatMessage(ts=1.0, user="a", text="clip it"),
        ChatMessage(ts=5000.0, user="b", text="way outside the window"),
    ]

    alignment = probe_alignment(
        tmp_path / "capture.ts",
        messages,
        window_seconds=10.0,
        search_seconds=2.0,
        bin_seconds=1.0,
    )

    assert captured == {"start": 0.0, "duration": 10.0}
    assert isinstance(alignment, TimelineAlignment)
    # One chat message inside the window is not enough evidence to claim an offset.
    assert alignment.offset_seconds == 0.0
    assert alignment.is_meaningful() is False


def test_estimate_timeline_offset_needs_enough_evidence():
    alignment = estimate_timeline_offset(
        [10.0, 20.0], [10.0, 20.0], [0.4, 0.4], search_seconds=5.0
    )
    assert alignment.offset_seconds == 0.0
    assert alignment.is_meaningful() is False


def test_audio_envelope_bins_rms():
    samples = np.ones(16000 * 2, dtype=np.float32)
    envelope = audio_envelope(samples, sample_rate=16000, frame_seconds=0.5)
    assert len(envelope) == 4
    assert np.allclose(envelope, 1.0)


def test_audio_envelope_handles_a_short_buffer():
    envelope = audio_envelope(np.ones(100, dtype=np.float32), frame_seconds=0.5)
    assert len(envelope) == 1


def test_envelope_offset_recovers_a_known_lag():
    wide = np.random.default_rng(0).random(2000)
    lag, score = envelope_offset(wide[700:1300], wide)
    assert lag == 700
    assert score > 0.99


def test_envelope_offset_rejects_degenerate_inputs():
    assert envelope_offset(np.zeros(10), np.ones(100)) == (0, 0.0)
    assert envelope_offset(np.ones(100), np.ones(10)) == (0, 0.0)


NEEDS_MEDIA = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)


@NEEDS_MEDIA
def test_verify_offset_with_clips_recovers_zero_offset(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    extract_window(FIXTURE, clip, start_seconds=100.0, duration_seconds=60.0)

    verification = verify_offset_with_clips(
        media_path=FIXTURE, clips=[(clip, 100.0)]
    )

    assert verification.checked == 1
    assert verification.is_confident()
    assert verification.offset_seconds == pytest.approx(0.0, abs=0.2)


@NEEDS_MEDIA
def test_verify_offset_with_clips_detects_a_30s_shift(tmp_path: Path):
    # The clip really starts at 100s but is claimed to start at 130s: the media clock
    # is 30s behind the stream clock, i.e. offset = -30. This is the exact error class
    # that a chat-vs-audio correlation got wrong on the real capture.
    clip = tmp_path / "clip.mp4"
    extract_window(FIXTURE, clip, start_seconds=100.0, duration_seconds=60.0)

    verification = verify_offset_with_clips(
        media_path=FIXTURE, clips=[(clip, 130.0)]
    )

    assert verification.is_confident()
    assert verification.offset_seconds == pytest.approx(-30.0, abs=0.2)


@NEEDS_MEDIA
def test_verify_offset_with_clips_ignores_missing_clips(tmp_path: Path):
    verification = verify_offset_with_clips(
        media_path=FIXTURE, clips=[(tmp_path / "nope.mp4", 0.0)]
    )
    assert verification.checked == 0
    assert verification.is_confident() is False



