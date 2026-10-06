from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clippy.extract.ffmpeg_cut import _plan_window, extract_window

FIXTURE = Path(__file__).resolve().parents[1] / "samples" / "sample_vod.mp4"
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4 (run scripts/make_sample_media.py)",
)


def _stream_starts(path: Path) -> dict[str, float]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout or "{}")
    return {
        str(stream["codec_type"]): float(stream["start_time"])
        for stream in payload.get("streams", [])
    }


def _frame_count(path: Path) -> int:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def test_extract_window_starts_every_stream_at_zero(tmp_path: Path):
    """
    Input seeking lands on the demuxer's sync point, which leaves the *video* stream's timestamps
    offset by the distance to the requested start - 0.8167s, or 49 frames at 60fps, on candidate
    28 - while the audio starts at 0. `-t` is enforced on that labelled timeline, so the offset
    also spent the last 49 video frames of the cut. The reset keeps `base.mp4` one aligned
    timeline for every later stage.
    """
    output = tmp_path / "clip.mp4"

    extract_window(FIXTURE, output, start_seconds=103.0, duration_seconds=4.0)

    assert _stream_starts(output) == {"video": 0.0, "audio": 0.0}


def test_extract_window_keeps_the_whole_requested_window(tmp_path: Path):
    """A cut holds `duration x fps` frames, so `-t` is not spent covering a seek offset."""
    output = tmp_path / "clip.mp4"

    extract_window(FIXTURE, output, start_seconds=103.0, duration_seconds=4.0)

    assert _frame_count(output) == 100  # 4.0s at the fixture's 25fps


def test_extract_window_asks_ffmpeg_for_the_aligned_window(monkeypatch, tmp_path: Path):
    """
    The cut has to seek early and then trim both streams to the same window: a video stream can
    only restart on a keyframe, so seeking straight to the start would begin the picture at the
    next one - ahead of the sound. The fixture's own tracks already start at 0, so this asserts the
    command rather than the media, which a later edit could quietly drop.
    """
    recorded: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("clippy.extract.ffmpeg_cut.subprocess.run", fake_run)

    result = extract_window(
        FIXTURE, tmp_path / "clip.mp4", start_seconds=100.0, duration_seconds=5.0, preroll_seconds=3.0
    )

    cmd = recorded[-1]
    assert cmd[cmd.index("-ss") + 1] == "97.000"  # start - preroll
    assert cmd.index("-ss") < cmd.index("-i")  # the decode has to start early
    assert cmd[cmd.index("-vf") + 1] == "trim=start=3.000:end=8.000,setpts=PTS-STARTPTS"
    assert cmd[cmd.index("-af") + 1] == "atrim=start=3.000:end=8.000,asetpts=PTS-STARTPTS"
    # A filter placed after the output path would never reach the encoder.
    assert cmd.index("-vf") < cmd.index(str(tmp_path / "clip.mp4"))
    assert result.head_seconds == 0.0


def test_plan_window_buys_a_longer_preroll_before_shifting_the_clip(monkeypatch, tmp_path: Path):
    """
    The probe says where the video really begins. A first answer that lands after the requested
    start has to buy a longer preroll rather than shift the clip, because the frames are still in
    the source - just behind a keyframe.
    """
    answers = iter([4.0, 0.0])
    monkeypatch.setattr(
        "clippy.extract.ffmpeg_cut.probe_video_head_seconds", lambda *a, **k: next(answers)
    )

    window, seek_at, head = _plan_window(FIXTURE, start=100.0, preroll=3.0, ffmpeg_path="ffmpeg")

    assert (window, seek_at, head) == (6.0, 94.0, 0.0)


def test_plan_window_reports_a_head_it_cannot_reach(monkeypatch):
    """When no preroll reaches the start, the clip shifts so the picture keeps up with the sound."""
    monkeypatch.setattr(
        "clippy.extract.ffmpeg_cut.probe_video_head_seconds", lambda *a, **k: 20.0
    )

    window, seek_at, head = _plan_window(FIXTURE, start=100.0, preroll=3.0, ffmpeg_path="ffmpeg")

    # Attempts of 3, 6 and 12 seconds: the last one leaves 8 seconds of the window unreachable.
    assert seek_at == 88.0
    assert head == pytest.approx(8.0)
    assert window == pytest.approx(20.0)


def test_plan_window_keeps_the_requested_window_when_the_probe_cannot_tell(monkeypatch):
    monkeypatch.setattr(
        "clippy.extract.ffmpeg_cut.probe_video_head_seconds", lambda *a, **k: None
    )

    window, seek_at, head = _plan_window(FIXTURE, start=100.0, preroll=3.0, ffmpeg_path="ffmpeg")

    assert (window, seek_at, head) == (3.0, 97.0, 0.0)
