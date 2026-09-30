from __future__ import annotations

import builtins
import shutil
from pathlib import Path

import pytest

from clippy.config import Settings
from clippy.edit import faces
from clippy.edit.track import TrackPoint, track_subject

FIXTURE = Path(__file__).resolve().parents[1] / "samples" / "sample_vod.mp4"

HAVE_OPENCV = faces.available()
needs_opencv = pytest.mark.skipif(not HAVE_OPENCV, reason="needs opencv-python-headless")
needs_media = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {"data_dir": tmp_path}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_even_rounds_down_to_even_dimensions():
    assert faces.even(481) == 480
    assert faces.even(480) == 480
    # Never zero: a zero-sized decode would be a silent no-op.
    assert faces.even(1) == 2


def test_pick_face_prefers_the_largest():
    big = (0.6, 0.1, 0.2, 0.2)
    small = (0.05, 0.05, 0.05, 0.05)
    assert faces.pick_face([small, big]) == big
    assert faces.pick_face([]) is None


def test_pick_face_prefers_the_previous_face_when_sizes_are_similar():
    """Two similar faces must not make the crop flicker between them."""
    left = (0.1, 0.2, 0.2, 0.2)
    right = (0.7, 0.2, 0.21, 0.21)  # marginally larger, but far from the previous face
    assert faces.pick_face([left, right], previous=left) == left
    assert faces.pick_face([left, right]) == right


def test_available_reports_a_missing_opencv(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2 here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert faces.available() is False


@needs_opencv
def test_the_cascade_finds_nothing_in_blank_or_noisy_frames():
    """The cheapest guard against a detector that hallucinates faces."""
    import numpy as np

    blank = np.zeros((270, 480), dtype=np.uint8)
    assert faces.detect_faces(blank) == []

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (270, 480), dtype=np.uint8)
    assert faces.detect_faces(noise) == []

    assert faces.detect_faces(np.zeros((0, 0), dtype=np.uint8)) == []


@needs_opencv
@needs_media
def test_face_track_rejects_a_clip_with_no_face(tmp_path: Path):
    """
    The fixture has no faces, so the whole decode/detect/gate path must come back empty.

    This is the conservative outcome the layout depends on: no face means the caller keeps
    motion tracking.
    """
    points = faces.face_track(FIXTURE, duration_seconds=4.0)
    assert points == []


@needs_opencv
@needs_media
def test_track_subject_falls_back_to_motion_when_no_face_is_found(tmp_path: Path):
    settings = _settings(tmp_path, layout_track_backend="opencv")
    points = track_subject(FIXTURE, duration_seconds=4.0, settings=settings)

    assert points  # motion tracking produced something
    assert all(point.height == 0.0 for point in points)  # and it is not face evidence


@needs_media
def test_track_subject_without_opencv_uses_motion(monkeypatch, tmp_path: Path):
    """A requested face backend must degrade to motion when the package is missing."""
    monkeypatch.setattr(faces, "available", lambda: False)
    monkeypatch.setattr(
        "clippy.edit.faces.face_track",
        lambda *a, **k: pytest.fail("the face backend must not run without opencv"),
    )
    settings = _settings(tmp_path, layout_track_backend="opencv")

    points = track_subject(FIXTURE, duration_seconds=1.0, settings=settings)
    assert all(point.height == 0.0 for point in points)  # motion evidence carries no box


def test_track_subject_returns_face_points_when_the_detector_finds_a_face(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(faces, "available", lambda: True)
    canned = [
        TrackPoint(t=0.5, x=0.75, y=0.18, width=0.1, height=0.12),
        TrackPoint(t=1.0, x=0.75, y=0.19, width=0.1, height=0.12),
    ]
    monkeypatch.setattr(faces, "face_track", lambda *a, **k: canned)
    settings = _settings(tmp_path, layout_track_backend="opencv")

    points = track_subject(Path("whatever.mp4"), duration_seconds=1.0, settings=settings)
    assert points == canned


@needs_media
def test_track_subject_survives_a_detector_failure(monkeypatch, tmp_path: Path):
    """A detector that throws must degrade to motion, never fail the render."""
    monkeypatch.setattr(faces, "available", lambda: True)

    def boom(*args, **kwargs):
        raise RuntimeError("cascade exploded")

    monkeypatch.setattr(faces, "face_track", boom)
    settings = _settings(tmp_path, layout_track_backend="opencv")

    points = track_subject(FIXTURE, duration_seconds=1.0, settings=settings)
    assert all(point.height == 0.0 for point in points)  # fell back to motion

