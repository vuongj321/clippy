from __future__ import annotations

import builtins
import shutil
from pathlib import Path

import numpy as np
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


def test_cluster_faces_keeps_the_larger_box_when_two_claim_one_frame():
    """Two boxes in one frame that fall in the same cluster: the bigger one owns the frame."""
    big = (0.10, 0.50, 0.14, 0.16)
    small = (0.12, 0.52, 0.11, 0.13)  # similar enough in size to join, but smaller
    clusters = faces._cluster_faces([[big, small]])

    assert len(clusters) == 1
    assert clusters[0].by_frame[0] == big


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


def _fake_frames(count: int) -> np.ndarray:
    """Blank sampled frames: the detections are faked, so the pixels carry no information."""
    return np.zeros((count, 36, 64), dtype=np.uint8)


def _detector_per_frame(boxes_by_index):
    """A `detect_faces` stand-in that returns a scripted list of boxes per call, in order."""
    calls = {"index": 0}

    def fake_detect(frame, **kwargs):  # noqa: ARG001 - the image is irrelevant here
        index = calls["index"]
        calls["index"] += 1
        return list(boxes_by_index(index))

    return fake_detect


def test_cluster_faces_merges_the_same_face_across_frames():
    box = (0.1, 0.5, 0.1, 0.12)
    clusters = faces._cluster_faces([[box] for _ in range(30)])

    assert len(clusters) == 1
    assert clusters[0].persistence(30) == pytest.approx(1.0)


def test_cluster_faces_splits_two_positions_that_never_overlap():
    left = (0.1, 0.5, 0.1, 0.12)
    right = (0.8, 0.5, 0.1, 0.12)
    clusters = faces._cluster_faces([[left if i % 2 == 0 else right] for i in range(40)])

    assert len(clusters) == 2
    assert sorted(cluster.persistence(40) for cluster in clusters) == [
        pytest.approx(0.5),
        pytest.approx(0.5),
    ]


def test_cluster_faces_keeps_a_much_larger_box_separate():
    """A face-sized box and a poster-sized one at the same spot are two candidates, not one."""
    small = (0.1, 0.5, 0.1, 0.12)
    huge = (0.12, 0.5, 0.4, 0.45)
    clusters = faces._cluster_faces([[small], [small], [huge]])

    assert len(clusters) == 2


def test_cluster_faces_keeps_a_drifting_face_as_one_cluster():
    """A face that crosses the frame in small steps is one candidate, not several fragments."""
    boxes = [[(0.2 + index * 0.01, 0.5, 0.1, 0.12)] for index in range(30)]
    clusters = faces._cluster_faces(boxes)

    assert len(clusters) == 1
    assert clusters[0].persistence(30) == pytest.approx(1.0)


def test_cluster_faces_keeps_a_size_jittering_face_as_one_cluster():
    """Detected box size wobbles frame to frame; that must not split the face in two."""
    sizes = [(0.10, 0.12), (0.14, 0.17)] * 15  # ~2x area swings, within the ratio
    clusters = faces._cluster_faces([[(0.4, 0.5, w, h)] for w, h in sizes])

    assert len(clusters) == 1


def test_track_from_cluster_carries_its_own_position_forward():
    cluster = faces._FaceCluster()
    cluster.add(0, (0.1, 0.5, 0.1, 0.12))
    cluster.add(5, (0.2, 0.5, 0.1, 0.12))

    points = faces._track_from_cluster(cluster, total_frames=8, sample_fps=6.0)

    assert len(points) == 8
    assert points[0].x == pytest.approx(0.15)
    assert points[4].x == pytest.approx(0.15)  # carried forward, not the frame centre
    assert points[5].x == pytest.approx(0.25)


def test_track_from_cluster_is_centred_before_its_first_detection():
    cluster = faces._FaceCluster()
    cluster.add(3, (0.6, 0.4, 0.1, 0.12))

    points = faces._track_from_cluster(cluster, total_frames=5, sample_fps=6.0)

    assert points[0].x == pytest.approx(0.5) and points[0].width == 0.0
    assert points[3].x == pytest.approx(0.65)


def test_face_track_picks_the_persistent_face_over_a_larger_transient_one(monkeypatch):
    """The clip-wide vote is what stops one oversized detection hijacking the crop."""
    monkeypatch.setattr(faces, "sample_gray_frames", lambda *a, **k: _fake_frames(40))
    small = (0.1, 0.5, 0.1, 0.12)  # the facecam, seen in every frame
    huge = (0.5, 0.4, 0.45, 0.5)  # a bigger face in the first three frames only
    monkeypatch.setattr(
        faces, "detect_faces", _detector_per_frame(lambda i: [small, huge] if i < 3 else [small])
    )

    points = faces.face_track(Path("whatever.mp4"), duration_seconds=7.0, sample_fps=6.0)

    assert len(points) == 40
    assert all(point.x == pytest.approx(0.15) for point in points)
    assert all(point.width == pytest.approx(0.1) for point in points)


def test_face_track_stays_on_one_face_when_two_alternate(monkeypatch):
    """Two similar faces must not average into a crop that sits between them."""
    monkeypatch.setattr(faces, "sample_gray_frames", lambda *a, **k: _fake_frames(40))
    big = (0.1, 0.5, 0.12, 0.14)  # marginally larger, so the tie is decided, not random
    small = (0.8, 0.5, 0.1, 0.12)
    monkeypatch.setattr(
        faces, "detect_faces", _detector_per_frame(lambda i: [big] if i % 2 == 0 else [small])
    )

    points = faces.face_track(Path("whatever.mp4"), duration_seconds=7.0, sample_fps=6.0)

    assert all(point.x == pytest.approx(0.16) for point in points)  # the bigger face's centre


def test_face_track_rejects_a_face_that_only_flashes_by(monkeypatch):
    monkeypatch.setattr(faces, "sample_gray_frames", lambda *a, **k: _fake_frames(100))
    box = (0.1, 0.5, 0.1, 0.12)
    monkeypatch.setattr(
        faces, "detect_faces", _detector_per_frame(lambda i: [box] if i < 10 else [])
    )

    points = faces.face_track(Path("whatever.mp4"), duration_seconds=17.0, sample_fps=6.0)

    # Seen in 10% of frames, below the 20% floor: no usable face, so motion takes over.
    assert points == []


def test_face_track_prefers_the_larger_face_over_a_smaller_more_persistent_one(monkeypatch):
    """
    The webcam outranks a smaller face inside on-screen content even when it is seen less often.

    This is the gaming case that regressed in practice: a big frontal face inside a screenshot or
    post thumbnail is detected more often than the (larger) real webcam, so a persistence-first
    vote picked the on-screen face and cropped the facecam panel onto it.
    """
    monkeypatch.setattr(faces, "sample_gray_frames", lambda *a, **k: _fake_frames(120))
    webcam = (0.78, 0.20, 0.10, 0.17)  # area 0.017, seen in a third of frames
    thumbnail = (0.36, 0.25, 0.054, 0.096)  # area 0.005, seen in half the frames
    monkeypatch.setattr(
        faces,
        "detect_faces",
        _detector_per_frame(
            lambda i: ([thumbnail] if i % 2 == 0 else []) + ([webcam] if i % 3 == 0 else [])
        ),
    )

    points = faces.face_track(Path("whatever.mp4"), duration_seconds=20.0, sample_fps=6.0)

    assert len(points) == 120
    assert all(point.x == pytest.approx(0.83) for point in points)
    assert all(point.width == pytest.approx(0.10) for point in points)


def test_face_track_ignores_a_face_seen_in_too_few_frames(monkeypatch):
    """A large face that only shows up in a couple of samples cannot win the webcam vote."""
    monkeypatch.setattr(faces, "sample_gray_frames", lambda *a, **k: _fake_frames(60))
    steady = (0.1, 0.5, 0.1, 0.12)  # the webcam, in every frame
    flash = (0.6, 0.4, 0.30, 0.34)  # bigger, but only three frames
    monkeypatch.setattr(
        faces,
        "detect_faces",
        _detector_per_frame(lambda i: [steady, flash] if i < 3 else [steady]),
    )

    points = faces.face_track(Path("whatever.mp4"), duration_seconds=10.0, sample_fps=6.0)

    assert len(points) == 60
    assert all(point.x == pytest.approx(0.15) for point in points)


