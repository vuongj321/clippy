"""Face detection for vertical framing and caption avoidance (opt-in `opencv` backend).

The default tracker follows *motion*, which answers "where is something happening" but cannot
tell a face from a HUD. This module answers the narrower and more useful question - "where is the
person" - using OpenCV's bundled Haar cascade, so there is no model to download and one optional
dependency (`opencv-python-headless`, installed with the project).

Everything here is deliberately defensive:

- ``cv2`` is imported *inside* the functions, so the edit package stays importable - and the test
  suite stays runnable - on a machine without OpenCV.
- Detection runs on small grayscale frames sampled at a low rate, because the answer only has to
  be good enough to bias a crop and to choose which caption band to avoid.
- A clip whose chosen face appears in fewer than `min_hit_ratio` of the sampled frames is rejected
  outright (`face_track` returns ``[]``), so a busy background or one lucky hit can never steer the
  framing. The caller then falls back to motion.

Detections are grouped into spatial clusters across the whole clip, and the cluster that behaves
like a webcam - one face, seen consistently, at a stable spot - is chosen. A per-frame greedy pick
lets a single oversized or one-frame detection hijack the crop; the vote over the clip is what
stops a game character, an on-screen poster or a second person steering the framing.

Frames where the chosen face is not detected carry its previous position forward, so downstream
segment logic sees a dense, calm track rather than a sparse, jumpy one.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from clippy.edit.track import DEFAULT_SAMPLE_FPS, TrackPoint

logger = logging.getLogger(__name__)

DETECT_WIDTH = 480
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE_RATIO = 0.06
MIN_HIT_RATIO = 0.2

# Clustering constants. A detection joins a cluster when its centre is within `CLUSTER_MAX_JUMP`
# of that cluster's centre (a face cannot teleport between samples), and when its area is within
# `CLUSTER_SIZE_RATIO` of the cluster's median - so a much bigger box (poster, cutscene, thumbnail)
# starts its own cluster and has to earn the vote instead of merging into the real face.
CLUSTER_MAX_JUMP = 0.15
CLUSTER_SIZE_RATIO = 2.5

FaceBox = tuple[float, float, float, float]  # x, y, w, h as fractions of the frame


def _area(box: FaceBox) -> float:
    return box[2] * box[3]


def _center(box: FaceBox) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class _FaceCluster:
    """One candidate face: the detections that plausibly belong to it, by sampled-frame index."""

    by_frame: dict[int, FaceBox] = field(default_factory=dict)
    # The most recently added box. Association compares against this rather than the median, so a
    # face that drifts slowly keeps chaining into the same cluster instead of fragmenting.
    last: FaceBox | None = field(default=None, repr=False)

    def add(self, frame_index: int, box: FaceBox) -> None:
        self.by_frame[frame_index] = box
        self.last = box

    @property
    def center(self) -> tuple[float, float]:
        xs = sorted(_center(box)[0] for box in self.by_frame.values())
        ys = sorted(_center(box)[1] for box in self.by_frame.values())
        return (_median(xs), _median(ys))

    @property
    def median_area(self) -> float:
        return _median([_area(box) for box in self.by_frame.values()])

    def persistence(self, total_frames: int) -> float:
        return len(self.by_frame) / max(1, total_frames)


def available() -> bool:
    """Is OpenCV importable? The only reason this backend can be missing."""
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def _cascade() -> Any:
    """Load the bundled frontal-face cascade once (it ships with the wheel)."""
    import cv2
    import cv2.data

    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.exists():
        raise RuntimeError(
            "OpenCV is installed but ships no face cascade "
            f"({path}); OpenCV 5 removed the bundled cascades, so install "
            "opencv-python-headless<5."
        )
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError(f"could not load the face cascade: {path}")
    return cascade


def _require_ffmpeg(ffmpeg_path: str) -> str:
    ffmpeg = shutil.which(ffmpeg_path) or (
        ffmpeg_path if Path(ffmpeg_path).exists() else None
    )
    if not ffmpeg:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    return ffmpeg


def _source_size(media_path: Path, *, ffprobe_path: str = "ffprobe") -> tuple[int, int]:
    """Source pixel size, so the decode can keep the aspect ratio without a second guess."""
    ffprobe = shutil.which(ffprobe_path) or (
        ffprobe_path if Path(ffprobe_path).exists() else None
    )
    if not ffprobe:
        raise RuntimeError(
            f"ffprobe not found ({ffprobe_path!r}). Install FFmpeg and ensure it is on PATH."
        )
    proc = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0",
            str(media_path),
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ffprobe failed on "
            f"{media_path}: {proc.stderr.decode('utf-8', errors='replace')}"
        )
    fields = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()[0].split(",")
    return int(fields[0]), int(fields[1])


def even(value: int) -> int:
    """Round down to an even number: h264 and the rawvideo path both prefer even dimensions."""
    return max(2, int(value) - (int(value) % 2))


def sample_gray_frames(
    media_path: Path,
    *,
    duration_seconds: float,
    start_seconds: float = 0.0,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    width: int = DETECT_WIDTH,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
) -> np.ndarray:
    """
    Decode small grayscale frames for detection, shaped ``(frames, height, width)``.

    The cascade needs far more pixels than the motion profile does, so this is a separate decode
    from `track.sample_motion_profiles`, at a resolution a face is still recognisable in. The
    output keeps the source aspect ratio, and only the sampled frames are held in memory (a 45 s
    clip at 6 fps is ~35 MB of uint8).
    """
    ffmpeg = _require_ffmpeg(ffmpeg_path)
    source_width, source_height = _source_size(media_path, ffprobe_path=ffprobe_path)
    frame_width = even(max(64, int(width)))
    frame_height = even(round(frame_width * source_height / max(1, source_width)))

    cmd = [ffmpeg, "-v", "error"]
    if start_seconds:
        cmd += ["-ss", f"{max(0.0, float(start_seconds)):.3f}"]
    cmd += [
        "-i",
        str(media_path),
        "-t",
        f"{max(0.1, float(duration_seconds)):.3f}",
        "-vf",
        f"fps={sample_fps},scale={frame_width}:{frame_height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    frame_bytes = frame_width * frame_height
    frame_count = len(proc.stdout) // frame_bytes
    if frame_count == 0:
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg face-frame decode failed: "
                f"{proc.stderr.decode('utf-8', errors='replace')}"
            )
        return np.zeros((0, frame_height, frame_width), dtype=np.uint8)
    if proc.returncode != 0:
        # Twitch VODs carry damaged packets, and ffmpeg reports those as a non-zero exit even
        # though it decoded usable frames. Frames on disk beat a clean exit code.
        logger.warning(
            "ffmpeg reported %s while sampling face frames from %s; using the %d frames it "
            "produced",
            proc.returncode,
            media_path.name,
            frame_count,
        )
    buffer = np.frombuffer(proc.stdout[: frame_count * frame_bytes], dtype=np.uint8)
    return buffer.reshape(frame_count, frame_height, frame_width)


def detect_faces(
    frame: np.ndarray,
    *,
    scale_factor: float = SCALE_FACTOR,
    min_neighbors: int = MIN_NEIGHBORS,
    min_size_ratio: float = MIN_SIZE_RATIO,
) -> list[FaceBox]:
    """
    Frontal faces in one grayscale frame, as fractions of the frame.

    Returns an empty list when nothing plausible is found; the caller decides what that means.
    """
    if frame.size == 0:
        return []
    height, width = frame.shape[:2]
    min_side = max(8, int(round(min(width, height) * max(0.01, min_size_ratio))))
    boxes = _cascade().detectMultiScale(
        frame,
        scaleFactor=max(1.01, float(scale_factor)),
        minNeighbors=max(1, int(min_neighbors)),
        minSize=(min_side, min_side),
    )
    return [(x / width, y / height, w / width, h / height) for x, y, w, h in boxes]


def _cluster_faces(
    boxes_per_frame: Sequence[Sequence[FaceBox]],
    *,
    max_jump: float = CLUSTER_MAX_JUMP,
    size_ratio: float = CLUSTER_SIZE_RATIO,
) -> list[_FaceCluster]:
    """
    Group per-frame detections into candidate faces (pure).

    Boxes are walked largest-first so a bigger face claims its cluster before a smaller one nearby
    can. A box joins the nearest cluster that is close enough to that cluster's **most recent** box
    (centre within `max_jump`) and similar enough in size (`size_ratio`), otherwise it starts a new
    cluster - so a game character, an on-screen image or a second person becomes a separate
    candidate rather than polluting the webcam. Comparing against the most recent box (not the
    cluster's median) is what keeps a slowly drifting face as one cluster instead of splitting it
    into several half-persistent fragments. Ties are broken deterministically by frame order, so the
    same clip always clusters the same way (which keeps the composition cache stable).
    """
    clusters: list[_FaceCluster] = []
    for frame_index, boxes in enumerate(boxes_per_frame):
        for box in sorted(boxes, key=_area, reverse=True):
            cx, cy = _center(box)
            area = _area(box)
            best: _FaceCluster | None = None
            best_distance = float("inf")
            for cluster in clusters:
                if cluster.last is None:
                    continue
                # Size is compared against the cluster's most recent box, so a face whose detected
                # box jitters or drifts in size stays connected, while a sudden poster-sized box is
                # still refused.
                last_area = _area(cluster.last)
                if last_area > 0 and not (
                    last_area / size_ratio <= area <= last_area * size_ratio
                ):
                    continue
                last_x, last_y = _center(cluster.last)
                distance = math.hypot(cx - last_x, cy - last_y)
                if distance <= max_jump and distance < best_distance:
                    best, best_distance = cluster, distance
            if best is None:
                best = _FaceCluster()
                clusters.append(best)
            # Boxes are visited largest-first, so when two in one frame claim the same cluster the
            # bigger one keeps it; the smaller is dropped rather than overwriting it.
            if frame_index not in best.by_frame:
                best.add(frame_index, box)
    return clusters


def _track_from_cluster(
    cluster: _FaceCluster,
    *,
    total_frames: int,
    sample_fps: float,
) -> list[TrackPoint]:
    """
    Replay one cluster across every sampled frame, carrying its last position forward.

    Only the chosen cluster's detections move the track, so the frames it missed - and any other
    face seen in those frames - cannot drag it. Before its first detection the track sits at the
    frame centre with an unknown box, exactly as the motion fallback would.
    """
    step = 1.0 / max(sample_fps, 1e-6)
    points: list[TrackPoint] = []
    last: FaceBox | None = None
    for index in range(total_frames):
        box = cluster.by_frame.get(index)
        if box is not None:
            last = box
        if last is None:
            centre_x, centre_y, width, height = 0.5, 0.5, 0.0, 0.0
        else:
            centre_x, centre_y = _center(last)
            width, height = last[2], last[3]
        points.append(
            TrackPoint(
                t=(index + 0.5) * step,
                x=min(max(centre_x, 0.0), 1.0),
                y=min(max(centre_y, 0.0), 1.0),
                width=width,
                height=height,
            )
        )
    return points


def face_track(
    media_path: Path,
    *,
    duration_seconds: float,
    start_seconds: float = 0.0,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    width: int = DETECT_WIDTH,
    min_size_ratio: float = MIN_SIZE_RATIO,
    min_hit_ratio: float = MIN_HIT_RATIO,
    min_neighbors: int = MIN_NEIGHBORS,
    scale_factor: float = SCALE_FACTOR,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
) -> list[TrackPoint]:
    """
    Follow the clip's most webcam-like face, or return ``[]`` when there is no usable face.

    Every sampled frame is detected once, then the detections are clustered across the whole clip
    and the most persistent, size-consistent face wins. One point per sampled frame: the winning
    cluster's centre when it has a detection, otherwise its previous position carried forward (and
    the frame centre before its first detection). The box size is carried too, so the caption stage
    can test the real face rectangle against the caption band instead of assuming a subject height.

    A clip whose best face appears in fewer than `min_hit_ratio` of frames is rejected, so a busy
    background - or a face that merely flashes past - cannot steer the framing; the caller then
    falls back to motion.
    """
    frames = sample_gray_frames(
        media_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
        width=width,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
    )
    total = int(frames.shape[0])
    if total == 0:
        return []

    boxes_per_frame = [
        detect_faces(
            frame,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size_ratio=min_size_ratio,
        )
        for frame in frames
    ]
    clusters = _cluster_faces(boxes_per_frame)
    if not clusters:
        logger.info(
            "No face detected anywhere in %s; the caller keeps motion tracking",
            media_path.name,
        )
        return []

    ranked = sorted(
        clusters, key=lambda cluster: (cluster.persistence(total), cluster.median_area), reverse=True
    )
    winner = ranked[0]
    persistence = winner.persistence(total)
    threshold = max(0.0, float(min_hit_ratio))
    if persistence < threshold:
        logger.info(
            "Face track rejected for %s: best face seen in %.0f%% of frames (< %.0f%%)",
            media_path.name,
            persistence * 100,
            threshold * 100,
        )
        return []

    runner_up = ranked[1].persistence(total) if len(ranked) > 1 else 0.0
    logger.info(
        "Face track for %s: chose 1 of %d candidate face(s), seen in %d/%d frames (%.0f%%), "
        "next best %.0f%%, centre (%.2f, %.2f)",
        media_path.name,
        len(clusters),
        len(winner.by_frame),
        total,
        persistence * 100,
        runner_up * 100,
        winner.center[0],
        winner.center[1],
    )
    return _track_from_cluster(winner, total_frames=total, sample_fps=sample_fps)


