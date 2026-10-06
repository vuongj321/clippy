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
like a webcam - a face seen consistently, and large enough to be a real camera rather than a face
embedded in on-screen content - is chosen. A per-frame greedy pick lets a single oversized or
one-frame detection hijack the crop; the vote over the clip is what stops a game character, an
on-screen poster or a second person steering the framing. A candidate must clear a minimum number
of detections, and the winner is the *time-weighted face area* (persistence x size), not raw
persistence, so a small but ever-present face inside a screenshot, a post thumbnail or an avatar
cannot outvote the larger webcam.

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
# A face must be seen in at least this many sampled frames before it can be the webcam. One or two
# lucky detections (a stray giant box, a face that only flashes past) are not evidence of a webcam;
# a handful of samples survives a couple of misses without letting a one-off detection win.
MIN_FACE_HITS = 4

# Clustering constants. A detection joins a cluster when its centre is within `CLUSTER_MAX_JUMP`
# of that cluster's most recent box (a face cannot teleport between samples), and when its area is
# within `CLUSTER_SIZE_RATIO` of that same box - so a much bigger box (poster, cutscene, thumbnail)
# starts its own cluster and has to earn the vote instead of merging into the real face. Matching
# the most recent box rather than the cluster's median is what keeps a slowly drifting face whole.
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
    min_hits: int = MIN_FACE_HITS,
    min_neighbors: int = MIN_NEIGHBORS,
    scale_factor: float = SCALE_FACTOR,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
) -> list[TrackPoint]:
    """
    Follow the clip's most webcam-like face, or return ``[]`` when there is no usable face.

    Every sampled frame is detected once, then the detections are clustered across the whole clip.
    Candidates seen in fewer than `min_hits` frames are dropped, and the survivor with the most
    *time-weighted face area* (persistence x median area) wins - so a larger, reasonably-persistent
    webcam beats a smaller face that merely appears more often inside on-screen content. One point
    per sampled frame: the winning cluster's centre when it has a detection, otherwise its previous
    position carried forward (and the frame centre before its first detection). The box size is
    carried too, so the caption stage can test the real face rectangle against the caption band
    instead of assuming a subject height.

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

    # Require a minimum number of detections before a cluster can be the webcam. This throws away
    # one-off giant boxes and faces that only flash past, so the ranking below chooses between
    # faces that are actually there rather than between lucky hits.
    floor = max(1, int(min_hits))
    eligible = [cluster for cluster in clusters if len(cluster.by_frame) >= floor]
    if not eligible:
        logger.info(
            "Face track for %s: no face seen in >= %d sampled frames; keeping motion tracking",
            media_path.name,
            floor,
        )
        return []

    # Rank by time-weighted face area (persistence x size): a bigger face seen often is the
    # webcam, while a small face that rides along inside a screenshot/post/avatar loses even when
    # it is detected more often. Persistence and area stay as tie-breakers for determinism.
    ranked = sorted(
        eligible,
        key=lambda cluster: (
            cluster.persistence(total) * cluster.median_area,
            cluster.persistence(total),
            cluster.median_area,
        ),
        reverse=True,
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
        "Face track for %s: chose 1 of %d candidate face(s) (%d eligible), seen in %d/%d frames "
        "(%.0f%%), median area %.4f, next best %.0f%%, centre (%.2f, %.2f)",
        media_path.name,
        len(clusters),
        len(eligible),
        len(winner.by_frame),
        total,
        persistence * 100,
        winner.median_area,
        runner_up * 100,
        winner.center[0],
        winner.center[1],
    )
    return _track_from_cluster(winner, total_frames=total, sample_fps=sample_fps)


def _median_face_box(track: Sequence[TrackPoint]) -> FaceBox | None:
    """The steady face rectangle - the guard a snapped tile still has to frame."""
    boxes = [point for point in track if point.width > 0 and point.height > 0]
    if not boxes:
        return None
    # `TrackPoint.x/y` is the face's *centre* and `width/height` is its size, so the guard is the
    # rectangle around that centre rather than a corner-anchored box.
    face_w = _median([point.width for point in boxes])
    face_h = _median([point.height for point in boxes])
    return (
        _median([point.x for point in boxes]) - face_w / 2.0,
        _median([point.y for point in boxes]) - face_h / 2.0,
        face_w,
        face_h,
    )


def _contains(inner: FaceBox | None, outer: FaceBox) -> bool:
    if inner is None:
        return True
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    slack = 1e-6
    return (
        ox <= ix + slack
        and oy <= iy + slack
        and ox + ow >= ix + iw - slack
        and oy + oh >= iy + ih - slack
    )


def _strongest_seam(
    scores: np.ndarray,
    rows: range,
    *,
    floor: float,
    ratio: float,
) -> tuple[int, float] | None:
    """
    The row with the strongest seam to the row below, when it is really a seam.

    A painted overlay edge is a *step* in the picture, so it has to clear both an absolute floor and
    a multiple of what the rest of the band looks like. Requiring the ratio as well is what keeps a
    merely busy stretch of the frame from being mistaken for the webcam's border.
    """
    band = [(row, float(scores[row])) for row in rows if 0 <= row < len(scores)]
    if not band:
        return None
    typical = float(np.median([score for _row, score in band]))
    row, strength = max(band, key=lambda item: item[1])
    if strength < floor or strength < ratio * max(typical, 1e-6):
        return None
    return row, strength


def snap_facecam_box_to_edge(
    frames: np.ndarray,
    box: FaceBox,
    *,
    track: Sequence[TrackPoint] | None = None,
    max_shift_ratio: float = 0.15,
    min_edge_strength: float = 8.0,
    min_edge_ratio: float = 2.5,
    min_height_ratio: float = 0.25,
) -> tuple[FaceBox, float]:
    """
    Move a derived facecam tile's horizontal edges onto the webcam overlay's own seams.

    A tile derived from the *face* is not the rectangle the streamer drew, so it can run past the
    webcam and put a strip of the game at the bottom of the panel - the panel is flush with the
    frame, so whatever the tile overshoots by ends up on screen. The overlay is painted over the
    frame, which makes its boundary a hard seam: for every row in a band around the tile's top and
    bottom, measure the mean absolute difference to the row below across the tile's width, take the
    median over the sampled frames, and snap an edge onto the strongest seam.

    Returns the (possibly moved) box and the seam strength that justified the move (0.0 when the box
    is untouched). The box is left exactly as it was when there is no frame to read, no seam that
    clearly beats the local noise, a move longer than `max_shift_ratio` of the tile, or a move that
    would stop the tile framing the tracked face.
    """
    if frames.size == 0:
        return box, 0.0
    _total, height, width = frames.shape
    ox, oy, ow, oh = box
    if width < 16 or height < 16 or ow <= 0 or oh <= 0:
        return box, 0.0

    x0 = max(0, min(int(round(ox * width)), width - 1))
    x1 = min(width, max(x0 + 1, int(round((ox + ow) * width))))
    y0 = max(0, min(int(round(oy * height)), height - 1))
    y1 = min(height, max(y0 + 1, int(round((oy + oh) * height))))
    tile_rows = y1 - y0
    if x1 - x0 < 8 or tile_rows < 8:
        return box, 0.0

    # Seam strength per row, median over frames so a single odd frame cannot invent an edge.
    deltas = np.abs(np.diff(frames.astype(np.int16), axis=1))[:, :, x0:x1]
    scores = np.median(deltas.mean(axis=2), axis=0)

    band = max(3, int(round(0.12 * tile_rows)))
    limit = max(1, int(round(max_shift_ratio * tile_rows)))
    min_rows = max(8, int(round(min_height_ratio * tile_rows)))
    protect = _median_face_box(track) if track else None

    top_seam = _strongest_seam(
        scores,
        range(max(0, y0 - band), min(len(scores) - 1, y0 + band) + 1),
        floor=min_edge_strength,
        ratio=min_edge_ratio,
    )
    bottom_seam = _strongest_seam(
        scores,
        range(max(0, y1 - 1 - band), min(len(scores) - 1, y1 - 1 + band) + 1),
        floor=min_edge_strength,
        ratio=min_edge_ratio,
    )

    edge_top, edge_bottom, strength = y0, y1, 0.0
    for seam, is_top in ((bottom_seam, False), (top_seam, True)):
        if seam is None:
            continue
        row, score = seam
        candidate = row + 1  # the first row *below* the seam
        if is_top:
            if abs(candidate - edge_top) > limit or edge_bottom - candidate < min_rows:
                continue
            new_top, new_bottom = candidate, edge_bottom
        else:
            if abs(candidate - edge_bottom) > limit or candidate - edge_top < min_rows:
                continue
            new_top, new_bottom = edge_top, candidate
        framed = (
            x0 / width,
            new_top / height,
            (x1 - x0) / width,
            (new_bottom - new_top) / height,
        )
        if not _contains(protect, framed):
            continue
        edge_top, edge_bottom = new_top, new_bottom
        strength = max(strength, score)

    if edge_top == y0 and edge_bottom == y1:
        return box, 0.0
    return (
        (x0 / width, edge_top / height, (x1 - x0) / width, (edge_bottom - edge_top) / height),
        strength,
    )


