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
- A clip where the face was seen in fewer than `min_hit_ratio` of sampled frames is rejected
  outright (`face_track` returns ``[]``), so a busy background or one lucky hit can never steer
  the framing. The caller then falls back to motion.

Frames where the face is not detected carry the previous position forward, so downstream segment
logic sees a dense, calm track rather than a sparse, jumpy one.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
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

FaceBox = tuple[float, float, float, float]  # x, y, w, h as fractions of the frame


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


def pick_face(
    boxes: Sequence[FaceBox],
    *,
    previous: FaceBox | None = None,
) -> FaceBox | None:
    """
    Choose the face to follow: the largest, with a nudge toward the previous one.

    Largest-first is the right prior for a facecam (it is the biggest face on screen), and the
    proximity nudge stops the choice flickering between two similar-sized faces.
    """
    if not boxes:
        return None

    def score(box: FaceBox) -> float:
        x, y, w, h = box
        area = w * h
        if previous is None:
            return area
        px, py, pw, ph = previous
        distance = abs((x + w / 2) - (px + pw / 2)) + abs((y + h / 2) - (py + ph / 2))
        return area / (1.0 + distance)

    return max(boxes, key=score)


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
    Follow the largest face across a clip, or return ``[]`` when there is no usable face.

    One point per sampled frame: the detected centre when there is a detection, otherwise the
    previous position carried forward (and the frame centre before the first detection). The box
    size is carried too, so the caption stage can test the real face rectangle against the
    caption band instead of assuming a subject height.
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
    if frames.shape[0] == 0:
        return []

    boxes: list[FaceBox | None] = []
    previous: FaceBox | None = None
    for frame in frames:
        found = pick_face(
            detect_faces(
                frame,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size_ratio=min_size_ratio,
            ),
            previous=previous,
        )
        if found is not None:
            previous = found
        boxes.append(found)

    hits = sum(1 for box in boxes if box is not None)
    hit_ratio = hits / len(boxes)
    threshold = max(0.0, float(min_hit_ratio))
    if hit_ratio < threshold:
        logger.info(
            "Face track rejected for %s: seen in %.0f%% of frames (< %.0f%%)",
            media_path.name,
            hit_ratio * 100,
            threshold * 100,
        )
        return []

    step = 1.0 / max(sample_fps, 1e-6)
    points: list[TrackPoint] = []
    last: FaceBox | None = None
    for index, box in enumerate(boxes):
        if box is not None:
            last = box
        centre = (last[0] + last[2] / 2.0, last[1] + last[3] / 2.0) if last else (0.5, 0.5)
        points.append(
            TrackPoint(
                t=(index + 0.5) * step,
                x=min(max(centre[0], 0.0), 1.0),
                y=min(max(centre[1], 0.0), 1.0),
                width=last[2] if last else 0.0,
                height=last[3] if last else 0.0,
            )
        )
    logger.info(
        "Face track for %s: %d/%d frames (%.0f%%)",
        media_path.name,
        hits,
        len(boxes),
        hit_ratio * 100,
    )
    return points


