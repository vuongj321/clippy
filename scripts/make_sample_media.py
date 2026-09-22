#!/usr/bin/env python
"""Generate a short silent sample MP4 for local pipeline smoke tests (requires ffmpeg)."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("samples/sample_vod.mp4"),
    )
    parser.add_argument("--duration", type=float, default=260.0)
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Tone burst around t=120 to roughly align with sample chat spike
    # aevalsrc: silence then louder tone
    filter_complex = (
        f"sine=frequency=440:sample_rate=44100:duration={args.duration},"
        f"volume='if(between(t,118,125),3,0.2)':eval=frame"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=640x360:d={args.duration}",
        "-f",
        "lavfi",
        "-i",
        filter_complex,
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(args.out),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
