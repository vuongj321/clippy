from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import uvicorn

from clippy.api.app import create_app
from clippy.chat.models import load_chat_json
from clippy.config import Settings, get_settings
from clippy.edit.plan import (
    CAPTION_EMPHASIS_MODES,
    CAPTION_STYLES,
    DEADAIR_MODES,
    STRATEGIES,
    EditOverrides,
)
from clippy.edit.pipeline import run_edit_pipeline
from clippy.eval.export import export_reviews_csv, export_reviews_json, precision_report
from clippy.ingest.align import (
    DEFAULT_ALIGN_WINDOW_SECONDS,
    OffsetVerification,
    alignment_warning,
    probe_alignment,
    verify_offset_with_clips,
)
from clippy.ingest.capture import (
    capture_vod,
    default_output_path,
    normalize_downloader,
    prune_sources,
)
from clippy.pipeline import run_live_pipeline, run_vod_pipeline
from clippy.store.db import Database


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_vod_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Clippy VOD detection pipeline")
    parser.add_argument("--media", type=Path, required=True, help="Path to VOD/media file")
    parser.add_argument("--chat", type=Path, required=True, help="Path to chat JSON")
    parser.add_argument("--streamer", required=True, help="Twitch login")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--vod-id", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    # Clear cached settings if config path provided
    get_settings.cache_clear()
    settings = get_settings(args.config)
    result = run_vod_pipeline(
        media_path=args.media,
        chat_path=args.chat,
        streamer_login=args.streamer,
        settings=settings,
        display_name=args.display_name,
        source_url=args.source_url,
        vod_id=args.vod_id,
    )
    print(json.dumps(result, indent=2))


def run_live_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Clippy live Twitch pipeline")
    parser.add_argument("--channel", required=True, help="Twitch channel login")
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Seconds to record before detect/extract (default 300)",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    get_settings.cache_clear()
    settings = get_settings(args.config)
    result = run_live_pipeline(
        channel_login=args.channel,
        settings=settings,
        duration_seconds=args.duration,
    )
    print(json.dumps(result, indent=2))


def serve_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve Clippy review UI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    get_settings.cache_clear()
    settings = get_settings(args.config)
    host = args.host or settings.host
    port = args.port or settings.port
    app = create_app(settings)
    uvicorn.run(app, host=host, port=port)


def export_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export reviews and print precision report")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args(argv)
    get_settings.cache_clear()
    settings = get_settings(args.config)
    settings.ensure_dirs()
    db = Database(settings.resolved_db_path())
    report = precision_report(db)
    print(json.dumps(report, indent=2))
    json_out = args.json_out or (settings.data_dir / "exports" / "reviews.json")
    csv_out = args.csv_out or (settings.data_dir / "exports" / "reviews.csv")
    export_reviews_json(db, json_out)
    export_reviews_csv(db, csv_out)
    print(f"Wrote {json_out}")
    print(f"Wrote {csv_out}")


def edit_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plan or render Clippy short-form edits for candidates"
    )
    parser.add_argument(
        "--candidate",
        type=int,
        action="append",
        default=None,
        help="Candidate id to edit (repeatable)",
    )
    parser.add_argument("--stream", type=int, default=None, help="Edit candidates of a stream id")
    parser.add_argument(
        "--all-pending", action="store_true", help="Edit every pending candidate"
    )
    parser.add_argument(
        "--status",
        default=None,
        choices=("pending", "approved", "rejected", "all"),
        help="Status filter for --stream/--all-pending (default: pending)",
    )
    parser.add_argument("--top", type=int, default=None, help="Cap the number of edits this run")
    parser.add_argument("--max-per-run", type=int, default=None, help="Alias for --top")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write plan.json and a renders row only (no ffmpeg, no network)",
    )
    parser.add_argument("--force", action="store_true", help="Redo cached artifacts")
    parser.add_argument("--source", type=Path, default=None, help="Override source media file")
    parser.add_argument(
        "--source-offset",
        type=float,
        default=None,
        help="Seconds to shift source timestamps (partial captures)",
    )
    parser.add_argument(
        "--chat",
        type=Path,
        default=None,
        help="Chat JSON used as boundary evidence (stream-relative timestamps)",
    )
    parser.add_argument("--strategy", default=None, choices=STRATEGIES)
    parser.add_argument("--caption-style", default=None, choices=CAPTION_STYLES)
    parser.add_argument("--caption-emphasis", default=None, choices=CAPTION_EMPHASIS_MODES)
    parser.add_argument("--deadair-mode", default=None, choices=DEADAIR_MODES)
    parser.add_argument("--no-captions", action="store_true", help="Render without captions")
    parser.add_argument(
        "--keep-intermediate", action="store_true", help="Keep stage artifacts (base/trimmed)"
    )
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if not (args.candidate or args.stream is not None or args.all_pending):
        parser.error("specify --candidate, --stream, or --all-pending")

    status = None if args.status == "all" else (args.status or "pending")
    overrides = EditOverrides(
        strategy=args.strategy,
        caption_style=args.caption_style,
        caption_emphasis=args.caption_emphasis,
        deadair_mode=args.deadair_mode,
        captions_enabled=False if args.no_captions else None,
        keep_intermediate=True if args.keep_intermediate else None,
    )

    get_settings.cache_clear()
    settings = get_settings(args.config)
    try:
        result = run_edit_pipeline(
            settings=settings,
            candidate_ids=args.candidate,
            stream_id=args.stream,
            status=None if args.candidate else status,
            max_per_run=args.top or args.max_per_run,
            dry_run=args.dry_run,
            force=args.force,
            source_path=args.source,
            source_offset_seconds=args.source_offset,
            overrides=overrides,
            chat_path=args.chat,
        )
    except ValueError as exc:
        print(f"clippy-edit: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2))


def _verify_offset_with_clips(
    *,
    settings: Settings,
    stream_id: int,
    media_path: Path,
    warnings: list[str],
) -> OffsetVerification | None:
    """
    Prove the capture's offset by correlating existing Phase 1 clips against it.

    Chat-vs-audio correlation is only a heuristic (spiky things correlate with spiky
    things), so a confidently *applied* offset must come from comparing identical
    audio content. Missing clips or an inconclusive score leaves the offset at 0.
    """
    db = Database(settings.resolved_db_path())
    candidates = db.list_candidates(stream_id=stream_id, status=None, limit=3)
    clips = [
        (
            Path(candidate.media_path),
            max(0.0, candidate.source_ts - candidate.pre_context_seconds),
        )
        for candidate in candidates
        if candidate.media_path and Path(candidate.media_path).exists()
    ]
    if not clips:
        warnings.append(
            "no Phase 1 clips available to verify the offset; leaving it at 0 "
            "(pass --source-offset if the capture did not start at the stream start)"
        )
        return None
    try:
        verification = verify_offset_with_clips(
            media_path=media_path, clips=clips, ffmpeg_path=settings.ffmpeg_path
        )
    except Exception as exc:  # verification must never fail a capture
        warnings.append(f"clip-based offset verification failed: {exc}")
        return None
    if verification.is_confident():
        warnings.append(
            f"clip-verified source offset {verification.offset_seconds:+.2f}s "
            f"(score {verification.score:.3f} over {verification.checked} clips)"
        )
    else:
        warnings.append(
            "clip-based offset verification was inconclusive "
            f"(score {verification.score:.3f}); leaving the offset at 0"
        )
    return verification


def capture_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Capture a Twitch VOD at high quality for short-form editing"
    )
    parser.add_argument("--vod", required=True, help="VOD id or full URL")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Output media path")
    parser.add_argument("--quality", default=None, help="Downloader quality (default: config)")
    parser.add_argument("--downloader", default=None, choices=("streamlink", "yt-dlp", "yt_dlp"))
    parser.add_argument(
        "--chat", type=Path, default=None, help="Chat JSON used to verify the timeline"
    )
    parser.add_argument("--no-align", action="store_true", help="Skip timeline verification")
    parser.add_argument("--align-window", type=float, default=DEFAULT_ALIGN_WINDOW_SECONDS)
    parser.add_argument(
        "--record-stream",
        type=int,
        default=None,
        help="Update this stream id with the capture path and source metadata",
    )
    parser.add_argument(
        "--source-offset",
        type=float,
        default=None,
        help="Override the detected stream->media offset (seconds)",
    )
    parser.add_argument("--prune", action="store_true", help="Enforce source_budget_gb")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    get_settings.cache_clear()
    settings = get_settings(args.config)
    settings.ensure_dirs()

    warnings: list[str] = []
    try:
        downloader = normalize_downloader(args.downloader or settings.capture_downloader)
        output = args.out or default_output_path(settings.resolved_source_dir(), args.vod)
        result = capture_vod(
            vod=args.vod,
            output=output,
            quality=args.quality or settings.capture_quality,
            downloader=downloader,
            ffprobe_path=settings.ffprobe_path,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"clippy-capture: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    alignment = None
    alignment = None
    verification = None
    if args.chat and not args.no_align:
        try:
            alignment = probe_alignment(
                result.path,
                load_chat_json(args.chat),
                window_seconds=args.align_window,
                ffmpeg_path=settings.ffmpeg_path,
            )
            if alignment.is_meaningful():
                message = alignment_warning(
                    alignment, tolerance_seconds=settings.alignment_tolerance_seconds
                )
                if message:
                    warnings.append(
                        message + " (chat vs audio is only a heuristic: nothing applied)"
                    )
            else:
                warnings.append(
                    "timeline verification found too little signal to judge the offset"
                )
        except Exception as exc:  # verification must never fail a capture
            warnings.append(f"timeline verification failed: {exc}")
    elif not args.no_align:
        warnings.append("no --chat given; timeline offset not verified (assuming 0)")

    if args.record_stream is not None:
        verification = _verify_offset_with_clips(
            settings=settings,
            stream_id=args.record_stream,
            media_path=result.path,
            warnings=warnings,
        )

    if args.source_offset is not None:
        applied_offset = args.source_offset
    elif verification is not None and verification.is_confident():
        applied_offset = verification.offset_seconds
    else:
        applied_offset = 0.0
        if (
            alignment is not None
            and alignment.is_meaningful()
            and abs(alignment.offset_seconds) > settings.alignment_tolerance_seconds
        ):
            warnings.append(
                "chat-based offset was NOT applied (heuristic); "
                f"pass --source-offset {alignment.offset_seconds:.2f} to force it"
            )

    recorded_stream = None
    if args.record_stream is not None:
        db = Database(settings.resolved_db_path())
        if db.get_stream(args.record_stream) is None:
            warnings.append(f"stream {args.record_stream} not found; source not recorded")
        else:
            db.update_stream_source(
                args.record_stream,
                media_path=str(result.path),
                source_width=result.media.width,
                source_height=result.media.height,
                source_fps=result.media.fps,
                capture_quality=result.quality,
                source_offset_seconds=applied_offset,
                source_bytes=result.bytes,
            )
            recorded_stream = args.record_stream

    pruned: list[str] = []
    if args.prune:
        pruned = [
            str(path)
            for path in prune_sources(
                settings.resolved_source_dir(),
                budget_gb=settings.source_budget_gb,
                keep=[result.path],
            )
        ]

    print(
        json.dumps(
            {
                **result.to_dict(),
                "alignment": None if alignment is None else alignment.to_dict(),
                "clip_verification": (
                    None if verification is None else verification.to_dict()
                ),
                "source_offset_seconds": applied_offset,
                "recorded_stream": recorded_stream,
                "pruned": pruned,
                "warnings": warnings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    print(
        "Use clippy-vod, clippy-live, clippy-serve, clippy-export, clippy-edit, "
        "or clippy-capture entry points.",
        file=sys.stderr,
    )
    sys.exit(1)
