from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import uvicorn

from clippy.api.app import create_app
from clippy.config import get_settings
from clippy.eval.export import export_reviews_csv, export_reviews_json, precision_report
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


if __name__ == "__main__":
    print("Use clippy-vod, clippy-live, clippy-serve, or clippy-export entry points.", file=sys.stderr)
    sys.exit(1)
