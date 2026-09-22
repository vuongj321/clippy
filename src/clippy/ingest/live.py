from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from clippy.buffer.rolling import RollingMediaBuffer
from clippy.chat.irc import TwitchIrcChat
from clippy.chat.models import ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class LiveIngestSession:
    """
    Live ingest via Streamlink (HLS→file segments) + Twitch IRC chat.

    Timeline: t=0 when the session starts (wall clock). Chat and media are
    aligned to that origin. Streamlink writes a continuously growing file;
    we treat that file as the source media for later window cuts from t=0.
    """

    channel_login: str
    buffer: RollingMediaBuffer
    output_path: Path
    nick: str
    oauth_token: str
    streamlink_path: str = "streamlink"
    quality: str = "best"
    chat: list[ChatMessage] = field(default_factory=list)
    timeline_origin: float = field(default_factory=time.time)
    _stop: threading.Event = field(default_factory=threading.Event)
    _stream_proc: subprocess.Popen | None = None
    _irc_thread: threading.Thread | None = None

    def start(self) -> None:
        self.timeline_origin = time.time()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer.set_source_media(self.output_path)
        self._start_streamlink()
        self._start_irc()

    def stop(self) -> None:
        self._stop.set()
        if self._stream_proc and self._stream_proc.poll() is None:
            self._stream_proc.terminate()
            try:
                self._stream_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._stream_proc.kill()

    def _start_streamlink(self) -> None:
        exe = shutil.which(self.streamlink_path)
        if not exe:
            raise RuntimeError(
                "streamlink not found on PATH. Install streamlink to record live Twitch."
            )
        url = f"https://twitch.tv/{self.channel_login}"
        cmd = [
            exe,
            url,
            self.quality,
            "-o",
            str(self.output_path),
            "--twitch-disable-ads",
            "--retry-streams",
            "5",
            "--retry-max",
            "10",
        ]
        logger.info("Starting streamlink: %s", " ".join(cmd))
        self._stream_proc = subprocess.Popen(cmd)

    def _start_irc(self) -> None:
        def _run() -> None:
            import asyncio

            client = TwitchIrcChat(
                self.channel_login,
                nick=self.nick,
                oauth_token=self.oauth_token,
                timeline_origin=self.timeline_origin,
                on_message=self._on_chat,
            )

            async def _main() -> None:
                task = asyncio.create_task(client.run())
                while not self._stop.is_set():
                    await asyncio.sleep(0.5)
                client.stop()
                task.cancel()

            asyncio.run(_main())

        self._irc_thread = threading.Thread(target=_run, name="twitch-irc", daemon=True)
        self._irc_thread.start()

    def _on_chat(self, msg: ChatMessage) -> None:
        self.chat.append(msg)

    def elapsed(self) -> float:
        return time.time() - self.timeline_origin


def create_live_session(
    channel_login: str,
    *,
    buffer_dir: Path,
    nick: str,
    oauth_token: str,
    retention_seconds: float = 600.0,
) -> LiveIngestSession:
    buffer = RollingMediaBuffer(buffer_dir, retention_seconds=retention_seconds)
    output = buffer_dir / f"{channel_login}_live.ts"
    return LiveIngestSession(
        channel_login=channel_login.lower(),
        buffer=buffer,
        output_path=output,
        nick=nick,
        oauth_token=oauth_token,
    )
