from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable

from clippy.chat.models import ChatMessage

logger = logging.getLogger(__name__)

TWITCH_IRC = "wss://irc-ws.chat.twitch.tv:443"


class TwitchIrcChat:
    """
    Minimal Twitch IRC chat client over WebSocket.

    Timeline: ts is seconds since `timeline_origin` (monotonic wall alignment).
    Caller should set timeline_origin when the live media session starts.
    """

    def __init__(
        self,
        channel_login: str,
        *,
        nick: str,
        oauth_token: str,
        timeline_origin: float,
        on_message: Callable[[ChatMessage], None] | None = None,
    ) -> None:
        self.channel_login = channel_login.lstrip("#").lower()
        self.nick = nick.lower()
        token = oauth_token
        if token.startswith("oauth:"):
            token = token[len("oauth:") :]
        self.oauth_token = token
        self.timeline_origin = timeline_origin
        self.on_message = on_message
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets package required for live chat") from exc

        uri = TWITCH_IRC
        while not self._stop.is_set():
            try:
                async with websockets.connect(uri) as ws:
                    await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
                    await ws.send(f"PASS oauth:{self.oauth_token}")
                    await ws.send(f"NICK {self.nick}")
                    await ws.send(f"JOIN #{self.channel_login}")
                    logger.info("Joined Twitch IRC #%s", self.channel_login)
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="replace")
                        for line in raw.split("\r\n"):
                            if not line:
                                continue
                            if line.startswith("PING"):
                                await ws.send("PONG :tmi.twitch.tv")
                                continue
                            msg = self._parse_privmsg(line)
                            if msg and self.on_message:
                                self.on_message(msg)
            except Exception:
                logger.exception("IRC disconnected; reconnecting in 3s")
                await asyncio.sleep(3)

    def _parse_privmsg(self, line: str) -> ChatMessage | None:
        # @tags :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
        if "PRIVMSG" not in line:
            return None
        try:
            tags_part = ""
            rest = line
            if line.startswith("@"):
                tags_part, rest = line[1:].split(" ", 1)
            prefix, rest2 = rest.split(" ", 1)
            user = prefix.lstrip(":").split("!", 1)[0]
            if " :" not in rest2:
                return None
            _, text = rest2.split(" :", 1)
            import time

            ts = time.time() - self.timeline_origin
            return ChatMessage(ts=ts, user=user, text=text)
        except Exception:
            logger.debug("Failed to parse IRC line: %s", line)
            return None


async def iter_irc_messages(
    channel_login: str,
    *,
    nick: str,
    oauth_token: str,
    timeline_origin: float,
) -> AsyncIterator[ChatMessage]:
    queue: asyncio.Queue[ChatMessage] = asyncio.Queue()
    client = TwitchIrcChat(
        channel_login,
        nick=nick,
        oauth_token=oauth_token,
        timeline_origin=timeline_origin,
        on_message=lambda m: queue.put_nowait(m),
    )
    task = asyncio.create_task(client.run())
    try:
        while True:
            msg = await queue.get()
            yield msg
    finally:
        client.stop()
        task.cancel()
