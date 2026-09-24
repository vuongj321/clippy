from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLIPPY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    db_path: Path | None = None
    media_dir: Path | None = None
    buffer_dir: Path | None = None

    pre_context_seconds: float = 30.0
    post_context_seconds: float = 30.0
    coalesce_gap_seconds: float = 20.0

    chat_window_seconds: float = 5.0
    chat_baseline_seconds: float = 60.0
    chat_spike_multiplier: float = 3.0
    chat_min_rate: float = 0.5
    chat_keywords: list[str] = Field(
        default_factory=lambda: ["clip it", "clip that", "clip this", "clip"]
    )
    chat_keyword_score: float = 0.7
    chat_spike_score: float = 0.85

    audio_frame_seconds: float = 0.5
    audio_baseline_seconds: float = 30.0
    audio_spike_multiplier: float = 2.5
    audio_min_rms: float = 0.02
    audio_spike_score: float = 0.75

    disk_budget_gb: float = 20.0
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    twitch_client_id: str = ""
    twitch_client_secret: str = ""
    twitch_irc_nick: str = ""
    twitch_irc_oauth: str = ""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    asr_model: str = "whisper-1"
    caption_model: str = "gpt-4o-mini"
    caption_max_chat_messages: int = 40
    caption_max_per_run: int = 20

    host: str = "127.0.0.1"
    port: int = 8000

    def resolved_db_path(self) -> Path:
        return self.db_path or (self.data_dir / "clippy.db")

    def resolved_media_dir(self) -> Path:
        return self.media_dir or (self.data_dir / "media")

    def resolved_buffer_dir(self) -> Path:
        return self.buffer_dir or (self.data_dir / "buffer")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_media_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_buffer_dir().mkdir(parents=True, exist_ok=True)


def load_yaml_overrides(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


@lru_cache
def get_settings(config_path: str | None = None) -> Settings:
    overrides: dict = {}
    if config_path:
        overrides = load_yaml_overrides(Path(config_path))
    else:
        default_yaml = PROJECT_ROOT / "config.yaml"
        if default_yaml.exists():
            overrides = load_yaml_overrides(default_yaml)
    return Settings(**overrides)
