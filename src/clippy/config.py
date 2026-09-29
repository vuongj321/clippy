from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
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
    edits_dir: Path | None = None
    source_dir: Path | None = None

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

    # --- Phase 2: HQ capture (M1) ---
    capture_quality: str = "best"
    capture_downloader: Literal["streamlink", "yt_dlp"] = "streamlink"
    capture_source_offset_seconds: float = 0.0
    source_budget_gb: float = 40.0
    alignment_tolerance_seconds: float = 1.0

    # --- Phase 2: clip boundaries ---
    clip_min_seconds: float = 10.0
    clip_max_seconds: float = 45.0
    clip_target_seconds: float = 30.0
    boundary_search_seconds: float = 20.0
    hook_lookback_seconds: float = 8.0
    min_context_seconds: float = 1.5
    reaction_tail_seconds: float = 3.0
    boundary_min_silence_seconds: float = 0.35
    word_gap_min_seconds: float = 0.08
    boundary_llm_refine: bool = False
    extract_duration_tolerance_seconds: float = 0.5

    # --- Phase 2: dead-air removal ---
    deadair_enabled: bool = True
    deadair_mode: Literal["cut", "speed"] = "cut"
    deadair_noise_db: float = -30.0
    deadair_min_gap_seconds: float = 0.8
    deadair_keep_pad_seconds: float = 0.2
    deadair_min_keep_seconds: float = 0.5
    deadair_max_removed_ratio: float = 0.4
    deadair_speed_factor: float = 1.5

    # --- Phase 2: burned-in captions ---
    caption_enabled: bool = True
    caption_style: Literal["karaoke_highlight", "block_pop", "minimal"] = "karaoke_highlight"
    caption_font: str = "Arial"
    caption_font_size: int = 54
    caption_max_chars_per_line: int = 18
    caption_max_lines: int = 2
    caption_max_cue_seconds: float = 2.2
    caption_min_cue_seconds: float = 0.5
    caption_break_gap_seconds: float = 0.35
    caption_emphasis: Literal["heuristic", "llm", "off"] = "heuristic"
    caption_primary_color: str = "&H00FFFFFF"
    caption_highlight_color: str = "&H0000D7FF"
    caption_margin_v: int = 260
    caption_safe_area: Literal["auto", "top", "middle", "bottom"] = "auto"
    caption_avoid_ratio: float = 0.25
    caption_uppercase: bool = True
    caption_fonts_dir: str = "C:/Windows/Fonts"
    asr_provider: Literal["api", "ffmpeg_whisper"] = "api"
    asr_word_timestamps: bool = True
    asr_whisper_model_path: str = ""

    # --- Phase 2: vertical formatting ---
    clip_target_width: int = 1080
    clip_target_height: int = 1920
    clip_fps: int = 30
    layout_strategy: Literal["auto", "fit_blur", "irl", "gaming", "conversation"] = "auto"
    layout_track_backend: Literal["none", "motion", "opencv", "mediapipe"] = "motion"
    layout_smoothing: float = 0.12
    layout_zoom: float = 1.0
    facecam_box: str = ""
    quality_warn_upscale: float = 2.0

    # --- Phase 2: render + audio ---
    render_dir: Path | None = None
    render_crf: int = 20
    intermediate_crf: int = 16
    render_preset: str = "veryfast"
    render_encoder: Literal["auto", "x264", "nvenc"] = "auto"
    render_disk_budget_gb: float = 20.0
    edit_max_per_run: int = 20
    keep_intermediate: bool = False
    audio_normalize: bool = True
    audio_target_lufs: float = -14.0
    audio_true_peak: float = -1.5
    audio_limiter: bool = True

    # --- Phase 2: metadata ---
    metadata_enabled: bool = True
    metadata_model: str = ""
    metadata_max_hashtags: int = 6
    metadata_title_max_chars: int = 60
    thumbnail_enabled: bool = True
    thumbnail_overlay_text: bool = False

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

    def resolved_edits_dir(self) -> Path:
        return self.edits_dir or (self.render_dir or (self.data_dir / "edits"))

    def resolved_source_dir(self) -> Path:
        return self.source_dir or (self.data_dir / "source")

    def parsed_facecam_box(self) -> tuple[float, float, float, float] | None:
        """Parse "x,y,w,h". Values are fractions of the source frame when <= 1, else pixels."""
        raw = (self.facecam_box or "").strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"facecam_box must be 'x,y,w,h' (got {self.facecam_box!r})"
            )
        try:
            values = tuple(float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"facecam_box must be numeric (got {self.facecam_box!r})") from exc
        return values  # type: ignore[return-value]

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_media_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_buffer_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_edits_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_source_dir().mkdir(parents=True, exist_ok=True)


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
