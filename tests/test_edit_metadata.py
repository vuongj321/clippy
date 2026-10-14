from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clippy.config import Settings
from clippy.edit.metadata import (
    ClipMetadata,
    _parse_json_object,
    build_hashtags,
    capture_thumbnail,
    fallback_metadata,
    generate_metadata,
    normalize_hashtag,
    source_words,
    validate_metadata,
)

FIXTURE = Path(__file__).resolve().parents[1] / "samples" / "sample_vod.mp4"


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "data_dir": tmp_path,
        "metadata_enabled": True,
        "metadata_max_hashtags": 6,
        "metadata_title_max_chars": 60,
        "chat_keywords": ["clip", "clip that"],
        "openai_api_key": "sk-test",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _completion(content: str) -> _FakeResponse:
    return _FakeResponse({"choices": [{"message": {"content": content}}]})


def test_normalize_hashtag_collapses_to_a_single_token():
    assert normalize_hashtag("#Clip It!") == "clipit"
    assert normalize_hashtag("gta") == "gta"
    assert normalize_hashtag("  ") == ""


def test_source_words_collects_lowercase_tokens():
    words = source_words("We played GTA RP!", "and roleplayed")
    assert {"we", "played", "gta", "rp", "and", "roleplayed"} <= words


def test_build_hashtags_leads_with_the_streamer_then_content():
    tags = build_hashtags(
        "the streamer talked about gta roleplay",
        "roleplay was wild",
        streamer_login="JasonV",
        keywords=["clip", "gta"],
        limit=5,
    )
    assert tags[0] == "jasonv"
    assert "gta" in tags  # a chat keyword the transcript supports
    assert "roleplay" in tags
    assert all(" " not in tag for tag in tags)


def test_validate_metadata_drops_unsupported_hashtags(tmp_path: Path):
    validated = validate_metadata(
        {
            "title": "GTA moment",
            "description": "A thing happened.",
            "hashtags": ["gta", "minecraft"],
        },
        transcript_text="we played gta all night",
        streamer_login="jason",
        settings=_settings(tmp_path),
    )
    assert validated is not None
    assert validated.hashtags == ["gta"]
    assert validated.source == "llm"


def test_validate_metadata_requires_a_title(tmp_path: Path):
    assert (
        validate_metadata(
            {"title": "  ", "hashtags": ["gta"]},
            transcript_text="gta",
            settings=_settings(tmp_path),
        )
        is None
    )


def test_validate_metadata_truncates_the_title_and_caps_hashtags(tmp_path: Path):
    long_title = "word " * 40
    validated = validate_metadata(
        {"title": long_title, "hashtags": ["gta", "roleplay", "stream", "night", "clip", "rp", "extra"]},
        transcript_text="gta roleplay stream night clip rp extra",
        settings=_settings(tmp_path, metadata_title_max_chars=30, metadata_max_hashtags=3),
    )
    assert validated is not None
    assert len(validated.title or "") <= 30
    assert len(validated.hashtags) <= 3


def test_fallback_metadata_uses_caption_and_transcript(tmp_path: Path):
    metadata = fallback_metadata(
        transcript_text="First sentence here. Second sentence here. Third one.",
        caption="Jason announces a collab",
        streamer_login="jason",
        streamer_display_name="Jason",
        keywords=["clip"],
        thumbnail_time=12.5,
        settings=_settings(tmp_path),
    )
    assert metadata.title == "Jason announces a collab"
    assert metadata.description is not None and "(via Jason)" in metadata.description
    assert metadata.description.count("sentence") == 2
    assert metadata.thumbnail_time == pytest.approx(12.5)
    assert metadata.source == "fallback"
    assert metadata.hashtags[0] == "jason"


def test_parse_json_object_handles_code_fences():
    assert _parse_json_object('```json\n{"title": "x"}\n```') == {"title": "x"}
    assert _parse_json_object("no json here") == {}
    assert _parse_json_object('{"broken": ') == {}


def test_generate_metadata_without_a_key_uses_the_fallback(tmp_path: Path):
    metadata = generate_metadata(
        transcript_text="we played gta",
        chat_context={},
        streamer_display_name="Jason",
        streamer_login="jason",
        settings=_settings(tmp_path),
        api_key="",
    )
    assert metadata.source == "fallback"
    assert metadata.reason is not None and "no API key" in metadata.reason


def test_generate_metadata_uses_a_validated_llm_response(monkeypatch, tmp_path: Path):
    content = json.dumps(
        {
            "title": "GTA chaos",
            "description": "Wild.",
            "hashtags": ["gta", "minecraft"],
        }
    )
    monkeypatch.setattr(
        "clippy.edit.metadata.httpx.post", lambda *a, **k: _completion(content)
    )
    metadata = generate_metadata(
        transcript_text="we played gta all night",
        chat_context={"window_count": 12},
        streamer_display_name="Jason",
        streamer_login="jason",
        settings=_settings(tmp_path),
        api_key="sk-test",
        thumbnail_time=3.0,
    )
    assert metadata.source == "llm"
    assert metadata.title == "GTA chaos"
    assert metadata.hashtags == ["gta"]  # the invented tag is dropped
    assert metadata.thumbnail_time == pytest.approx(3.0)


def test_generate_metadata_falls_back_on_unusable_output(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "clippy.edit.metadata.httpx.post", lambda *a, **k: _completion("sorry, no json")
    )
    metadata = generate_metadata(
        transcript_text="we played gta",
        chat_context={},
        streamer_display_name="Jason",
        streamer_login="jason",
        settings=_settings(tmp_path),
        api_key="sk-test",
    )
    assert metadata.source == "fallback"
    assert metadata.reason is not None and "fallback" in metadata.reason


def test_generate_metadata_falls_back_on_an_http_failure(monkeypatch, tmp_path: Path):
    def boom(*args, **kwargs):
        raise RuntimeError("429 too many requests")

    monkeypatch.setattr("clippy.edit.metadata.httpx.post", boom)
    metadata = generate_metadata(
        transcript_text="we played gta",
        chat_context={},
        streamer_display_name="Jason",
        streamer_login="jason",
        settings=_settings(tmp_path),
        api_key="sk-test",
    )
    assert metadata.source == "fallback"
    assert metadata.reason is not None and "llm metadata failed" in metadata.reason


def test_generate_metadata_can_be_disabled(monkeypatch, tmp_path: Path):
    def never(*args, **kwargs):
        raise AssertionError("metadata must not call the API when disabled")

    monkeypatch.setattr("clippy.edit.metadata.httpx.post", never)
    metadata = generate_metadata(
        transcript_text="we played gta",
        chat_context={},
        streamer_display_name="Jason",
        streamer_login="jason",
        settings=_settings(tmp_path, metadata_enabled=False),
        api_key="sk-test",
    )
    assert metadata.source == "fallback"
    assert metadata.reason == "metadata disabled"


def test_clip_metadata_round_trips():
    original = ClipMetadata(
        title="title",
        description="description",
        hashtags=["a", "b"],
        thumbnail_time=1.5,
        source="llm",
        reason=None,
    )
    assert ClipMetadata.from_dict(original.to_dict()).to_dict() == original.to_dict()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)
def test_capture_thumbnail_writes_a_cover(tmp_path: Path):
    output = tmp_path / "thumb.jpg"
    result = capture_thumbnail(FIXTURE, output, time_seconds=120.0)
    assert result == output
    assert output.exists() and output.stat().st_size > 1000


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)
def test_capture_thumbnail_with_overlay_text(tmp_path: Path):
    output = tmp_path / "thumb_overlay.jpg"
    capture_thumbnail(
        FIXTURE, output, time_seconds=120.0, overlay_text="GTA CHAOS: the moment"
    )
    assert output.exists() and output.stat().st_size > 1000
