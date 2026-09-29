from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clippy.api.app import create_app
from clippy.config import Settings
from clippy.edit.plan import EditPaths, EditPlan, MetadataPlan, build_plan
from clippy.store.db import Database


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {"data_dir": tmp_path, "openai_api_key": ""}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _seed(tmp_path: Path, *, with_final: bool = True, with_plan: bool = True):
    settings = _settings(tmp_path)
    db = Database(settings.resolved_db_path())
    streamer = db.get_or_create_streamer("jason", "Jason")
    stream = db.create_stream(
        streamer.id, "vod", media_path="vod.ts", source_width=1920, source_height=1080
    )
    candidate = db.create_candidate(
        stream.id,
        source_ts=100.0,
        pre_context_seconds=30.0,
        post_context_seconds=30.0,
        signals={"kind": "keyword"},
        score=0.9,
        caption="Jason announces a collab",
    )

    paths = EditPaths.for_candidate(settings, candidate.id)
    paths.ensure_root()
    if with_plan:
        plan = build_plan(
            candidate=candidate,
            stream=stream,
            source_path=Path("vod.ts"),
            settings=settings,
        )
        plan.metadata = MetadataPlan(
            enabled=True,
            source="fallback",
            title="Jason announces a collab",
            description="A collab is coming.",
            hashtags=["jason", "collab"],
        )
        plan.stage = "complete"
        plan.save(paths.plan)
        paths.captions.write_text(
            "\n".join(
                [
                    "[Script Info]",
                    "Dialogue: 0,0:00:00.00,0:00:02.00,Caption,,0,0,0,,{\\k50}CLIP{\\k50}THIS",
                    "Dialogue: 0,0:00:02.00,0:00:04.00,Caption,,0,0,0,,A second cue",
                ]
            ),
            encoding="utf-8",
        )

    render = None
    if with_final:
        paths.final.write_bytes(b"fake mp4 bytes")
        render = db.create_render(
            candidate.id,
            kind="final",
            path=str(paths.final),
            width=1080,
            height=1920,
            duration=12.5,
        )
        db.update_candidate_edit(
            candidate.id, edit_status="rendered", edited_media_path=str(paths.final)
        )
    return settings, db, candidate, render, paths


def test_index_shows_the_render_badge(tmp_path: Path):
    settings, _db, candidate, _render, _paths = _seed(tmp_path)
    client = TestClient(create_app(settings))

    response = client.get("/")

    assert response.status_code == 200
    assert f"/candidates/{candidate.id}" in response.text
    assert "rendered" in response.text
    assert "download" in response.text


def test_candidate_page_shows_the_edit(tmp_path: Path):
    settings, _db, candidate, render, _paths = _seed(tmp_path)
    client = TestClient(create_app(settings))

    response = client.get(f"/candidates/{candidate.id}")

    assert response.status_code == 200
    body = response.text
    assert f"/renders/{render.id}/media" in body
    assert "Edited clip" in body
    assert "Clip boundaries" in body
    assert "Dead air" in body
    assert "Re-render" in body
    assert f"/candidates/{candidate.id}/download" in body
    # Cue text is shown without ASS override tags.
    assert "CLIPTHIS" in body
    assert "{\\k50}" not in body
    assert "Metadata" in body


def test_candidate_page_without_a_plan_still_renders(tmp_path: Path):
    settings, _db, candidate, _render, _paths = _seed(
        tmp_path, with_final=False, with_plan=False
    )
    client = TestClient(create_app(settings))

    response = client.get(f"/candidates/{candidate.id}")

    assert response.status_code == 200
    assert "Edited clip" not in response.text


def test_download_serves_the_final_file(tmp_path: Path):
    settings, _db, candidate, _render, _paths = _seed(tmp_path)
    client = TestClient(create_app(settings))

    response = client.get(f"/candidates/{candidate.id}/download")

    assert response.status_code == 200
    assert response.content == b"fake mp4 bytes"


def test_download_without_a_render_is_404(tmp_path: Path):
    settings, _db, candidate, _render, _paths = _seed(tmp_path, with_final=False)
    client = TestClient(create_app(settings))

    assert client.get(f"/candidates/{candidate.id}/download").status_code == 404


def test_render_media_serves_by_revision(tmp_path: Path):
    settings, _db, _candidate, render, _paths = _seed(tmp_path)
    client = TestClient(create_app(settings))

    assert client.get(f"/renders/{render.id}/media").status_code == 200
    assert client.get("/renders/9999/media").status_code == 404


def test_metadata_edit_updates_the_plan(tmp_path: Path):
    settings, _db, candidate, _render, paths = _seed(tmp_path)
    client = TestClient(create_app(settings))

    response = client.post(
        f"/candidates/{candidate.id}/metadata",
        data={
            "title": "New title",
            "description": "New description",
            "hashtags": "#GTA, Role Play",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    plan = EditPlan.load(paths.plan)
    assert plan.metadata.source == "manual"
    assert plan.metadata.title == "New title"
    assert plan.metadata.description == "New description"
    assert plan.metadata.hashtags == ["gta", "roleplay"]


def test_metadata_edit_requires_a_plan(tmp_path: Path):
    settings, _db, candidate, _render, _paths = _seed(tmp_path, with_plan=False)
    client = TestClient(create_app(settings))

    response = client.post(
        f"/candidates/{candidate.id}/metadata",
        data={"title": "x"},
        follow_redirects=False,
    )
    assert response.status_code == 404


def test_render_route_passes_overrides(monkeypatch, tmp_path: Path):
    settings, _db, candidate, _render, _paths = _seed(tmp_path)
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"planned": 1}

    monkeypatch.setattr("clippy.api.app.run_edit_pipeline", fake_run)
    client = TestClient(create_app(settings))

    response = client.post(
        f"/candidates/{candidate.id}/render",
        data={
            "strategy": "gaming",
            "caption_style": "block_pop",
            "caption_emphasis": "off",
            "force": "1",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert captured["candidate_ids"] == [candidate.id]
    assert captured["dry_run"] is False
    assert captured["force"] is True
    assert captured["overrides"].strategy == "gaming"
    assert captured["overrides"].caption_style == "block_pop"
    assert captured["overrides"].caption_emphasis == "off"


def test_render_route_rejects_an_invalid_strategy(monkeypatch, tmp_path: Path):
    settings, _db, candidate, _render, _paths = _seed(tmp_path)

    def boom(**kwargs):
        raise ValueError("Invalid strategy 'nonsense'")

    monkeypatch.setattr("clippy.api.app.run_edit_pipeline", boom)
    client = TestClient(create_app(settings))

    response = client.post(
        f"/candidates/{candidate.id}/render", data={"strategy": "nonsense"}
    )
    assert response.status_code == 400


def test_render_route_404s_for_an_unknown_candidate(tmp_path: Path):
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    assert client.post("/candidates/4242/render", data={}).status_code == 404

