from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clippy.caption.asr import TranscriptPayload, TranscriptSegment, TranscriptWord
from clippy.config import Settings
from clippy.edit.plan import (
    EditOverrides,
    EditPaths,
    EditPlan,
    WARN_BOUNDARIES_PENDING,
    WARN_LAYOUT_PENDING,
)
from clippy.edit.pipeline import (
    EditJob,
    resolve_jobs,
    run_edit_pipeline,
    select_edit_jobs,
)
from clippy.store.db import Candidate, Database, Stream

FIXTURE = Path(__file__).resolve().parents[1] / "samples" / "sample_vod.mp4"


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = {"data_dir": tmp_path}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _seed(
    tmp_path: Path,
    *,
    media: Path | None,
    source_width: int = 1920,
    source_height: int = 1080,
    score: float = 0.9,
    source_ts: float = 120.0,
):
    settings = _settings(tmp_path)
    db = Database(settings.resolved_db_path())
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(
        streamer.id,
        "vod",
        media_path=str(media) if media is not None else None,
        source_width=source_width,
        source_height=source_height,
        source_fps=60.0,
        capture_quality="1080p60",
    )
    candidate = db.create_candidate(
        stream.id,
        source_ts=source_ts,
        pre_context_seconds=30.0,
        post_context_seconds=30.0,
        signals={"kind": "keyword"},
        score=score,
        extract_reason="Chat asked to clip it",
    )
    return settings, db, stream, candidate


def _media_file(tmp_path: Path, name: str = "vod.ts") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x00" * 32)
    return path


def test_dry_run_writes_plan_and_render_row(tmp_path: Path):
    media = _media_file(tmp_path)
    settings, db, stream, candidate = _seed(tmp_path, media=media)

    result = run_edit_pipeline(settings=settings, candidate_ids=[candidate.id], dry_run=True)

    assert result["dry_run"] is True
    assert result["candidates"] == 1
    assert result["selected"] == 1
    assert result["planned"] == 1
    assert result["skipped"] == 0

    paths = EditPaths.for_candidate(settings, candidate.id)
    assert paths.plan.exists()
    plan = EditPlan.load(paths.plan)
    assert plan.candidate_id == candidate.id
    assert plan.stream_id == stream.id
    assert plan.source_path == str(media)
    assert plan.bounds.method == "phase1_window"
    assert plan.bounds.duration == pytest.approx(45.0)
    assert plan.stage == "planned"

    renders = db.list_renders(candidate.id)
    assert len(renders) == 1
    assert renders[0].kind == "plan"
    assert renders[0].status == "ok"
    assert renders[0].path == str(paths.plan)
    assert json.loads(renders[0].plan_json)["bounds"]["method"] == "phase1_window"

    # A dry run never claims the candidate has a finished edit.
    reloaded = db.get_candidate(candidate.id)
    assert reloaded is not None
    assert reloaded.edit_status == "unrendered"
    assert reloaded.edited_media_path is None


def test_dry_run_records_failure_for_missing_source(tmp_path: Path):
    missing = tmp_path / "gone.ts"
    settings, db, _stream, candidate = _seed(tmp_path, media=missing)

    result = run_edit_pipeline(settings=settings, candidate_ids=[candidate.id], dry_run=True)

    assert result["planned"] == 0
    assert result["skipped"] == 1
    assert any("source media missing" in warning for warning in result["warnings"])

    renders = db.list_renders(candidate.id)
    assert len(renders) == 1
    assert renders[0].status == "failed"
    assert renders[0].error is not None and "source media missing" in renders[0].error
    assert not EditPaths.for_candidate(settings, candidate.id).plan.exists()


def test_extraction_failure_is_recorded_not_fatal(tmp_path: Path):
    # A 32-byte non-media file cannot be cut: the batch must survive and record it.
    media = _media_file(tmp_path)
    settings, db, _stream, candidate = _seed(tmp_path, media=media)

    result = run_edit_pipeline(
        settings=settings, candidate_ids=[candidate.id], dry_run=False
    )

    assert result["dry_run"] is False
    assert result["extracted"] == 0
    assert result["planned"] == 0
    assert result["failed"] == 1

    renders = db.list_renders(candidate.id)
    assert len(renders) == 1
    assert renders[0].status == "failed"
    assert renders[0].error is not None and "render failed" in renders[0].error


def test_max_per_run_caps_selection(tmp_path: Path):
    media = _media_file(tmp_path)
    settings, db, _stream, _candidate = _seed(tmp_path, media=media, score=0.5)
    stream_id = db.list_candidates()[0].stream_id
    for index, score in enumerate((0.4, 0.9)):
        db.create_candidate(
            stream_id,
            source_ts=200.0 + index,
            pre_context_seconds=30.0,
            post_context_seconds=30.0,
            signals={"kind": "keyword"},
            score=score,
        )

    result = run_edit_pipeline(
        settings=settings, stream_id=stream_id, status="pending", max_per_run=2, dry_run=True
    )
    assert result["candidates"] == 3
    assert result["selected"] == 2
    assert result["planned"] == 2
    # Highest score wins the capped slots.
    plan_paths = sorted(Path(d).name for d in result["edit_dirs"])
    assert plan_paths == ["1", "3"]


def test_overrides_flow_into_plan(tmp_path: Path):
    media = _media_file(tmp_path)
    settings, _db, _stream, candidate = _seed(tmp_path, media=media)
    run_edit_pipeline(
        settings=settings,
        candidate_ids=[candidate.id],
        dry_run=True,
        overrides=EditOverrides(strategy="gaming", caption_style="block_pop"),
    )
    plan = EditPlan.load(EditPaths.for_candidate(settings, candidate.id).plan)
    assert plan.layout.strategy == "gaming"
    assert plan.layout.resolved_strategy == "gaming"
    assert plan.captions.style == "block_pop"


def test_source_offset_override_is_recorded(tmp_path: Path):
    media = _media_file(tmp_path)
    settings, _db, _stream, candidate = _seed(tmp_path, media=media)
    run_edit_pipeline(
        settings=settings,
        candidate_ids=[candidate.id],
        dry_run=True,
        source_path=media,
        source_offset_seconds=4.25,
    )
    plan = EditPlan.load(EditPaths.for_candidate(settings, candidate.id).plan)
    assert plan.source_offset_seconds == pytest.approx(4.25)


def test_chat_evidence_drives_boundaries(tmp_path: Path):
    media = _media_file(tmp_path)
    settings, _db, _stream, candidate = _seed(
        tmp_path, media=media, source_ts=100.0
    )
    chat = tmp_path / "chat.json"
    chat.write_text(
        json.dumps(
            [
                {"ts": 95.0 + index * 0.5, "user": "chatter", "text": "clip it"}
                for index in range(21)
            ]
        ),
        encoding="utf-8",
    )

    result = run_edit_pipeline(
        settings=settings,
        candidate_ids=[candidate.id],
        dry_run=True,
        chat_path=chat,
    )

    assert result["chat_evidence"] == 21
    assert result["evidence_bounds"] == 1

    plan = EditPlan.load(EditPaths.for_candidate(settings, candidate.id).plan)
    assert plan.bounds.method == "signal_evidence"
    assert plan.bounds.start == pytest.approx(73.0)
    assert plan.bounds.end == pytest.approx(103.0)
    assert plan.bounds.payoff_ts == pytest.approx(103.0)
    assert plan.boundary_evidence is not None
    evidence = plan.boundary_evidence["evidence"]
    assert evidence["chat_burst_end_ts"] == pytest.approx(103.0)
    # Evidence-driven bounds must not be labelled as a Phase 1 placeholder.
    assert WARN_BOUNDARIES_PENDING not in plan.warning_codes()


def test_plan_round_trip_keeps_boundary_evidence(tmp_path: Path):
    media = _media_file(tmp_path)
    settings, _db, _stream, candidate = _seed(tmp_path, media=media, source_ts=100.0)
    chat = tmp_path / "chat.json"
    chat.write_text(
        json.dumps(
            [{"ts": 96.0 + index * 0.5, "user": "c", "text": "clip"} for index in range(11)]
        ),
        encoding="utf-8",
    )
    run_edit_pipeline(
        settings=settings, candidate_ids=[candidate.id], dry_run=True, chat_path=chat
    )
    plan = EditPlan.load(EditPaths.for_candidate(settings, candidate.id).plan)
    assert EditPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_replan_without_chat_keeps_the_chat_derived_bounds(tmp_path: Path):
    media = _media_file(tmp_path)
    settings, _db, _stream, candidate = _seed(tmp_path, media=media, source_ts=100.0)
    chat = tmp_path / "chat.json"
    chat.write_text(
        json.dumps(
            [
                {"ts": 95.0 + index * 0.5, "user": "chatter", "text": "clip it"}
                for index in range(21)
            ]
        ),
        encoding="utf-8",
    )

    run_edit_pipeline(
        settings=settings, candidate_ids=[candidate.id], dry_run=True, chat_path=chat
    )
    paths = EditPaths.for_candidate(settings, candidate.id)
    first = EditPlan.load(paths.plan)
    assert first.bounds.method == "signal_evidence"

    # A render submitted without a chat dump cannot re-derive the cut, so a re-plan must not reset the
    # clip back to the Phase 1 window it was cut away from.
    run_edit_pipeline(settings=settings, candidate_ids=[candidate.id], dry_run=True)

    again = EditPlan.load(paths.plan)
    assert again.bounds.method == "signal_evidence"
    assert again.bounds.start == pytest.approx(first.bounds.start)
    assert again.bounds.end == pytest.approx(first.bounds.end)
    assert again.boundary_evidence == first.boundary_evidence
    assert WARN_BOUNDARIES_PENDING not in again.warning_codes()


def test_replan_from_a_different_source_ignores_the_recorded_bounds(tmp_path: Path):
    media = _media_file(tmp_path)
    settings, _db, _stream, candidate = _seed(tmp_path, media=media, source_ts=100.0)
    chat = tmp_path / "chat.json"
    chat.write_text(
        json.dumps(
            [
                {"ts": 95.0 + index * 0.5, "user": "chatter", "text": "clip it"}
                for index in range(21)
            ]
        ),
        encoding="utf-8",
    )
    run_edit_pipeline(
        settings=settings, candidate_ids=[candidate.id], dry_run=True, chat_path=chat
    )

    # A different capture means the recorded bounds describe footage that is gone, so they must be
    # discarded rather than re-cut positionally onto the new source.
    other = tmp_path / "other.ts"
    other.write_bytes(b"\x00" * 32)
    run_edit_pipeline(
        settings=settings,
        candidate_ids=[candidate.id],
        dry_run=True,
        source_path=other,
    )

    switched = EditPlan.load(EditPaths.for_candidate(settings, candidate.id).plan)
    assert switched.source_path == str(other)
    assert switched.bounds.method == "phase1_window"
    assert WARN_BOUNDARIES_PENDING in switched.warning_codes()


def _job(candidate_id: int, *, score: float, ts: float) -> EditJob:
    candidate = Candidate(
        id=candidate_id,
        stream_id=1,
        source_ts=ts,
        pre_context_seconds=30.0,
        post_context_seconds=30.0,
        signals={},
        score=score,
        media_path=None,
        status="pending",
        created_at="2026-01-01T00:00:00+00:00",
    )
    stream = Stream(
        id=1,
        streamer_id=1,
        mode="vod",
        source_url=None,
        vod_id=None,
        media_path="vod.ts",
        started_at="2026-01-01T00:00:00+00:00",
        created_at="2026-01-01T00:00:00+00:00",
    )
    return EditJob(candidate=candidate, stream=stream, source_path=Path("vod.ts"))


def test_select_edit_jobs_ranks_by_score_then_time():
    jobs = [
        _job(1, score=0.5, ts=10.0),
        _job(2, score=0.9, ts=50.0),
        _job(3, score=0.9, ts=20.0),
    ]
    selected = select_edit_jobs(jobs, max_per_run=2)
    assert [job.candidate.id for job in selected] == [3, 2]


def test_select_edit_jobs_handles_empty_and_zero_cap():
    jobs = [_job(1, score=0.5, ts=10.0)]
    assert select_edit_jobs(jobs, max_per_run=0) == []
    assert select_edit_jobs([], max_per_run=5) == []


def test_resolve_jobs_warns_for_unknown_candidate(tmp_path: Path):
    media = _media_file(tmp_path)
    _settings_obj, db, _stream, candidate = _seed(tmp_path, media=media)
    jobs, warnings = resolve_jobs(db, candidate_ids=[candidate.id, 4242])
    assert [job.candidate.id for job in jobs] == [candidate.id]
    assert any("4242" in warning for warning in warnings)


def test_resolve_jobs_warns_for_stream_without_media(tmp_path: Path):
    _settings_obj, db, _stream, candidate = _seed(tmp_path, media=None)
    jobs, warnings = resolve_jobs(db, candidate_ids=[candidate.id])
    assert jobs == []
    assert any("no source media" in warning for warning in warnings)


def test_cli_dry_run_end_to_end(tmp_path: Path, capsys):
    media = _media_file(tmp_path)
    settings, _db, _stream, candidate = _seed(tmp_path, media=media)
    config = tmp_path / "cli-config.yaml"
    config.write_text(f'data_dir: "{tmp_path.as_posix()}"\n', encoding="utf-8")

    from clippy.cli import edit_main

    edit_main(["--candidate", str(candidate.id), "--dry-run", "--config", str(config)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["planned"] == 1
    assert EditPaths.for_candidate(settings, candidate.id).plan.exists()


def test_cli_requires_a_selector(tmp_path: Path):
    from clippy.cli import edit_main

    with pytest.raises(SystemExit):
        edit_main(["--dry-run"])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)
def test_render_path_produces_captions_end_to_end(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.resolved_db_path())
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(
        streamer.id, "vod", media_path=str(FIXTURE), source_width=640, source_height=360
    )
    candidate = db.create_candidate(
        stream.id,
        source_ts=120.0,
        pre_context_seconds=30.0,
        post_context_seconds=30.0,
        signals={"kind": "keyword"},
        score=0.9,
    )
    transcript = TranscriptPayload(
        text="clip this",
        language="en",
        duration=2.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="clip this",
                words=[TranscriptWord(0.0, 0.5, "CLIP"), TranscriptWord(0.6, 1.0, "this")],
            )
        ],
    )
    monkeypatch.setattr(
        "clippy.edit.captions.transcribe_words", lambda *a, **k: transcript
    )

    result = run_edit_pipeline(
        settings=settings, candidate_ids=[candidate.id], dry_run=False
    )

    assert result["extracted"] == 1
    assert result["captioned"] == 1
    assert result["failed"] == 0

    paths = EditPaths.for_candidate(settings, candidate.id)
    assert paths.base.exists() and paths.trimmed.exists()
    assert paths.captions.exists() and paths.transcript.exists()
    assert paths.layout.exists() and paths.vertical.exists() and paths.final.exists()

    finals = [row for row in db.list_renders(candidate.id) if row.kind == "final"]
    assert len(finals) == 1
    assert finals[0].captions_path == str(paths.captions)
    assert finals[0].width == 1080
    assert finals[0].height == 1920
    assert finals[0].is_current == 1

    plan = EditPlan.load(paths.plan)
    assert plan.stage == "complete"
    assert plan.captions.cue_count > 0
    assert plan.captions.reason is None
    assert plan.layout.width == 1080 and plan.layout.height == 1920
    assert plan.layout.layers  # the layout layers are recorded for review
    assert plan.layout.resolved_strategy != "auto"
    # Composition resolves the strategy, so the pre-composition warning must be gone.
    assert WARN_LAYOUT_PENDING not in plan.warning_codes()
    # 640x360 cannot support a tracked crop, so framing is fit_blur and the band stays low.
    assert plan.layout.resolved_strategy == "fit_blur"
    assert plan.captions.anchor == "bottom"
    assert plan.audio.applied is True

    reloaded = db.get_candidate(candidate.id)
    assert reloaded is not None
    assert reloaded.edit_status == "rendered"
    assert reloaded.edited_media_path == str(paths.final)

    # A re-run with identical inputs must reuse the composed video rather than re-encode it.
    composed_at = paths.vertical.stat().st_mtime_ns
    final_at = paths.final.stat().st_mtime_ns
    again = run_edit_pipeline(settings=settings, candidate_ids=[candidate.id], dry_run=False)
    assert again["failed"] == 0
    assert paths.vertical.stat().st_mtime_ns == composed_at
    assert paths.final.stat().st_mtime_ns == final_at

    # Changing only the caption style makes the cached render stale, so the new style reaches
    # the deliverable without `--force` - and therefore without paying for ASR again.
    restyled = run_edit_pipeline(
        settings=settings,
        candidate_ids=[candidate.id],
        dry_run=False,
        overrides=EditOverrides(caption_style="block_pop"),
    )
    assert restyled["failed"] == 0
    assert paths.vertical.stat().st_mtime_ns != composed_at
    assert paths.final.stat().st_mtime_ns != final_at

    restyled_plan = EditPlan.load(paths.plan)
    assert restyled_plan.captions.style == "block_pop"
    style_line = next(
        line
        for line in paths.captions.read_text(encoding="utf-8").splitlines()
        if line.startswith("Style: Caption")
    )
    assert style_line.split(",")[2] == "60"  # block_pop's font size, not karaoke_highlight's 54


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not FIXTURE.exists(),
    reason="needs ffmpeg and samples/sample_vod.mp4",
)
def test_auto_captions_follow_the_resolved_framing(monkeypatch, tmp_path: Path):
    """
    `caption_safe_area: auto` reads the frame, so framing is resolved before captions.

    A facecam box makes `auto` resolve to `gaming`, whose panel owns the bottom of the
    canvas - so the caption band has to move up instead of sitting on the face.
    """
    settings = _settings(tmp_path, facecam_box="0.7,0.6,0.3,0.4")
    db = Database(settings.resolved_db_path())
    streamer = db.get_or_create_streamer("tester", "Tester")
    stream = db.create_stream(
        streamer.id, "vod", media_path=str(FIXTURE), source_width=640, source_height=360
    )
    candidate = db.create_candidate(
        stream.id,
        source_ts=120.0,
        pre_context_seconds=30.0,
        post_context_seconds=30.0,
        signals={"kind": "keyword"},
        score=0.9,
    )
    transcript = TranscriptPayload(
        text="clip this",
        language="en",
        duration=2.0,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="clip this",
                words=[TranscriptWord(0.0, 0.5, "CLIP"), TranscriptWord(0.6, 1.0, "this")],
            )
        ],
    )
    monkeypatch.setattr(
        "clippy.edit.captions.transcribe_words", lambda *a, **k: transcript
    )

    result = run_edit_pipeline(
        settings=settings, candidate_ids=[candidate.id], dry_run=False
    )

    assert result["failed"] == 0
    paths = EditPaths.for_candidate(settings, candidate.id)
    plan = EditPlan.load(paths.plan)
    assert settings.caption_safe_area == "auto"
    assert plan.layout.resolved_strategy == "gaming"
    assert plan.captions.anchor == "top"

    # The burned track really is the top band (ASS alignment 8), not just a plan field.
    style_line = next(
        line
        for line in paths.captions.read_text(encoding="utf-8").splitlines()
        if line.startswith("Style: Caption")
    )
    assert style_line.split(",")[18] == "8"


