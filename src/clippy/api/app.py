from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from clippy.config import Settings, get_settings
from clippy.edit.pipeline import run_edit_pipeline
from clippy.edit.plan import (
    CAPTION_EMPHASIS_MODES,
    CAPTION_STYLES,
    STRATEGIES,
    EditOverrides,
    EditPaths,
    EditPlan,
    MetadataPlan,
)
from clippy.store.db import REJECTION_REASONS, Database

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
TEMPLATES = Jinja2Templates(directory=str(UI_DIR / "templates"))


def _normalize_tag(raw: str) -> str:
    """Human-typed hashtags are forgiving: strip decoration, collapse spaces."""
    token = raw.strip().lstrip("#").lower().replace(" ", "").replace("-", "")
    return "".join(ch if (ch.isalnum() or ch == "_") else "" for ch in token)


def _caption_preview(ass_path: Path, *, limit: int = 6) -> list[str]:
    """Readable cue text out of the ASS track, for the review page."""
    import re

    if not ass_path.exists():
        return []
    preview: list[str] = []
    tag_re = re.compile(r"\{[^}]*\}")
    for line in ass_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        text = line.split(",", 9)[-1]
        cleaned = tag_re.sub("", text).replace("\\N", " ").strip()
        if cleaned:
            preview.append(cleaned)
        if len(preview) >= limit:
            break
    return preview


def _resolve_chat_path(raw: str) -> Path | None:
    """
    Optional chat evidence from the review form.

    A path the reviewer typed that is not on disk is a 400 naming it, not a 500. Anything that is
    there but not a chat dump is rejected by `load_chat_json` and reaches the client the same way.
    """
    cleaned = raw.strip().strip("\"'")
    if not cleaned:
        return None
    path = Path(cleaned).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Chat JSON not found: {path}")
    return path


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    db = Database(settings.resolved_db_path())

    app = FastAPI(title="Clippy Review", version="0.1.0")
    app.state.db = db
    app.state.settings = settings

    static_dir = UI_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, status: str = "pending") -> HTMLResponse:
        if status not in ("pending", "approved", "rejected", "all"):
            status = "pending"
        status_filter = None if status == "all" else status  # type: ignore[assignment]
        views = db.list_candidate_views(status=status_filter)  # type: ignore[arg-type]
        stats = db.stats()
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "candidates": views,
                "stats": stats,
                "status": status,
                "reasons": REJECTION_REASONS,
            },
        )

    @app.get("/candidates/{candidate_id}", response_class=HTMLResponse)
    def candidate_detail(request: Request, candidate_id: int) -> HTMLResponse:
        views = db.list_candidate_views(status=None, limit=10_000)
        view = next((v for v in views if v.candidate.id == candidate_id), None)
        if not view:
            raise HTTPException(status_code=404, detail="Candidate not found")
        stats = db.stats()

        # `POST /candidates/{id}/render` cuts from the stream's source capture, so the page has to
        # know whether that file is still there: without it there is nothing to create, and the
        # pipeline would only record a failed render.
        stream = db.get_stream(view.candidate.stream_id)
        source_path = Path(stream.media_path) if stream and stream.media_path else None
        source_ready = bool(source_path and source_path.exists())
        last_render = next(iter(db.list_renders(candidate_id, limit=1)), None)

        paths = EditPaths.for_candidate(settings, candidate_id)
        plan = None
        if paths.plan.exists():
            try:
                plan = EditPlan.load(paths.plan)
            except Exception:  # a corrupt plan must not break the review page
                plan = None
        return TEMPLATES.TemplateResponse(
            request,
            "candidate.html",
            {
                "view": view,
                "stats": stats,
                "reasons": REJECTION_REASONS,
                "render": db.get_current_render(candidate_id),
                "plan": plan,
                "last_render": last_render,
                "source_path": str(source_path) if source_path else None,
                "source_ready": source_ready,
                "clip_width": settings.clip_target_width,
                "clip_height": settings.clip_target_height,
                "caption_preview": _caption_preview(paths.captions),
                "strategies": STRATEGIES,
                "caption_styles": CAPTION_STYLES,
                "caption_emphasis_modes": CAPTION_EMPHASIS_MODES,
            },
        )

    @app.post("/candidates/{candidate_id}/review")
    def review(
        candidate_id: int,
        decision: str = Form(...),
        reason_code: str = Form(""),
        notes: str = Form(""),
    ) -> RedirectResponse:
        if decision not in ("approved", "rejected"):
            raise HTTPException(status_code=400, detail="Invalid decision")
        reason = reason_code or None
        if decision == "approved":
            reason = None
        db.review_candidate(
            candidate_id,
            decision,  # type: ignore[arg-type]
            reason_code=reason,
            notes=notes or None,
        )
        return RedirectResponse(url="/", status_code=303)

    @app.get("/media/{candidate_id}")
    def media(candidate_id: int) -> FileResponse:
        candidate = db.get_candidate(candidate_id)
        if not candidate or not candidate.media_path:
            raise HTTPException(status_code=404, detail="Media not found")
        path = Path(candidate.media_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Media file missing on disk")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/stats")
    def api_stats() -> dict:
        return db.stats()

    @app.post("/candidates/{candidate_id}/render")
    def render_candidate(
        candidate_id: int,
        strategy: str = Form(""),
        caption_style: str = Form(""),
        caption_emphasis: str = Form(""),
        chat_path: str = Form(""),
        force: str = Form(""),
    ) -> RedirectResponse:
        """
        Create or re-render one candidate (the human escape hatch).

        A candidate that has never been planned is cut from scratch: `run_edit_pipeline` writes
        `plan.json` and renders it, falling back to the Phase 1 detection window unless a chat dump
        is supplied. Options left unset stay at the `config.yaml` defaults.
        """
        if db.get_candidate(candidate_id) is None:
            raise HTTPException(status_code=404, detail="Candidate not found")
        overrides = EditOverrides(
            strategy=strategy or None,
            caption_style=caption_style or None,
            caption_emphasis=caption_emphasis or None,
        )
        chat = _resolve_chat_path(chat_path)
        try:
            run_edit_pipeline(
                settings=settings,
                candidate_ids=[candidate_id],
                dry_run=False,
                force=bool(force),
                overrides=overrides,
                chat_path=chat,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=303)

    @app.post("/candidates/{candidate_id}/metadata")
    def update_metadata(
        candidate_id: int,
        title: str = Form(""),
        description: str = Form(""),
        hashtags: str = Form(""),
    ) -> RedirectResponse:
        """Persist human edits; the plan file stays the source of truth."""
        if db.get_candidate(candidate_id) is None:
            raise HTTPException(status_code=404, detail="Candidate not found")
        paths = EditPaths.for_candidate(settings, candidate_id)
        if not paths.plan.exists():
            raise HTTPException(status_code=404, detail="No plan for this candidate")

        plan = EditPlan.load(paths.plan)
        tags = [_normalize_tag(item) for item in hashtags.split(",") if item.strip()]
        plan.metadata = MetadataPlan(
            enabled=plan.metadata.enabled,
            source="manual",
            title=title.strip() or None,
            description=description.strip() or None,
            hashtags=[tag for tag in tags if tag],
            thumbnail=plan.metadata.thumbnail,
            reason="edited by a reviewer",
        )
        plan.save(paths.plan)

        current = db.get_current_render(candidate_id)
        if current is not None:
            with db.connection() as conn:
                conn.execute(
                    "UPDATE renders SET metadata_json = ? WHERE id = ?",
                    (json.dumps(plan.metadata.to_dict()), current.id),
                )
        return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=303)

    @app.get("/renders/{render_id}/media")
    def render_media(render_id: int) -> FileResponse:
        render = db.get_render(render_id)
        if render is None or not render.path:
            raise HTTPException(status_code=404, detail="Render not found")
        path = Path(render.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Render file missing on disk")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/candidates/{candidate_id}/download")
    def download_final(candidate_id: int) -> FileResponse:
        render = db.get_current_render(candidate_id)
        if render is None or not render.path:
            raise HTTPException(status_code=404, detail="No finished render yet")
        path = Path(render.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Render file missing on disk")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    return app
