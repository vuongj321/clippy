from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from clippy.config import Settings, get_settings
from clippy.store.db import REJECTION_REASONS, Database

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
TEMPLATES = Jinja2Templates(directory=str(UI_DIR / "templates"))


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
        return TEMPLATES.TemplateResponse(
            request,
            "candidate.html",
            {
                "view": view,
                "stats": stats,
                "reasons": REJECTION_REASONS,
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

    return app
