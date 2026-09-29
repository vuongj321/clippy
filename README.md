# Clippy

Detect clip-worthy moments from Twitch streams (chat + audio), extract review windows, and collect human approve/reject labels.

## Requirements

- Python 3.12+
- [FFmpeg](https://ffmpeg.org/) on `PATH` (`ffmpeg` + `ffprobe`) for audio analysis and window extraction
- [Streamlink](https://streamlink.github.io/) on `PATH` for live mode **and** HQ VOD capture (`uv tool install streamlink`)
- For live mode only: Twitch IRC credentials in `.env`

## Setup

```bash
uv sync
copy config.example.yaml config.yaml
copy .env.example .env
```

## VOD pipeline (primary)

Provide a local media file and chat JSON (timestamps in **stream-relative seconds**):

```bash
uv run clippy-vod --media path\to\vod.mp4 --chat samples\chat_sample.json --streamer somechannel
uv run clippy-serve
```

Open http://127.0.0.1:8000 to review candidates.

Chat JSON may be a list of `{ "ts", "user", "text" }` objects, or `{ "messages": [...] }` / `{ "comments": [...] }`. Keyword hits use whole-phrase / word-boundary matching (`clip` does not match `clippers`).

Set `CLIPPY_OPENAI_API_KEY` in `.env` to generate a short UI caption and transcript on **new** runs. After extracts finish, only the top `caption_max_per_run` extracted clips (default 20, highest score first) get speech-to-text plus a caption. Disk-skipped rows and lower-ranked extracts keep `extract_reason` only. There is no later pass and rerunning does not backfill. Captions are shown in the review UI only — they are not burned into the MP4. The run summary JSON includes `annotated`.

## Live pipeline

```bash
# Set CLIPPY_TWITCH_IRC_NICK and CLIPPY_TWITCH_IRC_OAUTH in .env
uv run clippy-live --channel somechannel --duration 300
uv run clippy-serve
```

Records via Streamlink, collects IRC chat on the same wall-clock timeline, then runs the same detect/extract/annotate path.

## Phase 2 — automatic short-form edit

Phase 2 turns a reviewed candidate into a publishable vertical clip. Phase 2's own plan lives in
`.cursor/plans/clippy_phase2_auto_edit.plan.md`; M0 (edit skeleton) and M1 (HQ capture) are in
place. The encoding stages land in M2-M11, so `clippy-edit` currently runs planning only.

```bash
# 1. Capture the source VOD at high quality (160p review windows are too soft to export).
uv run clippy-capture --vod <vod-id-or-url> --quality best \
    --chat samples/chat_sample.json --record-stream 1 --prune

# 2. Plan (and later render) edits for candidates. --dry-run writes plan.json + a renders row
#    only: no ffmpeg, no network. --chat supplies boundary evidence (stream-relative timestamps).
uv run clippy-edit --candidate 5 --dry-run --chat path\to\chat.json
uv run clippy-edit --all-pending --top 10

# 3. Serve the review UI (unchanged from Phase 1)
uv run clippy-serve
```

Each candidate's edit lives in `data/edits/{candidate_id}/` and starts with `plan.json`: final
clip bounds, dead-air map, layout strategy, caption style, audio targets and any quality warnings.
`renders` rows in SQLite track every revision.

Notes:

- `clippy-capture` verifies that the captured media's clock matches the chat clock (stream-relative
  seconds) and reports any offset instead of silently mis-cutting. Use `--source-offset` to force
  one, or `--no-align` to skip verification.
- `clippy-edit --source <file>` re-cuts a candidate against any capture without re-running
  detection, because `candidates.source_ts` is stream-relative.
- `--dry-run` is honest about what has not happened yet: plans carry `boundaries_pending` and
  `source_resolution_unknown` warnings until M2 records real boundaries and dims.

## Export / eval

```bash
uv run clippy-export
```

Writes `data/exports/reviews.json` and `reviews.csv`, and prints approve-rate stats.

## Layout

```text
src/clippy/     application package
scripts/        thin CLI wrappers
samples/        example chat dump
data/           SQLite DB + extracted media (gitignored)
docs/           architecture notes
```

## Config

See `config.example.yaml`. Important knobs: `pre_context_seconds`, `post_context_seconds`, `coalesce_gap_seconds`, chat spike multiplier, audio spike multiplier, `disk_budget_gb`, `asr_model`, `caption_model`, `caption_max_per_run`, `caption_max_chat_messages`, `openai_base_url`.

---

# Phase 2 — vertical clip rendering

Phase 1 finds candidate moments. Phase 2 turns them into publishable 1080x1920 clips with
captions, normalized audio, and metadata — then stops for a human to approve.

## Capture the source at HQ

```bash
uv run clippy-capture --stream 1          # streamlink at `best` + yt-dlp fallback
uv run clippy-capture --stream 1 --verify-offset   # correlate clips against the capture
```

Chat-vs-audio alignment only *warns*; the authoritative offset comes from correlating known
clip material against the capture, so a bad chat guess can never shift every render.

## Render one candidate, or all pending ones

```bash
uv run clippy-edit --candidate 5 --chat data/chat/2876956941.json
uv run clippy-edit --all-pending --top 5
```

Useful flags: `--dry-run`, `--force`, `--strategy fit_blur|irl|gaming|conversation`,
`--caption-style karaoke_highlight|block_pop|minimal`, `--caption-emphasis heuristic|off`,
`--deadair-mode off|cut|speed`, `--no-captions`, `--keep-intermediate`.

Stages run in order and cache: `planned → extracted → trimmed → captioned → composed →
complete`. A re-run only redoes what is missing or invalid.

## Review and re-render

```bash
uv run clippy-serve        # http://127.0.0.1:8000
```

Each candidate page plays the finished clip, shows the chosen boundaries, the dead air that was
removed, the layout (strategy, upscale factor, layer list), the caption cues and the warnings the
render produced. From there you can edit the metadata, download `final.mp4`, or re-render with a
different layout/caption style.

## Artifacts

```text
data/edits/<candidate_id>/
  plan.json        full decision record (bounds, dead air, layout, audio, metadata, warnings)
  base.mp4         exact source window
  trimmed.mp4      dead air removed
  captions.ass     burned-in caption track
  transcript.json  word timings
  layout.json      vertical layout segments
  vertical.mp4     1080x1920 with captions
  final.mp4        loudness-normalized deliverable
  metadata.json    title/description/hashtags/thumbnail time
  thumbnail.jpg    cover frame
```

## Config

Phase 2 adds: `clip_target_width`/`clip_target_height`/`clip_fps`, `layout_strategy`,
`layout_track_backend`, `layout_smoothing`, `quality_warn_upscale`, `render_crf`,
`render_preset`, `caption_style`, `caption_emphasis`, `caption_*` sizing knobs, `deadair_*`,
`audio_normalize`/`audio_target_lufs`/`audio_true_peak`/`audio_limiter`, `metadata_enabled`,
`metadata_title_max_chars`, `metadata_max_hashtags`, `thumbnail_enabled`, `thumbnail_overlay_text`.

**Known gap:** dead air is cut without a speech veto, and `conversation` splits the frame rather
than tracking who is speaking — see section 19 of `docs/architecture.md` for why, and what a real
fix would need.
