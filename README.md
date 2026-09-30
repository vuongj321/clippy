# Clippy

Detect clip-worthy moments from Twitch streams (chat + audio), extract review windows, collect
human approve/reject labels, and render those moments as vertical clips for approval.

## Requirements

- Python 3.12+
- [FFmpeg](https://ffmpeg.org/) on `PATH` (`ffmpeg` + `ffprobe`) for audio analysis and window extraction
- [Streamlink](https://streamlink.github.io/) on `PATH` for live mode **and** HQ VOD capture (`uv tool install streamlink`)
- For live mode only: Twitch IRC credentials in `.env`
- `uv sync` also installs `opencv-python-headless`, used only by `layout_track_backend: opencv` (a
  bundled face cascade — no model download). Nothing else imports it.

## Setup

```bash
uv sync
copy config.example.yaml config.yaml
copy .env.example .env
```

## VOD pipeline

Provide a local media file and chat JSON (timestamps in **stream-relative seconds**):

```bash
uv run clippy-vod --media path\to\vod.mp4 --chat samples\chat_sample.json --streamer somechannel
uv run clippy-serve
```

Open http://127.0.0.1:8000 to review candidates.

Chat JSON may be a list of `{ "ts", "user", "text" }` objects, or `{ "messages": [...] }` / `{ "comments": [...] }`. Keyword hits use whole-phrase / word-boundary matching (`clip` does not match `clippers`).

Set `CLIPPY_OPENAI_API_KEY` in `.env` to generate a short UI caption and transcript on **new** runs. After extracts finish, only the top `caption_max_per_run` extracted clips (default 20, highest score first) get speech-to-text plus a caption. Disk-skipped rows and lower-ranked extracts keep `extract_reason` only. There is no later pass and rerunning does not backfill. These captions are review-UI and export labels: the detection pass never touches the media. The
render step writes its own burned-in caption track (see below). The run summary JSON includes
`annotated`.

## Live pipeline

```bash
# Set CLIPPY_TWITCH_IRC_NICK and CLIPPY_TWITCH_IRC_OAUTH in .env
uv run clippy-live --channel somechannel --duration 300
uv run clippy-serve
```

Records via Streamlink, collects IRC chat on the same wall-clock timeline, then runs the same detect/extract/annotate path.

## Vertical clips

`clippy-edit` turns a candidate into a publishable 1080x1920 clip — reframed, with burned-in
captions, normalized audio and metadata — and stops for a human to approve. Render from the HQ
capture rather than the 160p review window:

```bash
# 1. Capture the source VOD at high quality (160p review windows are too soft to export).
uv run clippy-capture --vod <vod-id-or-url> --quality best \
    --chat samples/chat_sample.json --record-stream 1 --prune

# 2. Render one candidate, a whole stream's candidates, or every pending one. --dry-run writes
#    plan.json + a renders row only: no ffmpeg, no network. --chat supplies boundary evidence
#    (stream-relative timestamps).
uv run clippy-edit --candidate 5 --chat path\to\chat.json
uv run clippy-edit --all-pending --top 10
uv run clippy-edit --candidate 5 --dry-run

# 3. Serve the review UI.
uv run clippy-serve
```

Each candidate's edit lives in `data/edits/{candidate_id}/` and starts with `plan.json`: final
clip bounds, dead-air map, layout, captions, audio targets and any warnings. `renders` rows in
SQLite track every revision.

Stages run in order and cache: `planned → extracted → trimmed → captioned → composed → complete`. A
re-run redoes only what changed: composition checks whether the pixels, the caption text, the
framing and the encode settings still match the video it finds, and the loudness pass refuses a
`final.mp4` that is older than the `vertical.mp4` it came from. `--force` rebuilds every stage,
including transcription.

Useful flags: `--dry-run`, `--force`, `--strategy auto|fit_blur|irl|gaming|conversation`,
`--caption-style karaoke_highlight|block_pop|minimal`, `--caption-emphasis heuristic|llm|off`,
`--deadair-mode cut|speed`, `--no-captions`, `--keep-intermediate`, `--source <file>` and
`--source-offset`.

Notes:

- `clippy-capture` verifies that the captured media's clock matches the chat clock (stream-relative
  seconds) and reports any offset instead of silently mis-cutting. Chat-vs-audio alignment only
  *warns*; the authoritative offset comes from correlating known clip material against the capture,
  so a bad chat guess can never shift every render. Use `--source-offset` to force one, or
  `--no-align` to skip verification.
- `clippy-edit --source <file>` re-cuts a candidate against any capture without re-running
  detection, because `candidates.source_ts` is stream-relative.
- `--dry-run` is honest about what has not happened yet: without chat evidence the bounds are the
  fixed detection window (`boundaries_pending`), and a source that was never probed leaves its
  dimensions unknown (`source_resolution_unknown`).

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
  plan.json            full decision record (bounds, dead air, layout, captions, audio, metadata, warnings)
  compose.inputs.json  fingerprint of what the composed video was made from
  base.mp4             exact source window
  trimmed.mp4          dead air removed
  captions.ass         caption track burned into the vertical render
  transcript.json      word timings
  layout.json          vertical layout segments
  vertical.mp4         1080x1920 with captions
  final.mp4            loudness-normalized deliverable
  metadata.json        title/description/hashtags/thumbnail time
  thumbnail.jpg        cover frame
```

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

See `config.example.yaml`.

Detection knobs: `pre_context_seconds`, `post_context_seconds`, `coalesce_gap_seconds`, chat spike
multiplier, audio spike multiplier, `disk_budget_gb`, `asr_model`, `caption_model`,
`caption_max_per_run`, `caption_max_chat_messages`, `openai_base_url`.

Editing knobs: `clip_target_width`/`clip_target_height`/`clip_fps`, `layout_strategy`,
`layout_track_backend`, `layout_smoothing`, `quality_warn_upscale`, `render_crf`, `render_preset`,
`caption_style`, `caption_emphasis`, `caption_*` sizing knobs, `deadair_*`,
`audio_normalize`/`audio_target_lufs`/`audio_true_peak`/`audio_limiter`, `metadata_enabled`,
`metadata_title_max_chars`, `metadata_max_hashtags`, `thumbnail_enabled`, `thumbnail_overlay_text`.

**What each editing setting actually does** — the layout strategies, the caption style presets, the
word-emphasis modes and the caption band — is documented in
[Changing the edit: the three surfaces](docs/architecture.md#changing-the-edit-the-three-surfaces).

**Known gaps:** dead air is cut without a speech veto, `conversation` splits the frame rather than
tracking who is speaking, and a vertical crop can only reframe a source that is taller than 9:16 —
see [Known gaps](docs/architecture.md#known-gaps) for why, and what a real fix would need.

