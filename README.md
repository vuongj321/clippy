# Clippy

Detect clip-worthy moments from Twitch streams (chat + audio), extract review windows, and collect human approve/reject labels.

## Requirements

- Python 3.12+
- [FFmpeg](https://ffmpeg.org/) on `PATH` (`ffmpeg` + `ffprobe`) for audio analysis and window extraction
- For live mode: [Streamlink](https://streamlink.github.io/) on `PATH`, plus Twitch IRC credentials in `.env`

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

Chat JSON may be a list of `{ "ts", "user", "text" }` objects, or `{ "messages": [...] }` / `{ "comments": [...] }`.

Set `CLIPPY_OPENAI_API_KEY` in `.env` to generate a short UI caption and transcript on **new** runs. After extracts finish, only the top `caption_max_per_run` extracted clips (default 20, highest score first) get speech-to-text plus a caption. Disk-skipped rows and lower-ranked extracts keep `extract_reason` only. There is no later pass and rerunning does not backfill. Captions are shown in the review UI only — they are not burned into the MP4.

## Live pipeline

```bash
# Set CLIPPY_TWITCH_IRC_NICK and CLIPPY_TWITCH_IRC_OAUTH in .env
uv run clippy-live --channel somechannel --duration 300
uv run clippy-serve
```

Records via Streamlink, collects IRC chat on the same wall-clock timeline, then runs detect/extract.

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

See `config.example.yaml`. Important knobs: `pre_context_seconds`, `post_context_seconds`, `coalesce_gap_seconds`, chat spike multiplier, audio spike multiplier, `disk_budget_gb`, `asr_model`, `caption_model`, `caption_max_per_run`.
