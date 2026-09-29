---
name: Clippy Phase 2 automatic short-form editing
overview: "Turn Phase 1 candidate events into publishable vertical short-form clips: capture HQ VODs, decide clip boundaries, remove dead air, generate synced captions, reframe to 1080x1920, normalize audio, generate metadata, and keep a human approve/reject gate before manual publishing."
todos:
  - id: m0-edit-skeleton
    content: EditPlan schema, data/edits + data/source artifact dirs, renders table + edit_status + stream source columns, new Settings fields, clippy-edit --dry-run
    status: completed
  - id: m1-hq-capture
    content: "HQ VOD capture (streamlink/yt-dlp), timeline-offset verification, chat reuse, --source re-cut path, source retention/prune, install docs, low-res source badge"
    status: completed
  - id: m2-boundaries
    content: Context analysis + start/end determination with hard non-cutting constraints and optional clamped LLM refinement
    status: completed
  - id: m3-extract
    content: Cut base.mp4 from the HQ source at final boundaries with duration assertion
    status: completed
  - id: m4-deadair
    content: Dead-air removal (silencedetect intersected with word gaps), protected main region, segment map in plan.json
    status: completed
  - id: m5-captions
    content: Word-level ASR timings, cue alignment, ASS writer with emphasis and configurable style presets
    status: pending
  - id: m6-vertical
    content: Vertical layouts (irl/gaming/conversation/fit_blur) + numpy subject tracking + filter graph builder
    status: pending
  - id: m7-composition
    content: Compose the 1080x1920 canvas and burn captions in a single encode
    status: pending
  - id: m8-audio
    content: Two-pass loudnorm measurement + limiter audio normalization to short-form targets
    status: pending
  - id: m9-metadata
    content: Title, description, hashtags, thumbnail with schema validation and deterministic fallback
    status: pending
  - id: m10-review
    content: Review UI/API for renders, metadata editing, re-render with overrides, download, new rejection reasons, export columns
    status: pending
  - id: m11-docs-smoke
    content: architecture.md/README/config.example.yaml updates, integration test, end-to-end HQ smoke run, deliberate-gaps list
    status: pending
isProject: false
---

# Clippy Phase 2 — Automatic Short-Form Editing

## Goal

Turn Phase 1's candidate events into polished, publishable vertical short-form videos
automatically, while keeping the human in the loop:

```text
Candidate -> Automatic Edit -> Human Review -> Approve / Reject
```

The deliverable is that a human can review a **completely edited** short-form video generated
from a livestream moment and publish it manually.

Framing inherited from Phase 1 and unchanged here: **score is detection confidence, not
clippability**. Phase 2 does not decide what is entertaining; it produces the best possible edit
of a moment a human flagged.

## Success criteria

Given a candidate event, the system must:

1. Determine final clip boundaries.
2. Extract the source footage.
3. Remove unnecessary material.
4. Generate captions.
5. Convert the video to vertical format.
6. Create a suitable composition.
7. Normalize audio.
8. Generate metadata.
9. Produce a final publishable video.

Traceability (criterion -> stage -> artifact -> gate):

| # | Criterion | Stage | Artifact | Gate |
| - | --------- | ----- | -------- | ---- |
| 0 | Source quality (enabling) | `clippy-capture` + `ingest/align.py` | `data/source/*.ts`, `streams.source_*` | M1 |
| 1 | Final clip boundaries | `edit/boundaries.py` | `plan.json` (`bounds`) | M2 |
| 2 | Extract source footage | `edit/render.py` + `extract_window` | `base.mp4` | M3 |
| 3 | Remove unnecessary material | `edit/deadair.py` | `trimmed.mp4`, `plan.deadair` | M4 |
| 4 | Captions | `caption/{asr,align,ass,styles}.py` | `captions.ass` | M5 |
| 5 | Vertical format | `edit/{layouts,track}.py` | `layout.json` | M6 |
| 6 | Composition | `edit/render.py` | `vertical.mp4` | M7 |
| 7 | Audio normalization | `edit/audio.py` | `final.mp4` | M8 |
| 8 | Metadata | `edit/metadata.py` | `metadata.json`, `thumbnail.jpg` | M9 |
| 9 | Publishable final video | `edit/pipeline.py` + API/UI | `final.mp4` | M11 |

## Locked decisions

1. **Capture at the highest available quality and edit from that.** Verified in this workspace:
   the only source VOD is `...TN-160p.ts` (284x160 @30, h264, aac 48 kHz stereo, 22211.8 s,
   811 MB) and every Phase 1 candidate is a 284x160 / 60 s cut. A true 9:16 crop of a 284x160
   frame is 90x160, so a 1080x1920 render would be a ~12x upscale. The 160p material is
   therefore **smoke-test fixture only**; real renders come from an HQ capture.
2. **An HQ source needs no re-detection.** `candidates.source_ts` is stream-relative seconds and
   a full VOD download starts at t=0, so existing scores, chat, extract reasons and review
   labels stay valid against the HQ file. Re-running Phase 1 detection is optional.
3. **Staged artifacts with idempotency.** Expensive work (ASR, LLM calls, encodes) is cached on
   disk and only redone with `--force`.
4. **Degrade, never drop.** A clip with no captions or a failed layout is still reviewable and
   downloadable; a metadata failure never blocks a render (metadata is an optimization layer).
5. **No new heavy runtime dependencies by default.** ASR and metadata reuse the existing
   OpenAI-compatible `httpx` path; framing/tracking is numpy-only with optional lazily imported
   adapters.
6. **Every knob lives in `Settings`** (`CLIPPY_` prefix, `config.yaml` override, documented in
   `config.example.yaml`), matching Phase 1 conventions.

## Prerequisites (verified state of this machine)

| Tool | Status | Action |
| ---- | ------ | ------ |
| `ffmpeg` / `ffprobe` | present, 9.0.2 full build (libass, drawtext, crop, scale, overlay, zscale, loudnorm, alimiter, silencedetect, h264_nvenc, whisper filter) | none |
| `streamlink` >= 6 | **missing** | `uv tool install streamlink` (PATH-visible binary, matches the existing `shutil.which` design) or `winget install --id Streamlink.Streamlink` |
| `yt-dlp` | **missing** | optional alternative downloader backend; `--write-chat` can also supply VOD chat JSON |
| Python 3.12 + `uv` | present (uv 0.12.0) | none |
| Burn-in fonts | present (`C:\Windows\Fonts\arialbd.ttf`, `impact.ttf`, `segoeuib.ttf`) | pass `fontsdir` |
| Package deps | numpy 2.5.3, httpx 0.28.1, fastapi 0.141.1, jinja2, pydantic-settings, pytest 9.1.1 | no new required deps |

## Pipeline

```text
VOD capture (HQ)
   -> Candidate (existing source_ts)
   -> Context Analysis
   -> Determine Start/End
   -> Extract Video
   -> Dead-Air Removal
   -> Caption Generation
   -> Vertical Formatting
   -> Composition
   -> Audio Processing
   -> Metadata Generation
   -> Final Clip
   -> Human Review (Approve / Reject)
   -> manual publish
```

Ordering note that matters: ASR runs on the **trimmed** clip, after dead-air removal, so
caption, layout and metadata timings already exist on the final timeline and need no remapping.
Boundary detection is the only stage that works in source time.

## Module layout (extends the existing one-way dependency style)

```text
src/clippy/
  edit/                      # NEW: orchestration + the edit itself
    __init__.py              # re-exports, same style as caption/__init__.py
    plan.py                  # EditPlan / ClipBounds / DeadAirPlan / LayoutPlan dataclasses + JSON round-trip
    boundaries.py            # context analysis + start/end determination (the "cut here" logic)
    deadair.py               # silence + word-gap detection -> keep segments / speed map
    layouts.py               # strategy registry: fit_blur | irl | gaming | conversation
    track.py                 # numpy motion-centroid tracking (+ lazily imported optional backends)
    audio.py                 # loudnorm measure pass + final filter arg builder
    metadata.py              # title/description/hashtags/thumbnail (LLM + deterministic fallback)
    render.py                # FilterGraph builder + single final encode
    pipeline.py              # run_edit_pipeline(...) -> Phase-1-style summary dict
  caption/                   # EXTEND existing package
    asr.py                   # + transcribe_words() (verbose_json + timestamp_granularities)
    align.py                 # NEW: words -> caption cues
    ass.py                   # NEW: ASS writer (styles, karaoke emphasis, escaping)
    styles.py                # NEW: style presets
  ingest/
    capture.py               # NEW: VOD download via streamlink/yt-dlp
    align.py                 # NEW: timeline-offset estimation between chat and audio
  api/app.py                 # + render / metadata / download routes
  ui/templates/              # + final clip, plan summary, metadata form, re-render form
  eval/export.py             # + render columns
```

Dependency direction stays intact: `cli -> pipeline -> {ingest, chat, audio, buffer, detect,
extract, caption, edit, store}` and `api -> {store, config}`. `edit/boundaries.py`,
`edit/deadair.py`, `caption/align.py`, `caption/ass.py` and `edit/metadata.py` must stay pure
logic (no ffmpeg, no network) so they are unit-testable without a media toolchain.

## Artifact layout

`data/edits/{candidate_id}/` (new `Settings.resolved_edits_dir()`, created in `ensure_dirs()`):

```text
plan.json        # EditPlan: bounds + evidence, dead-air map, layout, caption cfg, audio targets, warnings
base.mp4         # source range cut at the final boundaries
trimmed.mp4      # after dead-air removal (the timeline every later stage refers to)
transcript.json  # ASR segments + word timings (produced from trimmed.mp4)
captions.ass     # generated subtitles
layout.json      # per-keyframe crop/pan + layer rectangles
vertical.mp4     # composed 1080x1920 with captions burned (kept with --keep-intermediate)
final.mp4        # the publishable deliverable
thumbnail.jpg
metadata.json
```

`data/source/` holds captured HQ VODs (`{channel}_{vod_id}.ts` + optional `.chat.json`), with
`source_budget_gb` retention and `--prune`.

## Data model changes (`store/db.py`, additive)

New tables and columns, using the same idempotent-create + migration-tuple pattern as
`CANDIDATE_COLUMN_MIGRATIONS` (new `STREAM_COLUMN_MIGRATIONS` tuple for streams):

```sql
renders(
  id, candidate_id, revision, kind 'rough'|'final', path,
  plan_json, transcript_json, captions_path, layout_json, metadata_json,
  width, height, duration, status 'ok'|'failed', error, is_current, created_at
)
```

- `candidates.edit_status ('unrendered'|'rendering'|'rendered'|'failed')` and
  `candidates.edited_media_path` as denormalized copies for a fast review queue (mirrors the
  documented `status`-is-denormalized convention in `docs/architecture.md` section 8).
- `streams.source_width`, `source_height`, `source_fps`, `capture_quality`,
  `source_offset_seconds`, `source_bytes` so the UI can badge a low-res or offset source.
- `REJECTION_REASONS` gains `bad_edit` and `bad_captions` so edit-quality failures stay
  separable from content-quality failures.
- `renders.is_current` is a single-row invariant per candidate, switched inside one transaction.

## HQ capture and timeline verification (M1)

New CLI surface (thin wrapper in `scripts/`, entry point in `pyproject.toml`, matching
`clippy-vod`/`clippy-live`/`clippy-serve`/`clippy-export`):

```bash
clippy-capture --vod <vod-url-or-id> --quality best -o data/source/{channel}_{vod}.ts \
               [--downloader streamlink|yt-dlp] [--chat-out data/source/{channel}.chat.json] \
               [--source-offset 0.0] [--prune]
```

- `ingest/capture.py` builds the argv (pure function, unit-testable) and runs the downloader as a
  subprocess, exactly like `ingest/live.py` already does with streamlink. Streamlink backend:
  `streamlink <vod-url> <quality> -o <out> --stream-segment-threads 4 --retry-streams 5 --retry-max -1`.
  `-o` is required because streamlink VOD output is not seekable-free; `best` gives 1080p60 for
  most channels (1440p where the streamer pushes it).
- Missing binary raises `RuntimeError` naming the tool and the install command, matching the
  existing failure-message convention documented in `architecture.md` section 13.
- **Timeline contract:** a full VOD download starts at VOD t=0, so `candidates.source_ts`,
  `streams` chat and Phase 1 labels align with no arithmetic. Partial or offset captures are
  supported through `--source-offset` / `capture_source_offset_seconds`.
- `ingest/align.py: estimate_timeline_offset(chat_activity, rms_curve, *, search_seconds)` uses
  numpy cross-correlation of the chat-activity curve against the audio RMS curve and returns the
  lag that maximizes correlation. If `abs(lag) > alignment_tolerance_seconds`, the plan records a
  `timeline_offset` warning and applies the correction - never silently mis-cuts. This is pure
  numpy, so it is unit-testable with synthetic curves and a known injected lag.
- Chat handling: reuse `samples`-style chat JSON, an existing stream's chat, or
  `yt-dlp --write-chat` output. No new chat parsing is needed beyond what `chat/models.py`
  already tolerates.
- Retention: `source_budget_gb` with `--prune` (delete the source once every render for that
  stream is `ok`, or oldest-first when over budget).
- Recommended flow: `clippy-capture ...` then `clippy-edit --candidate N --source data/source/<hq>.ts`.
  Re-running Phase 1 detection against the HQ file is optional (it would change audio thresholds
  slightly and lose label continuity, so it is not the default).
- The existing stream row (284x160, quality `160p`) is badged **low-res source (smoke fixture)**
  in the UI; integration tests use it as a local, network-free fixture.
- Deferred alternative for future runs: a `--start/--duration` capture option backed by yt-dlp's
  `--download-sections`, for grabbing a single moment instead of a whole 6 h VOD. It would record
  the section start as `source_offset_seconds` automatically. Not needed while the full VOD is
  available, so it stays out of M1's scope.

## Stage 1-2: Context analysis and boundary determination (`edit/boundaries.py`)

Evidence timeline, all built from data Phase 1 already stores plus the window transcript:

| Element | Source | Use |
| ------- | ------ | --- |
| Signal peak | `candidates.source_ts` + `signals.events` | anchor for `main_ts` |
| Chat rate curve | `detect_chat_signals` windowing over the chat slice | reaction/payoff duration, burst decay |
| RMS curve | `audio.intensity.compute_rms_series` over the search window | payoff/reaction, loud-moment snapping |
| Utterances and words | ASR word timings for the search window | hook, payoff, mid-word protection |

Algorithm (deterministic core; the LLM is opt-in polish):

1. `main_ts` = signal peak, extended to the end of the utterance overlapping it (never cut
   mid-sentence).
2. `payoff_ts` = the last of {main utterance end, chat burst decay below baseline + margin, RMS
   decay}, capped by `reaction_tail_seconds`.
3. `hook_ts` = walk back to the last utterance boundary or silence >= `boundary_min_silence_seconds`,
   bounded by `hook_lookback_seconds` and floored at `main_ts - min_context_seconds`
   (never cut before necessary context).
4. `end_ts` = first natural boundary at or after payoff + tail, clamped to
   `[clip_min_seconds, clip_max_seconds]`.
5. **Constraint pass** encodes the "do not cut" rules as hard, individually tested invariants:
   no split inside a word (snap to a word gap >= `word_gap_min_seconds`), never start after the
   first clause of the setup utterance, never end before payoff + tail, never start after payoff,
   and min/max duration clamps.
6. Optional `boundary_llm_refine: true` sends transcript segments + chat context (via the
   existing `build_chat_context`) + a signal summary to one chat completion, expecting strict
   JSON `{hook, main, payoff, end, rationale}`. The response is validated and re-clamped through
   step 5; any failure or out-of-bounds value falls back to the deterministic result, following
   the never-raise pattern in `caption/generate.py`. Default off until measured.

## Stage 3: Extraction (`edit/render.py` + `extract/ffmpeg_cut.py`)

- Reuse `extract_window` against the HQ source to produce `base.mp4`, with the same
  `-c:v libx264 -preset veryfast -crf 23 -c:a aac -movflags +faststart` re-encode rationale
  already documented in `architecture.md` section 7 (stream-copy seeks drift on VODs).
- Add an ffprobe assertion via `audio.intensity.probe_duration_seconds`: if the produced duration
  deviates by more than `extract_duration_tolerance_seconds`, record a plan warning and clamp the
  plan to the actual media instead of shipping a silently short clip.
- Candidate row is unchanged in Phase 2 (Phase 1's rough window stays the review anchor); the
  final render path is tracked on `renders`.

## Stage 4: Dead-air removal (`edit/deadair.py`)

- `ffmpeg -af silencedetect=noise={deadair_noise_db}dB:d={deadair_min_gap_seconds}` on
  `base.mp4`, parsing stderr with the same subprocess idiom as `audio/intensity.py`.
- Intersect with ASR word gaps: a silence is only cuttable if no word straddles it.
- Keep segments = complement of cuttable gaps, padded by `deadair_keep_pad_seconds` so word
  onsets are not clipped; keeps shorter than `deadair_min_keep_seconds` are merged away.
- **Protected region**: nothing between `main_ts` and `payoff_ts` (+ `reaction_tail_seconds`) is
  ever removed.
- `deadair_mode: cut` (default; `trim/atrim` + `concat` in one re-encode) or `speed` (compress
  gaps with `setpts/atempo` to preserve flow and audible continuity).
- Emits the segment map into `plan.json`, so the UI and `clippy-export` can show
  `removed_seconds` and "how much filler was dropped".
- Degrade path: if `silencedetect` yields nothing or ASR is unavailable, `trimmed.mp4` is a copy
  of `base.mp4` and the plan records `deadair: {applied: false, reason: ...}`.

## Stage 5: Captions (`caption/asr.py`, `align.py`, `ass.py`, `styles.py`)

- Extend `caption/asr.py` with `transcribe_words()` using the existing OpenAI-compatible
  `/audio/transcriptions` call but `response_format=verbose_json` and
  `timestamp_granularities[]=["word","segment"]`. If the endpoint rejects word granularity
  (server-side `whisper-1` limits), retry with segments only and interpolate word times
  proportionally inside each segment. `transcribe_media()` stays byte-for-byte compatible so
  Phase 1 tests and the existing UI caption pass keep working.
- Configurable offline escape hatch, documented but not default: ffmpeg's `whisper` filter with a
  local ggml model (`asr_provider: api|ffmpeg_whisper`); the ffmpeg build here has the filter.
- `caption/align.py` turns words into cues with rules: `caption_max_chars_per_line`,
  `caption_max_lines`, `caption_max_cue_seconds`, `caption_min_cue_seconds`,
  `caption_break_gap_seconds`, sentence-punctuation breaks, cue boundaries snapped to word
  edges, and a guarantee that cues are monotonic and non-overlapping.
- Emphasis (`caption_emphasis: heuristic|llm|off`): `heuristic` (default) flags ALL-CAPS words,
  numbers, `chat_keywords` hits, an emotion list and content words via a small stoplist; `llm`
  asks the existing chat-completions endpoint for words to highlight and validates every returned
  word is a substring of the transcript; `off` disables.
- `caption/ass.py` writes `captions.ass` with libass karaoke `\k` tags so the style's
  `SecondaryColour` is the base color and `PrimaryColour` is the highlight, i.e. words light up as
  they are spoken. Fallback if a libass karaoke glitch shows up: split one event per word. All
  text is escaped for ASS (`{`, `}`, `\`, and newlines), and emoji/emote-only text is dropped.
- `caption/styles.py` ships presets (`karaoke_highlight`, `block_pop`, `minimal`) with font, size,
  colors, outline, shadow, uppercase and `caption_margin_v` all configurable.
- **Avoid covering important visual information:** the layout stage exposes a per-frame subject
  `content_box`; if the caption band intersects it in more than `caption_avoid_ratio` of frames,
  the anchor flips between bottom and top bands (`caption_safe_area: auto|top|middle|bottom`).
- Burn-in technique: `captions.ass` lives in the artifact dir and libass is invoked with a
  **relative filename while `cwd` is set to that directory**, plus `fontsdir=C:/Windows/Fonts`.
  This dodges the classic Windows ffmpeg colon-escaping trap in `ass=`/`subtitles=` filters.

## Stage 6: Vertical formatting (`edit/layouts.py`, `edit/track.py`)

Target `clip_target_width x clip_target_height` = `1080 x 1920` (9:16), `clip_fps` (default 30,
`0` = follow source), lanczos scaling, `-pix_fmt yuv420p`.

| Strategy | Behavior | Requirement |
| -------- | -------- | ----------- |
| `fit_blur` | full frame scaled to fit, blurred/cropped copy as background fill | works at any resolution; the honest fallback for the 160p fixtures |
| `irl` | tracked crop of a `source_h x 9/16` region, subject held near center, smoothed pan, zoom clamp | needs >= ~720p to be worth it |
| `gaming` | gameplay crop plus facecam slot (PiP or stacked), optional dynamic layout switching | `facecam_box` config or a persistent-corner motion heuristic |
| `conversation` | two-panel split or active-speaker follow from per-region audio energy + motion | needs >= 2 stable motion regions |

- `layout_strategy: auto` chooses from evidence (motion-region count, facecam presence, source
  resolution, per-streamer override) and **falls back to `fit_blur`** when
  `upscale_factor > quality_warn_upscale`. The factor is written into `plan.json` warnings and
  shown in the UI, so a bad render is visible, not silent.
- Note for expectations: even from a 1080p16:9 source, a 9:16 crop to 1080x1920 is a ~1.8x
  upscale; that is industry-normal, handled with lanczos plus mild `unsharp`. Only 1440p/4K
  capture makes it near-native. This is a documented tradeoff, not a bug.
- `edit/track.py`: ffmpeg samples frames to raw grayscale (~6 fps) -> numpy motion energy map plus
  center-bias prior -> weighted centroid -> exponential/one-euro smoothing -> bounded crop
  offsets. Backends are `none|motion|opencv|mediapipe` behind a lazily imported adapter so the core
  stays numpy-only and importable without CV packages.
- `layout.json` is a serializable `LayoutPlan` (layers with time-varying source/destination
  rectangles and a z-order). Both `edit/render.py` and the review UI (manual override + re-render)
  consume the same structure.
- Manual escape hatches for mis-framing: `crop_bias`, `facecam_box` and `zoom` overrides settable
  per render from the UI or CLI.

## Stage 7: Composition (`edit/render.py`)

- One `FilterGraph` builder emits a `filter_complex` (via a script file rather than a giant argv
  string): background fill -> layer scale/crop/pan -> overlay chain -> `ass` caption burn ->
  `format=yuv420p`.
- Single final encode: `libx264 -preset veryfast -crf 20 -r {clip_fps} -c:a aac -b:a 192k
  -movflags +faststart`, consistent with `extract_window`'s choices. `render_encoder: auto|x264|nvenc`
  with a runtime capability probe (this ffmpeg build exposes `h264_nvenc`).
- `vertical.mp4` is retained only with `--keep-intermediate`; `final.mp4` is the deliverable.
- Render budget: `render_disk_budget_gb` (a 30-45 s 1080x1920 clip is roughly 20-40 MB) plus
  pruning of superseded revisions (`is_current = 0`) before failing a render on disk pressure.

## Stage 8: Audio processing (`edit/audio.py`)

- Two-pass EBU R128: measure with `loudnorm=...:print_format=json -f null -`, parse
  `input_i/input_tp/input_lra/input_thresh`, then apply
  `loudnorm=I=-14:TP=-1.5:LRA=11:measured_*` plus an `alimiter` guard; 48 kHz stereo output.
- Targets come from `audio_target_lufs` (default -14, the short-form platform norm),
  `audio_true_peak` (-1.5) and `audio_limiter`; `audio_normalize: false` = passthrough for
  comparison renders.
- Measured and achieved values are stored in `plan.json` and asserted by an integration test
  (within +/- 1 LU of target).
- Degrade path: a measurement failure logs, warns, and falls back to single-pass `loudnorm`.

## Stage 9: Metadata generation (`edit/metadata.py`)

- Output: `{title, description, hashtags[], thumbnail_time}` written to `metadata.json`.
- LLM path reuses the `caption/generate.py` httpx + JSON-payload pattern and the configured model.
  Validation is strict: title <= `metadata_title_max_chars`, hashtags 3..`metadata_max_hashtags`
  normalized (`#` prefix, lowercase, no spaces), description length-capped, and any proper noun or
  hashtag that does not appear in the transcript/chat/streamer identity is **rejected** (anti-
  hallucination, mirroring the existing caption prompt discipline).
- Deterministic fallback when there is no API key or the call fails: title from the Phase 1
  `caption` else the strongest utterance; hashtags from the streamer handle plus top n-grams of
  chat/transcript filtered by a stoplist; description assembled from the first two sentences.
- Thumbnail: the frame at the peak motion+RMS timestamp via `ffmpeg -ss t -frames:v 1 -q:v 3`;
  optional `drawtext` overlay using `C:/Windows/Fonts/arialbd.ttf`, default off.
- Explicitly an optimization layer: `edit/pipeline.py` never fails a render because metadata
  failed, and `plan.json` records `metadata: {applied: false, reason: ...}`.

## Stage 10: Human review (`store/db.py`, `api/app.py`, `ui/`)

- API additions (plain HTML forms, no JavaScript, matching the existing UI style):
  - `POST /candidates/{id}/render` - form fields for `strategy`, `caption_style`,
    `caption_emphasis`, `crop_bias`, `zoom`, `deadair_mode`, `force`; runs synchronously (a 30-45 s
    clip is tens of seconds) and writes a new `renders` revision.
  - `POST /candidates/{id}/metadata` - edit title/description/hashtags.
  - `GET /renders/{id}/media` - serve a specific revision.
  - `GET /candidates/{id}/download` - `final.mp4` as an attachment.
- `ui/templates/candidate.html` gains: the vertical `<video>` for the current render, a plan
  summary (hook/main/payoff/end timestamps, `removed_seconds`, strategy, `upscale_factor` and any
  warnings), a caption preview, the editable metadata form, re-render and download actions, and the
  existing Approve / Reject forms.
- `ui/templates/index.html` gains an `edit_status` + strategy badge, the low-res source badge, and a
  "rendered / unrendered" filter alongside the existing status filter.
- Review semantics: approve/reject still labels the moment, now against the edited artifact. The new
  `bad_edit` and `bad_captions` reasons keep edit-quality failures separable from content-quality
  failures, so the Phase 1 approve-rate metric stays comparable while the edit loop gets its own
  signal.
- `GET /api/stats` and `db.stats()` gain render counts; `eval/export.py` adds `edit_status`,
  `edited_media_path`, `render_duration`, `removed_seconds`, `metadata` fields, and
  `precision_report` gains edit-stage counters.

## CLI surface

```bash
# HQ source
clippy-capture --vod <vod-url-or-id> --quality best -o data/source/chan_vod.ts [--prune]

# one candidate, one stream batch, or the whole pending queue
clippy-edit --candidate 5 --source data/source/chan_vod.ts
clippy-edit --stream 1 --top 10
clippy-edit --all-pending --max-per-run 20
clippy-edit --candidate 5 --dry-run            # plan.json + renders row only, no encode
clippy-edit --candidate 5 --force --strategy gaming --caption-style block_pop
clippy-edit --candidate 5 --no-captions --keep-intermediate
```

- Both are added to `[project.scripts]` in `pyproject.toml` with thin `scripts/` wrappers, matching
  `clippy-vod`.
- `clippy-edit` prints a Phase-1-style JSON summary
  (`{rendered, failed, skipped, warnings, edit_dir}`) and reuses the `_select_annotation_jobs`
  ranking pattern (`-score, source_ts, id`) for batch selection with `edit_max_per_run`.
- `--dry-run` is the M0 gate: plan + DB row, no ffmpeg, no network.

## Configuration additions (`Settings` + `config.example.yaml`)

| Group | Keys |
| ----- | ---- |
| Capture | `source_dir`, `capture_quality`, `capture_downloader`, `capture_source_offset_seconds`, `source_budget_gb`, `alignment_tolerance_seconds` |
| Bounds | `clip_min_seconds`, `clip_max_seconds`, `clip_target_seconds`, `boundary_search_seconds`, `hook_lookback_seconds`, `min_context_seconds`, `reaction_tail_seconds`, `boundary_min_silence_seconds`, `word_gap_min_seconds`, `boundary_llm_refine`, `extract_duration_tolerance_seconds` |
| Dead air | `deadair_enabled`, `deadair_mode`, `deadair_noise_db`, `deadair_min_gap_seconds`, `deadair_keep_pad_seconds`, `deadair_min_keep_seconds` |
| Captions | `caption_enabled`, `caption_style`, `caption_font`, `caption_font_size`, `caption_max_chars_per_line`, `caption_max_lines`, `caption_max_cue_seconds`, `caption_min_cue_seconds`, `caption_break_gap_seconds`, `caption_emphasis`, `caption_primary_color`, `caption_highlight_color`, `caption_margin_v`, `caption_safe_area`, `caption_avoid_ratio`, `caption_uppercase`, `asr_provider`, `asr_word_timestamps` |
| Vertical | `clip_target_width`, `clip_target_height`, `clip_fps`, `layout_strategy`, `layout_track_backend`, `layout_smoothing`, `layout_zoom`, `facecam_box`, `quality_warn_upscale` |
| Render | `render_dir`, `render_crf`, `render_preset`, `render_encoder`, `render_disk_budget_gb`, `edit_max_per_run`, `keep_intermediate` |
| Audio | `audio_normalize`, `audio_target_lufs`, `audio_true_peak`, `audio_limiter` |
| Metadata | `metadata_enabled`, `metadata_model`, `metadata_max_hashtags`, `metadata_title_max_chars`, `thumbnail_enabled`, `thumbnail_overlay_text` |

`Settings.ensure_dirs()` creates `data/edits` and `data/source`; a new
`Settings.resolved_edits_dir()` sits beside `resolved_media_dir()`/`resolved_buffer_dir()`.
Existing keys (`pre_context_seconds`, `post_context_seconds`, `disk_budget_gb`) keep their Phase 1
meaning so nothing regresses.

## Milestones and gates

| # | Milestone | Deliverable | Gate |
| - | --------- | ----------- | ---- |
| M0 | Edit skeleton | `edit/plan.py`, `data/edits` + `data/source` dirs, `renders` table + `edit_status` + stream source columns, `Settings` fields, `clippy-edit --dry-run` | `uv run pytest -q` green (34 existing + new); dry-run writes `plan.json` + a `renders` row for candidate 5 with no ffmpeg/network |
| M1 | HQ capture and source upgrade | `clippy-capture`, `ingest/align.py`, `--source`/`--source-offset` re-cut path, chat reuse, source retention + `--prune`, install docs, low-res source badge | argv-builder and offset-estimation unit tests pass (synthetic curves recover an injected lag within 0.5 s); a real VOD capture plus `clippy-edit --candidate 5 --source <hq>` yields a clip from the HQ file |
| M2 | Boundary detection | `edit/boundaries.py` deterministic core + constraint pass + optional clamped LLM refinement | unit tests: mid-word protection, context floor, payoff tail, min/max clamps, start-after-payoff rejection, unknown-signal safe default |
| M3 | Extraction | `base.mp4` from the HQ source | ffprobe duration within `extract_duration_tolerance_seconds` of the plan |
| M4 | Dead-air removal | `trimmed.mp4` + `plan.deadair` segment map | synthetic-silence tests; protected main region untouched; no segment below the min-keep floor |
| M5 | Captions | `transcribe_words`, `align.py`, `ass.py`, `styles.py` | cue-rule and ASS-escaping tests; captions.ass renders correctly in a player |
| M6 | Vertical formatting | `layouts.py`, `track.py`, graph builder | crop windows stay in frame for extreme centroids; layer rects tile the canvas without overlap; 3 s synthetic clip renders at 1080x1920 |
| M7 | Composition | `vertical.mp4` with burned captions | a real candidate renders; captions legible and in sync; caption anchor flip works |
| M8 | Audio processing | normalized `final.mp4` | measured loudness within +/- 1 LU of `audio_target_lufs`; no clipping (`alimiter`) |
| M9 | Metadata | `metadata.json` + `thumbnail.jpg` | fallback works with no API key; caps enforced; invented names/hashtags rejected |
| M10 | Human review | API/UI render + metadata + download, new reasons, export columns | 5 real candidates reviewed end to end in `clippy-serve` |
| M11 | Docs and end-to-end smoke | `docs/architecture.md` Phase 2 section, README, `config.example.yaml`, integration test, deliberate-gaps list | reviewer downloads `final.mp4`, copies the metadata, and publishes manually |

Build order is the milestone order. Each milestone keeps the existing suite green and adds its own
tests before moving on. M0 and M1 come first because every later gate needs an HQ source to be
meaningful.

## Testing plan

- New flat test modules, mirroring `tests/test_core.py` / `tests/test_caption.py` conventions (no
  fixtures directory, no pytest config, plain functions):
  - `tests/test_edit_boundaries.py` - boundary constraints and clamps.
  - `tests/test_edit_deadair.py` - keep-segment math, padding, protection, idempotence.
  - `tests/test_edit_captions.py` - cue grouping, interpolation, monotonicity, ASS escaping and
    emphasis tags.
  - `tests/test_edit_layouts.py` - crop bounds, smoothing, layer tiling, caption anchor flip.
  - `tests/test_edit_audio_metadata.py` - loudnorm arg builder, metadata caps, fallback, entity
    rejection.
  - `tests/test_edit_store.py` - `renders` round-trip, `is_current` switching, `edit_status`
    denormalization, legacy-DB migration (extends the existing legacy-migration test pattern).
  - `tests/test_ingest_capture_align.py` - downloader argv builder, missing-binary error, offset
    estimation on synthetic curves.
- Integration tests are gated on tool availability (`shutil.which("ffmpeg")`) and on the local 160p
  fixture in `data/media`; capture integration tests skip unless a VOD URL is provided via env, so
  `uv run pytest -q` stays offline and fast.
- Manual smoke sequence after M11:

```bash
uv tool install streamlink
uv run clippy-capture --vod <vod-id> --quality best -o data/source/chan_vod.ts
uv run clippy-edit --candidate 5 --source data/source/chan_vod.ts
uv run clippy-serve           # review the vertical clip, edit metadata, approve/reject
uv run clippy-export          # includes render columns
```

## Risks and mitigations

| Risk | Mitigation |
| ---- | ---------- |
| `streamlink`/`yt-dlp` not installed | M1 prerequisite step; `RuntimeError` naming the binary and install command |
| Long capture time and 8-12 GB per 1080p60 VOD | `source_budget_gb`, `--prune`, and re-cut-from-existing-candidates so a re-download is never needed for a re-render |
| Offset/partial capture would mis-cut silently | correlation-based offset check with an explicit warning plus correction |
| Even 1080p needs a ~1.8x crop upscale for 9:16 | lanczos + mild `unsharp`, `quality_warn_upscale` surfaced in the plan and UI, gaming/conversation layouts reduce it, 1440p capture makes it near-native |
| Tracking is a heuristic with no face ground truth | smoothing + zoom clamp, `crop_bias`/`facecam_box`/`zoom` overrides with one-click re-render, no auto-approval |
| LLM drift in boundaries, emphasis, metadata | strict schema validation, clamping through the deterministic constraint pass, defaults off/fallback on, never load-bearing |
| Windows ffmpeg filter-path quoting | relative ASS filename + `cwd` + `fontsdir`, documented in architecture.md |
| CPU cost of 1080x1920 encodes | `veryfast` + `crf 20`, `render_encoder: auto` (this build has `h264_nvenc`), `edit_max_per_run` batching |
| Disk growth from revisions and renders | `render_disk_budget_gb`, prune superseded revisions (`is_current = 0`) before failing on disk pressure |
| Caption/audio desync | ASR runs on the post-dead-air, post-speed artifact, so all downstream timings share one timeline (this is why the pipeline order matters) |

## Out of scope for this phase

- Publishing/uploading to any platform (manual publish only, by design).
- Learned ranking from performance metrics.
- Continuous streaming edit (Phase 2 edits finished candidates from a captured VOD).
- Face/emotion/scene CV models as hard dependencies (available only as optional adapters).
- Auto-approval and any autonomous quality judgement.
- Twitch Clip API usage.

## Status

**M0-M11 are complete. Full suite: 239 passing.** M5-M11 all ran against the real VOD (candidate 5) as well as the fixture.

## M5-M11 delivered

- **M5 captions** - `caption/asr.py::transcribe_words` (`verbose_json` + word/segment granularity, segment-only retry, segment interpolation fallback, top-level `words` folding), `caption/align.py` (`Cue`, `build_cues` with character/duration/gap/sentence breaks, `_normalise` merge+stretch+clamp, `pick_emphasis_words`), `caption/styles.py` (`karaoke_highlight`/`block_pop`/`minimal` presets + `Settings` overrides + `caption_anchor`), `caption/ass.py` (escaping, `H:MM:SS.cc` timestamps, `\k` karaoke and recoloured-plain modes, `write_ass`), `edit/captions.py::generate_captions` (cached transcript, style/emphasis resolution, degrade paths), pipeline stage `captioned`, `renders.captions_path`.
- **M6 layouts + tracking** - `edit/track.py` (`centroid_from_energy` with centre prior, `smooth_track`, `resample_track`, ffmpeg motion-profile decode, `track_subject`), `edit/layouts.py` (`LayoutLayer`/`LayoutSegment`/`CompositionPlan`, `crop_rect`, `resolve_strategy` with low-res downgrade, `segments_from_track` deadband+capped spans, `_fit_layers`/`_irl_layers`/`_gaming_layers`/`_conversation_layers`, `plan_layout`, upscale-factor warnings).
- **M7 composition** - `edit/render.py::build_composition_filter` (per-segment canvas + layers + overlay + concat + `ass` burn, audio trimmed in step) and `compose_vertical` (track -> layout -> `layout.json` -> single-pass `vertical.mp4`, plan summary + warnings, length check).
- **M8 audio** - `edit/audio.py::parse_loudnorm_output`, `build_loudnorm_filter` (measured two-pass with `linear=true`, `alimiter` guard), `measure_loudness`, `normalize_audio` (`-c:v copy`, single-pass fallback, never unnormalized).
- **M9 metadata** - `edit/metadata.py::ClipMetadata`, `normalize_hashtag`, `build_hashtags`, `validate_metadata` (drops hashtags the transcript/chat cannot support, caps title/hashtag counts), `fallback_metadata`, `generate_metadata` (LLM + validation + fallback with reasons), `capture_thumbnail` (frame grab, optional `drawtext` overlay), `metadata.json`.
- **M10 review** - `api/app.py`: candidate page now renders the finished clip, boundaries, dead air, layout and caption cues, plus `POST /candidates/{id}/render` (overrides + force), `POST /candidates/{id}/metadata`, `GET /renders/{id}/media`, `GET /candidates/{id}/download`; `ui/` templates gained the render badge, re-render form and metadata editor.
- **M11 docs** - `docs/architecture.md` sections 17-20 (stage table, decisions, deliberate gaps, testing), `README.md` Phase 2 workflow/artifacts/config, `config.example.yaml` completed.

### Verified against the real VOD

- M5: Groq returned 200 for word granularity; **13 cues** written to `data/edits/5/captions.ass`, plan stage `captioned`.
- M6-M9: the pipeline test renders the fixture end-to-end (`final.mp4` at 1080x1920 with burned captions, loudness-normalized, `metadata.json` + `thumbnail.jpg`).

### Bugs found by real data (and fixed)

1. **Groq accepts `timestamp_granularities` but returns no word timings** (0 words, 9 segments). The interpolation fallback kept every cue in sync; `_attach_top_level_words` now also folds timings returned beside `segments` instead of discarding them.
2. **`clip that` split into tokens** emphasised every "that" in the clip; multi-word chat keywords no longer leak their parts.
3. **Windows drive colon in `fontsdir` broke the whole filter graph** ("No option name near '/Windows/Fonts'"). Verified empirically against ffmpeg 9.0.2 - a *double* backslash is required - and `escape_filter_path` now does that.
4. **`normalize_hashtag` deleted the characters it meant to keep** (a `re.sub` with a keep-pattern), which silently produced empty tags; caught by the metadata tests.


## M4 delivered

- `src/clippy/edit/deadair.py` - span math (`merge_spans`, `complement_spans`, `subtract_span`, `shrink_spans`), `parse_silences`/`detect_silences` (ffmpeg `silencedetect`, dangling trailing silence closed at the clip duration), `plan_deadair` (speech veto, payoff protection, guard band, minimum gap/keep, total-removal cap, `cut`/`speed` modes), `build_deadair_filter` (single-pass `trim`/`atrim` + `concat`), `segments_for_render`.
- `edit/render.py` - `apply_deadair` writes `trimmed.mp4`, records the segment map/`removed_seconds`/reason on the plan, keeps the filter graph to streams that exist, and reports a length mismatch instead of hiding it.
- `edit/pipeline.py` - the render path runs extract -> dead air, marks the plan stage `trimmed`, records a `rough` render row for `trimmed.mp4`, and reports `deadair_removed_seconds`.
- `extract/ffmpeg_cut.extract_window` - optional `crf`/`preset`; intermediates now use `intermediate_crf` (16) so the final encode is not a second lossy generation.
- Tests: 14 pure tests in `tests/test_edit_deadair.py` plus 3 real-media tests (cut, speed mode, nothing-cuttable copy). Full suite: 160 passing.

Two real bugs caught by the tests: `merge_spans` admitted a zero-length span after clipping negatives, and the removal cap's tie-break depended on float ordering (now deterministic, restoring the earlier gap so opening context survives).

## M1 delivered and verified against the real VOD

- Capture: `data/source/2876956941.ts` - **19.18 GB, 1920x1080 @60, 22211.77 s** (matches the VOD length exactly), 1080p60 for a 6.2 h stream, ~8.1 MB/s, ~39 min.
- Stream 1 recorded with `capture_quality=best`, real dimensions/fps/bytes and `source_offset_seconds=0.0`, so `clippy-edit` needs no `--source` flag.
- Chat dump parsed: 127,553 messages, ts 0 -> 22210, 31 active 1-minute bins in the first 30 min (well above `MIN_ACTIVE_BINS`).

### Bug found on real data (and fixed)

The chat-vs-audio estimator reported `+30.0s` (score 0.14) and `clippy-capture` **applied** it, writing a wrong offset onto the stream row - which would have shifted every clip by 30 seconds. Verification against ground truth (correlating Phase 1's 60 s clips, cut from the original source, against the same nominal range of the HQ capture) showed the true offset is **0 s** at scores of 0.94-0.99 across five candidates.

Fixes:

1. `capture_main` no longer applies a chat-derived offset; that check may only warn. Only `--source-offset` or a confident `verify_offset_with_clips` result is applied.
2. New `verify_offset_with_clips` in `ingest/align.py` - RMS-envelope correlation of identical audio content (median of confident clips), with `OffsetVerification.is_confident()`. It is decisive in a way chat-vs-audio correlation never can be, because chat bursts and loud moments are both spiky.
3. The recorded `source_offset_seconds` was corrected back to 0 for stream 1.
4. Tests cover both directions: recovering a 0 s offset and **detecting an injected 30 s shift** on real audio.

## M3 delivered

- `src/clippy/edit/render.py` - `extract_base`: cached, duration-verified cut at the planned bounds, clamps the plan to the media that actually exists and records an `extract_short` warning.
- `edit/pipeline.py` - `dry_run=False` now cuts `base.mp4`, marks the plan stage `extracted`, and records an extraction failure on the render row instead of aborting the batch; the summary gained `extracted`.
- `tests/test_edit_render.py` + pipeline tests - 6 new tests (real ffmpeg cuts against the local fixture, early-source clamping, caching, zero-length and missing-source guards).

## M2 delivered

- `src/clippy/edit/boundaries.py` - `Word`/`Utterance`/`ContextEvidence`/`BoundaryEvidence`/`BoundaryDecision`, `words_to_utterances`, `speech_gaps`, `adjust_start`/`adjust_end` (never mid-word), `chat_burst_end_ts`, `audio_decay_ts`, `enforce_constraints`, `detect_bounds`, `apply_llm_bounds` + `refine_bounds_with_llm` (injected, clamped), `evidence_from_chat`.
- `edit/plan.py` - plans carry `boundary_evidence` (evidence + every adjustment); `build_plan` accepts evidence-driven bounds and only flags `boundaries_pending` when they are missing or fell back.
- `edit/pipeline.py` - `run_edit_pipeline(..., chat_path=...)` builds chat evidence per candidate; summary gains `chat_evidence` and `evidence_bounds`.
- `cli.py` - `clippy-edit --chat <json>`.
- `tests/test_edit_boundaries.py` - 33 tests.

Verified: candidate 5 planned from the real 127,553-message chat dump now yields `start 360.0 / end 390.0 / 30.0s` with `method: signal_evidence` and `chat_burst_end_ts: 390.0`, where before it was a 45 s Phase 1 window. Full suite: 129 passing.

Three real bugs the tests caught and fixed: the reaction-tail cap could truncate a sentence still being spoken; the "hook" rule picked the main utterance instead of the setup line; the duration ceiling could pull the start past the main event (now floored at `main - min_context`).

## M0 delivered

- `src/clippy/edit/plan.py` - `EditPlan`/`ClipBounds`/`DeadAirPlan`/`LayoutPlan`/`CaptionsPlan`/`AudioPlan`/`MetadataPlan`, `EditOverrides`, `EditPaths`, `phase1_bounds`, `build_plan`.
- `src/clippy/edit/pipeline.py` - `run_edit_pipeline(..., dry_run=True)` (plan + `renders` row, no ffmpeg/network), `resolve_jobs`, `select_edit_jobs`.
- `store/db.py` - `renders` table, `candidates.edit_status`/`edited_media_path`, `streams.source_*`, generalized `TABLE_MIGRATIONS`, `Render` dataclass, `bad_edit`/`bad_captions` rejection reasons.
- `cli.py` + `pyproject.toml` + `scripts/run_edit.py` - the `clippy-edit` entry point.
- `tests/test_edit_plan.py`, `tests/test_edit_store.py`, `tests/test_edit_pipeline.py` - 36 new tests.
- `config.example.yaml` + `.gitignore` updates.

M1 delivered (code complete, real capture pending):

- `src/clippy/ingest/capture.py` - `vod_url`, `normalize_downloader`, `build_capture_argv` (streamlink + yt-dlp), `capture_vod`, `probe_video_metadata`, `prune_sources`, `default_output_path`.
- `src/clippy/ingest/align.py` - `estimate_timeline_offset` (numpy cross-correlation of chat activity vs audio RMS), `alignment_warning`, `probe_alignment` with a bounded decode window and a `MIN_ACTIVE_BINS` evidence guard.
- `audio/intensity.extract_mono_pcm` - optional `start_seconds`/`duration_seconds` so alignment decodes a window instead of a whole 6 h VOD.
- `store/db.update_stream_source` - records the capture path, real dimensions/fps/size and offset on the stream row.
- `cli.py` + `pyproject.toml` + `scripts/capture.py` - the `clippy-capture` entry point.
- `tests/test_ingest_capture_align.py` + a store test - 20 new tests.

Verified: full suite at 94 passing; `streamlink 8.6.1` installed via `uv tool install streamlink`; every flag in the generated argv confirmed against streamlink's own option parser; an invalid VOD now fails in ~8 s with `clippy-capture: ... failed with exit code 1` instead of hanging ~50 s; `clippy-capture --help` and `probe_video_metadata` against the local fixture work.

**Environment finding:** the Phase 1 source VOD `...TN-160p.ts` has been deleted from `C:\Users\Jason\Downloads`, so no existing candidate has playable source footage. The real M1 gate (capture a VOD, then `clippy-edit --candidate 5 --source <hq>`) needs a VOD URL/id from the user.

Remaining: M1's real capture, then M2-M11 in order.



