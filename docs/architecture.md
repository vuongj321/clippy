# Architecture

Clippy monitors a Twitch stream, finds moments that may be worth clipping, extracts short video windows around those moments, and presents them for human review.

The MVP answers one question: **Can the system consistently find moments a human would clip?** It does not publish, auto-edit for short-form platforms, or learn from post metrics.

## End-to-end flow

```text
Twitch (VOD or live)
        │
        ▼
   Stream ingest
   (video + audio + chat)
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
  Rolling buffer   Chat ingest   Audio intensity
        │              │              │
        │              └──────┬───────┘
        │                     ▼
        │              Candidate detector
        │                     │
        │                     ▼
        │              Coalesce overlaps
        │                     │
        └─────────────────────┤
                              ▼
                    Extract window (FFmpeg)
                              │
                              ▼
                    SQLite + media files
                              │
                              ▼
                       Review UI
                              │
                              ▼
                  Approve / reject + reason
```

1. **Ingest** pulls media and chat for one stream (VOD first; live later).
2. **Buffer** keeps recent media so a detection can be cut into a playable file.
3. **Signals** turn chat and audio into time-stamped features on a shared timeline.
4. **Detector** flags spikes/anomalies as candidate moments and merges nearby ones.
5. **Extract** cuts a configurable window (default ~30s before and after) with FFmpeg.
6. **Store** saves candidate metadata and the media path in SQLite.
7. **Review** lets a human watch, approve or reject, and pick a reason code.

## Design principles

- **Cheap signals first.** Chat rate and audio intensity run continuously. Expensive AI (ASR, multimodal) is out of MVP scope and, if added later, should run only on candidates.
- **One shared clock.** Every event is stored as seconds relative to stream start. Chat, audio frames, and media cuts must use the same timeline.
- **Detection ≠ clippability.** The score is detection confidence (how strong the signals were). A human decides whether the moment is actually worth clipping.
- **Local-first.** MVP uses local files, SQLite, and in-process workers — not Redis, Celery, or S3.
- **One stream.** Multi-stream scaling is deferred; the pipeline is still shaped so another stream is mostly another config + process later.

## Components

### Config

Loaded from env/YAML. Controls:

| Setting | Role |
|--------|------|
| Pre/post window (e.g. 30s) | How much media to cut around a detection |
| Coalesce gap (e.g. 20s) | Merge detections that fall within this gap |
| Chat spike multiplier | Current chat rate vs baseline to fire a candidate |
| Keyword list | Phrases like `clip`, `CLIP IT` |
| Audio intensity threshold | RMS/peak vs rolling baseline |
| Media/DB paths | Where SQLite and extracted clips live |
| Disk budget | Skip new extracts when local media exceeds a max size |

### Ingest

Two modes share the same downstream pipeline:

**VOD (primary for MVP)**  
- Input: Twitch VOD, or a local MP4 plus a chat dump (JSON) for fully offline runs.  
- Output: media available for seeking/cutting, and chat messages with timestamps aligned to VOD-relative seconds.  
- Used to tune thresholds without fighting live reconnects and ads.

**Live (secondary)**  
- Video via HLS (or Streamlink).  
- Chat via Twitch IRC or EventSub.  
- Must handle reconnects and keep the rolling buffer bounded so disk/RAM do not grow without limit.

### Rolling media buffer

Holds enough recent media that when a candidate fires at time `T`, the system can still read `[T - pre, T + post]`.

- **VOD:** may be the full file on disk (seeking is enough); optional segment cache for speed.  
- **Live:** segment files or a time-bounded cache; old segments are deleted past the retention window.

Without a reliable buffer, detections cannot become reviewable clips.

### Chat ingest and signals

Chat is treated as a strong Twitch signal.

**Features:**

- **Message rate** — messages per second (or per short window) vs a rolling baseline. A large jump is an anomaly.
- **Keywords / phrases** — hits on configured terms (e.g. `CLIP IT`) boost or directly create candidates.
- **Optional later:** emote spikes, repeated phrases (not required for MVP).

Each chat event is mapped to stream-relative time so it lines up with audio and video.

### Audio intensity

Lightweight analysis (e.g. short-frame RMS/peak via FFmpeg or a small audio library):

- Maintain a rolling baseline of loudness.
- Flag sudden spikes (laughter, yelling, crowd noise, bangs) relative to that baseline.

Audio does not try to classify *what* happened — only that something acoustically unusual occurred.

### Candidate detector

Combines chat and audio features into candidate events.

Example intuition:

```text
Chat rate 4× baseline  +  audio spike  →  high-confidence candidate
Chat "CLIP IT" only                   →  candidate (keyword path)
Audio spike alone                     →  lower-confidence candidate
```

Each raw detection includes at least:

- `source_ts` — stream-relative time of the peak  
- contributing signal values (chat rate, keywords hit, audio level)  
- a numeric **score** (weighted combination; weights are config, not learned)

### Coalescing

One funny moment often produces many overlapping spikes. Before extract:

- Sort detections by time.
- Merge any pair (or chain) within the coalesce gap into a **single** candidate.
- Keep the strongest score / representative timestamp (e.g. peak of the cluster).

Goal: one review card per moment, not ten.

### Window extraction

For each coalesced candidate at time `T`:

1. Compute `[max(0, T - pre), T + post]`.
2. Cut that range with FFmpeg into `data/media/{candidate_id}.mp4`.
3. Persist the path on the candidate row.
4. If disk budget is exceeded, skip extract (or drop lowest-score pending files) — config decides the policy.

Final short-form boundaries (tighter start/end, captions, vertical crop) are **not** done here. The window is deliberately generous so a human (or a later editor stage) can judge the moment.

### Storage

**SQLite** holds structured state; **filesystem** holds media.

| Table | Purpose |
|-------|---------|
| `streamers` | Twitch identity (login, display name) |
| `streams` | One run: VOD or live, source id/url, start time |
| `candidates` | Detection time, window sizes, signals JSON, score, media path, status |
| `reviews` | Decision, reason code, notes, timestamp |

Candidate status: `pending` → `approved` | `rejected`.

**Rejection reason codes:** `false_alarm`, `needs_context`, `too_long`, `boring`, `unsafe`, `other`.

Reviews are the training signal for later phases (threshold tuning, optional rankers). They are not used for autonomous publishing in the MVP.

### Review UI

A simple FastAPI + HTML app:

- Ranked list of pending candidates (by score).
- Video player for the extracted window.
- Display: stream time, score, signal summary (transcript placeholder empty for now).
- Actions: approve / reject + reason.
- Stats: approve rate, counts by reason — enough to judge whether detection is useful.

### Eval (lightweight)

After a review session:

- Export reviews (CSV/JSON).
- **Precision** ≈ approved / reviewed for that run.
- Optional weak recall proxy: how often a chat “CLIP IT” peak produced a candidate.

Success is measurable human agreement, not full automation.

## Timeline contract

All pipeline stages must agree on time:

```text
t = 0  →  stream start (VOD start or live session start marker)
chat message at wall clock W  →  t = W - stream_start (or VOD-native offset)
audio frame at media PTS P    →  t = P (or P + known offset)
candidate.source_ts           →  t in seconds (float)
extract                       →  seek media using the same t
```

Never mix raw wall-clock chat times with unadjusted media PTS. Document the offset for each ingest adapter.

## Runtime topology (MVP)

```text
scripts/run_vod.py  or  scripts/run_live.py
        │
        ▼
  In-process pipeline
  (ingest → detect → extract → SQLite)
        │
        ▼
  uvicorn FastAPI review app
  (reads SQLite + serves media files)
```

No distributed queue in MVP. A later multi-stream design can split detector / extract / review into workers without changing the conceptual stages above.

## What this architecture deliberately omits

- Publishing to TikTok, YouTube Shorts, Instagram Reels  
- Continuous ASR or vision / scene-change / face-emotion models  
- Learned ranking from view metrics  
- Polished short-form editing (captions, vertical layout, subject tracking)  
- Cloud object storage and multi-stream orchestration  

Those can hang off the same candidate + review records later without redesigning the core loop: **ingest → signal → detect → extract → human label**.
