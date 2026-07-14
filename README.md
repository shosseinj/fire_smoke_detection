# Fire/Smoke CCTV — MP4-Only Incident Output

## Final output

After an incident ends, the final media directory contains only MP4 files:

```text
saved_fire_smoke_videos/
├── gate-01__3d65c73c-a573-446d-846d-dab967309cc5.mp4
├── gate-03__8de83221-7305-42e8-9242-8622fb4eece6.mp4
└── ...
```

The project no longer writes:

```text
event_*.json
frame_batch_*.json
video_status.json
video_completed.json
confirmation_snapshot.jpg
```

Alert metadata remains in memory and is available through:

```text
GET /api/v1/fire-smoke/latest-alerts
```

In production, replace the in-memory alert handler with a database insert or a
message-broker publish.

## Why the previous output contained JSON

The object containing:

```json
{
  "event_type": "alert_started",
  "incident_id": "...",
  "cause": "Smoke reached high severity..."
}
```

is alert metadata. It is not a video frame and cannot itself be converted into
a video.

The detector separately receives continuous incident frames. This build now
uses those frames to create one MP4 after `INCIDENT_ENDED`.

## Asynchronous video flow

```text
incident active
→ JPEG frames placed in hidden temporary spool

INCIDENT_ENDED
→ VIDEO_SAVE_QUEUED
→ background MP4 finalizer

VIDEO_SAVED
→ final MP4 placed in saved_fire_smoke_videos/
→ temporary JPEG spool deleted
```

Temporary files are stored under:

```text
.fire_smoke_video_spool/
```

They are not stored in the final video directory and are deleted after a
successful MP4 save.

## Logs

```text
INCIDENT_ENDED |
camera=gate-01
incident=...

VIDEO_SAVE_QUEUED |
camera=gate-01
incident=...
frames=742

VIDEO_SAVED |
camera=gate-01
incident=...
frames=742
duration=29.680s
path=.../saved_fire_smoke_videos/gate-01__....mp4
```

## API

Full status:

```text
GET /api/v1/fire-smoke/status
```

Video storage status:

```text
GET /api/v1/fire-smoke/video-status
```

Latest alert metadata, memory only:

```text
GET /api/v1/fire-smoke/latest-alerts
```

## Start

```powershell
python run_server.py
```

Do not use `--reload`.

## Test

```powershell
python smoke_test.py
```

Expected:

```text
SMOKE TEST PASSED
```
