from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI

from async_video_storage import AsyncIncidentVideoStorage

from fire_smoke_service import (
    CameraConfig,
    initialize,
)
from frame_extractor import (
    CameraSourceConfig,
    MultiCameraFrameExtractor,
)


class _IgnoreGeneratedFileChangeLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return (
            "changes detected"
            not in record.getMessage().lower()
        )


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | %(message)s"
    ),
)

for _logger_name in (
    "watchfiles",
    "watchfiles.main",
):
    _logger = logging.getLogger(
        _logger_name
    )
    _logger.setLevel(logging.WARNING)
    _logger.addFilter(
        _IgnoreGeneratedFileChangeLogs()
    )

BASE_DIR = Path(__file__).resolve().parent
PATH_STATICS = (BASE_DIR / "../fire").resolve()
EVENT_ROOT = (
    BASE_DIR / "saved_fire_smoke_events"
).resolve()
EVENT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


class InMemoryAlertEventHandler:
    """
    Receives alert/incident payloads without writing JSON or snapshots.

    Replace this handler with a database insert or message-broker publish when
    integrating the service into the main backend.
    """

    def __init__(
        self,
        maximum_events: int = 100,
    ) -> None:
        from collections import deque

        self.events = deque(
            maxlen=max(
                1,
                maximum_events,
            )
        )

    def handle(
        self,
        payload: dict[str, Any],
        snapshot: bytes | None,
    ) -> None:
        # Do not persist JSON or image files.
        self.events.append(
            {
                **payload,
                "snapshot_received": (
                    snapshot is not None
                ),
            }
        )

    def latest(
        self,
    ) -> list[dict[str, Any]]:
        return list(self.events)


alert_events = (
    InMemoryAlertEventHandler()
)

VIDEO_OUTPUT_DIR = (
    BASE_DIR
    / "saved_fire_smoke_videos"
).resolve()
VIDEO_SPOOL_DIR = (
    BASE_DIR
    / ".fire_smoke_video_spool"
).resolve()

video_storage = AsyncIncidentVideoStorage(
    VIDEO_OUTPUT_DIR,
    spool_dir=VIDEO_SPOOL_DIR,
    queue_size=512,
    finalizer_workers=2,
    cleanup_spool_after_success=True,
)



# ---------------------------------------------------------------------------
# 1. FASTAPI FRAME SOURCES
# Only this section contains video paths or RTSP URLs.
# FastAPI opens these sources and extracts frames.
# ---------------------------------------------------------------------------
CAMERA_SOURCES = [
    CameraSourceConfig(
        camera_id="gate-01",
        name="Main gate",
        source=str(
            PATH_STATICS
            / "input_videos/fire/fire1.mp4"
        ),
        loop=True,
        metadata={
            "location": "north gate",
        },
    ),
    CameraSourceConfig(
        camera_id="gate-02",
        name="Main gate",
        source=str(
            PATH_STATICS
            / "input_videos/fire/fire2.mp4"
        ),
        loop=True,
        metadata={
            "location": "north gate",
        },
    ),
    CameraSourceConfig(
        camera_id="gate-03",
        name="Main gate",
        source=str(
            PATH_STATICS
            / "input_videos/fire/fire3.mp4"
        ),
        loop=True,
        metadata={
            "location": "north gate",
        },
    ),
    CameraSourceConfig(
        camera_id="gate-04",
        name="Warehouse",
        source=str(
            PATH_STATICS
            / "input_videos/smoke/smoke1.mp4"
        ),
        loop=True,
        metadata={
            "location": "warehouse",
        },
    ),
]


# ---------------------------------------------------------------------------
# 2. DETECTOR SETTINGS
# No video paths exist here. These settings are matched by camera_id.
# ---------------------------------------------------------------------------
CAMERAS = [
    CameraConfig(
        camera_id="gate-01",
        name="Main gate",
        metadata={
            "location": "north gate",
        },
        severity_timeline_seconds=5.0,
        low_severity_min_count=5,
        medium_severity_min_count=15,
        high_severity_min_count=30,
        log_severity_changes=False,
        log_risk_windows=False,
    ),
    CameraConfig(
        camera_id="gate-02",
        name="Main gate",
        metadata={
            "location": "north gate",
        },
        severity_timeline_seconds=5.0,
        low_severity_min_count=5,
        medium_severity_min_count=15,
        high_severity_min_count=30,
        log_severity_changes=False,
        log_risk_windows=False,
    ),
    CameraConfig(
        camera_id="gate-03",
        name="Main gate",
        metadata={
            "location": "north gate",
        },
        severity_timeline_seconds=5.0,
        low_severity_min_count=5,
        medium_severity_min_count=15,
        high_severity_min_count=30,
        log_severity_changes=False,
        log_risk_windows=False,
    ),
    CameraConfig(
        camera_id="gate-04",
        name="Warehouse",
        metadata={
            "location": "warehouse",
        },
        severity_timeline_seconds=5.0,
        low_severity_min_count=5,
        medium_severity_min_count=15,
        high_severity_min_count=30,
        log_severity_changes=False,
        log_risk_windows=False,
    ),
]


# FastAPI owns the extractor.
frame_extractor = MultiCameraFrameExtractor(
    CAMERA_SOURCES
)

# The detector receives only lists of extracted frames.
application = initialize(
    extractor=frame_extractor,
    model_path=(
        PATH_STATICS
        / "fire_models/engine/"
        "best_nano_111_batch8.engine"
    ),
    cameras=CAMERAS,
    event_handler=alert_events.handle,
    frame_batch_handler=(
        video_storage.save_frame_batch
    ),
    device=0,
    batch_size=8,
    imgsz=640,
    show_preview=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    video_storage.start()
    application.start()
    yield
    # The detector/recorder first enqueue every final frame batch.
    application.stop()
    # Then wait for asynchronous MP4 jobs during graceful shutdown.
    video_storage.close()


app = FastAPI(lifespan=lifespan)


@app.get("/api/v1/fire-smoke/status")
async def status() -> dict[str, Any]:
    result = application.status()
    result["video_storage"] = (
        video_storage.status()
    )
    return result


@app.get(
    "/api/v1/fire-smoke/latest-alerts"
)
async def latest_alerts() -> list[
    dict[str, Any]
]:
    """
    Alert metadata retained in memory only.

    Production integration should save these payloads to the application
    database instead of local JSON files.
    """
    return alert_events.latest()


@app.get("/api/v1/fire-smoke/video-status")
async def video_status() -> dict[str, Any]:
    return video_storage.status()


@app.get("/api/v1/fire-smoke/risk")
async def risk() -> dict[str, Any]:
    return application.detector.status()[
        "cameras"
    ]



@app.get("/api/v1/fire-smoke/alert-policy")
async def alert_policy() -> dict[str, Any]:
    """
    Return the exact conditions that create an incident and send an alert.
    """
    return {
        camera.camera_id: {
            "timeline_seconds": (
                camera.severity_timeline_seconds
            ),
            "incident_starts_at": (
                camera.incident_start_severity
            ),
            "alert_is_sent_at": (
                camera.alert_start_severity
            ),
            "alert_only_once_per_incident": True,
            "high_rules": {
                "fire": {
                    "minimum_positive_count": (
                        camera.high_severity_min_count
                    ),
                    "minimum_positive_ratio": (
                        camera.high_severity_min_ratio
                    ),
                    "minimum_average_confidence": (
                        camera.fire_high_severity_confidence
                    ),
                },
                "smoke": {
                    "minimum_positive_count": (
                        camera.high_severity_min_count
                    ),
                    "minimum_positive_ratio": (
                        camera.high_severity_min_ratio
                    ),
                    "minimum_average_confidence": (
                        camera.smoke_high_severity_confidence
                    ),
                },
                "combined_rule": (
                    "Fire and smoke both at medium or higher "
                    "escalate overall severity to high."
                ),
            },
        }
        for camera in CAMERAS
    }


@app.get("/api/v1/fire-smoke/last-frame-list")
async def last_frame_list() -> dict[str, Any]:
    """
    Shows the last list passed to the model.

    Example:
    {
      "camera_ids": ["gate-01", "gate-02", "gate-03", "gate-04"],
      "frame_indexes": [15, 15, 15, 15],
      "frame_count": 4
    }
    """
    return application.last_frame_list
