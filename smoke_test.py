from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np

# The smoke test validates orchestration and alert rules without loading an
# actual Ultralytics model. Production still installs ultralytics normally.
if "ultralytics" not in sys.modules:
    _fake_ultralytics = types.ModuleType(
        "ultralytics"
    )
    _fake_ultralytics.YOLO = object
    sys.modules[
        "ultralytics"
    ] = _fake_ultralytics

from frame_extractor import (
    CameraSourceConfig,
    MultiCameraFrameExtractor,
)
from severity import (
    CameraRiskState,
    HazardSeverity,
    LabelRiskSnapshot,
    SeverityAnalyzer,
    SeverityPolicy,
    SeverityThreshold,
)
from fire_smoke_service import (
    CameraConfig,
    CameraRuntimeState,
    FireSmokeBatchDetector,
)
from tracking import StableObjectTracker
from async_video_storage import AsyncIncidentVideoStorage


def make_video(
    path: Path,
    base_value: int,
    frames: int = 8,
    fps: float = 10.0,
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (32, 24),
    )
    assert writer.isOpened()

    for index in range(frames):
        writer.write(
            np.full(
                (24, 32, 3),
                base_value + index * 5,
                dtype=np.uint8,
            )
        )

    writer.release()


def test_strict_frame_lists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        video_paths = [
            folder / f"video_{index}.mp4"
            for index in range(4)
        ]

        for index, path in enumerate(video_paths):
            make_video(
                path,
                20 + index * 40,
            )

        extractor = MultiCameraFrameExtractor(
            [
                CameraSourceConfig(
                    camera_id=f"camera-{index + 1}",
                    name=f"Camera {index + 1}",
                    source=str(path),
                    loop=False,
                    reader_queue_size=8,
                )
                for index, path
                in enumerate(video_paths)
            ]
        )
        extractor.start()

        try:
            batch_1 = extractor.extract_next_batch(
                3.0
            )
            batch_2 = extractor.extract_next_batch(
                3.0
            )

            assert batch_1 is not None
            assert batch_2 is not None

            assert batch_1.camera_ids == [
                "camera-1",
                "camera-2",
                "camera-3",
                "camera-4",
            ]
            assert batch_2.camera_ids == [
                "camera-1",
                "camera-2",
                "camera-3",
                "camera-4",
            ]

            # Exact required shape:
            # [f1_v1, f1_v2, f1_v3, f1_v4]
            # [f2_v1, f2_v2, f2_v3, f2_v4]
            assert batch_1.frame_indexes == [
                0, 0, 0, 0
            ]
            assert batch_2.frame_indexes == [
                1, 1, 1, 1
            ]
            assert len(batch_1.frames) == 4
            assert len(batch_2.frames) == 4

        finally:
            extractor.stop()


def test_demotion_candidate_reset() -> None:
    policy = SeverityPolicy(
        timeline_seconds=5.0,
        low=SeverityThreshold(
            1, 0.10, 0.10
        ),
        medium=SeverityThreshold(
            2, 0.20, 0.20
        ),
        high=SeverityThreshold(
            3, 0.30, 0.30
        ),
        demotion_hold_seconds=1.0,
    )

    analyzer = SeverityAnalyzer(
        fire_policy=policy,
        smoke_policy=policy,
    )
    state = CameraRiskState()

    # Promote to high.
    for index in range(3):
        result = analyzer.update(
            state=state,
            timestamp=index * 0.1,
            fire_confidence=0.9,
            smoke_confidence=0.0,
            fire_area_ratio=0.1,
            smoke_area_ratio=0.0,
            fire_track_count=1,
            smoke_track_count=0,
            fire_positive_threshold=0.3,
            smoke_positive_threshold=0.3,
        )

    assert result["overall"] == (
        HazardSeverity.HIGH
    )


def _risk_snapshot(
    label: str,
    severity: HazardSeverity,
    count: int,
    total: int,
    ratio: float,
    average_confidence: float,
) -> LabelRiskSnapshot:
    return LabelRiskSnapshot(
        label=label,
        severity=severity,
        positive_count=count,
        total_count=total,
        positive_ratio=ratio,
        average_confidence=average_confidence,
        max_confidence=average_confidence,
        average_area_ratio=0.1,
        contributing_track_count=2,
    )


def test_alert_cause() -> None:
    camera = CameraConfig(
        camera_id="test",
        name="Test",
    )
    state = CameraRuntimeState(
        camera=camera,
        tracker=StableObjectTracker(
            camera.tracker_settings(),
            camera.tracker_policy(),
        ),
        risk=CameraRiskState(),
        analyzer=SeverityAnalyzer(
            fire_policy=camera.severity_policy(
                "fire"
            ),
            smoke_policy=camera.severity_policy(
                "smoke"
            ),
        ),
    )
    detector = object.__new__(
        FireSmokeBatchDetector
    )
    risk = {
        "fire": _risk_snapshot(
            "fire",
            HazardSeverity.MEDIUM,
            20,
            100,
            0.20,
            0.50,
        ),
        "smoke": _risk_snapshot(
            "smoke",
            HazardSeverity.HIGH,
            80,
            100,
            0.80,
            0.70,
        ),
        "overall": HazardSeverity.HIGH,
    }

    cause = detector._alert_cause(
        state,
        risk,
    )
    assert (
        cause["code"]
        == "smoke_high_severity"
    )
    assert (
        "Smoke reached high severity"
        in cause["message"]
    )
    assert (
        cause["smoke"][
            "positive_count"
        ]
        == 80
    )


def test_async_video_saved_after_final_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        output = base / "videos"
        spool = base / ".spool"

        storage = AsyncIncidentVideoStorage(
            output,
            spool_dir=spool,
            queue_size=16,
            finalizer_workers=1,
        )
        storage.start()

        incident_id = "async-incident"
        frames: list[bytes] = []

        for index in range(6):
            image = np.full(
                (32, 48, 3),
                index * 30,
                dtype=np.uint8,
            )
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
            )
            assert ok
            frames.append(
                encoded.tobytes()
            )

        base_payload = {
            "schema_version": 5,
            "event_type": (
                "incident_frame_batch"
            ),
            "incident_id": incident_id,
            "hazard_type": "fire",
            "severity": "high",
            "camera": {
                "camera_id": "gate-test",
                "name": "Test",
                "source": "test.mp4",
                "metadata": {},
            },
            "recording_fps": 10.0,
        }

        storage.save_frame_batch(
            {
                **base_payload,
                "batch_index": 0,
                "is_first": True,
                "is_final": False,
                "frame_count": 4,
                "frames": [],
            },
            frames[:4],
        )
        storage.save_frame_batch(
            {
                **base_payload,
                "batch_index": 1,
                "is_first": False,
                "is_final": True,
                "frame_count": 2,
                "frames": [],
            },
            frames[4:],
        )

        final_video = (
            output
            / (
                "gate-test__"
                "async-incident.mp4"
            )
        )

        deadline = time.time() + 10.0

        while (
            not final_video.exists()
            and time.time() < deadline
        ):
            time.sleep(0.05)

        assert final_video.exists()
        assert final_video.stat().st_size > 0

        storage.close()

        # Final output contains MP4 only.
        output_files = [
            path
            for path in output.rglob("*")
            if path.is_file()
        ]
        assert output_files == [
            final_video
        ]
        assert not list(
            output.rglob("*.json")
        )
        assert not list(
            spool.rglob("*.json")
        )
        assert not (
            spool / incident_id
        ).exists()


if __name__ == "__main__":
    test_strict_frame_lists()
    test_demotion_candidate_reset()
    test_alert_cause()
    test_async_video_saved_after_final_batch()
    print("SMOKE TEST PASSED")
