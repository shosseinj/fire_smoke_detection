from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from frame_extractor import (
    ExtractedFrame,
    MultiCameraFrameExtractor,
    SynchronizedFrameBatch,
)
from incident_recorder import (
    FrameBatchHandler,
    QueuedIncidentRecorder,
    RecordingPolicy,
)
from severity import (
    CameraRiskState,
    HazardSeverity,
    SeverityAnalyzer,
    SeverityPolicy,
    SeverityThreshold,
)
from tracking import (
    Detection,
    StableObjectTracker,
    Track,
    TrackerPolicy,
    TrackerSettings,
)


LOGGER = logging.getLogger("fire-smoke-service")
EventHandler = Callable[
    [dict[str, Any], bytes | None],
    None | Awaitable[None],
]


@dataclass(frozen=True)
class CameraConfig:
    """
    Detection policy only.

    There is intentionally no `source` field here. Video/RTSP paths belong to
    CameraSourceConfig inside FastAPI's frame extractor.
    """

    camera_id: str
    name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    rolling_history: int = 15
    required_positive_detections: int = 8
    required_positive_ratio: float = 0.60
    consecutive_detections: int = 3
    fire_average_confidence: float = 0.45
    smoke_average_confidence: float = 0.40
    alert_release_after_missing: int = 10
    track_removal_after_missing: int = 18

    track_iou: float = 0.20
    track_center_distance: float = 0.75
    bbox_smoothing_alpha: float = 0.65
    confidence_ema_alpha: float = 0.35
    camera_evidence_min_track_hits: int = 2

    severity_timeline_seconds: float = 5.0
    low_severity_min_count: int = 5
    low_severity_min_ratio: float = 0.20
    medium_severity_min_count: int = 15
    medium_severity_min_ratio: float = 0.45
    high_severity_min_count: int = 30
    high_severity_min_ratio: float = 0.70

    fire_low_severity_confidence: float = 0.38
    fire_medium_severity_confidence: float = 0.48
    fire_high_severity_confidence: float = 0.60

    smoke_low_severity_confidence: float = 0.32
    smoke_medium_severity_confidence: float = 0.42
    smoke_high_severity_confidence: float = 0.55

    severity_demotion_hold_seconds: float = 5.0
    severity_log_interval_seconds: float = 5.0

    # Quiet production logging. Alerts and incident completion remain visible.
    log_severity_changes: bool = False
    log_risk_windows: bool = False

    incident_start_severity: str = "medium"
    alert_start_severity: str = "high"
    incident_end_grace_seconds: float = 10.0

    video_pre_alert_seconds: float = 5.0
    video_post_alert_seconds: float = 5.0
    video_frame_batch_size: int = 8
    video_frame_stride: int = 1
    video_jpeg_quality: int = 80
    video_max_width: int | None = 1280

    reset_state_on_source_loop: bool = True

    def tracker_settings(self) -> TrackerSettings:
        return TrackerSettings(
            track_iou=self.track_iou,
            track_center_distance=self.track_center_distance,
            bbox_smoothing_alpha=self.bbox_smoothing_alpha,
            confidence_ema_alpha=self.confidence_ema_alpha,
        )

    def tracker_policy(self) -> TrackerPolicy:
        return TrackerPolicy(
            rolling_history=self.rolling_history,
            required_positive_detections=(
                self.required_positive_detections
            ),
            required_positive_ratio=(
                self.required_positive_ratio
            ),
            consecutive_detections=(
                self.consecutive_detections
            ),
            fire_average_confidence=(
                self.fire_average_confidence
            ),
            smoke_average_confidence=(
                self.smoke_average_confidence
            ),
            alert_release_after_missing=(
                self.alert_release_after_missing
            ),
            track_removal_after_missing=(
                self.track_removal_after_missing
            ),
        )

    def severity_policy(
        self,
        label: str,
    ) -> SeverityPolicy:
        if label == "fire":
            low_conf = self.fire_low_severity_confidence
            medium_conf = self.fire_medium_severity_confidence
            high_conf = self.fire_high_severity_confidence
        else:
            low_conf = self.smoke_low_severity_confidence
            medium_conf = self.smoke_medium_severity_confidence
            high_conf = self.smoke_high_severity_confidence

        return SeverityPolicy(
            timeline_seconds=self.severity_timeline_seconds,
            low=SeverityThreshold(
                self.low_severity_min_count,
                self.low_severity_min_ratio,
                low_conf,
            ),
            medium=SeverityThreshold(
                self.medium_severity_min_count,
                self.medium_severity_min_ratio,
                medium_conf,
            ),
            high=SeverityThreshold(
                self.high_severity_min_count,
                self.high_severity_min_ratio,
                high_conf,
            ),
            demotion_hold_seconds=(
                self.severity_demotion_hold_seconds
            ),
            summary_log_seconds=(
                self.severity_log_interval_seconds
            ),
        )

    def recording_policy(self) -> RecordingPolicy:
        return RecordingPolicy(
            pre_alert_seconds=self.video_pre_alert_seconds,
            post_alert_seconds=self.video_post_alert_seconds,
            frame_batch_size=self.video_frame_batch_size,
            frame_stride=self.video_frame_stride,
            jpeg_quality=self.video_jpeg_quality,
            max_width=self.video_max_width,
        )


@dataclass
class CameraIncident:
    incident_id: str
    started_at_utc: str
    started_monotonic: float
    hazard_type: str
    maximum_severity: HazardSeverity
    labels_seen: set[str] = field(default_factory=set)
    contributing_track_ids: set[int] = field(default_factory=set)
    alert_sent: bool = False
    inactive_since: float | None = None


@dataclass
class CameraRuntimeState:
    camera: CameraConfig
    tracker: StableObjectTracker
    risk: CameraRiskState
    analyzer: SeverityAnalyzer
    incident: CameraIncident | None = None
    last_loop_index: int = 0
    last_batch_sequence: int = 0
    last_result: dict[str, Any] = field(default_factory=dict)


class AsyncEventDispatcher:
    _STOP = object()

    def __init__(
        self,
        handler: EventHandler | None,
        queue_size: int = 1024,
    ) -> None:
        self.handler = handler
        self.queue: queue.Queue[object] = queue.Queue(
            maxsize=queue_size
        )
        self.thread: threading.Thread | None = None
        self.closed = False
        self.errors = 0

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return

        self.thread = threading.Thread(
            target=self._worker,
            name="event-dispatcher",
            daemon=True,
        )
        self.thread.start()

    def submit(
        self,
        payload: dict[str, Any],
        snapshot: bytes | None = None,
    ) -> None:
        if self.closed:
            return

        try:
            self.queue.put_nowait((payload, snapshot))
        except queue.Full:
            LOGGER.error(
                "Event queue full; dropped %s",
                payload.get("event_type"),
            )

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True
        self.queue.put(self._STOP)

        if self.thread is not None:
            self.thread.join(timeout=10)

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is self._STOP:
                    return

                payload, snapshot = item
                if self.handler is None:
                    LOGGER.info("EVENT %s", payload)
                else:
                    result = self.handler(
                        payload,
                        snapshot,
                    )
                    if inspect.isawaitable(result):
                        asyncio.run(result)

            except Exception:
                self.errors += 1
                LOGGER.exception(
                    "Event handler failed."
                )
            finally:
                self.queue.task_done()


class FireSmokeBatchDetector:
    """
    CUDA detector that accepts frames only.

    Main API:
        process_frames(
            frames=[f1_v1, f1_v2, ...],
            packets=[metadata_v1, metadata_v2, ...],
            batch_sequence=1,
        )
    """

    def __init__(
        self,
        *,
        model_path: Path,
        cameras: Sequence[CameraConfig],
        device: int,
        batch_size: int,
        imgsz: int,
        fire_class_id: int,
        smoke_class_id: int,
        fire_candidate_conf: float,
        smoke_candidate_conf: float,
        event_dispatcher: AsyncEventDispatcher,
        recorder: QueuedIncidentRecorder,
        show_preview: bool,
    ) -> None:
        self.model_path = model_path
        self.cameras = {
            camera.camera_id: camera
            for camera in cameras
        }
        self.camera_order = tuple(
            camera.camera_id
            for camera in cameras
        )
        self.device = device
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.fire_class_id = fire_class_id
        self.smoke_class_id = smoke_class_id
        self.fire_candidate_conf = fire_candidate_conf
        self.smoke_candidate_conf = smoke_candidate_conf
        self.event_dispatcher = event_dispatcher
        self.recorder = recorder
        self.show_preview = show_preview

        self.model: YOLO | None = None
        self.processed_batches = 0
        self.processed_frames = 0
        self.last_inference_ms = 0.0
        self.last_error: str | None = None

        self.states = {
            camera.camera_id: CameraRuntimeState(
                camera=camera,
                tracker=StableObjectTracker(
                    camera.tracker_settings(),
                    camera.tracker_policy(),
                ),
                risk=CameraRiskState(),
                analyzer=SeverityAnalyzer(
                    fire_policy=camera.severity_policy("fire"),
                    smoke_policy=camera.severity_policy("smoke"),
                ),
            )
            for camera in cameras
        }

    def initialize(self) -> None:
        if self.batch_size != 8:
            raise ValueError(
                "TensorRT inference batch_size must be 8."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; CPU inference is disabled."
            )
        if not 0 <= self.device < torch.cuda.device_count():
            raise ValueError("Invalid CUDA device.")

        torch.cuda.set_device(self.device)
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

        if hasattr(
            torch.backends.cuda.matmul,
            "allow_tf32",
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

        self.model = YOLO(
            str(self.model_path),
            task="detect",
        )

        dummy = np.zeros(
            (self.imgsz, self.imgsz, 3),
            dtype=np.uint8,
        )
        for _ in range(3):
            self._predict(
                [dummy] * self.batch_size
            )

    def _predict(
        self,
        frames: list[np.ndarray],
    ) -> list[Any]:
        assert self.model is not None

        kwargs: dict[str, Any] = {
            "source": frames,
            "batch": self.batch_size,
            "imgsz": self.imgsz,
            "conf": min(
                self.fire_candidate_conf,
                self.smoke_candidate_conf,
            ),
            "iou": 0.45,
            "max_det": 50,
            "classes": [
                self.fire_class_id,
                self.smoke_class_id,
            ],
            "device": str(self.device),
            "augment": False,
            "rect": False,
            "stream": False,
            "verbose": False,
        }

        if self.model_path.suffix.lower() == ".pt":
            kwargs["half"] = True

        return list(
            self.model.predict(**kwargs)
        )

    def _infer(
        self,
        frames: list[np.ndarray],
    ) -> list[Any]:
        real_count = len(frames)
        if not 1 <= real_count <= 8:
            raise ValueError(
                "Frame list must contain 1-8 frames."
            )

        padded = list(frames)
        template = frames[0]

        while len(padded) < self.batch_size:
            padded.append(
                np.zeros_like(template)
            )

        started = time.perf_counter()
        results = self._predict(padded)
        self.last_inference_ms = (
            time.perf_counter() - started
        ) * 1000.0

        if len(results) < real_count:
            raise RuntimeError(
                "Model returned fewer results than real frames."
            )

        return results[:real_count]

    def _extract_detections(
        self,
        result: Any,
    ) -> list[Detection]:
        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return []

        rows = (
            result.boxes.data
            .detach()
            .cpu()
            .numpy()
        )
        detections: list[Detection] = []

        for row in rows:
            if len(row) < 6:
                continue

            x1, y1, x2, y2, confidence, class_value = row[:6]
            class_id = int(class_value)

            if class_id == self.fire_class_id:
                label = "fire"
                threshold = self.fire_candidate_conf
            elif class_id == self.smoke_class_id:
                label = "smoke"
                threshold = self.smoke_candidate_conf
            else:
                continue

            if float(confidence) < threshold:
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=float(confidence),
                    bbox=np.asarray(
                        [x1, y1, x2, y2],
                        dtype=np.float32,
                    ),
                )
            )

        return detections

    @staticmethod
    def _track_area_ratio(
        track: Track,
        frame: np.ndarray,
    ) -> float:
        x1, y1, x2, y2 = track.bbox.tolist()
        area = (
            max(0.0, x2 - x1)
            * max(0.0, y2 - y1)
        )
        frame_area = max(
            float(
                frame.shape[0]
                * frame.shape[1]
            ),
            1.0,
        )
        return area / frame_area

    def process_frames(
        self,
        *,
        frames: list[np.ndarray],
        packets: Sequence[ExtractedFrame],
        batch_sequence: int,
        created_monotonic: float,
    ) -> None:
        """
        Receives the exact list created by FastAPI's extractor.

        Round N:
            frames = [fN_v1, fN_v2, fN_v3, ...]
        """
        if len(frames) != len(packets):
            raise ValueError(
                "frames and packets must have equal lengths."
            )

        camera_ids = tuple(
            packet.camera_id
            for packet in packets
        )
        if camera_ids != self.camera_order:
            raise ValueError(
                "Frame list camera order does not match detector "
                f"configuration. Expected {self.camera_order}, "
                f"received {camera_ids}."
            )

        batch = SynchronizedFrameBatch(
            batch_sequence=batch_sequence,
            created_monotonic=created_monotonic,
            items=tuple(packets),
            configured_camera_ids=camera_ids,
        )

        policies = {
            camera_id: camera.recording_policy()
            for camera_id, camera
            in self.cameras.items()
        }
        self.recorder.capture_batch(
            batch,
            policies,
        )

        results = self._infer(frames)
        self.processed_batches += 1

        for packet, result in zip(
            packets,
            results,
        ):
            self._process_camera(
                packet=packet,
                result=result,
                batch_sequence=batch_sequence,
            )
            self.processed_frames += 1

        if self.show_preview:
            cv2.waitKey(1)

    def _process_camera(
        self,
        *,
        packet: ExtractedFrame,
        result: Any,
        batch_sequence: int,
    ) -> None:
        state = self.states[packet.camera_id]
        camera = state.camera
        state.last_batch_sequence = batch_sequence

        if (
            camera.reset_state_on_source_loop
            and packet.loop_index
            != state.last_loop_index
        ):
            self._end_incident(
                state,
                packet,
                "source_loop_restart",
            )
            state.tracker.reset()
            state.risk = CameraRiskState()
            state.last_loop_index = packet.loop_index

        tracks, _ = state.tracker.update(
            self._extract_detections(result),
            packet.frame_index,
            packet.source_time_seconds,
        )

        credible = [
            track
            for track in tracks
            if (
                track.missed_updates == 0
                and track.total_hits
                >= camera.camera_evidence_min_track_hits
            )
        ]
        fire_tracks = [
            track
            for track in credible
            if track.label == "fire"
        ]
        smoke_tracks = [
            track
            for track in credible
            if track.label == "smoke"
        ]

        fire_confidence = max(
            (
                track.confidence_ema
                for track in fire_tracks
            ),
            default=0.0,
        )
        smoke_confidence = max(
            (
                track.confidence_ema
                for track in smoke_tracks
            ),
            default=0.0,
        )

        fire_area_ratio = sum(
            self._track_area_ratio(
                track,
                packet.frame,
            )
            for track in fire_tracks
        )
        smoke_area_ratio = sum(
            self._track_area_ratio(
                track,
                packet.frame,
            )
            for track in smoke_tracks
        )

        risk = state.analyzer.update(
            state=state.risk,
            timestamp=packet.captured_monotonic,
            fire_confidence=fire_confidence,
            smoke_confidence=smoke_confidence,
            fire_area_ratio=fire_area_ratio,
            smoke_area_ratio=smoke_area_ratio,
            fire_track_count=len(fire_tracks),
            smoke_track_count=len(smoke_tracks),
            fire_positive_threshold=(
                self.fire_candidate_conf
            ),
            smoke_positive_threshold=(
                self.smoke_candidate_conf
            ),
        )

        track_ids = {
            track.track_id
            for track in credible
        }
        hazard_type = self._hazard_type(risk)

        self._update_incident(
            state,
            packet,
            risk,
            hazard_type,
            track_ids,
        )
        self._log_risk(
            state,
            packet,
            risk,
            hazard_type,
        )
        state.last_result = self._risk_payload(
            state,
            packet,
            risk,
            hazard_type,
            track_ids,
        )

        if self.show_preview:
            preview = packet.frame.copy()
            cv2.putText(
                preview,
                (
                    f"{camera.camera_id} "
                    f"{risk['overall'].label.upper()} "
                    f"{hazard_type}"
                ),
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(
                camera.camera_id,
                preview,
            )

    @staticmethod
    def _hazard_type(
        risk: dict[str, Any],
    ) -> str:
        fire = (
            risk["fire"].severity
            >= HazardSeverity.LOW
        )
        smoke = (
            risk["smoke"].severity
            >= HazardSeverity.LOW
        )

        if fire and smoke:
            return "fire_and_smoke"
        if fire:
            return "fire"
        if smoke:
            return "smoke"
        return "none"

    @staticmethod
    def _severity_value(
        name: str,
    ) -> HazardSeverity:
        return HazardSeverity[
            name.upper()
        ]


    @staticmethod
    def _threshold_payload(
        state: CameraRuntimeState,
        label: str,
        severity: HazardSeverity,
    ) -> dict[str, Any]:
        camera = state.camera

        if severity == HazardSeverity.HIGH:
            min_count = camera.high_severity_min_count
            min_ratio = camera.high_severity_min_ratio
            min_confidence = (
                camera.fire_high_severity_confidence
                if label == "fire"
                else camera.smoke_high_severity_confidence
            )
        elif severity == HazardSeverity.MEDIUM:
            min_count = camera.medium_severity_min_count
            min_ratio = camera.medium_severity_min_ratio
            min_confidence = (
                camera.fire_medium_severity_confidence
                if label == "fire"
                else camera.smoke_medium_severity_confidence
            )
        else:
            min_count = camera.low_severity_min_count
            min_ratio = camera.low_severity_min_ratio
            min_confidence = (
                camera.fire_low_severity_confidence
                if label == "fire"
                else camera.smoke_low_severity_confidence
            )

        return {
            "severity": severity.label,
            "minimum_positive_count": min_count,
            "minimum_positive_ratio": min_ratio,
            "minimum_average_confidence": min_confidence,
        }

    def _alert_cause(
        self,
        state: CameraRuntimeState,
        risk: dict[str, Any],
    ) -> dict[str, Any]:
        fire = risk["fire"]
        smoke = risk["smoke"]
        camera = state.camera

        fire_high = fire.severity >= HazardSeverity.HIGH
        smoke_high = smoke.severity >= HazardSeverity.HIGH
        fire_medium = fire.severity >= HazardSeverity.MEDIUM
        smoke_medium = smoke.severity >= HazardSeverity.MEDIUM

        if fire_high and smoke_high:
            code = "fire_and_smoke_high_severity"
            triggered_by = ["fire", "smoke"]
            message = (
                "Fire and smoke both reached high severity during the "
                f"{camera.severity_timeline_seconds:.1f}-second window."
            )
        elif fire_high:
            code = "fire_high_severity"
            triggered_by = ["fire"]
            message = (
                "Fire reached high severity: "
                f"{fire.positive_count}/{fire.total_count} positive "
                f"observations ({fire.positive_ratio:.1%}) with average "
                f"confidence {fire.average_confidence:.3f}."
            )
        elif smoke_high:
            code = "smoke_high_severity"
            triggered_by = ["smoke"]
            message = (
                "Smoke reached high severity: "
                f"{smoke.positive_count}/{smoke.total_count} positive "
                f"observations ({smoke.positive_ratio:.1%}) with average "
                f"confidence {smoke.average_confidence:.3f}."
            )
        elif fire_medium and smoke_medium:
            code = "combined_medium_fire_smoke_escalation"
            triggered_by = ["fire", "smoke"]
            message = (
                "Fire and smoke each reached at least medium severity in "
                f"the same {camera.severity_timeline_seconds:.1f}-second "
                "window, so the combined-hazard rule escalated the alert "
                "to high severity."
            )
        else:
            code = "overall_alert_threshold_reached"
            triggered_by = []
            message = (
                f"Overall risk reached the configured "
                f"{camera.alert_start_severity} alert threshold."
            )

        return {
            "code": code,
            "message": message,
            "timeline_seconds": camera.severity_timeline_seconds,
            "alert_threshold": camera.alert_start_severity,
            "triggered_by": triggered_by,
            "fire": {
                "effective_severity": fire.severity.label,
                "positive_count": fire.positive_count,
                "total_count": fire.total_count,
                "positive_ratio": round(
                    fire.positive_ratio,
                    6,
                ),
                "average_confidence": round(
                    fire.average_confidence,
                    6,
                ),
                "high_threshold": self._threshold_payload(
                    state,
                    "fire",
                    HazardSeverity.HIGH,
                ),
                "medium_threshold": self._threshold_payload(
                    state,
                    "fire",
                    HazardSeverity.MEDIUM,
                ),
            },
            "smoke": {
                "effective_severity": smoke.severity.label,
                "positive_count": smoke.positive_count,
                "total_count": smoke.total_count,
                "positive_ratio": round(
                    smoke.positive_ratio,
                    6,
                ),
                "average_confidence": round(
                    smoke.average_confidence,
                    6,
                ),
                "high_threshold": self._threshold_payload(
                    state,
                    "smoke",
                    HazardSeverity.HIGH,
                ),
                "medium_threshold": self._threshold_payload(
                    state,
                    "smoke",
                    HazardSeverity.MEDIUM,
                ),
            },
        }

    def _update_incident(
        self,
        state: CameraRuntimeState,
        packet: ExtractedFrame,
        risk: dict[str, Any],
        hazard_type: str,
        track_ids: set[int],
    ) -> None:
        severity = risk["overall"]
        start_level = self._severity_value(
            state.camera.incident_start_severity
        )
        alert_level = self._severity_value(
            state.camera.alert_start_severity
        )
        incident = state.incident
        just_started = False

        if (
            incident is None
            and severity >= start_level
        ):
            incident = CameraIncident(
                incident_id=str(uuid.uuid4()),
                started_at_utc=self._utc_now(),
                started_monotonic=(
                    packet.captured_monotonic
                ),
                hazard_type=hazard_type,
                maximum_severity=severity,
                labels_seen=set(
                    hazard_type.split("_and_")
                ),
                contributing_track_ids=set(
                    track_ids
                ),
            )
            state.incident = incident
            just_started = True

            self.recorder.start_incident(
                camera_id=state.camera.camera_id,
                camera_name=state.camera.name,
                source=packet.source,
                metadata={
                    **dict(packet.metadata),
                    **dict(state.camera.metadata),
                },
                incident_id=incident.incident_id,
                hazard_type=hazard_type,
                severity=severity.label,
                source_fps=packet.source_fps,
                policy=state.camera.recording_policy(),
            )
            self._send_event(
                "incident_started",
                state,
                packet,
                risk,
                hazard_type,
                track_ids,
                snapshot=True,
            )

        if incident is None:
            return

        previous_hazard = incident.hazard_type
        previous_maximum = (
            incident.maximum_severity
        )

        incident.maximum_severity = max(
            incident.maximum_severity,
            severity,
        )
        if hazard_type != "none":
            incident.labels_seen.update(
                hazard_type.split("_and_")
            )

        if {"fire", "smoke"}.issubset(
            incident.labels_seen
        ):
            incident.hazard_type = (
                "fire_and_smoke"
            )
        elif "fire" in incident.labels_seen:
            incident.hazard_type = "fire"
        else:
            incident.hazard_type = "smoke"

        incident.contributing_track_ids.update(
            track_ids
        )

        if severity > HazardSeverity.NONE:
            incident.inactive_since = None
        elif incident.inactive_since is None:
            incident.inactive_since = (
                packet.captured_monotonic
            )

        if (
            severity >= alert_level
            and not incident.alert_sent
        ):
            incident.alert_sent = True
            alert_cause = self._alert_cause(
                state,
                risk,
            )
            self._send_event(
                "alert_started",
                state,
                packet,
                risk,
                hazard_type,
                track_ids,
                snapshot=False,
                extra={
                    "cause": alert_cause["message"],
                    "alert_cause": alert_cause,
                    "alert_decision": {
                        "should_alert": True,
                        "current_severity": severity.label,
                        "required_severity": (
                            state.camera.alert_start_severity
                        ),
                        "only_once_per_incident": True,
                    },
                },
            )
            LOGGER.warning(
                "ALERT_STARTED | camera=%s incident=%s "
                "hazard=%s severity=%s cause=%s",
                state.camera.camera_id,
                incident.incident_id,
                incident.hazard_type,
                severity.label,
                alert_cause["message"],
            )

        meaningful_update = (
            risk["overall_changed"]
            or previous_hazard
            != incident.hazard_type
            or previous_maximum
            != incident.maximum_severity
        )

        if not just_started and meaningful_update:
            self.recorder.update_incident(
                camera_id=state.camera.camera_id,
                incident_id=incident.incident_id,
                hazard_type=incident.hazard_type,
                severity=(
                    incident.maximum_severity.label
                ),
            )
            self._send_event(
                "incident_updated",
                state,
                packet,
                risk,
                hazard_type,
                track_ids,
                snapshot=False,
            )

        if incident.inactive_since is not None:
            close_after = (
                state.camera.incident_end_grace_seconds
                + state.camera.video_post_alert_seconds
            )
            if (
                packet.captured_monotonic
                - incident.inactive_since
                >= close_after
            ):
                self._end_incident(
                    state,
                    packet,
                    "risk_inactive_timeout",
                    risk=risk,
                    hazard_type=hazard_type,
                    track_ids=track_ids,
                )

    def _end_incident(
        self,
        state: CameraRuntimeState,
        packet: ExtractedFrame,
        reason: str,
        *,
        risk: dict[str, Any] | None = None,
        hazard_type: str = "none",
        track_ids: set[int] | None = None,
    ) -> None:
        incident = state.incident
        if incident is None:
            return

        if risk is None:
            risk = {
                "fire": state.analyzer._snapshot(
                    "fire",
                    state.risk.fire,
                ),
                "smoke": state.analyzer._snapshot(
                    "smoke",
                    state.risk.smoke,
                ),
                "overall": (
                    state.risk.overall_severity
                ),
                "overall_changed": False,
            }

        self._send_event(
            "incident_ended",
            state,
            packet,
            risk,
            hazard_type,
            track_ids or set(),
            snapshot=False,
            end_reason=reason,
        )
        self.recorder.stop_incident(
            camera_id=state.camera.camera_id,
            incident_id=incident.incident_id,
            hazard_type=incident.hazard_type,
            severity=(
                incident.maximum_severity.label
            ),
        )

        LOGGER.info(
            "INCIDENT_ENDED | camera=%s incident=%s "
            "hazard=%s reason=%s tracks=%d",
            state.camera.camera_id,
            incident.incident_id,
            incident.hazard_type,
            reason,
            len(
                incident.contributing_track_ids
            ),
        )
        state.incident = None

    def _send_event(
        self,
        event_type: str,
        state: CameraRuntimeState,
        packet: ExtractedFrame,
        risk: dict[str, Any],
        hazard_type: str,
        track_ids: set[int],
        *,
        snapshot: bool,
        end_reason: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload = self._risk_payload(
            state,
            packet,
            risk,
            hazard_type,
            track_ids,
        )
        payload.update(
            {
                "event_type": event_type,
                "occurred_at_utc": self._utc_now(),
                "incident_id": (
                    state.incident.incident_id
                    if state.incident
                    else None
                ),
                "end_reason": end_reason,
            }
        )
        if extra:
            payload.update(dict(extra))

        image: bytes | None = None
        if snapshot:
            ok, encoded = cv2.imencode(
                ".jpg",
                packet.frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    90,
                ],
            )
            if ok:
                image = encoded.tobytes()

        self.event_dispatcher.submit(
            payload,
            image,
        )

    def _risk_payload(
        self,
        state: CameraRuntimeState,
        packet: ExtractedFrame,
        risk: dict[str, Any],
        hazard_type: str,
        track_ids: set[int],
    ) -> dict[str, Any]:
        return {
            "schema_version": 5,
            "camera": {
                "camera_id": state.camera.camera_id,
                "name": state.camera.name,
                "source": packet.source,
                "metadata": {
                    **dict(packet.metadata),
                    **dict(state.camera.metadata),
                },
            },
            "batch_sequence": (
                state.last_batch_sequence
            ),
            "frame_index": packet.frame_index,
            "source_time_seconds": round(
                packet.source_time_seconds,
                3,
            ),
            "captured_monotonic": round(
                packet.captured_monotonic,
                6,
            ),
            "hazard_type": hazard_type,
            "incident_hazard_type": (
                state.incident.hazard_type
                if state.incident is not None
                else hazard_type
            ),
            "current_severity": (
                risk["overall"].label
            ),
            "maximum_incident_severity": (
                state.incident
                .maximum_severity.label
                if state.incident is not None
                else risk["overall"].label
            ),
            "fire": SeverityAnalyzer.snapshot_dict(
                risk["fire"]
            ),
            "smoke": SeverityAnalyzer.snapshot_dict(
                risk["smoke"]
            ),
            "contributing_track_count": len(
                track_ids
            ),
            "contributing_track_ids": sorted(
                track_ids
            ),
        }

    def _log_risk(
        self,
        state: CameraRuntimeState,
        packet: ExtractedFrame,
        risk: dict[str, Any],
        hazard_type: str,
    ) -> None:
        if (
            state.camera.log_severity_changes
            and risk["overall_changed"]
        ):
            LOGGER.warning(
                "SEVERITY_CHANGED | camera=%s "
                "incident=%s previous=%s current=%s "
                "hazard=%s",
                state.camera.camera_id,
                (
                    state.incident.incident_id
                    if state.incident
                    else "none"
                ),
                risk[
                    "previous_overall"
                ].label,
                risk["overall"].label,
                hazard_type,
            )

        if (
            state.camera.log_risk_windows
            and packet.captured_monotonic
            - state.risk.last_summary_log_at
            >= state.camera.severity_log_interval_seconds
        ):
            state.risk.last_summary_log_at = (
                packet.captured_monotonic
            )
            LOGGER.info(
                "RISK_WINDOW | camera=%s incident=%s "
                "window=%.1fs fire=%d/%d ratio=%.3f "
                "avg=%.3f smoke=%d/%d ratio=%.3f "
                "avg=%.3f severity=%s hazard=%s",
                state.camera.camera_id,
                (
                    state.incident.incident_id
                    if state.incident
                    else "none"
                ),
                state.camera.severity_timeline_seconds,
                risk["fire"].positive_count,
                risk["fire"].total_count,
                risk["fire"].positive_ratio,
                risk["fire"].average_confidence,
                risk["smoke"].positive_count,
                risk["smoke"].total_count,
                risk["smoke"].positive_ratio,
                risk["smoke"].average_confidence,
                risk["overall"].label,
                hazard_type,
            )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    def status(self) -> dict[str, Any]:
        return {
            "model": str(self.model_path),
            "device": self.device,
            "batch_size": self.batch_size,
            "expected_camera_order": list(
                self.camera_order
            ),
            "processed_batches": (
                self.processed_batches
            ),
            "processed_frames": (
                self.processed_frames
            ),
            "last_inference_ms": round(
                self.last_inference_ms,
                3,
            ),
            "last_error": self.last_error,
            "cameras": {
                camera_id: {
                    "severity": (
                        state.risk
                        .overall_severity.label
                    ),
                    "incident_id": (
                        state.incident.incident_id
                        if state.incident
                        else None
                    ),
                    "alert_sent": (
                        state.incident.alert_sent
                        if state.incident
                        else False
                    ),
                    "last_result": (
                        state.last_result
                    ),
                }
                for camera_id, state
                in self.states.items()
            },
        }

    def close(self) -> None:
        if self.show_preview:
            cv2.destroyAllWindows()


class FireSmokeApplication:
    """
    FastAPI background pipeline.

    FastAPI extracts frames first, then passes the ordered list to the detector:
        frames = batch.frames
        detector.process_frames(frames=frames, packets=batch.items, ...)
    """

    def __init__(
        self,
        *,
        extractor: MultiCameraFrameExtractor,
        detector: FireSmokeBatchDetector,
        recorder: QueuedIncidentRecorder,
        events: AsyncEventDispatcher,
        extraction_timeout_seconds: float = 1.0,
    ) -> None:
        self.extractor = extractor
        self.detector = detector
        self.recorder = recorder
        self.events = events
        self.extraction_timeout_seconds = (
            extraction_timeout_seconds
        )

        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.running = False
        self.last_error: str | None = None
        self.last_frame_list: dict[str, Any] = {}

    def start(self) -> None:
        self.events.start()
        self.recorder.start()
        self.extractor.start()
        self.detector.initialize()

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="fastapi-frame-list-pipeline",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        self.running = True

        try:
            while not self.stop_event.is_set():
                batch = (
                    self.extractor.extract_next_batch(
                        self.extraction_timeout_seconds
                    )
                )
                if batch is None:
                    continue

                # Exact requested data path:
                # [fN_video1, fN_video2, fN_video3, ...]
                frames = batch.frames

                self.last_frame_list = {
                    "batch_sequence": (
                        batch.batch_sequence
                    ),
                    "camera_ids": (
                        batch.camera_ids
                    ),
                    "frame_indexes": (
                        batch.frame_indexes
                    ),
                    "frame_count": len(frames),
                }

                self.detector.process_frames(
                    frames=frames,
                    packets=batch.items,
                    batch_sequence=(
                        batch.batch_sequence
                    ),
                    created_monotonic=(
                        batch.created_monotonic
                    ),
                )

        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception(
                "FastAPI frame-list pipeline failed."
            )
        finally:
            self.running = False

    def stop(self) -> None:
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=15)

        self.extractor.stop()
        self.detector.close()
        self.recorder.close()
        self.events.close()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "last_error": self.last_error,
            "last_frame_list": (
                self.last_frame_list
            ),
            "extractor": (
                self.extractor.status()
            ),
            "detector": (
                self.detector.status()
            ),
            "recorder": (
                self.recorder.status()
            ),
        }


def initialize(
    *,
    extractor: MultiCameraFrameExtractor,
    model_path: str | Path,
    cameras: Sequence[
        CameraConfig | Mapping[str, Any]
    ],
    event_handler: EventHandler | None,
    frame_batch_handler: FrameBatchHandler,
    device: int = 0,
    batch_size: int = 8,
    imgsz: int = 640,
    fire_class_id: int = 0,
    smoke_class_id: int = 1,
    fire_candidate_conf: float = 0.30,
    smoke_candidate_conf: float = 0.25,
    show_preview: bool = False,
    frame_queue_size: int = 512,
) -> FireSmokeApplication:
    configs = [
        value
        if isinstance(value, CameraConfig)
        else CameraConfig(**dict(value))
        for value in cameras
    ]

    if not 1 <= len(configs) <= 8:
        raise ValueError(
            "Provide between 1 and 8 detector camera configurations."
        )

    source_order = tuple(
        source.camera_id
        for source in extractor.sources
    )
    detector_order = tuple(
        camera.camera_id
        for camera in configs
    )

    if source_order != detector_order:
        raise ValueError(
            "FastAPI source order and detector camera order "
            f"must match. Sources={source_order}, "
            f"detector={detector_order}."
        )

    events = AsyncEventDispatcher(
        event_handler
    )
    recorder = QueuedIncidentRecorder(
        frame_batch_handler,
        queue_size=frame_queue_size,
    )
    detector = FireSmokeBatchDetector(
        model_path=Path(model_path)
        .expanduser()
        .resolve(),
        cameras=configs,
        device=device,
        batch_size=batch_size,
        imgsz=imgsz,
        fire_class_id=fire_class_id,
        smoke_class_id=smoke_class_id,
        fire_candidate_conf=fire_candidate_conf,
        smoke_candidate_conf=smoke_candidate_conf,
        event_dispatcher=events,
        recorder=recorder,
        show_preview=show_preview,
    )

    return FireSmokeApplication(
        extractor=extractor,
        detector=detector,
        recorder=recorder,
        events=events,
    )
