from __future__ import annotations

import asyncio
import inspect
import logging
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

import cv2
import numpy as np

from frame_extractor import SynchronizedFrameBatch


LOGGER = logging.getLogger("incident-recorder")
FrameBatchHandler = Callable[
    [dict[str, Any], list[bytes]],
    None | Awaitable[None],
]


@dataclass(frozen=True)
class RecordingPolicy:
    pre_alert_seconds: float = 5.0
    post_alert_seconds: float = 5.0
    frame_batch_size: int = 8
    frame_stride: int = 1
    jpeg_quality: int = 80
    max_width: int | None = 1280


@dataclass(frozen=True)
class EncodedFrame:
    jpeg: bytes
    frame_index: int
    source_time_seconds: float
    captured_monotonic: float
    loop_index: int


@dataclass(frozen=True)
class CaptureBatchTask:
    batch: SynchronizedFrameBatch
    policies: Mapping[str, RecordingPolicy]


@dataclass(frozen=True)
class StartTask:
    camera_id: str
    camera_name: str
    source: str
    metadata: Mapping[str, Any]
    incident_id: str
    hazard_type: str
    severity: str
    source_fps: float
    policy: RecordingPolicy


@dataclass(frozen=True)
class UpdateTask:
    camera_id: str
    incident_id: str
    hazard_type: str
    severity: str


@dataclass(frozen=True)
class StopTask:
    camera_id: str
    incident_id: str
    hazard_type: str
    severity: str


@dataclass
class ActiveRecording:
    camera_id: str
    camera_name: str
    source: str
    metadata: Mapping[str, Any]
    incident_id: str
    hazard_type: str
    severity: str
    source_fps: float
    policy: RecordingPolicy
    pending: list[EncodedFrame] = field(default_factory=list)
    batch_index: int = 0
    first_pending: bool = True


class QueuedIncidentRecorder:
    _STOP = object()

    def __init__(
        self,
        handler: FrameBatchHandler,
        queue_size: int = 512,
    ) -> None:
        self.handler = handler
        self.queue: queue.Queue[object] = queue.Queue(
            maxsize=queue_size
        )
        self.thread: threading.Thread | None = None
        self.closed = False
        self.prebuffers: dict[
            str,
            deque[EncodedFrame],
        ] = {}
        self.active: dict[
            str,
            ActiveRecording,
        ] = {}
        self.lock = threading.Lock()
        self.metrics = {
            "submitted_batches": 0,
            "encoded_frames": 0,
            "delivered_frames": 0,
            "delivered_batches": 0,
            "dropped_batches": 0,
            "errors": 0,
        }

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._worker,
            name="incident-recorder",
            daemon=True,
        )
        self.thread.start()

    def capture_batch(
        self,
        batch: SynchronizedFrameBatch,
        policies: Mapping[str, RecordingPolicy],
    ) -> None:
        if self.closed:
            return

        try:
            self.queue.put_nowait(
                CaptureBatchTask(
                    batch=batch,
                    policies=policies,
                )
            )
            with self.lock:
                self.metrics["submitted_batches"] += 1
        except queue.Full:
            with self.lock:
                self.metrics["dropped_batches"] += 1

    def start_incident(self, **kwargs: Any) -> None:
        self.queue.put(StartTask(**kwargs))

    def update_incident(self, **kwargs: Any) -> None:
        self.queue.put(UpdateTask(**kwargs))

    def stop_incident(self, **kwargs: Any) -> None:
        self.queue.put(StopTask(**kwargs))

    def close(self, timeout: float = 20.0) -> None:
        if self.closed:
            return

        self.closed = True
        self.queue.put(self._STOP)

        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                **self.metrics,
                "queue_depth": self.queue.qsize(),
                "active_incidents": len(self.active),
            }

    def _call_handler(
        self,
        payload: dict[str, Any],
        frames: list[bytes],
    ) -> None:
        result = self.handler(payload, frames)
        if inspect.isawaitable(result):
            asyncio.run(result)

    @staticmethod
    def _resize(
        frame: np.ndarray,
        max_width: int | None,
    ) -> np.ndarray:
        if (
            max_width is None
            or max_width <= 0
            or frame.shape[1] <= max_width
        ):
            return frame

        scale = max_width / frame.shape[1]
        return cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    def _encode(
        self,
        frame: np.ndarray,
        packet: Any,
        policy: RecordingPolicy,
    ) -> EncodedFrame:
        resized = self._resize(frame, policy.max_width)
        ok, encoded = cv2.imencode(
            ".jpg",
            resized,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                policy.jpeg_quality,
            ],
        )
        if not ok:
            raise RuntimeError(
                "Could not JPEG-encode incident frame."
            )

        with self.lock:
            self.metrics["encoded_frames"] += 1

        return EncodedFrame(
            jpeg=encoded.tobytes(),
            frame_index=packet.frame_index,
            source_time_seconds=packet.source_time_seconds,
            captured_monotonic=packet.captured_monotonic,
            loop_index=packet.loop_index,
        )

    @staticmethod
    def _prebuffer_capacity(
        policy: RecordingPolicy,
        fps: float,
    ) -> int:
        recording_fps = fps / max(policy.frame_stride, 1)
        return max(
            1,
            int(
                math.ceil(
                    recording_fps
                    * policy.pre_alert_seconds
                )
            ),
        )

    def _capture(self, task: CaptureBatchTask) -> None:
        for packet in task.batch.items:
            policy = task.policies[packet.camera_id]

            if packet.frame_index % policy.frame_stride != 0:
                continue

            encoded = self._encode(
                packet.frame,
                packet,
                policy,
            )
            capacity = self._prebuffer_capacity(
                policy,
                packet.source_fps,
            )

            prebuffer = self.prebuffers.get(
                packet.camera_id
            )
            if (
                prebuffer is None
                or prebuffer.maxlen != capacity
            ):
                old = list(prebuffer or ())[-capacity:]
                prebuffer = deque(
                    old,
                    maxlen=capacity,
                )
                self.prebuffers[
                    packet.camera_id
                ] = prebuffer

            prebuffer.append(encoded)
            active = self.active.get(
                packet.camera_id
            )
            if active is not None:
                active.pending.append(encoded)
                self._flush(active)

    def _start(self, task: StartTask) -> None:
        existing = self.active.get(task.camera_id)
        if existing is not None:
            existing.hazard_type = task.hazard_type
            existing.severity = task.severity
            return

        active = ActiveRecording(**task.__dict__)
        active.pending.extend(
            list(
                self.prebuffers.get(
                    task.camera_id,
                    (),
                )
            )
        )
        self.active[task.camera_id] = active
        self._flush(active)

    def _update(self, task: UpdateTask) -> None:
        active = self.active.get(task.camera_id)
        if (
            active is not None
            and active.incident_id == task.incident_id
        ):
            active.hazard_type = task.hazard_type
            active.severity = task.severity

    def _stop(self, task: StopTask) -> None:
        active = self.active.get(task.camera_id)
        if (
            active is None
            or active.incident_id != task.incident_id
        ):
            return

        active.hazard_type = task.hazard_type
        active.severity = task.severity
        self._deliver(
            active,
            active.pending[:],
            is_final=True,
        )
        del self.active[task.camera_id]

    def _flush(self, active: ActiveRecording) -> None:
        size = active.policy.frame_batch_size
        while len(active.pending) >= size:
            batch = active.pending[:size]
            del active.pending[:size]
            self._deliver(
                active,
                batch,
                is_final=False,
            )

    def _deliver(
        self,
        active: ActiveRecording,
        batch: list[EncodedFrame],
        *,
        is_final: bool,
    ) -> None:
        payload = {
            "schema_version": 5,
            "event_type": "incident_frame_batch",
            "incident_id": active.incident_id,
            "hazard_type": active.hazard_type,
            "severity": active.severity,
            "camera": {
                "camera_id": active.camera_id,
                "name": active.camera_name,
                "source": active.source,
                "metadata": dict(active.metadata),
            },
            "batch_index": active.batch_index,
            "is_first": active.first_pending,
            "is_final": is_final,
            "frame_count": len(batch),
            "recording_fps": (
                active.source_fps
                / max(
                    active.policy.frame_stride,
                    1,
                )
            ),
            "frames": [
                {
                    "position": index,
                    "frame_index": frame.frame_index,
                    "source_time_seconds": round(
                        frame.source_time_seconds,
                        3,
                    ),
                    "captured_monotonic": round(
                        frame.captured_monotonic,
                        6,
                    ),
                    "loop_index": frame.loop_index,
                    "content_type": "image/jpeg",
                }
                for index, frame in enumerate(batch)
            ],
        }

        self._call_handler(
            payload,
            [frame.jpeg for frame in batch],
        )
        active.batch_index += 1
        active.first_pending = False

        with self.lock:
            self.metrics["delivered_frames"] += len(
                batch
            )
            self.metrics["delivered_batches"] += 1

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is self._STOP:
                    for active in list(
                        self.active.values()
                    ):
                        self._deliver(
                            active,
                            active.pending[:],
                            is_final=True,
                        )
                    self.active.clear()
                    return

                if isinstance(item, CaptureBatchTask):
                    self._capture(item)
                elif isinstance(item, StartTask):
                    self._start(item)
                elif isinstance(item, UpdateTask):
                    self._update(item)
                elif isinstance(item, StopTask):
                    self._stop(item)

            except Exception:
                with self.lock:
                    self.metrics["errors"] += 1
                LOGGER.exception(
                    "Incident recorder worker failed."
                )
            finally:
                self.queue.task_done()
