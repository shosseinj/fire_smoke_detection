from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


LOGGER = logging.getLogger("frame-extractor")


@dataclass(frozen=True)
class CameraSourceConfig:
    """
    Source configuration owned only by FastAPI's frame extractor.

    The fire/smoke detector never opens this path or RTSP URL.
    """

    camera_id: str
    name: str
    source: str
    enabled: bool = True
    loop: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reconnect_seconds: float = 1.0
    reader_queue_size: int = 3


@dataclass(frozen=True)
class ExtractedFrame:
    camera_id: str
    camera_name: str
    source: str
    frame: np.ndarray
    camera_sequence: int
    frame_index: int
    source_time_seconds: float
    captured_monotonic: float
    loop_index: int
    source_fps: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SynchronizedFrameBatch:
    """
    One ordered extraction round.

    Example with four configured cameras:
        batch 1 -> [f1_v1, f1_v2, f1_v3, f1_v4]
        batch 2 -> [f2_v1, f2_v2, f2_v3, f2_v4]
    """

    batch_sequence: int
    created_monotonic: float
    items: tuple[ExtractedFrame, ...]
    configured_camera_ids: tuple[str, ...]

    @property
    def frames(self) -> list[np.ndarray]:
        return [item.frame for item in self.items]

    @property
    def camera_ids(self) -> list[str]:
        return [item.camera_id for item in self.items]

    @property
    def frame_indexes(self) -> list[int]:
        return [item.frame_index for item in self.items]


class CameraFrameReader:
    """
    Reads one source in its own thread.

    A small bounded queue prevents unlimited latency. When the queue is full,
    the oldest frame is dropped so CCTV processing stays close to real time.
    """

    _STOP = object()

    def __init__(self, config: CameraSourceConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frames: queue.Queue[ExtractedFrame | object] = queue.Queue(
            maxsize=max(1, config.reader_queue_size)
        )

        self._camera_sequence = 0
        self._frame_index = -1
        self._loop_index = 0
        self._source_fps = 25.0
        self._connected = False
        self._last_error: str | None = None
        self._dropped_frames = 0
        self._is_static_file = Path(config.source).expanduser().exists()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-reader-{self.config.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._release()

    def get_next(self, timeout: float) -> ExtractedFrame | None:
        try:
            value = self._frames.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

        try:
            if value is self._STOP:
                return None
            assert isinstance(value, ExtractedFrame)
            return value
        finally:
            self._frames.task_done()

    def status(self) -> dict[str, Any]:
        return {
            "camera_id": self.config.camera_id,
            "name": self.config.name,
            "source": self.config.source,
            "connected": self._connected,
            "last_error": self._last_error,
            "queue_depth": self._frames.qsize(),
            "camera_sequence": self._camera_sequence,
            "frame_index": self._frame_index,
            "loop_index": self._loop_index,
            "source_fps": round(self._source_fps, 3),
            "dropped_frames": self._dropped_frames,
        }

    def _open(self) -> bool:
        self._release()

        source_value: str | int = self.config.source
        if self.config.source.isdigit():
            source_value = int(self.config.source)

        capture = cv2.VideoCapture(source_value)
        if not capture.isOpened():
            capture.release()
            self._connected = False
            self._last_error = f"Could not open source: {self.config.source}"
            return False

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if np.isfinite(fps) and fps > 0.0:
            self._source_fps = fps

        self._capture = capture
        self._connected = True
        self._last_error = None
        return True

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._connected = False

    def _restart_static_source(self) -> bool:
        if self._capture is None:
            return False
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_index = -1
        self._loop_index += 1
        return True

    def _offer_frame(self, packet: ExtractedFrame) -> None:
        try:
            self._frames.put_nowait(packet)
            return
        except queue.Full:
            pass

        # Drop the oldest queued frame, then keep the newest frame.
        try:
            old = self._frames.get_nowait()
            self._frames.task_done()
            if old is not self._STOP:
                self._dropped_frames += 1
        except queue.Empty:
            pass

        try:
            self._frames.put_nowait(packet)
        except queue.Full:
            self._dropped_frames += 1

    def _run(self) -> None:
        next_due = time.monotonic()

        while not self._stop.is_set():
            if self._capture is None and not self._open():
                self._stop.wait(self.config.reconnect_seconds)
                continue

            assert self._capture is not None
            ok, frame = self._capture.read()

            if not ok:
                if (
                    self._is_static_file
                    and self.config.loop
                    and self._restart_static_source()
                ):
                    continue

                self._last_error = "Frame read failed; reconnecting."
                self._release()
                self._stop.wait(self.config.reconnect_seconds)
                continue

            self._frame_index += 1
            self._camera_sequence += 1
            captured_at = time.monotonic()
            source_time = (
                self._frame_index / self._source_fps
                if self._source_fps > 0.0
                else 0.0
            )

            self._offer_frame(
                ExtractedFrame(
                    camera_id=self.config.camera_id,
                    camera_name=self.config.name,
                    source=self.config.source,
                    frame=frame,
                    camera_sequence=self._camera_sequence,
                    frame_index=self._frame_index,
                    source_time_seconds=source_time,
                    captured_monotonic=captured_at,
                    loop_index=self._loop_index,
                    source_fps=self._source_fps,
                    metadata=self.config.metadata,
                )
            )

            # Pace static videos as live CCTV.
            if self._is_static_file and self._source_fps > 0.0:
                next_due += 1.0 / self._source_fps
                delay = next_due - time.monotonic()
                if delay > 0.0:
                    self._stop.wait(delay)
                elif delay < -1.0:
                    next_due = time.monotonic()

        self._release()


class MultiCameraFrameExtractor:
    """
    FastAPI-owned synchronized frame extractor.

    The extractor waits for one frame from every configured camera, preserving
    configuration order. It never returns a partial list, so camera/result
    alignment cannot shift.
    """

    def __init__(
        self,
        sources: Sequence[CameraSourceConfig],
    ) -> None:
        enabled = [source for source in sources if source.enabled]
        if not 1 <= len(enabled) <= 8:
            raise ValueError("Provide between 1 and 8 enabled camera sources.")

        ids = [source.camera_id for source in enabled]
        if len(ids) != len(set(ids)):
            raise ValueError("camera_id values must be unique.")

        self.sources = tuple(enabled)
        self.readers = tuple(CameraFrameReader(source) for source in self.sources)
        self._pending: dict[str, ExtractedFrame] = {}
        self._batch_sequence = 0
        self._timeouts = 0

    def start(self) -> None:
        for reader in self.readers:
            reader.start()

    def stop(self) -> None:
        for reader in self.readers:
            reader.stop()

    def extract_next_batch(
        self,
        timeout_seconds: float = 1.0,
    ) -> SynchronizedFrameBatch | None:
        deadline = time.monotonic() + max(0.0, timeout_seconds)

        # Keep already-collected frames in _pending if another camera times out.
        for reader in self.readers:
            camera_id = reader.config.camera_id
            if camera_id in self._pending:
                continue

            remaining = max(0.0, deadline - time.monotonic())
            packet = reader.get_next(remaining)
            if packet is None:
                self._timeouts += 1
                return None

            self._pending[camera_id] = packet

        ordered_items = tuple(
            self._pending[source.camera_id]
            for source in self.sources
        )
        self._pending.clear()
        self._batch_sequence += 1

        return SynchronizedFrameBatch(
            batch_sequence=self._batch_sequence,
            created_monotonic=time.monotonic(),
            items=ordered_items,
            configured_camera_ids=tuple(
                source.camera_id for source in self.sources
            ),
        )

    def status(self) -> dict[str, Any]:
        return {
            "batch_sequence": self._batch_sequence,
            "timeouts": self._timeouts,
            "configured_camera_ids": [
                source.camera_id for source in self.sources
            ],
            "waiting_camera_ids": [
                source.camera_id
                for source in self.sources
                if source.camera_id not in self._pending
            ],
            "pending_camera_ids": list(self._pending),
            "readers": [reader.status() for reader in self.readers],
        }
