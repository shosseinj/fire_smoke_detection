from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


LOGGER = logging.getLogger("async-video-storage")


@dataclass(frozen=True)
class FrameBatchSaveTask:
    payload: dict[str, Any]
    frames: list[bytes]


class AsyncIncidentVideoStorage:
    """
    Asynchronously saves one MP4 file per completed incident.

    Final media directory:
        saved_fire_smoke_videos/
            gate-01__<incident-id>.mp4

    Temporary JPEG frames:
        .fire_smoke_video_spool/<incident-id>/*.jpg

    No event JSON, batch JSON, video-status JSON, or completion JSON is written.
    Temporary JPEG files are removed after successful MP4 finalization.
    """

    _STOP = object()

    def __init__(
        self,
        output_dir: Path,
        *,
        spool_dir: Path | None = None,
        queue_size: int = 512,
        finalizer_workers: int = 2,
        cleanup_spool_after_success: bool = True,
        completed_history_size: int = 100,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.spool_dir = (
            spool_dir.resolve()
            if spool_dir is not None
            else (
                self.output_dir.parent
                / ".fire_smoke_video_spool"
            ).resolve()
        )
        self.spool_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.queue: queue.Queue[
            FrameBatchSaveTask | object
        ] = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self.finalizer = ThreadPoolExecutor(
            max_workers=max(
                1,
                finalizer_workers,
            ),
            thread_name_prefix=(
                "incident-video-finalizer"
            ),
        )
        self.cleanup_spool_after_success = (
            cleanup_spool_after_success
        )

        self.worker: threading.Thread | None = None
        self.closed = False
        self.lock = threading.RLock()

        self.futures: dict[
            str,
            Future[dict[str, Any]],
        ] = {}
        self.incident_frame_counts: dict[
            str,
            int,
        ] = {}
        self.latest_payloads: dict[
            str,
            dict[str, Any],
        ] = {}
        self.completed_videos: deque[
            dict[str, Any]
        ] = deque(
            maxlen=max(
                1,
                completed_history_size,
            )
        )
        self.failed_videos: deque[
            dict[str, Any]
        ] = deque(
            maxlen=max(
                1,
                completed_history_size,
            )
        )

        self.metrics = {
            "queued_batches": 0,
            "spooled_batches": 0,
            "spooled_frames": 0,
            "finalization_queued": 0,
            "finalizing": 0,
            "videos_ready": 0,
            "videos_failed": 0,
            "errors": 0,
        }

    def start(self) -> None:
        if (
            self.worker is not None
            and self.worker.is_alive()
        ):
            return

        self.worker = threading.Thread(
            target=self._spool_worker,
            name="incident-frame-spooler",
            daemon=True,
        )
        self.worker.start()

    def save_frame_batch(
        self,
        payload: dict[str, Any],
        frames: list[bytes],
    ) -> None:
        """
        Queue already JPEG-encoded incident frames.

        The method does not create an MP4 and does not write JSON files.
        """

        if self.closed:
            raise RuntimeError(
                "Video storage is closed."
            )

        task = FrameBatchSaveTask(
            payload=dict(payload),
            frames=list(frames),
        )

        # Blocking queue insertion preserves incident frame order.
        self.queue.put(task)

        with self.lock:
            self.metrics[
                "queued_batches"
            ] += 1

    def status(self) -> dict[str, Any]:
        with self.lock:
            active_jobs = {
                incident_id: (
                    "done"
                    if future.done()
                    else "processing"
                )
                for incident_id, future
                in self.futures.items()
            }

            return {
                **self.metrics,
                "queue_depth": (
                    self.queue.qsize()
                ),
                "output_directory": str(
                    self.output_dir
                ),
                "temporary_spool_directory": (
                    str(self.spool_dir)
                ),
                "active_finalization_jobs": (
                    active_jobs
                ),
                "completed_videos": list(
                    self.completed_videos
                ),
                "failed_videos": list(
                    self.failed_videos
                ),
            }

    def close(
        self,
        timeout: float = 60.0,
    ) -> None:
        if self.closed:
            return

        self.closed = True
        self.queue.put(self._STOP)

        if self.worker is not None:
            self.worker.join(
                timeout=timeout
            )

        self.finalizer.shutdown(
            wait=True,
            cancel_futures=False,
        )

    def _incident_spool(
        self,
        incident_id: str,
    ) -> Path:
        folder = (
            self.spool_dir
            / incident_id
        )
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )
        return folder

    @staticmethod
    def _safe_component(
        value: str,
    ) -> str:
        allowed = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_"
        )
        cleaned = "".join(
            character
            if character in allowed
            else "_"
            for character in value
        )
        return cleaned or "camera"

    def _final_video_path(
        self,
        payload: dict[str, Any],
    ) -> Path:
        camera_id = self._safe_component(
            str(
                payload["camera"][
                    "camera_id"
                ]
            )
        )
        incident_id = self._safe_component(
            str(
                payload[
                    "incident_id"
                ]
            )
        )
        return (
            self.output_dir
            / (
                f"{camera_id}__"
                f"{incident_id}.mp4"
            )
        )

    def _spool_task(
        self,
        task: FrameBatchSaveTask,
    ) -> None:
        payload = task.payload
        incident_id = str(
            payload["incident_id"]
        )
        batch_index = int(
            payload["batch_index"]
        )
        spool = self._incident_spool(
            incident_id
        )

        frame_count = 0

        for position, content in enumerate(
            task.frames
        ):
            frame_path = (
                spool
                / (
                    f"{batch_index:09d}_"
                    f"{position:05d}.jpg"
                )
            )
            frame_path.write_bytes(
                content
            )
            frame_count += 1

        with self.lock:
            self.metrics[
                "spooled_batches"
            ] += 1
            self.metrics[
                "spooled_frames"
            ] += frame_count
            self.incident_frame_counts[
                incident_id
            ] = (
                self.incident_frame_counts.get(
                    incident_id,
                    0,
                )
                + frame_count
            )
            self.latest_payloads[
                incident_id
            ] = dict(payload)

        if bool(
            payload.get("is_final")
        ):
            with self.lock:
                self.metrics[
                    "finalization_queued"
                ] += 1

            LOGGER.info(
                "VIDEO_SAVE_QUEUED | "
                "camera=%s incident=%s "
                "frames=%d",
                payload["camera"][
                    "camera_id"
                ],
                incident_id,
                self.incident_frame_counts.get(
                    incident_id,
                    0,
                ),
            )

            future = (
                self.finalizer.submit(
                    self._finalize_video,
                    dict(payload),
                )
            )

            with self.lock:
                self.futures[
                    incident_id
                ] = future

            future.add_done_callback(
                lambda completed,
                current_incident=(
                    incident_id
                ):
                self._finalization_done(
                    current_incident,
                    completed,
                )
            )

    def _finalization_done(
        self,
        incident_id: str,
        future: Future[
            dict[str, Any]
        ],
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:
            failure = {
                "incident_id": (
                    incident_id
                ),
                "status": "failed",
                "error": str(exc),
            }

            with self.lock:
                self.metrics[
                    "videos_failed"
                ] += 1
                self.metrics[
                    "errors"
                ] += 1
                self.failed_videos.append(
                    failure
                )

            LOGGER.exception(
                "VIDEO_SAVE_FAILED | "
                "incident=%s",
                incident_id,
            )
            return

        with self.lock:
            self.metrics[
                "videos_ready"
            ] += 1
            self.completed_videos.append(
                result
            )

        LOGGER.info(
            "VIDEO_SAVED | "
            "camera=%s incident=%s "
            "frames=%d duration=%.3fs "
            "path=%s",
            result["camera_id"],
            incident_id,
            result["frame_count"],
            result[
                "duration_seconds"
            ],
            result["video_path"],
        )

    def _finalize_video(
        self,
        final_payload: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        incident_id = str(
            final_payload[
                "incident_id"
            ]
        )
        spool = self._incident_spool(
            incident_id
        )

        with self.lock:
            self.metrics[
                "finalizing"
            ] += 1

        started = time.perf_counter()
        final_path = (
            self._final_video_path(
                final_payload
            )
        )
        temporary_path = (
            final_path.with_name(
                final_path.stem
                + ".partial.mp4"
            )
        )

        try:
            frame_paths = sorted(
                spool.glob("*.jpg")
            )

            if not frame_paths:
                raise RuntimeError(
                    "No temporary incident "
                    "frames were found."
                )

            first = cv2.imread(
                str(
                    frame_paths[0]
                ),
                cv2.IMREAD_COLOR,
            )

            if first is None:
                raise RuntimeError(
                    "Could not decode the "
                    "first incident frame."
                )

            height, width = (
                first.shape[:2]
            )
            fps = max(
                1.0,
                float(
                    final_payload[
                        "recording_fps"
                    ]
                ),
            )

            writer = cv2.VideoWriter(
                str(temporary_path),
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                fps,
                (width, height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    "Could not create "
                    f"{temporary_path}"
                )

            written = 0

            try:
                for frame_path in (
                    frame_paths
                ):
                    frame = cv2.imread(
                        str(
                            frame_path
                        ),
                        cv2.IMREAD_COLOR,
                    )

                    if frame is None:
                        continue

                    if (
                        frame.shape[1]
                        != width
                        or frame.shape[0]
                        != height
                    ):
                        frame = cv2.resize(
                            frame,
                            (
                                width,
                                height,
                            ),
                            interpolation=(
                                cv2.INTER_AREA
                            ),
                        )

                    writer.write(frame)
                    written += 1

            finally:
                writer.release()

            if written == 0:
                raise RuntimeError(
                    "No frames were written "
                    "to the MP4 file."
                )

            if final_path.exists():
                final_path.unlink()

            temporary_path.replace(
                final_path
            )

            duration = written / fps
            elapsed = (
                time.perf_counter()
                - started
            )

            result = {
                "incident_id": (
                    incident_id
                ),
                "camera_id": (
                    final_payload[
                        "camera"
                    ][
                        "camera_id"
                    ]
                ),
                "hazard_type": (
                    final_payload[
                        "hazard_type"
                    ]
                ),
                "severity": (
                    final_payload[
                        "severity"
                    ]
                ),
                "status": "ready",
                "video_path": str(
                    final_path
                ),
                "frame_count": written,
                "fps": fps,
                "duration_seconds": round(
                    duration,
                    3,
                ),
                "file_size_bytes": (
                    final_path.stat(
                    ).st_size
                ),
                "finalization_seconds": (
                    round(
                        elapsed,
                        3,
                    )
                ),
            }

            if (
                self.cleanup_spool_after_success
            ):
                shutil.rmtree(
                    spool,
                    ignore_errors=True,
                )

            with self.lock:
                self.incident_frame_counts.pop(
                    incident_id,
                    None,
                )
                self.latest_payloads.pop(
                    incident_id,
                    None,
                )

            return result

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink(
                    missing_ok=True
                )
            raise

        finally:
            with self.lock:
                self.metrics[
                    "finalizing"
                ] = max(
                    0,
                    self.metrics[
                        "finalizing"
                    ]
                    - 1,
                )

    def _spool_worker(
        self,
    ) -> None:
        while True:
            item = self.queue.get()

            try:
                if item is self._STOP:
                    return

                assert isinstance(
                    item,
                    FrameBatchSaveTask,
                )
                self._spool_task(
                    item
                )

            except Exception:
                with self.lock:
                    self.metrics[
                        "errors"
                    ] += 1

                LOGGER.exception(
                    "Incident frame spool "
                    "task failed."
                )

            finally:
                self.queue.task_done()
