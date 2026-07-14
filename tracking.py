from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TrackerSettings:
    track_iou: float = 0.20
    track_center_distance: float = 0.75
    bbox_smoothing_alpha: float = 0.65
    confidence_ema_alpha: float = 0.35


@dataclass(frozen=True)
class TrackerPolicy:
    rolling_history: int = 15
    required_positive_detections: int = 8
    required_positive_ratio: float = 0.60
    consecutive_detections: int = 3
    fire_average_confidence: float = 0.45
    smoke_average_confidence: float = 0.40
    alert_release_after_missing: int = 10
    track_removal_after_missing: int = 18


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: np.ndarray


@dataclass
class Track:
    track_id: int
    class_id: int
    label: str
    bbox: np.ndarray
    first_frame: int
    last_frame: int
    first_source_time_s: float
    last_source_time_s: float
    confidence_ema: float
    max_confidence: float

    age_updates: int = 1
    total_hits: int = 1
    consecutive_hits: int = 1
    missed_updates: int = 0
    confidence_sum: float = 0.0
    confirmed: bool = False
    alert_active: bool = False
    ever_alerted: bool = False
    confidence_history: deque[float] = field(default_factory=deque)
    hit_history: deque[int] = field(default_factory=deque)

    def history_hits(self) -> int:
        return int(sum(self.hit_history))

    def history_ratio(self) -> float:
        return (
            self.history_hits() / len(self.hit_history)
            if self.hit_history
            else 0.0
        )

    def history_average_confidence(self) -> float:
        positives = [value for value in self.confidence_history if value > 0.0]
        return float(sum(positives) / len(positives)) if positives else 0.0


@dataclass(frozen=True)
class TrackTransition:
    status: str
    track_id: int
    label: str
    frame_index: int
    source_time_s: float
    reason: str


class StableObjectTracker:
    def __init__(
        self,
        settings: TrackerSettings,
        policy: TrackerPolicy,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(
            0.0, float(a[3] - a[1])
        )
        area_b = max(0.0, float(b[2] - b[0])) * max(
            0.0, float(b[3] - b[1])
        )
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _center_distance_ratio(a: np.ndarray, b: np.ndarray) -> float:
        ax = float(a[0] + a[2]) * 0.5
        ay = float(a[1] + a[3]) * 0.5
        bx = float(b[0] + b[2]) * 0.5
        by = float(b[1] + b[3]) * 0.5
        distance = math.hypot(ax - bx, ay - by)

        diagonal_a = math.hypot(
            max(1.0, float(a[2] - a[0])),
            max(1.0, float(a[3] - a[1])),
        )
        diagonal_b = math.hypot(
            max(1.0, float(b[2] - b[0])),
            max(1.0, float(b[3] - b[1])),
        )
        return distance / max(diagonal_a, diagonal_b, 1.0)

    def _confirmation_threshold(self, label: str) -> float:
        if label == "fire":
            return self.policy.fire_average_confidence
        return self.policy.smoke_average_confidence

    def _is_confirmed(self, track: Track) -> bool:
        return (
            track.total_hits >= self.policy.required_positive_detections
            and track.history_hits()
            >= self.policy.required_positive_detections
            and track.consecutive_hits
            >= self.policy.consecutive_detections
            and track.history_ratio()
            >= self.policy.required_positive_ratio
            and track.history_average_confidence()
            >= self._confirmation_threshold(track.label)
        )

    def _create(
        self,
        detection: Detection,
        frame_index: int,
        source_time_s: float,
    ) -> None:
        confidence_history: deque[float] = deque(
            [detection.confidence],
            maxlen=self.policy.rolling_history,
        )
        hit_history: deque[int] = deque(
            [1],
            maxlen=self.policy.rolling_history,
        )

        self.tracks[self.next_track_id] = Track(
            track_id=self.next_track_id,
            class_id=detection.class_id,
            label=detection.label,
            bbox=detection.bbox.astype(np.float32, copy=True),
            first_frame=frame_index,
            last_frame=frame_index,
            first_source_time_s=source_time_s,
            last_source_time_s=source_time_s,
            confidence_ema=detection.confidence,
            max_confidence=detection.confidence,
            confidence_sum=detection.confidence,
            confidence_history=confidence_history,
            hit_history=hit_history,
        )
        self.next_track_id += 1

    def _match(
        self,
        track: Track,
        detection: Detection,
        frame_index: int,
        source_time_s: float,
    ) -> None:
        bbox_alpha = self.settings.bbox_smoothing_alpha
        confidence_alpha = self.settings.confidence_ema_alpha

        track.bbox = (
            bbox_alpha * detection.bbox
            + (1.0 - bbox_alpha) * track.bbox
        ).astype(np.float32)

        track.confidence_ema = (
            confidence_alpha * detection.confidence
            + (1.0 - confidence_alpha) * track.confidence_ema
        )
        track.max_confidence = max(track.max_confidence, detection.confidence)
        track.confidence_sum += detection.confidence
        track.last_frame = frame_index
        track.last_source_time_s = source_time_s
        track.age_updates += 1
        track.total_hits += 1
        track.consecutive_hits += 1
        track.missed_updates = 0
        track.confidence_history.append(detection.confidence)
        track.hit_history.append(1)

    @staticmethod
    def _mark_missed(
        track: Track,
        frame_index: int,
        source_time_s: float,
    ) -> None:
        track.last_frame = frame_index
        track.last_source_time_s = source_time_s
        track.age_updates += 1
        track.consecutive_hits = 0
        track.missed_updates += 1
        track.confidence_history.append(0.0)
        track.hit_history.append(0)

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        source_time_s: float,
    ) -> tuple[list[Track], list[TrackTransition]]:
        existing_ids = list(self.tracks)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        candidates: list[tuple[float, int, int]] = []

        for track_id in existing_ids:
            track = self.tracks[track_id]

            for detection_index, detection in enumerate(detections):
                if detection.label != track.label:
                    continue

                iou = self._iou(track.bbox, detection.bbox)
                center_ratio = self._center_distance_ratio(
                    track.bbox, detection.bbox
                )

                if (
                    iou < self.settings.track_iou
                    and center_ratio > self.settings.track_center_distance
                ):
                    continue

                center_score = max(
                    0.0,
                    1.0
                    - center_ratio
                    / max(self.settings.track_center_distance, 1e-6),
                )
                candidates.append(
                    (max(iou, center_score), track_id, detection_index)
                )

        candidates.sort(key=lambda item: item[0], reverse=True)

        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            self._match(
                self.tracks[track_id],
                detections[detection_index],
                frame_index,
                source_time_s,
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id in existing_ids:
            if track_id not in matched_tracks:
                self._mark_missed(
                    self.tracks[track_id],
                    frame_index,
                    source_time_s,
                )

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._create(detection, frame_index, source_time_s)

        transitions: list[TrackTransition] = []

        for track in list(self.tracks.values()):
            track.confirmed = self._is_confirmed(track)

            if track.confirmed and not track.alert_active:
                track.alert_active = True
                track.ever_alerted = True
                transitions.append(
                    TrackTransition(
                        status="started",
                        track_id=track.track_id,
                        label=track.label,
                        frame_index=frame_index,
                        source_time_s=source_time_s,
                        reason="stable_track_confirmed",
                    )
                )

            if (
                track.alert_active
                and track.missed_updates
                >= self.policy.alert_release_after_missing
            ):
                track.alert_active = False
                transitions.append(
                    TrackTransition(
                        status="ended",
                        track_id=track.track_id,
                        label=track.label,
                        frame_index=frame_index,
                        source_time_s=source_time_s,
                        reason="missing_release_threshold",
                    )
                )

        for track_id, track in list(self.tracks.items()):
            if (
                track.missed_updates
                <= self.policy.track_removal_after_missing
            ):
                continue

            if track.alert_active:
                transitions.append(
                    TrackTransition(
                        status="ended",
                        track_id=track.track_id,
                        label=track.label,
                        frame_index=frame_index,
                        source_time_s=source_time_s,
                        reason="track_expired",
                    )
                )
            del self.tracks[track_id]

        return list(self.tracks.values()), transitions

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
