from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class HazardSeverity(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class SeverityThreshold:
    min_count: int
    min_ratio: float
    min_average_confidence: float


@dataclass(frozen=True)
class SeverityPolicy:
    timeline_seconds: float
    low: SeverityThreshold
    medium: SeverityThreshold
    high: SeverityThreshold
    demotion_hold_seconds: float = 5.0
    summary_log_seconds: float = 5.0


@dataclass(frozen=True)
class HazardObservation:
    timestamp: float
    positive: bool
    confidence: float
    affected_area_ratio: float
    contributing_track_count: int


@dataclass(frozen=True)
class LabelRiskSnapshot:
    label: str
    severity: HazardSeverity
    positive_count: int
    total_count: int
    positive_ratio: float
    average_confidence: float
    max_confidence: float
    average_area_ratio: float
    contributing_track_count: int


@dataclass
class LabelRiskState:
    observations: deque[HazardObservation] = field(default_factory=deque)
    severity: HazardSeverity = HazardSeverity.NONE
    lower_candidate: HazardSeverity | None = None
    lower_candidate_since: float | None = None


@dataclass
class CameraRiskState:
    fire: LabelRiskState = field(default_factory=LabelRiskState)
    smoke: LabelRiskState = field(default_factory=LabelRiskState)
    overall_severity: HazardSeverity = HazardSeverity.NONE
    last_summary_log_at: float = 0.0


class SeverityAnalyzer:
    def __init__(
        self,
        *,
        fire_policy: SeverityPolicy,
        smoke_policy: SeverityPolicy,
    ) -> None:
        self.fire_policy = fire_policy
        self.smoke_policy = smoke_policy

    @staticmethod
    def _trim(
        state: LabelRiskState,
        now: float,
        timeline_seconds: float,
    ) -> None:
        cutoff = now - timeline_seconds
        while state.observations and state.observations[0].timestamp < cutoff:
            state.observations.popleft()

    @staticmethod
    def _snapshot(
        label: str,
        state: LabelRiskState,
    ) -> LabelRiskSnapshot:
        values = list(state.observations)
        positives = [value for value in values if value.positive]
        total_count = len(values)
        positive_count = len(positives)
        positive_ratio = (
            positive_count / total_count
            if total_count
            else 0.0
        )
        average_confidence = (
            sum(value.confidence for value in positives) / positive_count
            if positive_count
            else 0.0
        )
        max_confidence = max(
            (value.confidence for value in positives),
            default=0.0,
        )
        average_area_ratio = (
            sum(value.affected_area_ratio for value in positives)
            / positive_count
            if positive_count
            else 0.0
        )
        contributing_track_count = max(
            (
                value.contributing_track_count
                for value in positives
            ),
            default=0,
        )

        return LabelRiskSnapshot(
            label=label,
            severity=state.severity,
            positive_count=positive_count,
            total_count=total_count,
            positive_ratio=positive_ratio,
            average_confidence=average_confidence,
            max_confidence=max_confidence,
            average_area_ratio=average_area_ratio,
            contributing_track_count=contributing_track_count,
        )

    @staticmethod
    def _candidate(
        snapshot: LabelRiskSnapshot,
        policy: SeverityPolicy,
    ) -> HazardSeverity:
        for severity, threshold in (
            (HazardSeverity.HIGH, policy.high),
            (HazardSeverity.MEDIUM, policy.medium),
            (HazardSeverity.LOW, policy.low),
        ):
            if (
                snapshot.positive_count >= threshold.min_count
                and snapshot.positive_ratio >= threshold.min_ratio
                and snapshot.average_confidence
                >= threshold.min_average_confidence
            ):
                return severity

        return HazardSeverity.NONE

    def _update_label(
        self,
        *,
        label: str,
        state: LabelRiskState,
        policy: SeverityPolicy,
        observation: HazardObservation,
    ) -> tuple[LabelRiskSnapshot, HazardSeverity, HazardSeverity]:
        now = observation.timestamp
        state.observations.append(observation)
        self._trim(state, now, policy.timeline_seconds)

        before = state.severity
        candidate = self._candidate(
            self._snapshot(label, state),
            policy,
        )

        if candidate >= state.severity:
            state.severity = candidate
            state.lower_candidate = None
            state.lower_candidate_since = None
        else:
            # The hold timer applies to one specific lower candidate.
            # A change from medium candidate to none restarts the timer.
            if candidate != state.lower_candidate:
                state.lower_candidate = candidate
                state.lower_candidate_since = now
            elif (
                state.lower_candidate_since is not None
                and now - state.lower_candidate_since
                >= policy.demotion_hold_seconds
            ):
                state.severity = candidate
                state.lower_candidate = None
                state.lower_candidate_since = None

        return (
            self._snapshot(label, state),
            before,
            state.severity,
        )

    def update(
        self,
        *,
        state: CameraRiskState,
        timestamp: float,
        fire_confidence: float,
        smoke_confidence: float,
        fire_area_ratio: float,
        smoke_area_ratio: float,
        fire_track_count: int,
        smoke_track_count: int,
        fire_positive_threshold: float,
        smoke_positive_threshold: float,
    ) -> dict[str, Any]:
        fire_snapshot, fire_before, fire_after = self._update_label(
            label="fire",
            state=state.fire,
            policy=self.fire_policy,
            observation=HazardObservation(
                timestamp=timestamp,
                positive=fire_confidence >= fire_positive_threshold,
                confidence=fire_confidence,
                affected_area_ratio=fire_area_ratio,
                contributing_track_count=fire_track_count,
            ),
        )

        smoke_snapshot, smoke_before, smoke_after = self._update_label(
            label="smoke",
            state=state.smoke,
            policy=self.smoke_policy,
            observation=HazardObservation(
                timestamp=timestamp,
                positive=smoke_confidence >= smoke_positive_threshold,
                confidence=smoke_confidence,
                affected_area_ratio=smoke_area_ratio,
                contributing_track_count=smoke_track_count,
            ),
        )

        previous_overall = state.overall_severity
        overall = max(fire_after, smoke_after)

        if (
            fire_after >= HazardSeverity.MEDIUM
            and smoke_after >= HazardSeverity.MEDIUM
        ):
            overall = HazardSeverity.HIGH

        state.overall_severity = overall

        return {
            "fire": fire_snapshot,
            "smoke": smoke_snapshot,
            "fire_changed": fire_before != fire_after,
            "smoke_changed": smoke_before != smoke_after,
            "previous_overall": previous_overall,
            "overall": overall,
            "overall_changed": previous_overall != overall,
        }

    @staticmethod
    def snapshot_dict(
        snapshot: LabelRiskSnapshot,
    ) -> dict[str, Any]:
        return {
            "severity": snapshot.severity.label,
            "positive_count": snapshot.positive_count,
            "total_count": snapshot.total_count,
            "positive_ratio": round(snapshot.positive_ratio, 6),
            "average_confidence": round(snapshot.average_confidence, 6),
            "max_confidence": round(snapshot.max_confidence, 6),
            "average_area_ratio": round(snapshot.average_area_ratio, 6),
            "contributing_track_count": (
                snapshot.contributing_track_count
            ),
        }
