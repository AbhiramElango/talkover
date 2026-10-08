"""Small interval algebra helpers for timestamp-based audio evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class TimeInterval:
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("start_seconds cannot be negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def merge_intervals(intervals: Iterable[TimeInterval]) -> tuple[TimeInterval, ...]:
    """Merge overlapping or touching intervals into sorted disjoint intervals."""

    ordered = sorted(intervals)
    if not ordered:
        return ()
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start_seconds <= previous.end_seconds:
            merged[-1] = TimeInterval(
                previous.start_seconds,
                max(previous.end_seconds, current.end_seconds),
            )
        else:
            merged.append(current)
    return tuple(merged)


def intersection_duration(
    left: Iterable[TimeInterval], right: Iterable[TimeInterval]
) -> float:
    """Return total duration covered by both sets of intervals."""

    a = merge_intervals(left)
    b = merge_intervals(right)
    i = 0
    j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(
            0.0,
            min(a[i].end_seconds, b[j].end_seconds)
            - max(a[i].start_seconds, b[j].start_seconds),
        )
        if a[i].end_seconds <= b[j].end_seconds:
            i += 1
        else:
            j += 1
    return total


def subtract_intervals(
    base: Iterable[TimeInterval], exclusions: Iterable[TimeInterval]
) -> tuple[TimeInterval, ...]:
    """Remove all excluded coverage from the base interval union."""

    remaining: list[TimeInterval] = []
    blocked = merge_intervals(exclusions)
    for source in merge_intervals(base):
        cursor = source.start_seconds
        for exclusion in blocked:
            if exclusion.end_seconds <= cursor:
                continue
            if exclusion.start_seconds >= source.end_seconds:
                break
            if exclusion.start_seconds > cursor:
                remaining.append(
                    TimeInterval(cursor, min(exclusion.start_seconds, source.end_seconds))
                )
            cursor = max(cursor, exclusion.end_seconds)
            if cursor >= source.end_seconds:
                break
        if cursor < source.end_seconds:
            remaining.append(TimeInterval(cursor, source.end_seconds))
    return tuple(remaining)
