"""Interval algebra and confidence intervals for evaluation."""

from talkover.evaluation.bootstrap import RateEstimate, cluster_bootstrap_rate
from talkover.evaluation.intervals import (
    TimeInterval,
    intersection_duration,
    merge_intervals,
    subtract_intervals,
)

__all__ = [
    "RateEstimate",
    "TimeInterval",
    "cluster_bootstrap_rate",
    "intersection_duration",
    "merge_intervals",
    "subtract_intervals",
]
