"""Cluster bootstrap confidence intervals for rate metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RateEstimate:
    value: float
    low: float
    high: float
    successes: int
    total: int


def cluster_bootstrap_rate(
    outcomes: Sequence[bool],
    clusters: Sequence[str],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RateEstimate:
    """Rate with a percentile CI that resamples whole clusters (e.g. meetings)."""

    if len(outcomes) != len(clusters):
        raise ValueError("outcomes and clusters must align")
    if not outcomes:
        return RateEstimate(float("nan"), float("nan"), float("nan"), 0, 0)
    grouped: dict[str, list[bool]] = defaultdict(list)
    for outcome, cluster in zip(outcomes, clusters):
        grouped[cluster].append(outcome)
    hits = np.array([sum(values) for values in grouped.values()], dtype=float)
    sizes = np.array([len(values) for values in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sizes), size=(resamples, len(sizes)))
    rates = hits[picks].sum(axis=1) / np.maximum(sizes[picks].sum(axis=1), 1)
    tail = (1 - confidence) / 2
    return RateEstimate(
        value=float(hits.sum() / sizes.sum()),
        low=float(np.quantile(rates, tail)),
        high=float(np.quantile(rates, 1 - tail)),
        successes=int(hits.sum()),
        total=int(sizes.sum()),
    )
