"""TurnBench dev (Mundo AI / Sesame): consensus caller events while the other speaker talks.

Evaluation only (Dataset Public License v1.0: non-commercial, no voice cloning).
"""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from talkover.interruption.events import Label


LABEL_MAP: dict[str, Label | None] = {
    "Floor-taking Cooperative Interruption": "interrupt",
    "Floor-taking Competitive Interruption": "interrupt",
    "Continuer Backchannel": "backchannel",
    "Acknowledgement Backchannel": "backchannel",
    "Reaction Backchannel": "backchannel",
    "Laughter": "ignore",
    "Non-Speech Noise": "ignore",
    "Channel Bleed": "ignore",
    "Speech, Non-Linguistic": "ignore",
}
SPEECH_LABELS = frozenset({
    "Normal Turn", "Strong Floor Hold", "Bounded Response", "Overlap", "Filler",
    "Floor-taking Cooperative Interruption", "Floor-taking Competitive Interruption",
    "Non-floor Taking Cooperative Interruption", "Non-floor Taking Competitive Interruption",
})
ANNOTATORS = "abc"


@dataclass(frozen=True)
class ConsensusEvent:
    conversation_id: str
    caller: int
    onset_seconds: float
    label: Label
    source_labels: tuple[str, ...]


@dataclass(frozen=True)
class ConsensusConfig:
    tolerance_seconds: float = 0.3
    min_annotators: int = 2
    min_agent_lead_seconds: float = 0.5
    min_agent_remaining_seconds: float = 0.5


def consensus_events(row: dict, config: ConsensusConfig = ConsensusConfig()) -> list[ConsensusEvent]:
    events = []
    for caller, agent in ((1, 2), (2, 1)):
        tracks = [row[f"speaker_{caller}_annotation_{a}"] or [] for a in ANNOTATORS]
        agent_spans = [
            (e["start_s"], e["end_s"])
            for a in ANNOTATORS for e in row[f"speaker_{agent}_annotation_{a}"] or [] if e["label"] in SPEECH_LABELS
        ]
        marks = sorted(
            (e["start_s"], annotator, e["label"])
            for annotator, track in enumerate(tracks) for e in track
        )
        used = [False] * len(marks)
        for i, (start, _, _) in enumerate(marks):
            if used[i]:
                continue
            group = [j for j in range(i, len(marks)) if not used[j] and marks[j][0] - start <= config.tolerance_seconds]
            by_annotator = {}
            for j in group:
                by_annotator.setdefault(marks[j][1], j)
            for j in by_annotator.values():
                used[j] = True
            if len(by_annotator) < config.min_annotators:
                continue
            labels = [marks[j][2] for j in by_annotator.values()]
            votes = Counter(LABEL_MAP.get(label) for label in labels)
            label, count = votes.most_common(1)[0]
            if label is None or count < config.min_annotators:
                continue
            onset = float(np.median([marks[j][0] for j in by_annotator.values()]))
            lead, remaining = onset - config.min_agent_lead_seconds, onset + config.min_agent_remaining_seconds
            if any(s <= lead and e >= remaining for s, e in agent_spans):
                events.append(ConsensusEvent(row["conversation_id"], caller, round(onset, 3), label, tuple(sorted(labels))))
    return events


def decode_channel(audio: dict, target_rate: int = 16_000) -> NDArray[np.float32]:
    samples, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != target_rate:
        divisor = np.gcd(rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, rate // divisor)
    return mono.astype(np.float32)


def read_conversations(files: Sequence[Path]) -> Iterator[dict]:
    import pyarrow.parquet as pq

    columns = ["conversation_id", "metadata", "speaker_1_audio", "speaker_2_audio"] + [
        f"speaker_{s}_annotation_{a}" for s in (1, 2) for a in ANNOTATORS
    ]
    for path in files:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            yield from parquet.read_row_group(group, columns=columns).to_pylist()
