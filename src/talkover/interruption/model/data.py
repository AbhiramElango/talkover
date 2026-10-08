"""Training and evaluation windows cut from stereo clips (ch0 caller, ch1 agent)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from talkover.interruption.augment import AugmentationPipeline, AugmentContext
from talkover.interruption.model.windows import (
    CLASSES,
    INTERRUPT,
    WindowConfig,
    cut_window,
    envelope_noise,
    model_inputs,
)


@dataclass(frozen=True)
class ClipRow:
    path: Path
    label: str
    act: str
    meeting_id: str


def read_manifest(path: Path) -> list[ClipRow]:
    rows = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        rows.append(ClipRow(path.parent / row["clip"], row["label"], row["act"], row["meeting_id"]))
    return rows


def window_label(row_label: str, decision_seconds: float) -> int:
    return CLASSES.index("ignore") if decision_seconds <= 0 else CLASSES.index(row_label)


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha1(":".join(map(str, parts)).encode()).digest()[:8], "little")


class TrainWindows(Dataset):
    """Clean train clips, freshly augmented per epoch, at a random decision point."""

    def __init__(
        self,
        rows: Sequence[ClipRow],
        speech: AugmentationPipeline,
        silent_ignore: AugmentationPipeline,
        context: AugmentContext,
        config: WindowConfig = WindowConfig(),
        seed: int = 0,
    ) -> None:
        self.rows, self.speech, self.silent_ignore = list(rows), speech, silent_ignore
        self.context, self.config, self.seed = context, config, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        rng = np.random.default_rng(_seed(self.seed, self.epoch, index))
        clip, _ = sf.read(row.path, dtype="float32")
        pipeline = self.silent_ignore if row.act == "crosstalk" else self.speech
        clip, _ = pipeline(clip, self.context, rng)
        config = self.config
        if rng.random() < config.pre_onset_probability:
            decision = float(rng.uniform(config.earliest_pre_onset_seconds, 0.0))
        else:
            decision = float(rng.uniform(config.min_decision_seconds, config.max_decision_seconds))
        return torch.from_numpy(cut_window(clip, decision, config)), window_label(row.label, decision)


class FixedWindows(Dataset):
    """Rendered eval clips at fixed decision points; returns (window, label, clip index, decision)."""

    def __init__(self, rows: Sequence[ClipRow], decisions: Sequence[float], config: WindowConfig = WindowConfig()) -> None:
        self.rows, self.decisions, self.config = list(rows), tuple(decisions), config

    def __len__(self) -> int:
        return len(self.rows) * len(self.decisions)

    def __getitem__(self, index: int):
        clip_index, decision_index = divmod(index, len(self.decisions))
        row = self.rows[clip_index]
        decision = self.decisions[decision_index]
        clip, _ = sf.read(row.path, dtype="float32")
        window = cut_window(clip, decision, self.config)
        return torch.from_numpy(window), window_label(row.label, decision), clip_index, decision


def class_balanced_weights(rows: Sequence[ClipRow]) -> torch.Tensor:
    counts = {label: sum(row.label == label for row in rows) for label in CLASSES}
    return torch.tensor([1.0 / counts[row.label] for row in rows], dtype=torch.double)
